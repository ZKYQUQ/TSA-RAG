import argparse
import json
import os
import pandas as pd
from tqdm import tqdm
import torch
import transformers
from typing import List, Dict, Any
import copy
import numpy as np
from transformers import AutoTokenizer

torch.manual_seed(29)

# 定义提示模板
PROMPT_DICT = {
    # asqa, eli5
    "reader": (
        "Instruction: Write an accurate, engaging, and concise answer for the given question. Use an "
        "unbiased and journalistic tone.\n\n"
        "## Paragraph\n{paragraph}\n\n"
        "## Question\n{question}\n\n"
        "## Response"
    ),
    # qampari
    "reader_short_ans": (
        "Instruction: Provide a list of accurate answers for the given question. Separate answers by "
        "commas. Do not explain yourself or output anything else.\n\n"
        "## Paragraph\n{paragraph}\n\n"
        "## Question\n{question}\n\n"
        "## Response"
    ),
    # short form qa
    "reader_short_form": (
        "Instruction: Provide one accurate answer for the given question. "
        "Do not explain yourself or output anything else.\n\n"
        "## Paragraph\n{paragraph}\n\n"
        "## Question\n{question}\n\n"
        "## Response"
    )
    # "reader_short_form": (
    #     "Instruction: Provide one accurate answer for the given question. List the answer as a single line. "
    #     "Do not explain yourself or output anything else.\n\n"
    #     "## Paragraph\n{paragraph}\n\n"
    #     "## Question\n{question}\n\n"
    #     "## Response"
    # )
}

def read_file(input_path):
    """读取输入文件"""
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"不支持的文件格式: {input_path}")
    
    print(f"\n***加载数据集大小: {len(df)}条记录***\n")
    return df

def get_question_field(data_item, input_file):
    """根据数据集类型获取问题字段"""
    if "asqa" in input_file.lower():
        return data_item.get("ambiguous_question", "")
    elif "eli5" in input_file.lower():
        return data_item.get("title", "")
    elif "qampari" in input_file.lower():
        return data_item.get("question_text", "")
    else:
        return data_item["question"]
        # raise ValueError("Unsupported input file type for document subtree extraction")
    # elif "hotpotqa" in input_file.lower():
    #     return data_item.get("question", "")
    # elif "triviaqa" in input_file.lower():
    #     return data_item.get("question", "")
    

def get_context(
    data_item,
    retrieve,
    extract,
    mode=-1,
    # max_context_length=3800,
    max_context_length=4000,
    language_model=None,
    tokenizer=None,
    retrieve_top_k=-1,
    limit_context_window=False,
):
    """根据模式获取上下文
    
    Args:
        data_item: 数据项
        retrieve: 是否使用检索模式
        extract: 是否使用提取模式
        mode: 上下文组织模式：
            1: 匹配pre_retrieved_passages的title，将对应tree_extract_results放在前面
            2: 按total_score对tree_extract_results进行首尾交替排序
        max_context_length: 最大上下文长度（token数量，默认3800）
        language_model: 使用的语言模型路径
        limit_context_window: 是否强制限制上下文长度
    """
    # Llama-2-13b-chat-hf has a 4k context window, so keep the historical
    # default behavior. Other models can opt in with --limit_context_window.
    language_model = language_model or ""
    use_length_limit = limit_context_window or ("Llama-2-13b-chat-hf" in language_model)
    
    # 定义计算token数量的函数
    def count_tokens(text):
        return len(tokenizer.encode(text))
    
    if retrieve and extract:
        # 模式2.3: retrieve+extract+read
        if "tree_extract_results" in data_item and len(data_item["tree_extract_results"]) > 0:
            contexts = []
            current_token_count = 0  # 仅在需要限制长度时使用
            
            if mode == 1 and "pre_retrieved_passages" in data_item and len(data_item["pre_retrieved_passages"]) > 0:
                # 模式1：先匹配pre_retrieved_passages的title
                retrieved_titles = [passage.get('title', '').strip() for passage in data_item["pre_retrieved_passages"]]
                matched_results = []
                unmatched_results = []
                
                for result in data_item["tree_extract_results"]:
                    if "formated_subtree" in result and result["formated_subtree"]:
                        result_title = result.get("title", "").strip()
                        if result_title in retrieved_titles:
                            matched_results.append(result)
                        else:
                            unmatched_results.append(result)
                
                # 先添加匹配到的结果
                for result in matched_results:
                    contexts.append(result["formated_subtree"])
                
                # 再添加未匹配到的结果
                for result in unmatched_results:
                    contexts.append(result["formated_subtree"])
            
            elif mode == 2:
                # 模式2：按total_score进行首尾交替排序
                valid_results = [
                    result for result in data_item["tree_extract_results"]
                    if "formated_subtree" in result and result["formated_subtree"]
                ]
                
                # 先按照total_score排序（从高到低）
                sorted_results = sorted(
                    valid_results, 
                    key=lambda x: x["total_score"],
                    reverse=True  # 从高到低排序
                )
                
                reordered_contexts = []
                
                # 分别获取奇数位置和偶数位置的结果
                n = len(sorted_results)
                odd_indices = list(range(0, n, 2))  # 0, 2, 4...
                even_indices = list(range(1, n, 2))  # 1, 3, 5...
                
                # 先添加所有奇数位置（高分组）
                for i in odd_indices:
                    reordered_contexts.append(sorted_results[i]["formated_subtree"])
                
                # 再添加所有偶数位置（低分组）- 但是是倒序添加
                for i in reversed(even_indices):
                    reordered_contexts.append(sorted_results[i]["formated_subtree"])
                
                contexts = reordered_contexts
            
            else:
                # 原始模式：直接添加所有结果，但为Llama-2-13b限制长度
                for result in data_item["tree_extract_results"]:
                    if "formated_subtree" in result and result["formated_subtree"]:
                        if use_length_limit:
                            subtree = result["formated_subtree"]
                            subtree_token_count = count_tokens(subtree)
                            
                            if current_token_count + subtree_token_count <= max_context_length:
                                contexts.append(subtree)
                                current_token_count += subtree_token_count
                            else:
                                remaining_tokens = max_context_length - current_token_count
                                # 精确截取token
                                tokens = tokenizer.encode(subtree)
                                partial_tokens = tokens[:remaining_tokens]
                                partial_subtree = tokenizer.decode(partial_tokens, skip_special_tokens=True)
                                contexts.append(partial_subtree)
                                break
                        else:
                            contexts.append(result["formated_subtree"])
            
            return "\n\n".join(contexts)
        return ""
    
    elif retrieve:
        # 模式2.2: retrieve+read
        if retrieve_top_k > 0:
            contexts = []
            for passage in data_item["pre_retrieved_passages"][:retrieve_top_k]:
                passage_text = f"{passage.get('title', '')}\n{passage.get('text', '')}"
                contexts.append(passage_text)
            return "\n\n".join(contexts)
        if "pre_retrieved_passages" in data_item and len(data_item["pre_retrieved_passages"]) > 0:
            contexts = []
            for passage in data_item["pre_retrieved_passages"]:
                passage_text = f"{passage.get('title', '')}\n{passage.get('text', '')}"
                contexts.append(passage_text)
            return "\n\n".join(contexts)
        return ""
    
    else:
        # 模式2.1: 只有read
        return ""

def load_existing_results(output_file):
    """加载已有的结果文件"""
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                
                if "data" in existing_data:
                    # 格式化的数据集结果
                    processed_ids = set()
                    for item in existing_data["data"]:
                        if "id" in item:
                            processed_ids.add(item["id"])
                        elif "sample_id" in item:
                            processed_ids.add(item["sample_id"])
                    return existing_data, processed_ids
                else:
                    # 简单列表格式结果
                    if isinstance(existing_data, list):
                        processed_ids = set()
                        for item in existing_data:
                            processed_ids.add(item["id"])
                        return existing_data, processed_ids
                    else:
                        raise ValueError("Invalid existing data format")
        except Exception as e:
            print(f"加载现有结果失败: {e}")
    
    return None, set()

def format_asqa_result(row, question, answer):
    """格式化ASQA结果 - 严格按照format_eval.py的格式"""
    # 准备qa_pairs
    qa_pairs = []
    if "qa_pairs" in row:
        qa_pairs = [{
            # "context": qa_pair.get("context", ""),
            "question": qa_pair["question"],
            "short_answers": qa_pair["short_answers"].tolist(),
            # "wikipage": qa_pair.get("wikipage", ""),
        } for qa_pair in row["qa_pairs"]]
    else:
        raise ValueError("ASQA数据集缺少qa_pairs字段")
    
    # 准备annotations
    annotations = []
    if "annotations" in row:
        annotations = [{
            "knowledge": annotation["knowledge"].tolist(),
            "long_answer": annotation["long_answer"],
        } for annotation in row["annotations"]]
    
    # # 准备docs
    # if "pre_retrieved_passages" in row:
    #     docs = row["pre_retrieved_passages"]
    # else:
    #     docs = []
    
    result = {
        "sample_id": row["sample_id"],
        "question": question.strip(),
        "qa_pairs": qa_pairs,
        "annotations": annotations,
        # "docs": docs,
        "answer": row["annotations"][0]["long_answer"],
        "output": answer.strip(),
    }
    return result

def format_eli5_result(row, question, answer):
    """格式化ELI5结果 - 严格按照format_eval.py的格式"""
    # 处理answers
    if "answers" in row and isinstance(row["answers"], dict):
        a_ids = row["answers"].get("a_id", [])
        a_texts = row["answers"].get("text", [])
        a_scores = row["answers"].get("score", [])
        
        if len(a_scores) > 0 and len(a_texts) > 0:
            answer_idx = np.argmax(a_scores)
            original_answer = a_texts[answer_idx]
        else:
            raise ValueError("ELI5数据集缺少有效的答案")
    else:
        raise ValueError("ELI5数据集缺少有效的答案")
    
    # 处理claims
    claims = row["claims"].tolist()
    
    # # 准备docs
    # if "pre_retrieved_passages" in row:
    #     docs = row["pre_retrieved_passages"]
    # else:
    #     docs = []
    
    result = {
        "id": row["q_id"],
        "question": question.strip(),
        "answer": original_answer.strip(),
        # "docs": docs,
        "output": answer.strip(),
        "claims": claims,
    }
    return result

def format_qampari_result(row, question, answer):
    """格式化QAMPARI结果 - 严格按照format_eval.py的格式"""
    # 处理answers
    answers = []
    if "answer_list" in row:
        for ans in row["answer_list"]:
            local_answers = ans["aliases"].tolist()
            local_answers.append(ans["answer_text"])
            answers.append(list(set(local_answers)))
    else:
        raise ValueError("QAMPARI数据集缺少有效的答案")
    
    # # 准备docs
    # if "pre_retrieved_passages" in row:
    #     docs = row["pre_retrieved_passages"]
    # else:
    #     docs = []
    
    result = {
        "id": str(row.get("qid", "") or row.get("id", "")),
        "question": question.strip(),
        "answers": answers,
        # "docs": docs,
        "output": answer.strip(),
    }
    return result

def format_generic_result(row, question, prompt, answer):
    """格式化通用结果"""
    result = {
        "id": row["id"],
        "question": question,
        "output": answer,
        "golden_answers": row["golden_answers"].tolist()
    }
    return result

def save_batch_results(results, output_file, dataset_type):
    """保存批次结果"""
    if not results:
        return

    existing_data, _ = load_existing_results(output_file)
    
    if dataset_type in ["asqa", "eli5", "qampari"]:
        # 格式化的数据集
        if existing_data is None:
            # 创建新的文件
            output_data = {
                "data": results,
                "config": {}
            }
        else:
            # 添加到现有文件
            output_data = copy.deepcopy(existing_data)
            output_data["data"].extend(results)
    else:
        # 简单列表格式
        if existing_data is None:
            output_data = results
        else:
            output_data = existing_data + results
    
    # 创建临时文件
    temp_file = f"{output_file}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # 原子替换
    if os.path.exists(output_file):
        os.replace(temp_file, output_file)
    else:
        os.rename(temp_file, output_file)
    
    print(f"保存了 {len(results)} 条结果到 {output_file}")

def inference(
    input_file, 
    output_file, 
    language_model,
    retrieve=False,
    extract=False,
    read=True,
    max_new_tokens=256,
    batch_size=10,
    debug=None,
    retrieve_top_k=-1,
    max_context_length=3800,
    limit_context_window=False,
):
    """执行推理过程"""
    # 加载数据
    df = read_file(input_file)
    
    # 确定数据集类型
    if "asqa" in input_file.lower():
        dataset_type = "asqa"
    elif "eli5" in input_file.lower():
        dataset_type = "eli5"
    elif "qampari" in input_file.lower():
        dataset_type = "qampari"
    else:
        dataset_type = "generic"
    
    print(f"数据集类型: {dataset_type}")
    
    # 加载已处理的结果
    existing_data, processed_ids = load_existing_results(output_file)
    
    # 初始化语言模型
    if read:
        print(f"加载语言模型: {language_model}")
        pipeline = transformers.pipeline(
            "text-generation",
            model=language_model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(language_model)
    
    batch_results = []
    processed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理样本"):
        # 检查是否已处理过
        # row_id = str(row.get("id", "")) or str(row.get("sample_id", ""))
        row_id = str(row.get("id", "")) or str(row.get("sample_id", "")) or str(row.get("q_id", "")) or str(row.get("qid", ""))
        if row_id in processed_ids:
            print(f"跳过已处理的样本 ID: {row_id}")
            continue
        
        # 获取问题
        question = get_question_field(row, input_file)
        
        if not question:
            print(f"警告: 样本 {idx} 没有找到问题字段，跳过")
            continue
            
        # 获取上下文
        # context = get_context(row, retrieve, extract, mode=2)
        context = get_context(
            row,
            retrieve,
            extract,
            max_context_length=max_context_length,
            language_model=language_model,
            tokenizer=tokenizer,
            retrieve_top_k=retrieve_top_k,
            limit_context_window=limit_context_window,
        )
        
        # 确定使用的提示模板
        if "qampari" in input_file.lower():
            prompt_template = PROMPT_DICT["reader_short_ans"]
        elif any(ds in input_file.lower() for ds in ["asqa", "eli5"]):
            prompt_template = PROMPT_DICT["reader"]
        else:
            prompt_template = PROMPT_DICT["reader_short_form"]
        
        # 构建提示
        if context:
            prompt = prompt_template.format(paragraph=context, question=question)
        else:
            # 对于没有上下文的情况，移除paragraph部分
            prompt = prompt_template.format(paragraph="", question=question).replace("## Paragraph\n\n\n", "")
        
        # 执行推理
        try:
            messages = [{"role": "user", "content": prompt}]
            outputs = pipeline(
                messages,
                max_new_tokens=max_new_tokens,
            )
            
            # 提取回答
            answer = outputs[0]["generated_text"][-1]["content"]
            
            # 根据数据集类型格式化结果
            if dataset_type == "asqa":
                result = format_asqa_result(row, question, answer)
                result["prompt"] = prompt
            elif dataset_type == "eli5":
                result = format_eli5_result(row, question, answer)
            elif dataset_type == "qampari":
                result = format_qampari_result(row, question, answer)
            else:
                result = format_generic_result(row, question, prompt, answer)
            
            batch_results.append(result)
            processed_count += 1
            
            # print(f"[{processed_count}/{len(df)}] Q: {question}\nA: {answer}\n")
            
            # 批量保存
            if len(batch_results) >= batch_size:
                save_batch_results(batch_results, output_file, dataset_type)
                batch_results = []
            
            # Debug模式下限制处理的样本数
            if debug is not None and processed_count >= debug:
                print(f"已处理 {debug} 条数据，Debug模式已完成")
                break
                
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
    
    # 保存剩余的结果
    if batch_results:
        save_batch_results(batch_results, output_file, dataset_type)
    
    print(f"\n***处理完成，共保存 {processed_count} 条新结果***\n")
    return processed_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reader推理程序")
    
    # 基本参数
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True,
        help="输入的parquet文件路径"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True,
        help="输出的json文件路径"
    )
    parser.add_argument(
        "--language_model", 
        type=str, 
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="语言模型路径"
    )
    
    # 模式参数
    parser.add_argument(
        "--retrieve", 
        action="store_true", 
        help="使用检索模式"
    )
    parser.add_argument(
        "--extract", 
        action="store_true", 
        help="使用提取模式（需要与retrieve一起使用）"
    )
    parser.add_argument(
        "--read", 
        action="store_true", 
        default=True,
        help="使用阅读模式（默认启用）"
    )
    
    # 批量处理参数
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=10,
        help="批量保存大小"
    )
    
    # Debug参数
    parser.add_argument(
        "--debug", 
        type=int, 
        default=None,
        help="Debug模式：只处理指定数量的样本"
    )

    parser.add_argument(
        "--retrieve_top_k", 
        type=int, 
        default=-1,
        help="Only use the first K pre-retrieved passages in retrieve-only mode."
    )

    parser.add_argument(
        "--max_context_length",
        type=int,
        default=3800,
        help="最大输入上下文token数量；Llama-2默认启用，也可配合--limit_context_window强制启用"
    )

    parser.add_argument(
        "--limit_context_window",
        action="store_true",
        help="强制按--max_context_length限制输入上下文长度"
    )
    
    # 其他参数
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256,
        help="最大生成token数量"
    )
    
    args = parser.parse_args()
    
    # 输出参数信息
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("推理程序启动")

    # 执行推理
    inference(
        input_file=args.input_file,
        output_file=args.output_file,
        language_model=args.language_model,
        retrieve=args.retrieve,
        extract=args.extract,
        read=args.read,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        debug=args.debug,
        retrieve_top_k=args.retrieve_top_k,
        max_context_length=args.max_context_length,
        limit_context_window=args.limit_context_window,
    )
