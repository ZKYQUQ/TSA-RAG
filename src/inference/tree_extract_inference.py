import pandas as pd
import json
import os
from collections import defaultdict, deque
from tqdm import tqdm
import copy
import argparse
import transformers
import torch
import tempfile
import shutil

torch.manual_seed(29)

class TreeNode:
    def __init__(self, nid: int, ntext: str, ntype: str, span: list, lighted: bool = False):
        self.id = nid
        self.text = ntext
        self.type = ntype
        self.parent = None
        self.children = []
        self.span = span
        self.lighted = lighted
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        self.children.sort(key=lambda x: x.id)

class DocumentTree:
    def __init__(self, raw_data: dict, with_intro: bool=True, include_parent_siblings: bool = False):
        self.nodes = {}
        self.leaves = []
        self.full_text = raw_data.get("full_text", "")
        self.with_intro = with_intro
        self.include_parent_siblings = include_parent_siblings
        self.root = self.build_document_tree(raw_data)
    
    def build_document_tree(self, raw_data: dict):
        """构建文档树"""
        self.nodes = {}
        self.leaves = []
        
        # 创建所有节点
        for data in raw_data["extracted_nodes"]:
            node = TreeNode(data["id"], data["text"], data["type"], data["span"])
            
            # 根节点（title）和intro内容点亮
            if data["id"] == 0 or (self.with_intro and data["type"] == "content" and data["relation"]["up_id"] == 0):
                node.lighted = True
            
            # 记录叶子节点
            if len(data["relation"]["down_ids"]) == 0:
                self.leaves.append(node)
                
            self.nodes[node.id] = node
        
        self.nodes[0].text = "=" + self.nodes[0].text + "="

        # 建立父子关系
        for data in raw_data["extracted_nodes"]:
            current_node = self.nodes[data["id"]]
            up_id = data["relation"]["up_id"]
            
            if up_id != -1:
                parent_node = self.nodes[up_id]
                parent_node.add_child(current_node)
        
        # 排序叶子节点
        self.leaves.sort(key=lambda x: x.id)
        
        # 找到根节点
        root = None
        for node in self.nodes.values():
            if node.parent is None:
                root = node
                break
        
        return root
    
    def get_siblings(self, node):
        """获取节点的兄弟节点"""
        if node.parent:
            return node.parent.children
        return [node]
    
    def light_heading_descendants(self, node):
        """递归点亮节点的所有后代"""
        for child in node.children:
            child.lighted = True
            self.light_heading_descendants(child)
    
    def light_nodes_by_heading_ids(self, heading_ids):
        """根据heading IDs点亮相关节点"""
        for heading_id in heading_ids:
            if heading_id not in self.nodes:
                continue

            if heading_id == 0:
                self.root.lighted = True
                continue
                
            target_node = self.nodes[heading_id]
            
            # 点亮目标节点
            target_node.lighted = True
            
            # 点亮所有后代节点
            self.light_heading_descendants(target_node)
            
            # 向上点亮到根节点
            current_node = target_node
            while current_node.parent is not None:
                current_node = current_node.parent
                current_node.lighted = True
                
                # 如果启用include_parent_siblings，点亮title节点的content兄弟节点
                # if self.include_parent_siblings and current_node.type == "title":
                if self.include_parent_siblings:
                    siblings = self.get_siblings(current_node)
                    for sibling in siblings:
                        if sibling.type == "content":
                            sibling.lighted = True
    
    def format_data(self, node, level: int = 0):
        """格式化节点数据"""
        if node.type == "title":
            # 对于标题节点，保持原有的=格式
            return f"{node.id}: {node.text}"
        elif node.type == "content":
            # 对于内容节点，直接显示文本
            return f"{node.id}: {node.text}"
        else:
            return f"{node.id}: {node.text}"
    
    def traverse(self, node=None, level: int = 0, result: str = ""):
        """遍历并格式化文档树"""
        if node is None:
            node = self.root
            
        if node.lighted:
            # 添加缩进
            indent = "    " * level
            result += indent + self.format_data(node, level) + "\n"
            
            # 遍历子节点
            for child in node.children:
                if child.lighted:
                    result = self.traverse(child, level + 1, result)
        
        return result

def load_existing_results(output_file):
    """加载已有的结果文件"""
    if os.path.exists(output_file):
        try:
            # 确定query字段
            if "asqa" in output_file:
                query_str = 'ambiguous_question'
            elif "eli5" in output_file:
                query_str = 'title'
            elif "qampari" in output_file:
                query_str = 'question_text'
            else:
                raise ValueError("Unsupported output file type for loading existing results")
            
            existing_df = pd.read_parquet(output_file)
            processed_sample_queries = set(existing_df[query_str].tolist())
            print(f"Found existing results with {len(processed_sample_queries)} processed samples")
            return existing_df, processed_sample_queries
        except Exception as e:
            print(f"Error loading existing results: {e}")
            return None, set()
    return None, set()

# def save_batch_results(all_processed_rows, output_file):
#     """保存批次结果"""
#     if all_processed_rows:
#         result_df = pd.DataFrame(all_processed_rows)
#         result_df.to_parquet(output_file, index=False)
#         print(f"Saved {len(all_processed_rows)} results to {output_file}")

def atomic_write_parquet(df, output_path):
        """原子性写入parquet文件"""
        # 创建临时文件
        temp_dir = os.path.dirname(output_path)
        temp_file = tempfile.NamedTemporaryFile(
            dir=temp_dir, 
            suffix='.parquet.tmp', 
            delete=False
        )
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # 写入临时文件
            print(f"写入临时文件: {temp_path}")
            df.to_parquet(temp_path, index=False)
            
            # 原子性移动到最终位置
            print(f"移动到最终文件: {output_path}")
            shutil.move(temp_path, output_path)
            print(f"成功保存到: {output_path}")
            
        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"清理临时文件: {temp_path}")
            raise e
    
def save_batch_results(processed_rows, output_path):
    """保存批次结果"""
    if processed_rows:
        batch_df = pd.DataFrame(processed_rows)
        
        # 如果输出文件已存在，则追加
        if os.path.exists(output_path):
            existing_df = pd.read_parquet(output_path)
            combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
            atomic_write_parquet(combined_df, output_path)
        else:
            atomic_write_parquet(batch_df, output_path)

        print(f"保存了 {len(processed_rows)} 条记录")

def extract_document_subtrees(input_file, output_file, with_intro=True, include_parent_siblings=True, batch_size=100, plain_tree=False):
    """构建document subtrees"""
    
    def build_plain_doc_info(source_doc_info):
        doc_info_copy = copy.deepcopy(source_doc_info)
        extracted_nodes = doc_info_copy.get("extracted_nodes", [])
        if len(extracted_nodes) == 0:
            return doc_info_copy

        root_node = None
        content_nodes = []
        for node in extracted_nodes:
            node_id = node.get("id")
            node_type = node.get("type", "")
            if node_id == 0:
                root_node = node
            elif node_type == "content":
                node["relation"] = {"up_id": 0, "down_ids": []}
                content_nodes.append(node)

        if root_node is None:
            return doc_info_copy

        content_nodes.sort(key=lambda n: n["id"])
        count = 1
        for node in content_nodes:
            node['id'] = count
            count += 1
        root_node["relation"] = {"up_id": -1, "down_ids": [n["id"] for n in content_nodes]}
        doc_info_copy["extracted_nodes"] = [root_node] + content_nodes
        headings = [node["id"] for node in doc_info_copy["extracted_nodes"]]
        return doc_info_copy, headings
    
    # 读取输入数据
    print("Loading input data...")
    df = pd.read_parquet(input_file)
    
    # 确定query字段
    if "asqa" in input_file:
        query_str = 'ambiguous_question'
    elif "eli5" in input_file:
        query_str = 'title'
    elif "qampari" in input_file:
        query_str = 'question_text'
    else:
        query_str = 'question'
        # raise ValueError("Unsupported input file type for document subtree extraction")
    
    # 加载已有结果
    existing_df, processed_sample_queries = load_existing_results(output_file)
    
    # 过滤未处理的数据
    if processed_sample_queries:
        unprocessed_mask = ~df[query_str].isin(processed_sample_queries)
        unprocessed_df = df[unprocessed_mask].reset_index(drop=True)
        print(f"Found {len(unprocessed_df)} unprocessed samples out of {len(df)} total samples")
    else:
        unprocessed_df = df
        print(f"Processing all {len(df)} samples")
    
    if len(unprocessed_df) == 0:
        print("All samples have been processed!")
        return
    
    # # 处理数据
    # all_processed_rows = []
    # if existing_df is not None:
    #     all_processed_rows = existing_df.to_dict('records')
    
    batch_processed = []
    
    for idx, row in tqdm(unprocessed_df.iterrows(), total=len(unprocessed_df), desc="Processing samples"):
        try:
            new_row = copy.deepcopy(row)
            path_extract_results = row.get('path_extract_results', [])
            doc_infos = row.get('doc_infos', [])
            
            # 构建文档标题到doc_info的映射
            doc_info_map = {}
            for doc_info in doc_infos:
                title = doc_info.get('title', '')
                if title:
                    doc_info_map[title] = doc_info
            
            document_subtrees = []
            
            # 处理每个path extract结果
            for extract_result in path_extract_results:
                doc_title = extract_result.get('title', '')
                headings = extract_result.get('headings', [])
                
                if not doc_title or len(headings) == 0:
                    continue
                
                # 查找对应的doc_info
                if doc_title not in doc_info_map:
                    raise ValueError(f"Warning: Document '{doc_title}' not found in doc_infos")
                    continue
                
                doc_info = doc_info_map[doc_title]
                
                # 构建DocumentTree
                try:
                    if plain_tree:
                        doc_info_for_tree, headings = build_plain_doc_info(doc_info)
                    else:
                        doc_info_for_tree = doc_info
                    
                    doc_tree = DocumentTree(doc_info_for_tree, with_intro=with_intro, include_parent_siblings=include_parent_siblings)
                    
                    # 点亮指定的headings
                    doc_tree.light_nodes_by_heading_ids(headings)
                    
                    # 生成文档树的文本表示
                    tree_text = doc_tree.traverse()
                    
                    if tree_text.strip():  # 只保存非空的树
                        document_subtrees.append({
                            "title": doc_title,
                            # "doc_info": doc_info,
                            "doc_info": doc_info_for_tree,
                            "path_extract_headings": headings,
                            "document_tree": tree_text.strip()
                        })
                        
                except Exception as e:
                    print(f"Error processing document tree for '{doc_title}': {e}")
                    continue
            
            # 添加新字段
            new_row['document_subtrees'] = document_subtrees
            
            batch_processed.append(new_row)
            
            # # 每batch_size条数据保存一次
            # if len(batch_processed) >= batch_size:
            #     # all_processed_rows.extend(batch_processed)
            #     # save_batch_results(all_processed_rows, output_file)
            #     # batch_processed = []
            #     # print(f"Processed {idx + 1}/{len(unprocessed_df)} samples")
            #     save_batch_results(batch_processed, output_file)
            #     batch_processed = []  # 清空缓存

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # 保存剩余的数据
    if batch_processed:
        # all_processed_rows.extend(batch_processed)
        # save_batch_results(all_processed_rows, output_file)
        save_batch_results(batch_processed, output_file)
        # batch_processed = []  # 清空缓存

    print(f"Document subtree construction complete! Results saved to {output_file}")


def construct_tree_extract_prompt(document_tree, question):
    prompt = f"""Given a document structure tree comprising a title, hierarchical headings and subordinate paragraphs, identify all paragraphs that answer the specified question. List each relevant paragraph with the tag [paragraph]. If none are relevant, reply exactly: "No relevant paragraphs".

## Document Structure Tree
{document_tree}

## Question
{question}

## Response"""
    return prompt

def model_inference(input_file, output_file, model_path, max_new_tokens=512, batch_size=10):
    """
    针对document_subtrees字段进行模型推理，添加tree_extract_results字段
    
    Args:
        input_file: 输入的parquet文件路径
        output_file: 输出的parquet文件路径
        model_path: 模型路径
        max_new_tokens: 最大生成token数量
        batch_size: 批处理大小
    """
    
    # 读取输入数据
    print("Loading input data...")
    df = pd.read_parquet(input_file)
    
    # 确定query字段
    if "asqa" in input_file:
        query_str = 'ambiguous_question'
    elif "eli5" in input_file:
        query_str = 'title'
    elif "qampari" in input_file:
        query_str = 'question_text'
    else:
        query_str = 'question'
        # raise ValueError("Unsupported input file type for model inference")
    
    # 加载已有结果
    existing_df, processed_sample_queries = load_existing_results(output_file)
    
    # 过滤未处理的数据
    if processed_sample_queries:
        unprocessed_mask = ~df[query_str].isin(processed_sample_queries)
        unprocessed_df = df[unprocessed_mask].reset_index(drop=True)
        print(f"Found {len(unprocessed_df)} unprocessed samples out of {len(df)} total samples")
    else:
        unprocessed_df = df
        print(f"Processing all {len(df)} samples")
    
    if len(unprocessed_df) == 0:
        print("All samples have been processed!")
        return
    
    # 初始化模型pipeline
    print(f"Loading model from {model_path}...")
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    

    # # 处理数据
    # all_processed_rows = []
    # if existing_df is not None:
    #     all_processed_rows = existing_df.to_dict('records')
    
    batch_processed = []
    
    for idx, row in tqdm(unprocessed_df.iterrows(), total=len(unprocessed_df), desc="Processing samples"):
        try:
            new_row = copy.deepcopy(row)
            question = row[query_str]
            document_subtrees = row.get('document_subtrees', [])
            
            # 对每个文档子树进行推理
            tree_extract_results = []
            
            for subtree in document_subtrees:
                title = subtree.get('title', '')
                doc_info = subtree.get('doc_info', {})
                document_tree = subtree.get('document_tree', '')
                path_extract_headings = subtree.get('path_extract_headings', [])
                
                if not title or not document_tree:
                    continue
                
                # 构造推理prompt
                prompt = construct_tree_extract_prompt(document_tree, question)
                
                # 模型推理
                messages = [{"role": "user", "content": prompt}]
                
                preds = pipeline(
                    messages,
                    max_new_tokens=max_new_tokens,
                    # do_sample=False,
                    # temperature=0.0,
                )
                raw_output = preds[0]["generated_text"][-1]["content"]
                
                # 保存结果
                tree_extract_results.append({
                    "title": title,
                    "doc_info": doc_info,
                    # "path_extract_headings": path_extract_headings,
                    # "document_tree": document_tree,
                    "raw_output": raw_output,
                    "local_score": doc_info.get("local_score", 0),
                    "global_score": doc_info.get("global_score", 0),
                    "total_score": doc_info.get("total_score", 0)
                })
            
            # 添加新字段
            new_row['tree_extract_results'] = tree_extract_results
            
            batch_processed.append(new_row)
            
            # 每batch_size条数据保存一次
            if len(batch_processed) >= batch_size:
                # all_processed_rows.extend(batch_processed)
                # save_batch_results(all_processed_rows, output_file)
                # batch_processed = []
                # print(f"Processed {idx + 1}/{len(unprocessed_df)} samples")
                save_batch_results(batch_processed, output_file)
                batch_processed = []  # 清空缓存

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # 保存剩余的数据
    if batch_processed:
        # all_processed_rows.extend(batch_processed)
        # save_batch_results(all_processed_rows, output_file)
        save_batch_results(batch_processed, output_file)
        # batch_processed = []  # 清空缓存

    print(f"Tree extract model inference complete! Results saved to {output_file}")

def parse_model_output(raw_output):
    if not raw_output or "No relevant paragraphs".lower() in raw_output.lower():
        return []
    
    paragraph_ids = []
    
    lines = raw_output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 检查是否是段落行
        if line.startswith('[paragraph]'):
            paragraph_content = line[len('[paragraph]'):].strip()
            
            # 解析paragraph_id
            parts = paragraph_content.split(':', 1)
            if len(parts) == 2:
                try:
                    paragraph_id = int(parts[0].strip())
                    paragraph_ids.append(paragraph_id)
                except ValueError:
                    print(f"Invalid paragraph ID format in line: {line}")
                    continue
                    
    return paragraph_ids

def parse_and_filter_results(input_file, output_file, batch_size=10, subtree_with_intro=False, subtree_include_parent_siblings=False):
    """
    解析tree_extract_results中的raw_output，过滤内容并构造子树
    
    Args:
        input_file: 输入的parquet文件路径
        output_file: 输出的parquet文件路径
        batch_size: 批处理大小
    """
    # 读取输入数据
    print("Loading input data...")
    df = pd.read_parquet(input_file)
    
    # 确定query字段
    if "asqa" in input_file:
        query_str = 'ambiguous_question'
    elif "eli5" in input_file:
        query_str = 'title'
    elif "qampari" in input_file:
        query_str = 'question_text'
    else:
        query_str = 'question'
        # raise ValueError("Unsupported input file type for parsing and filtering results")
    
    # 加载已有结果
    existing_df, processed_sample_queries = load_existing_results(output_file)
    
    # 过滤未处理的数据
    if processed_sample_queries:
        unprocessed_mask = ~df[query_str].isin(processed_sample_queries)
        unprocessed_df = df[unprocessed_mask].reset_index(drop=True)
        print(f"Found {len(unprocessed_df)} unprocessed samples out of {len(df)} total samples")
    else:
        unprocessed_df = df
        print(f"Processing all {len(df)} samples")
    
    if len(unprocessed_df) == 0:
        print("All samples have been processed!")
        return
    
    # # 处理数据
    # all_processed_rows = []
    # if existing_df is not None:
    #     all_processed_rows = existing_df.to_dict('records')
    
    batch_processed = []
    
    for idx, row in tqdm(unprocessed_df.iterrows(), total=len(unprocessed_df), desc="Processing samples"):
        try:
            new_row = copy.deepcopy(row)
            tree_extract_results = row.get('tree_extract_results', [])
            
            # 处理每个树提取结果
            for i, result in enumerate(tree_extract_results):
                # 检查是否已处理过
                if 'cleaned_output' in result and 'formated_subtree' in result:
                    continue
                
                raw_output = result.get('raw_output', '')
                doc_info = result.get('doc_info', {})
                
                # 解析模型输出
                paragraph_ids = parse_model_output(raw_output)
                
                # 验证并过滤paragraph_ids
                valid_paragraph_ids = []
                
                try:
                    # 使用DocumentTree验证节点
                    doc_tree = DocumentTree(doc_info, with_intro=True, include_parent_siblings=True)
                    
                    for paragraph_id in paragraph_ids:
                        # 检查节点是否存在且为content类型
                        if paragraph_id in doc_tree.nodes and doc_tree.nodes[paragraph_id].type == 'content':
                            valid_paragraph_ids.append(paragraph_id)
                            
                except Exception as e:
                    print(f"Error validating paragraphs: {e}")
                    valid_paragraph_ids = []
                
                # 保存清洗后的结果
                new_row['tree_extract_results'][i]['cleaned_output'] = sorted(valid_paragraph_ids)
                
                # 对于非空结果，构建格式化子树
                if valid_paragraph_ids:
                    try:
                        # 创建新的DocumentTree实例
                        doc_tree = DocumentTree(doc_info, with_intro=subtree_with_intro, include_parent_siblings=subtree_include_parent_siblings)
                        
                        # 重置所有节点的lighted状态为False
                        for node_id in doc_tree.nodes:
                            doc_tree.nodes[node_id].lighted = False
                        
                        # 点亮指定的paragraph节点及其所有父节点
                        for paragraph_id in valid_paragraph_ids:
                            if paragraph_id in doc_tree.nodes:
                                # 点亮节点
                                doc_tree.nodes[paragraph_id].lighted = True
                                
                                # 向上点亮所有父节点
                                current_node = doc_tree.nodes[paragraph_id]
                                while current_node.parent is not None:
                                    current_node = current_node.parent
                                    current_node.lighted = True
                        
                        # 生成格式化子树
                        formated_subtree = doc_tree.traverse()
                        new_row['tree_extract_results'][i]['formated_subtree'] = formated_subtree
                    
                    except Exception as e:
                        print(f"Error generating formatted subtree: {e}")
                        new_row['tree_extract_results'][i]['formated_subtree'] = ""
                else:
                    # 如果没有有效段落，设置为空字符串
                    new_row['tree_extract_results'][i]['formated_subtree'] = ""
            
            # 按total_score排序tree_extract_results
            new_row['tree_extract_results'] = sorted(
                new_row['tree_extract_results'], 
                key=lambda x: x['total_score'], 
                reverse=True
            )
            
            batch_processed.append(new_row)
            
            # # 每batch_size条数据保存一次
            # if len(batch_processed) >= batch_size:
            #     # all_processed_rows.extend(batch_processed)
            #     # save_batch_results(all_processed_rows, output_file)
            #     # batch_processed = []
            #     # print(f"Processed {idx + 1}/{len(unprocessed_df)} samples")
            #     save_batch_results(batch_processed, output_file)
            #     batch_processed = []  # 清空缓存

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # 保存剩余的数据
    if batch_processed:
        # all_processed_rows.extend(batch_processed)
        # save_batch_results(all_processed_rows, output_file)
        save_batch_results(batch_processed, output_file)
        # batch_processed = []  # 清空缓存

    print(f"Parse and filter complete! Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Tree Extract Inference Pipeline')
    
    # 必需参数
    parser.add_argument('--input_file', '-i', 
                        type=str, 
                        required=True,
                        help='输入的parquet文件路径')
    
    parser.add_argument('--output_file', '-o', 
                        type=str, 
                        required=True,
                        help='输出的parquet文件路径')
    
    parser.add_argument('--substage', 
                        type=str, 
                        required=True,
                        choices=['extract_document_subtrees', 'inference', 'parse_and_filter'],
                        help='处理阶段')
    
    parser.add_argument('--subtree_with_intro',
                        type=bool,
                        default=False,
                        help='是否包含intro内容节点')

    parser.add_argument('--subtree_include_parent_siblings',
                        type=bool,
                        default=False,
                        help='在向上点亮过程中是否包含父节点的content类型兄弟节点')

    parser.add_argument('--plain_tree',
                        type=bool,
                        default=False,
                        help='是否仅保留标题节点及所有内容节点构建两层树')

    # inference 阶段参数
    parser.add_argument('--model_path', 
                        type=str,
                        help='模型路径（inference阶段需要）')
    
    parser.add_argument('--max_new_tokens', 
                        type=int, 
                        default=512,
                        help='最大新生成token数（inference阶段）')
    
    # 通用参数
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=10,
                        help='批处理大小')
    
    args = parser.parse_args()

    # 打印全部参数
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    if args.substage == 'extract_document_subtrees':
        extract_document_subtrees(
            input_file=args.input_file,
            output_file=args.output_file,
            with_intro=True,
            include_parent_siblings=True,
            batch_size=args.batch_size,
            plain_tree=args.plain_tree
        )
    elif args.substage == 'inference':
        if not args.model_path:
            raise ValueError("inference阶段需要提供model_path参数")
        
        model_inference(
            input_file=args.input_file,
            output_file=args.output_file,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size
        )
    elif args.substage == 'parse_and_filter':
        parse_and_filter_results(
            input_file=args.input_file,
            output_file=args.output_file,
            batch_size=args.batch_size,
            subtree_with_intro=args.subtree_with_intro,
            subtree_include_parent_siblings=args.subtree_include_parent_siblings,
        )
    else:
        raise ValueError(f"Unknown substage: {args.substage}. Supported stages are: 'extract_document_subtrees', 'inference', 'parse_and_filter'.")


if __name__ == "__main__":
    main()