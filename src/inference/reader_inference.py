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
}

def read_file(input_path):
    """Read the input file."""
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")
    
    print(f"\n***Loaded dataset size: {len(df)} records.***\n")
    return df

def get_question_field(data_item, input_file):
    """Return the question field for the dataset type."""
    if "asqa" in input_file.lower():
        return data_item.get("ambiguous_question", "")
    elif "eli5" in input_file.lower():
        return data_item.get("title", "")
    elif "qampari" in input_file.lower():
        return data_item.get("question_text", "")
    else:
        return data_item["question"]
    

def get_context(
    data_item,
    retrieve,
    extract,
    mode=-1,
    max_context_length=4000,
    language_model=None,
    tokenizer=None,
    retrieve_top_k=-1,
    limit_context_window=False,
):
    """Build context according to the retrieval/extraction mode.
    
    Args:
        data_item: Input row.
        retrieve: Whether to use retrieval results.
        extract: Whether to use extracted tree contexts.
        mode: Context ordering mode:
            1: prioritize tree_extract_results whose titles match pre_retrieved_passages.
            2: sort tree_extract_results by total_score and interleave high/low ranks.
        max_context_length: Maximum context length in tokens.
        language_model: Language model path.
        limit_context_window: Whether to force the context length limit.
    """
    # Llama-2-13b-chat-hf has a 4k context window, so keep the historical
    # default behavior. Other models can opt in with --limit_context_window.
    language_model = language_model or ""
    use_length_limit = limit_context_window or ("Llama-2-13b-chat-hf" in language_model)
    
    def count_tokens(text):
        return len(tokenizer.encode(text))
    
    if retrieve and extract:
        # Mode 2.3: retrieve + extract + read.
        if "tree_extract_results" in data_item and len(data_item["tree_extract_results"]) > 0:
            contexts = []
            current_token_count = 0
            
            if mode == 1 and "pre_retrieved_passages" in data_item and len(data_item["pre_retrieved_passages"]) > 0:
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
                
                for result in matched_results:
                    contexts.append(result["formated_subtree"])
                
                for result in unmatched_results:
                    contexts.append(result["formated_subtree"])
            
            elif mode == 2:
                valid_results = [
                    result for result in data_item["tree_extract_results"]
                    if "formated_subtree" in result and result["formated_subtree"]
                ]
                
                sorted_results = sorted(
                    valid_results, 
                    key=lambda x: x["total_score"],
                    reverse=True
                )
                
                reordered_contexts = []
                
                n = len(sorted_results)
                odd_indices = list(range(0, n, 2))  # 0, 2, 4...
                even_indices = list(range(1, n, 2))  # 1, 3, 5...
                
                for i in odd_indices:
                    reordered_contexts.append(sorted_results[i]["formated_subtree"])
                
                for i in reversed(even_indices):
                    reordered_contexts.append(sorted_results[i]["formated_subtree"])
                
                contexts = reordered_contexts
            
            else:
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
        # Mode 2.2: retrieve + read.
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
        # Mode 2.1: read only.
        return ""

def load_existing_results(output_file):
    """Load an existing result file."""
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                
                if "data" in existing_data:
                    processed_ids = set()
                    for item in existing_data["data"]:
                        if "id" in item:
                            processed_ids.add(item["id"])
                        elif "sample_id" in item:
                            processed_ids.add(item["sample_id"])
                    return existing_data, processed_ids
                else:
                    if isinstance(existing_data, list):
                        processed_ids = set()
                        for item in existing_data:
                            processed_ids.add(item["id"])
                        return existing_data, processed_ids
                    else:
                        raise ValueError("Invalid existing data format")
        except Exception as e:
            print(f"Failed to load existing results: {e}")
    
    return None, set()

def format_asqa_result(row, question, answer):
    """Format ASQA results according to format_eval.py."""
    qa_pairs = []
    if "qa_pairs" in row:
        qa_pairs = [{
            # "context": qa_pair.get("context", ""),
            "question": qa_pair["question"],
            "short_answers": qa_pair["short_answers"].tolist(),
            # "wikipage": qa_pair.get("wikipage", ""),
        } for qa_pair in row["qa_pairs"]]
    else:
        raise ValueError("ASQA dataset is missing the qa_pairs field.")
    
    annotations = []
    if "annotations" in row:
        annotations = [{
            "knowledge": annotation["knowledge"].tolist(),
            "long_answer": annotation["long_answer"],
        } for annotation in row["annotations"]]
    
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
    """Format ELI5 results according to format_eval.py."""
    if "answers" in row and isinstance(row["answers"], dict):
        a_ids = row["answers"].get("a_id", [])
        a_texts = row["answers"].get("text", [])
        a_scores = row["answers"].get("score", [])
        
        if len(a_scores) > 0 and len(a_texts) > 0:
            answer_idx = np.argmax(a_scores)
            original_answer = a_texts[answer_idx]
        else:
            raise ValueError("ELI5 dataset is missing valid answers.")
    else:
        raise ValueError("ELI5 dataset is missing valid answers.")
    
    claims = row["claims"].tolist()
    
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
    """Format QAMPARI results according to format_eval.py."""
    answers = []
    if "answer_list" in row:
        for ans in row["answer_list"]:
            local_answers = ans["aliases"].tolist()
            local_answers.append(ans["answer_text"])
            answers.append(list(set(local_answers)))
    else:
        raise ValueError("QAMPARI dataset is missing valid answers.")
    
    result = {
        "id": str(row.get("qid", "") or row.get("id", "")),
        "question": question.strip(),
        "answers": answers,
        # "docs": docs,
        "output": answer.strip(),
    }
    return result

def format_generic_result(row, question, prompt, answer):
    """Format generic short-form QA results."""
    result = {
        "id": row["id"],
        "question": question,
        "output": answer,
        "golden_answers": row["golden_answers"].tolist()
    }
    return result

def save_batch_results(results, output_file, dataset_type):
    """Save a batch of results."""
    if not results:
        return

    existing_data, _ = load_existing_results(output_file)
    
    if dataset_type in ["asqa", "eli5", "qampari"]:
        if existing_data is None:
            output_data = {
                "data": results,
                "config": {}
            }
        else:
            output_data = copy.deepcopy(existing_data)
            output_data["data"].extend(results)
    else:
        if existing_data is None:
            output_data = results
        else:
            output_data = existing_data + results
    
    temp_file = f"{output_file}.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    if os.path.exists(output_file):
        os.replace(temp_file, output_file)
    else:
        os.rename(temp_file, output_file)
    
    print(f"Saved {len(results)} results to {output_file}")

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
    """Run reader inference."""
    df = read_file(input_file)
    
    if "asqa" in input_file.lower():
        dataset_type = "asqa"
    elif "eli5" in input_file.lower():
        dataset_type = "eli5"
    elif "qampari" in input_file.lower():
        dataset_type = "qampari"
    else:
        dataset_type = "generic"
    
    print(f"Dataset type: {dataset_type}")
    
    existing_data, processed_ids = load_existing_results(output_file)
    
    if read:
        print(f"Loading language model: {language_model}")
        pipeline = transformers.pipeline(
            "text-generation",
            model=language_model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(language_model)
    
    batch_results = []
    processed_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        row_id = str(row.get("id", "")) or str(row.get("sample_id", "")) or str(row.get("q_id", "")) or str(row.get("qid", ""))
        if row_id in processed_ids:
            print(f"Skipping processed sample ID: {row_id}")
            continue
        
        question = get_question_field(row, input_file)
        
        if not question:
            print(f"Warning: sample {idx} has no question field; skipping.")
            continue
            
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
        
        if "qampari" in input_file.lower():
            prompt_template = PROMPT_DICT["reader_short_ans"]
        elif any(ds in input_file.lower() for ds in ["asqa", "eli5"]):
            prompt_template = PROMPT_DICT["reader"]
        else:
            prompt_template = PROMPT_DICT["reader_short_form"]
        
        if context:
            prompt = prompt_template.format(paragraph=context, question=question)
        else:
            prompt = prompt_template.format(paragraph="", question=question).replace("## Paragraph\n\n\n", "")
        
        try:
            messages = [{"role": "user", "content": prompt}]
            outputs = pipeline(
                messages,
                max_new_tokens=max_new_tokens,
            )
            
            answer = outputs[0]["generated_text"][-1]["content"]
            
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
            
            if len(batch_results) >= batch_size:
                save_batch_results(batch_results, output_file, dataset_type)
                batch_results = []
            
            if debug is not None and processed_count >= debug:
                print(f"Processed {debug} samples; debug mode complete.")
                break
                
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
    
    if batch_results:
        save_batch_results(batch_results, output_file, dataset_type)
    
    print(f"\n***Processing complete. Saved {processed_count} new results.***\n")
    return processed_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reader inference utility")
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True,
        help="Input Parquet file path."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True,
        help="Output JSON file path."
    )
    parser.add_argument(
        "--language_model", 
        type=str, 
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Language model path."
    )
    
    parser.add_argument(
        "--retrieve", 
        action="store_true", 
        help="Use retrieval context."
    )
    parser.add_argument(
        "--extract", 
        action="store_true", 
        help="Use extracted context; requires --retrieve."
    )
    parser.add_argument(
        "--read", 
        action="store_true", 
        default=True,
        help="Use reader inference; enabled by default."
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=10,
        help="Batch size for incremental saves."
    )
    
    parser.add_argument(
        "--debug", 
        type=int, 
        default=None,
        help="Debug mode: process only this many samples."
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
        help="Maximum input context length in tokens. Enabled by default for Llama-2 and otherwise with --limit_context_window."
    )

    parser.add_argument(
        "--limit_context_window",
        action="store_true",
        help="Force truncation by --max_context_length."
    )
    
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256,
        help="Maximum number of generated tokens."
    )
    
    args = parser.parse_args()
    
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Starting reader inference.")

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
