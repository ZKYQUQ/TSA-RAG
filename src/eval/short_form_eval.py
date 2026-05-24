import re
import string
import numpy as np
from collections import Counter
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
import json
import pyarrow as pa

def normalize_answer(s):
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_acc(parquet_path):
    """
    Calculate accuracy (exact match) for the results in a parquet file.
    
    Args:
        parquet_path: Path to the parquet result file
        
    Returns:
        Accuracy score (float)
    """
    #df = pq.read_table(parquet_path).to_pandas()
    df = pd.read_json(parquet_path)
    correct = 0
    total = len(df)
    
    for _, row in df.iterrows():
        golden_answers = row['golden_answers'] if "golden_answers" in row else row["answers"]
        # model_answer = row['reader']['answer']
        # model_answer = row['output']
        model_answer = row['output'].strip("\n., ")
        # Normalize both model answer and golden answers
        norm_model_answer = normalize_answer(model_answer)
        norm_golden_answers = [normalize_answer(ans) for ans in golden_answers if normalize_answer(ans)]
        
        # Check if model answer matches any of the golden answers
        if norm_model_answer in norm_golden_answers:
            correct += 1
    
    return correct / total if total > 0 else 0.0

def calculate_em(parquet_path):
    """
    Calculate accuracy (exact match) for the results in a parquet file.
    
    Args:
        parquet_path: Path to the parquet result file
        
    Returns:
        Accuracy score (float)
    """
    #df = pq.read_table(parquet_path).to_pandas()
    df = pd.read_json(parquet_path)
    correct = 0
    total = len(df)
    
    for _, row in df.iterrows():
        golden_answers = row['golden_answers'] if "golden_answers" in row else row["answers"]
        # model_answer = row['reader']['answer']
        model_answer = row['output'].strip("\n., ")
        # Normalize both model answer and golden answers
        norm_model_answer = normalize_answer(model_answer)
        norm_golden_answers = [normalize_answer(ans) for ans in golden_answers if normalize_answer(ans)]
        
        for gold_answer in norm_golden_answers:
            if gold_answer in norm_model_answer:
                correct += 1
                break
    
    return correct / total if total > 0 else 0.0

def token_level_f1(prediction: str, ground_truths: list):
        final_metric = {"f1": 0, "precision": 0, "recall": 0}
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        for ground_truth in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_ground_truth = normalize_answer(ground_truth)

            if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
                continue
            if (
                normalized_ground_truth in ["yes", "no", "noanswer"]
                and normalized_prediction != normalized_ground_truth
            ):
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["f1", "precision", "recall"]:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

def calculate_char_f1(parquet_path):
    """
    Calculate character-level F1 score for the results in a parquet file.
    Follows the exact same logic as the reference F1_Score class implementation.
    
    Args:
        parquet_path: Path to the parquet result file
        
    Returns:
        Average character-level F1 score (float)
    """
    #df = pq.read_table(parquet_path).to_pandas()
    df = pd.read_json(parquet_path)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for _, row in df.iterrows():
        golden_answers = row['golden_answers'] if "golden_answers" in row else row["answers"]
        # model_answer = row['reader']['answer']
        model_answer = row['output'].strip("\n., ")
        metrics = token_level_f1(model_answer, golden_answers)
        precision_scores.append(metrics['precision'])
        recall_scores.append(metrics['recall'])
        f1_scores.append(metrics['f1'])

        rtn_dict = {
            "precision": np.mean(precision_scores) if precision_scores else 0.0,
            "recall": np.mean(recall_scores) if recall_scores else 0.0,
            "f1": np.mean(f1_scores) if f1_scores else 0.0
        }
    
    return rtn_dict



def find_and_save_special_cases(parquet_path):
    # 读取parquet文件
    #df = pq.read_table(parquet_path).to_pandas()
    df = pd.read_json(parquet_path)
    em_special_cases = []
    em_special_rows = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        golden_answers = row['golden_answers'] if "golden_answers" in row else row["answers"]
        # model_answer = row['reader']['answer']
        model_answer = row['output'].strip("\n., ")
        norm_model_answer = normalize_answer(model_answer)
        norm_golden_answers = [normalize_answer(ans) for ans in golden_answers if normalize_answer(ans)]

        em_flag = False
        acc_flag = False
        for gold_answer in norm_golden_answers:
            if gold_answer in norm_model_answer:
                em_flag = True
                break
        if norm_model_answer in norm_golden_answers:
            acc_flag = True
            
        if em_flag and not acc_flag:
            # 准备JSONL数据
            special_case = {
                "question": row['question'],
                "model_answer": norm_model_answer,
                "golden_answers": norm_golden_answers,
                "em_flag": em_flag,
                "acc_flag": acc_flag
            }
            em_special_cases.append(special_case)
            
            # 准备Parquet数据（去除reader字段）
            parquet_row = row.to_dict()
            del parquet_row['reader']
            em_special_rows.append(parquet_row)
    
    print(f"EM == True && Acc == False: {len(em_special_cases)}")
    
    # 保存EM特殊案例到JSONL文件
    output_jsonl_path = parquet_path.replace('.parquet', '_em_special_cases.jsonl')
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for case in em_special_cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    
    # 保存EM特殊案例到Parquet文件
    if em_special_rows:
        output_parquet_path = parquet_path.replace('.parquet', '_em_special_cases.parquet')
        special_df = pd.DataFrame(em_special_rows)
        
        # 确保列数据类型与原始数据一致
        original_schema = pq.read_schema(parquet_path)
        new_schema = pa.schema([field for field in original_schema if field.name != 'reader'])
        
        table = pa.Table.from_pandas(special_df, schema=new_schema)
        pq.write_table(table, output_parquet_path)
        print(f"Saved EM special cases to {output_parquet_path}")

def read_json_file(json_path):
    """
    读取JSON文件并返回DataFrame格式数据
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        pandas.DataFrame: 包含JSON数据的DataFrame
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为DataFrame
    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
    
    return df

def find_and_save_special_cases_json(json_path):
    # 读取json文件
    df = read_json_file(json_path)
    
    em_special_cases = []
    em_special_rows = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        golden_answers = row['golden_answers'] if "golden_answers" in row else row["answers"]
        model_answer = row['reader']['answer']

        norm_model_answer = normalize_answer(model_answer)
        norm_golden_answers = [normalize_answer(ans) for ans in golden_answers if normalize_answer(ans)]

        em_flag = False
        acc_flag = False
        for gold_answer in norm_golden_answers:
            if gold_answer in norm_model_answer:
                em_flag = True
                break
        if norm_model_answer in norm_golden_answers:
            acc_flag = True
            
        if em_flag and not acc_flag:
            # 准备JSONL数据
            special_case = {
                "question": row['question'],
                "model_answer": norm_model_answer,
                "golden_answers": norm_golden_answers,
                "em_flag": em_flag,
                "acc_flag": acc_flag
            }
            em_special_cases.append(special_case)
            
            # 准备JSON数据（去除reader字段）
            json_row = row.to_dict()
            del json_row['reader']
            em_special_rows.append(json_row)
    
    print(f"EM == True && Acc == False: {len(em_special_cases)}")
    
    # 保存EM特殊案例到JSONL文件
    output_jsonl_path = json_path.replace('.json', '_em_special_cases.jsonl')
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for case in em_special_cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    
    # 保存EM特殊案例到JSON文件
    if em_special_rows:
        output_json_path = json_path.replace('.json', '_em_special_cases.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(em_special_rows, f, ensure_ascii=False, indent=2)
        print(f"Saved EM special cases to {output_json_path}")
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate short-form QA output JSON.")
    parser.add_argument("--result_file", required=True)
    args = parser.parse_args()

    acc_score = calculate_acc(args.result_file)
    print(f"Accuracy: {acc_score:.4f}")

    char_f1_score = calculate_char_f1(args.result_file)
    for key, value in char_f1_score.items():
        print(f"{key.capitalize()} score: {value:.4f}")

    em_score = calculate_em(args.result_file)
    print(f"EM: {em_score:.4f}")

    # find_and_save_special_cases_json(args.result_file)
