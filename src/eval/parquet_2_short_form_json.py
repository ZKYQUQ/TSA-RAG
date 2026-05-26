import pandas as pd
import json
import numpy as np

def convert_parquet_to_json(parquet_path, json_path):
    """
    Read a Parquet file, extract and rename fields row by row, and save JSON.
    This version handles each row's 'golden_answers' field explicitly.

    Args:
        parquet_path (str): Input Parquet file path.
        json_path (str): Output JSON file path.
    """
    try:
        print(f"Reading Parquet file: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        required_cols = ['id', 'question', 'golden_answers', 'reader']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: missing required columns. Required: {required_cols}, found: {list(df.columns)}")
            return

        data_list = []

        print("Processing rows...")
        for index, row in df.iterrows():
            # Convert golden_answers to a plain list regardless of its array-like type.
            golden_answers_list = list(row['golden_answers'])

            record = {
                'id': row['id'],
                'question': row['question'],
                'golden_answers': golden_answers_list,
                'output': row['reader']["answer"]
            }
            data_list.append(record)

        print(f"Writing JSON file: {json_path}")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)
        
        print("Conversion complete.")

    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert short-form parquet results to JSON.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    convert_parquet_to_json(args.input_file, args.output_file)
