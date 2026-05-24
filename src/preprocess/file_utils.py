import os
import json
import unicodedata
import pandas as pd
import time
from datetime import timedelta
import argparse

def construct_file_index(jsonl_dir, output_index_file):
    title_index = {}

    for filename in os.listdir(jsonl_dir):
        # if filename.endswith(".jsonl"):
        if '.jsonl' in filename:
            filepath = os.path.join(jsonl_dir, filename)
            
            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    data = json.loads(line.strip())
                    title = data["title"]
                    
                    if title not in title_index:
                        title_index[title] = {"filename": filename, "line_num": line_num}

    with open(output_index_file, "w", encoding="utf-8") as f:
        json.dump(title_index, f, ensure_ascii=False, indent=4)


def title_to_jsonTree(title, index_file, jsonl_dir):
    with open(index_file, "r", encoding="utf-8") as f:
        title_index = json.load(f)
    
    if title in title_index:
        jsonl_filename = title_index[title]["filename"]
        line_num = title_index[title]["line_num"]
        jsonl_filepath = os.path.join(jsonl_dir, jsonl_filename)
        
        with open(jsonl_filepath, "r", encoding="utf-8") as f:
            for current_line_num, line in enumerate(f):
                if current_line_num == line_num:
                    return json.loads(line.strip())
    else:
        return None


def get_nodes_from_title(title, title_index, path_to_document_trees):
    title = unicodedata.normalize("NFC", title)  # normalize title
    if title in title_index:
        jsonl_filename = title_index[title]["filename"]
        line_num = title_index[title]["line_num"]
        jsonl_filepath = os.path.join(path_to_document_trees, jsonl_filename)
        
        with open(jsonl_filepath, "r", encoding="utf-8") as f:
            for current_line_num, line in enumerate(f):
                if current_line_num == line_num:
                    print(f"=====json.loads=====")
                    print("jsonl_filepath: " + jsonl_filepath)
                    return json.loads(line.strip())
    else:
        print("=====None=====")
        return None


def read_file(input_path):
    data = []
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif input_path.endswith(".jsonl"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]
    elif input_path.endswith(".txt"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = [line.strip() for line in f.readlines]
    elif input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
        data = df.to_dict(orient="records")
    else:
        raise NotImplementedError
    
    print(f"\n***Loaded dataset size: {len(data)}.***\n")
    return data


def main():
    parser = argparse.ArgumentParser(description="Build title-to-file index for document-tree jsonl corpus.")
    parser.add_argument("--jsonl_dir", required=True, help="Folder containing document-tree jsonl files.")
    parser.add_argument("--output_index_file", required=True, help="Output title index json path.")
    args = parser.parse_args()

    start_time = time.time()
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    construct_file_index(args.jsonl_dir, args.output_index_file)
    elapsed = time.time() - start_time
    print(f"Done: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed: {timedelta(seconds=elapsed)}")


if __name__ == "__main__":
    main()
