import pandas as pd
import json
import numpy as np

def convert_parquet_to_json(parquet_path, json_path):
    """
    从 Parquet 文件中读取数据，通过遍历每一行来提取和重命名字段，然后保存为 JSON 文件。
    该版本手动处理每一行的 'golden_answers' 字段。

    Args:
        parquet_path (str): 输入的 Parquet 文件路径。
        json_path (str): 输出的 JSON 文件路径。
    """
    try:
        # 1. 读取 Parquet 文件
        print(f"正在读取 Parquet 文件: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        # 2. 检查所需列是否存在
        required_cols = ['id', 'question', 'golden_answers', 'reader']
        if not all(col in df.columns for col in required_cols):
            print(f"错误: Parquet 文件缺少必需的列。需要: {required_cols}, 存在: {list(df.columns)}")
            return

        # 3. 初始化一个空列表来存储转换后的数据
        data_list = []

        # 4. 遍历 DataFrame 的每一行
        print("正在逐行处理数据...")
        for index, row in df.iterrows():
            # 手动将 golden_answers 转换为 list
            # 无论原始类型是 ndarray 还是其他可迭代对象，list() 都能稳健处理
            golden_answers_list = list(row['golden_answers'])

            # 创建新的字典
            record = {
                'id': row['id'],
                'question': row['question'],
                'golden_answers': golden_answers_list,
                'output': row['reader']["answer"]
            }
            data_list.append(record)

        # 5. 将列表写入 JSON 文件
        print(f"正在写入 JSON 文件: {json_path}")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)
        
        print("转换成功！")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert short-form parquet results to JSON.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    convert_parquet_to_json(args.input_file, args.output_file)
