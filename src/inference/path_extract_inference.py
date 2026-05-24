import pandas as pd
import json
import os
import unicodedata
from collections import defaultdict, deque
from tqdm import tqdm
import copy
import argparse
import transformers
import torch
import re
import tempfile
import shutil

torch.manual_seed(29)

def collect_required_documents(input_file):
    """预先统计输入文件中需要的所有文档标题"""
    print("Collecting required documents...")
    df = pd.read_parquet(input_file)
    
    required_titles = set()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing required docs"):
        graph_propagated_documents = row.get('graph_propagated_documents', [])
        
        for doc_result in graph_propagated_documents:
            propagation_result = doc_result.get('result')
            if not propagation_result:
                continue
                
            # 收集根节点标题
            root_title = propagation_result.get('title', '')
            if root_title:
                # required_titles.add(root_title)
                required_titles.add(unicodedata.normalize("NFC", root_title))
            
            # 收集所有hop节点的标题
            for hop in range(1, 11):  # hop-1到hop-5
                hop_key = f'hop-{hop}'
                hop_nodes = propagation_result.get(hop_key, [])
                for node in hop_nodes:
                    node_title = node.get('title', '')
                    if node_title:
                        # required_titles.add(node_title)
                        required_titles.add(unicodedata.normalize("NFC", node_title))
    
    print(f"Found {len(required_titles)} unique required documents")
    return required_titles

class OptimizedDocumentLoader:
    """优化的文档加载器，只加载需要的文档"""
    def __init__(self, title_index_file, document_trees_path, required_titles=None):
        print("Loading title index...")
        with open(title_index_file, "r", encoding="utf-8") as f:
            self.title_index = json.load(f)
        
        self.document_trees_path = document_trees_path
        self.documents = {}
        self.required_titles = required_titles or set()
        
        if self.required_titles:
            self._load_required_documents()
        else:
            print("Warning: No required titles provided, will load on demand")
    
    def _load_required_documents(self):
        """只加载需要的文档"""
        print(f"Loading {len(self.required_titles)} required documents...")
        
        # 构建文件到所需文档的映射
        files_to_docs = defaultdict(list)
        
        for title in self.required_titles:
            if title not in self.title_index:
                raise ValueError(f"Warning: Title '{title}' not found in title index")
                continue
            if title in self.title_index:
                title_info = self.title_index[title]
                filename = title_info["filename"]
                line_num = title_info["line_num"]
                files_to_docs[filename].append((title, line_num))
        
        print(f"Documents spread across {len(files_to_docs)} files")
        
        # loaded_count = 0
        
        # 按文件顺序加载，每个文件只读一次
        for filename in tqdm(files_to_docs.keys(), desc="Loading required documents"):
            loaded_count = 0
            filepath = os.path.join(self.document_trees_path, filename)
            required_lines = {line_num: title for title, line_num in files_to_docs[filename]}
            print(f"\nLoading {len(required_lines)} required lines from {filename}")
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f):
                        if line_num in required_lines:
                            try:
                                doc = json.loads(line.strip())
                                title = required_lines[line_num]
                                assert doc.get('title') == title, f"Title mismatch: {doc.get('title')} != {title}"
                                
                                # 只保留必要字段，减少内存占用
                                optimized_doc = {
                                    'title': doc.get('title', ''),
                                    'extracted_nodes': doc.get('extracted_nodes', [])
                                }
                                
                                self.documents[title] = optimized_doc
                                loaded_count += 1
                                
                                # 如果该文件的所有文档都已加载，提前退出
                                if loaded_count >= len(required_lines):
                                    break
                                    
                            except json.JSONDecodeError as e:
                                raise ValueError(f"Error loading line {line_num} in {filename}: {e}")
                                continue
                                
            except Exception as e:
                raise ValueError(f"Error loading file {filename}: {e}")
                continue
        
            print(f"Successfully loaded {loaded_count} documents")
        # print(f"Memory usage reduced by loading only required documents")
    
    def get_nodes_from_title(self, title):
        """从内存中获取文档信息"""
        title = unicodedata.normalize("NFC", title)
        if title not in self.documents:
            raise ValueError(f"Title '{title}' not found in loaded documents")
        return self.documents[title]


def extract_headings_from_nodes(extracted_nodes):
    """从extracted_nodes中提取所有标题节点"""
    if not extracted_nodes:
        return []
    
    headings = []
    for node in extracted_nodes[1:]:
        if node.get("type") == "title":
            headings.append({"id": node["id"] , "text": node["text"]})
    
    return headings

def format_doc_content(title, intro, headings):
    """格式化单个文档的内容"""
    content = f"0: ={title}=\n"
    
    if intro and intro.strip():
        # 将intro按行分割，每行前加4个空格
        intro_lines = intro.strip().split('\n')
        for i, line in enumerate(intro_lines):
            if line.strip():
                content += f"    {i + 1}: {line.strip()}\n"
    
    # 添加标题
    for heading in headings:
        if heading["text"] and heading["text"].strip():
            # 计算标题等级（通过=的数量）
            heading_text = heading["text"].strip()
            if heading_text.startswith('=') and heading_text.endswith('='):
                # 已经有=标记，直接添加缩进
                level = 0
                temp = heading_text
                while temp.startswith('='):
                    level += 1
                    temp = temp[1:]
                
                # 每个等级缩进4个空格
                indent = "    " * (level - 1)
                content += f"{indent}{heading['id']}: {heading_text}\n"
            else:
                raise ValueError(f"Heading '{heading_text}' does not start and end with '='")
    
    return content + "\n"

def format_path_prompt(path_docs_info):
    """格式化路径为prompt格式"""
    if not path_docs_info:
        return ""
    
    prompt = ""
    
    doc_titles = []

    for i, doc_info in enumerate(path_docs_info):
        title = doc_info['title']
        intro = doc_info['intro']
        headings = doc_info['headings']

        doc_titles.append(title)
        
        if i == 0:
            prompt += "[Root Doc]\n"
        else:
            prompt += f"[Doc{i}]\n"
        
        doc_content = format_doc_content(title, intro, headings)
        prompt += doc_content

    return {"path_text": prompt.strip(), "doc_titles": doc_titles, "num_titles": len(doc_titles)}

def build_paths_from_propagation_result(propagation_result, doc_loader):
    """从图传播结果构建所有路径"""
    if not propagation_result:
        return [], [], None

    # 获取根节点信息
    root_title = propagation_result['title']
    root_intro = propagation_result['intro']
    
    # 获取根节点的完整信息
    root_doc_info = doc_loader.get_nodes_from_title(root_title)
    if root_doc_info:
        root_headings = extract_headings_from_nodes(root_doc_info['extracted_nodes'])
    else:
        raise ValueError(f"Root node {root_title} not found in document trees")
    
    root_info = {
        'title': root_title,
        'intro': root_intro,
        'headings': root_headings,
        'full_info': root_doc_info
    }
    
    # 收集所有文档信息
    root_doc_info["hop"] = 0
    root_doc_info["local_score"] = 100
    root_doc_info["global_score"] = 100
    root_doc_info["total_score"] = 100
    all_docs_info = [root_doc_info]
    
    # 检查是否有hop-1节点
    hop1_nodes = propagation_result['hop-1']
    
    if len(hop1_nodes) == 0:
        # 只有根节点的情况
        formatted_paths = [format_path_prompt([root_info])]
        return all_docs_info, formatted_paths, root_info

    # 构建节点映射和父子关系
    node_map = {root_title: root_info}
    children_map = defaultdict(list)
    
    # 处理所有hop的节点，构建完整的图结构
    max_hops = 5
    for hop in range(1, max_hops + 1):
        hop_key = f'hop-{hop}'
        hop_nodes = propagation_result[hop_key]
        
        for node in hop_nodes:
            node_title = node['title']
            node_intro = node['intro']
            source_title = node['source_title']
            
            # 获取节点的完整信息
            node_doc_info = doc_loader.get_nodes_from_title(node_title)
            if node_doc_info:
                node_headings = extract_headings_from_nodes(node_doc_info['extracted_nodes'])
                node_doc_info["hop"] = hop
                node_doc_info["local_score"] = node["local_score"]
                node_doc_info["global_score"] = node["global_score"]
                node_doc_info["total_score"] = node["total_score"]
                all_docs_info.append(node_doc_info)
            else:
                raise ValueError(f"Node {node_title} not found in document trees")
            
            node_info = {
                'title': node_title,
                'intro': node_intro,
                'headings': node_headings,
                'full_info': node_doc_info
            }
            
            node_map[node_title] = node_info
            
            # 建立父子关系
            children_map[source_title].append(node_title)
    
    # 为每个hop-1节点构建路径
    formatted_paths = []
    
    for hop1_node in hop1_nodes:
        hop1_title = hop1_node['title']
        
        if hop1_title not in node_map:
            raise ValueError(f"Hop-1 node {hop1_title} not found in node map")
            continue
        
        # 构建以hop1_title为根的子树的所有节点（BFS顺序）
        path_nodes = [root_info]  # 路径从根节点开始
        
        # BFS遍历以hop1_title为根的子树
        queue = deque([hop1_title])
        visited = {hop1_title}
        
        while queue:
            current_title = queue.popleft()
            
            # 添加当前节点到路径
            path_nodes.append(node_map[current_title])
            
            # 将子节点加入队列
            for child_title in children_map[current_title]:
                if child_title not in visited:
                    visited.add(child_title)
                    queue.append(child_title)
        
        formatted_paths.append(format_path_prompt(path_nodes))
    
    return all_docs_info, formatted_paths, root_info

def load_existing_results(output_file):
    """加载已有的结果文件"""
    if os.path.exists(output_file):
        try:
            
            if "asqa" in output_file:
                query_str = 'ambiguous_question'
            elif "eli5" in output_file:
                query_str = 'title'
            elif "qampari" in output_file:
                query_str = 'question_text'
            # elif "hotpotqa" in output_file:
            #     query_str = 'question'
            else:
                raise ValueError("Unsupported output file type for loading existing results")
            
            existing_df = pd.read_parquet(output_file)
            # processed_sample_ids = set(existing_df['sample_id'].tolist())
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

def extract_paths(input_file, output_file, title_index_file, document_trees_path, batch_size=100):
    """处理parquet文件，添加路径信息"""
    
    # # 初始化文档加载器
    # doc_loader = DocumentLoader(title_index_file, document_trees_path)

    # 预先收集需要的文档
    required_titles = collect_required_documents(input_file)
    
    # 初始化优化的文档加载器
    doc_loader = OptimizedDocumentLoader(title_index_file, document_trees_path, required_titles)
    
    # 读取输入数据
    print("Loading input data...")
    df = pd.read_parquet(input_file)
    
    # 加载已有结果
    existing_df, processed_sample_quries = load_existing_results(output_file)

    if "asqa" in output_file:
        query_str = 'ambiguous_question'
    elif "eli5" in output_file:
        query_str = 'title'
    elif "qampari" in output_file:
        query_str = 'question_text'
    else:
        query_str = 'question'

    # 过滤未处理的数据
    if processed_sample_quries:
        # unprocessed_mask = ~df['sample_id'].isin(processed_sample_ids)
        unprocessed_mask = ~df[query_str].isin(processed_sample_quries)
        unprocessed_df = df[unprocessed_mask].reset_index(drop=True)
        print(f"Found {len(unprocessed_df)} unprocessed samples out of {len(df)} total samples")
    else:
        unprocessed_df = df
        print(f"Processing all {len(df)} samples")
    
    if len(unprocessed_df) == 0:
        print("All samples have been processed!")
        return

    batch_processed = []
    for idx, row in tqdm(unprocessed_df.iterrows(), total=len(unprocessed_df), desc="Processing samples"):
        # try:
        # 创建新行数据
        new_row = copy.deepcopy(row)

        # 获取图传播结果
        graph_propagated_documents = row['graph_propagated_documents']
        
        sample_doc_infos = []
        sample_formatted_paths = []
        
        # 处理每个检索文档的图传播结果
        for doc_result in graph_propagated_documents:
            doc_title = doc_result['title']
            propagation_result = doc_result['result']
            
            if propagation_result is None:
                print(f"Propagation result for {doc_title} is None(not found root doc), skipping...")
                # sample_doc_infos.append([])
                # sample_formatted_paths.append([])
                continue
            
            # 构建路径
            doc_infos, formatted_paths, root_info = build_paths_from_propagation_result(
                propagation_result, doc_loader
            )

            if root_info is not None:
                formatted_paths.append(format_path_prompt([root_info]))
            
            # sample_doc_infos.append(doc_infos)
            # sample_formatted_paths.append(formatted_paths)
            sample_doc_infos.extend(doc_infos)
            sample_formatted_paths.extend(formatted_paths)
        
        # 添加新字段
        new_row['doc_infos'] = sample_doc_infos
        new_row['formatted_paths'] = sample_formatted_paths

        del new_row['graph_propagated_documents']
        
        batch_processed.append(new_row)
        
        # # 每batch_size条数据保存一次
        # if len(batch_processed) >= batch_size:
        #     # all_processed_rows.extend(batch_processed)
        #     # save_batch_results(all_processed_rows, output_file)
        #     # batch_processed = []
        #     # print(f"Processed {idx + 1}/{len(unprocessed_df)} samples")
        #     save_batch_results(batch_processed, output_file)
        #     batch_processed = []  # 清空缓存

        # except Exception as e:
        #     print(f"Error processing sample {idx} (query: {row[query_str]}): {e}")
        #     # 跳过出错的样本，继续处理下一个
        #     continue
    
    # 保存剩余的数据
    if batch_processed:
        # all_processed_rows.extend(batch_processed)
        # save_batch_results(all_processed_rows, output_file)
        save_batch_results(batch_processed, output_file)
    
    print(f"Processing complete! Final results saved to {output_file}")


def model_inference(input_file, output_file, model_path, max_new_tokens=512, batch_size=10):
    """Stage 1 substage 2: 模型推理，针对所有path抽取相关headings"""
    
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
            # new_row = row.to_dict()
            new_row = copy.deepcopy(row)
            question = row[query_str]
            formatted_paths = row['formatted_paths']
            
            # 处理每个检索文档的路径
            path_extract_raw_output = []
            
            # for doc_paths in formatted_paths:
            #     doc_path_outputs = []
                
            for path_info in formatted_paths:
                if not path_info or not path_info.get('path_text'):
                    continue
                
                # 构造推理prompt
                prompt = construct_inference_prompt(path_info['path_text'], question)
                
                # 模型推理
                messages = [{"role": "user", "content": prompt}]
                
                # try:
                preds = pipeline(
                    messages,
                    max_new_tokens=max_new_tokens,
                    # do_sample=False,
                    # temperature=0.0,
                )
                raw_output = preds[0]["generated_text"][-1]["content"]
                # except Exception as e:
                #     print(f"Model inference error: {e}")
                    
                
                # 保存结果
                path_result = {
                    "path_text": path_info['path_text'],
                    "doc_titles": path_info['doc_titles'],
                    "num_titles": path_info['num_titles'],
                    "raw_output": raw_output
                }
                
                path_extract_raw_output.append(path_result)
            
            # 添加新字段
            new_row['path_extract_raw_output'] = path_extract_raw_output
            
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
    
    print(f"Model inference complete! Results saved to {output_file}")



def construct_inference_prompt(path_text, question):
    """构造推理prompt"""
    instruction = ("Given a document tree composed of multiple structured documents, each consisting of a title, an introduction, and hierarchical headings, identify all headings that may contain answers to the specified question. List relevant headings with the tag [title] and [heading]. If none are relevant, reply exactly: \"No relevant headings\".")
    
    prompt = f"{instruction}\n\n## Document Tree\n{path_text}\n\n## Question\n{question}\n\n## Response"
    return prompt




def parse_heading_hierarchy(doc_outline):
    """
    解析文档大纲的层级结构，返回heading之间的父子关系
    """
    lines = doc_outline.strip().split('\n')
    hierarchy = {}
    intro_ids = []
    stack = []  # (heading_id, level)
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 解析heading格式：id: =title= 或 id: content
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
            
        heading_id = int(parts[0].strip())
        content = parts[1].strip()
        
        # 判断是否为标题（包含=号）
        title_match = re.match(r'(=+)(.+?)\1', content)
        if title_match:
            level = len(title_match.group(1))
        else:
            # 非标题内容
            intro_ids.append(heading_id)
            continue
        
        # 维护层级关系
        while stack and stack[-1][1] >= level:
            stack.pop()
        
        parent_id = stack[-1][0] if stack else -1
        
        hierarchy[heading_id] = {
            'level': level,
            'parent_id': parent_id,
            'children_ids': []
        }
        
        # 更新父节点的children_ids
        if parent_id != -1:
            hierarchy[parent_id]['children_ids'].append(heading_id)
        
        stack.append((heading_id, level))
    
    return hierarchy, intro_ids

def filter_path_extract_fun(doc_outline, raw_headings):
    """过滤单个文档的heading提取结果"""
    if not raw_headings:
        return []
    
    # 解析文档层级结构
    hierarchy, intro_ids = parse_heading_hierarchy(doc_outline)
    
    # 获取所有有效的heading_ids
    valid_heading_ids = set(hierarchy.keys())
    
    # 过滤掉不存在的heading_id
    valid_raw_headings = [hid for hid in raw_headings if hid in valid_heading_ids]
    
    # 处理父子关系：只保留最内层的headings
    filtered_headings = set(valid_raw_headings)
    
    for heading_id in valid_raw_headings:
        if heading_id not in hierarchy:
            continue

        # 如果当前heading_id是0（标题），则跳过
        if heading_id == 0:
            continue
        
        # 如果当前heading的任何子节点也在raw_headings中，则移除当前heading
        children_ids = hierarchy[heading_id]['children_ids']
        if any(child_id in valid_raw_headings for child_id in children_ids):
            filtered_headings.discard(heading_id)
    
    # 处理title heading（通常是0号）
    final_headings = set(filtered_headings)
    
    if 0 not in final_headings:
        for intro_id in intro_ids:
            if intro_id in raw_headings:
                final_headings.add(0)
                break
    
    return sorted(list(final_headings))


def parse_model_output(raw_output):
    """解析模型输出，提取标题和headings"""
    if not raw_output or "No relevant headings".lower() in raw_output.lower():
        return []
    
    results = []
    current_title = None
    
    lines = raw_output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 检查是否是标题行
        if line.startswith('[title]'):
            if len(line.strip().split(':', 1)) == 2:
                current_title = line.strip().split(':', 1)[-1].strip().strip('=').strip()
            else:
                current_title = line[7:].strip("0").strip().strip("=").strip()
        elif line.startswith('[heading]'):
            # 提取heading信息
            heading_content = line[9:].strip()
            
            # 解析heading_id
            parts = heading_content.split(':', 1)
            if len(parts) == 2:
                try:
                    heading_id = int(parts[0].strip().strip("="))
                    if current_title:
                        results.append({
                            "doc_title": current_title,
                            "heading_id": heading_id
                        })
                except ValueError:
                    print(f"Invalid heading ID format in line: {line}")
                    continue
        else:
            print(f"Unexpected line format: {line}")
    
    return results


def parse_path_documents_with_order(path_content):
    """解析path内容，提取各个文档的标题和大纲，同时保持文档出现的顺序"""
    docs = {}
    doc_order = []
    
    # 按[Root Doc]、[Doc1]等分割文档
    doc_sections = re.split(r'(?:^|\n)\[([^\]]+)\]\n', path_content.strip())
    
    for i in range(1, len(doc_sections), 2):
        if i + 1 < len(doc_sections):
            doc_name = doc_sections[i]
            doc_content = doc_sections[i + 1].strip()
            
            # 提取文档标题（第一行通常是标题）
            lines = doc_content.split('\n')
            if lines:
                # 解析第一行获取标题
                first_line = lines[0].strip()
                # title_match = re.match(r'0:\s*=(.+?)=', first_line)
                title_match = re.match(r'0:\s*=(.+)=', first_line)
                if title_match:
                    doc_title = title_match.group(1).strip()
                    docs[doc_title] = doc_content
                    doc_order.append(doc_title)
    
    return docs, doc_order


def validate_heading_id_in_outline(heading_id, doc_outline):
    """验证heading_id是否在文档大纲中存在"""
    lines = doc_outline.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith(f"{heading_id}:"):
            return True
    return False


def parse_and_filter_results(input_file, output_file, batch_size=10):
    """Stage 1 substage 3: 推理结果解析、过滤"""
    
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
            # new_row = row.to_dict()
            new_row = copy.deepcopy(row)
            path_extract_raw_output = row['path_extract_raw_output']
            # doc_infos = row['doc_infos']
            
            # 用于合并同一个样本下所有path的结果
            doc_results = {}  # {doc_title: {"doc_outline": str, "raw_headings": set, "scores": dict}}
            
            # 处理每个检索文档的路径输出
            for doc_idx, path_output in enumerate(path_extract_raw_output):
                # for path_output in doc_path_outputs:
                raw_output = path_output.get('raw_output', '')
                path_text = path_output.get('path_text', '')
                
                if not raw_output or not path_text:
                    continue
                
                # 解析模型输出
                extracted_results = parse_model_output(raw_output)
                
                if not extracted_results:
                    continue
                
                # 解析path中的文档
                path_docs, doc_order = parse_path_documents_with_order(path_text)
                
                # 验证并合并结果
                for result_item in extracted_results:
                    doc_title = result_item.get('doc_title', '').strip()
                    heading_id = result_item.get('heading_id')
                    
                    if not doc_title or heading_id is None:
                        continue
                    
                    # 检查doc_title和heading_id是否在path中存在
                    if doc_title not in path_docs:
                        print(f"idx: {idx},  doc_idx: {doc_idx}")
                        print(f"raw_output: {raw_output}")
                        print(f"\nDocument title '{doc_title}' not found in path docs, skipping...")
                        print(f"Path docs available: {list(path_docs.keys())}")
                        print(f"Path text: {path_text}")
                        continue
                    
                    doc_outline = path_docs[doc_title]
                    if not validate_heading_id_in_outline(heading_id, doc_outline):
                        continue
                    
                    # 合并到样本级别结果
                    if doc_title not in doc_results:
                        doc_results[doc_title] = {
                            "doc_outline": doc_outline,
                            "raw_headings": set(),
                            # "scores": doc_info_map.get(doc_title, {
                            #     "local_score": 0,
                            #     "global_score": 0,
                            #     "total_score": 0
                            # })
                        }
                    doc_results[doc_title]["raw_headings"].add(heading_id)
            
            # 过滤每个文档的结果
            path_extract_results = []
            
            for doc_title, doc_data in doc_results.items():
                doc_outline = doc_data["doc_outline"]
                raw_headings = list(doc_data["raw_headings"])
                
                # 过滤headings
                filtered_headings = filter_path_extract_fun(doc_outline, raw_headings)
                
                if filtered_headings:
                    path_extract_results.append({
                        "title": doc_title,
                        "headings": filtered_headings,
                        # "local_score": scores["local_score"],
                        # "global_score": scores["global_score"],
                        # "total_score": scores["total_score"]
                    })
            
            # # 按total_score排序
            # path_extract_results.sort(key=lambda x: x["total_score"], reverse=True)
            
            # 添加新字段
            new_row['path_extract_results'] = path_extract_results
            
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
    parser = argparse.ArgumentParser(description='Path Extract Inference Pipeline')
    
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
                        choices=['extract_paths', 'inference', 'parse_and_filter'],
                        help='处理阶段')
    
    # extract_paths 阶段参数
    parser.add_argument('--title_index_file', 
                        type=str,
                        default="data/wiki/v2_title_index_w_links.json",
                        help='标题索引文件路径（extract_paths阶段需要）')
    
    parser.add_argument('--document_trees_path', 
                        type=str,
                        default="data/wiki/document_tree_w_links",
                        help='文档树目录路径（extract_paths阶段需要）')
    
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
    
    # 输出到新parquet文件
    # 这里batch_size设置大一点，不然频繁io写入速度非常慢，可以设置为100或更大
    if args.substage == 'extract_paths':
        if not args.title_index_file or not args.document_trees_path:
            raise ValueError("extract_paths阶段需要提供title_index_file和document_trees_path参数")
        
        extract_paths(
            input_file=args.input_file,
            output_file=args.output_file,
            title_index_file=args.title_index_file,
            document_trees_path=args.document_trees_path,
            batch_size=args.batch_size
        )
    
    # 输出到新parquet文件
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
    
    # 输出到新parquet文件
    elif args.substage == 'parse_and_filter':
        parse_and_filter_results(
            input_file=args.input_file,
            output_file=args.output_file,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
