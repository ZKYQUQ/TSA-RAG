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
    """Collect all document titles needed by the input file."""
    print("Collecting required documents...")
    df = pd.read_parquet(input_file)
    
    required_titles = set()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing required docs"):
        graph_propagated_documents = row.get('graph_propagated_documents', [])
        
        for doc_result in graph_propagated_documents:
            propagation_result = doc_result.get('result')
            if not propagation_result:
                continue
                
            root_title = propagation_result.get('title', '')
            if root_title:
                required_titles.add(unicodedata.normalize("NFC", root_title))
            
            for hop in range(1, 11):
                hop_key = f'hop-{hop}'
                hop_nodes = propagation_result.get(hop_key, [])
                for node in hop_nodes:
                    node_title = node.get('title', '')
                    if node_title:
                        required_titles.add(unicodedata.normalize("NFC", node_title))
    
    print(f"Found {len(required_titles)} unique required documents")
    return required_titles

class OptimizedDocumentLoader:
    """Document loader that only loads required documents."""
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
        """Load only the required documents."""
        print(f"Loading {len(self.required_titles)} required documents...")
        
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
                                
                                optimized_doc = {
                                    'title': doc.get('title', ''),
                                    'extracted_nodes': doc.get('extracted_nodes', [])
                                }
                                
                                self.documents[title] = optimized_doc
                                loaded_count += 1
                                
                                if loaded_count >= len(required_lines):
                                    break
                                    
                            except json.JSONDecodeError as e:
                                raise ValueError(f"Error loading line {line_num} in {filename}: {e}")
                                continue
                                
            except Exception as e:
                raise ValueError(f"Error loading file {filename}: {e}")
                continue
        
            print(f"Successfully loaded {loaded_count} documents")
    def get_nodes_from_title(self, title):
        """Fetch document nodes from memory."""
        title = unicodedata.normalize("NFC", title)
        if title not in self.documents:
            raise ValueError(f"Title '{title}' not found in loaded documents")
        return self.documents[title]


def extract_headings_from_nodes(extracted_nodes):
    """Extract all heading nodes from extracted_nodes."""
    if not extracted_nodes:
        return []
    
    headings = []
    for node in extracted_nodes[1:]:
        if node.get("type") == "title":
            headings.append({"id": node["id"] , "text": node["text"]})
    
    return headings

def format_doc_content(title, intro, headings):
    """Format one document for the path prompt."""
    content = f"0: ={title}=\n"
    
    if intro and intro.strip():
        intro_lines = intro.strip().split('\n')
        for i, line in enumerate(intro_lines):
            if line.strip():
                content += f"    {i + 1}: {line.strip()}\n"
    
    for heading in headings:
        if heading["text"] and heading["text"].strip():
            heading_text = heading["text"].strip()
            if heading_text.startswith('=') and heading_text.endswith('='):
                level = 0
                temp = heading_text
                while temp.startswith('='):
                    level += 1
                    temp = temp[1:]
                
                indent = "    " * (level - 1)
                content += f"{indent}{heading['id']}: {heading_text}\n"
            else:
                raise ValueError(f"Heading '{heading_text}' does not start and end with '='")
    
    return content + "\n"

def format_path_prompt(path_docs_info):
    """Format a document path for prompting."""
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
    """Build all document paths from a graph propagation result."""
    if not propagation_result:
        return [], [], None

    root_title = propagation_result['title']
    root_intro = propagation_result['intro']
    
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
    
    root_doc_info["hop"] = 0
    root_doc_info["local_score"] = 100
    root_doc_info["global_score"] = 100
    root_doc_info["total_score"] = 100
    all_docs_info = [root_doc_info]
    
    hop1_nodes = propagation_result['hop-1']
    
    if len(hop1_nodes) == 0:
        formatted_paths = [format_path_prompt([root_info])]
        return all_docs_info, formatted_paths, root_info

    node_map = {root_title: root_info}
    children_map = defaultdict(list)
    
    max_hops = 5
    for hop in range(1, max_hops + 1):
        hop_key = f'hop-{hop}'
        hop_nodes = propagation_result[hop_key]
        
        for node in hop_nodes:
            node_title = node['title']
            node_intro = node['intro']
            source_title = node['source_title']
            
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
            
            children_map[source_title].append(node_title)
    
    formatted_paths = []
    
    for hop1_node in hop1_nodes:
        hop1_title = hop1_node['title']
        
        if hop1_title not in node_map:
            raise ValueError(f"Hop-1 node {hop1_title} not found in node map")
            continue
        
        path_nodes = [root_info]
        
        queue = deque([hop1_title])
        visited = {hop1_title}
        
        while queue:
            current_title = queue.popleft()
            
            path_nodes.append(node_map[current_title])
            
            for child_title in children_map[current_title]:
                if child_title not in visited:
                    visited.add(child_title)
                    queue.append(child_title)
        
        formatted_paths.append(format_path_prompt(path_nodes))
    
    return all_docs_info, formatted_paths, root_info

def load_existing_results(output_file):
    """Load existing results."""
    if os.path.exists(output_file):
        try:
            
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

def atomic_write_parquet(df, output_path):
        """Atomically write a Parquet file."""
        temp_dir = os.path.dirname(output_path)
        temp_file = tempfile.NamedTemporaryFile(
            dir=temp_dir, 
            suffix='.parquet.tmp', 
            delete=False
        )
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            print(f"Writing temporary file: {temp_path}")
            df.to_parquet(temp_path, index=False)
            
            print(f"Moving to final file: {output_path}")
            shutil.move(temp_path, output_path)
            print(f"Saved to: {output_path}")
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"Removed temporary file: {temp_path}")
            raise e
    
def save_batch_results(processed_rows, output_path):
    """Save a batch of processed rows."""
    if processed_rows:
        batch_df = pd.DataFrame(processed_rows)
        
        if os.path.exists(output_path):
            existing_df = pd.read_parquet(output_path)
            combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
            atomic_write_parquet(combined_df, output_path)
        else:
            atomic_write_parquet(batch_df, output_path)

        print(f"Saved {len(processed_rows)} records.")

def extract_paths(input_file, output_file, title_index_file, document_trees_path, batch_size=100):
    """Process a Parquet file and add path information."""
    
    required_titles = collect_required_documents(input_file)
    
    doc_loader = OptimizedDocumentLoader(title_index_file, document_trees_path, required_titles)
    
    print("Loading input data...")
    df = pd.read_parquet(input_file)
    
    existing_df, processed_sample_quries = load_existing_results(output_file)

    if "asqa" in output_file:
        query_str = 'ambiguous_question'
    elif "eli5" in output_file:
        query_str = 'title'
    elif "qampari" in output_file:
        query_str = 'question_text'
    else:
        query_str = 'question'

    if processed_sample_quries:
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
        new_row = copy.deepcopy(row)

        graph_propagated_documents = row['graph_propagated_documents']
        
        sample_doc_infos = []
        sample_formatted_paths = []
        
        for doc_result in graph_propagated_documents:
            doc_title = doc_result['title']
            propagation_result = doc_result['result']
            
            if propagation_result is None:
                print(f"Propagation result for {doc_title} is None(not found root doc), skipping...")
                continue
            
            doc_infos, formatted_paths, root_info = build_paths_from_propagation_result(
                propagation_result, doc_loader
            )

            if root_info is not None:
                formatted_paths.append(format_path_prompt([root_info]))
            
            sample_doc_infos.extend(doc_infos)
            sample_formatted_paths.extend(formatted_paths)
        
        new_row['doc_infos'] = sample_doc_infos
        new_row['formatted_paths'] = sample_formatted_paths

        del new_row['graph_propagated_documents']
        
        batch_processed.append(new_row)
        
    if batch_processed:
        save_batch_results(batch_processed, output_file)
    
    print(f"Processing complete! Final results saved to {output_file}")


def model_inference(input_file, output_file, model_path, max_new_tokens=512, batch_size=10):
    """Stage 1 substage 2: run model inference to extract relevant headings from paths."""
    
    print("Loading input data...")
    df = pd.read_parquet(input_file)
    
    if "asqa" in input_file:
        query_str = 'ambiguous_question'
    elif "eli5" in input_file:
        query_str = 'title'
    elif "qampari" in input_file:
        query_str = 'question_text'
    else:
        query_str = 'question'
    
    existing_df, processed_sample_queries = load_existing_results(output_file)
    
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
    
    print(f"Loading model from {model_path}...")
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    batch_processed = []
    
    for idx, row in tqdm(unprocessed_df.iterrows(), total=len(unprocessed_df), desc="Processing samples"):
        try:
            new_row = copy.deepcopy(row)
            question = row[query_str]
            formatted_paths = row['formatted_paths']
            
            path_extract_raw_output = []
            
            for path_info in formatted_paths:
                if not path_info or not path_info.get('path_text'):
                    continue
                
                prompt = construct_inference_prompt(path_info['path_text'], question)
                
                messages = [{"role": "user", "content": prompt}]
                
                preds = pipeline(
                    messages,
                    max_new_tokens=max_new_tokens,
                )
                raw_output = preds[0]["generated_text"][-1]["content"]
                path_result = {
                    "path_text": path_info['path_text'],
                    "doc_titles": path_info['doc_titles'],
                    "num_titles": path_info['num_titles'],
                    "raw_output": raw_output
                }
                
                path_extract_raw_output.append(path_result)
            
            new_row['path_extract_raw_output'] = path_extract_raw_output
            
            batch_processed.append(new_row)
            
            if len(batch_processed) >= batch_size:
                save_batch_results(batch_processed, output_file)
                batch_processed = []

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    if batch_processed:
        save_batch_results(batch_processed, output_file)
    
    print(f"Model inference complete! Results saved to {output_file}")



def construct_inference_prompt(path_text, question):
    """Construct the inference prompt."""
    instruction = ("Given a document tree composed of multiple structured documents, each consisting of a title, an introduction, and hierarchical headings, identify all headings that may contain answers to the specified question. List relevant headings with the tag [title] and [heading]. If none are relevant, reply exactly: \"No relevant headings\".")
    
    prompt = f"{instruction}\n\n## Document Tree\n{path_text}\n\n## Question\n{question}\n\n## Response"
    return prompt




def parse_heading_hierarchy(doc_outline):
    """
    Parse a document outline and return parent-child heading relationships.
    """
    lines = doc_outline.strip().split('\n')
    hierarchy = {}
    intro_ids = []
    stack = []  # (heading_id, level)
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
            
        heading_id = int(parts[0].strip())
        content = parts[1].strip()
        
        title_match = re.match(r'(=+)(.+?)\1', content)
        if title_match:
            level = len(title_match.group(1))
        else:
            intro_ids.append(heading_id)
            continue
        
        while stack and stack[-1][1] >= level:
            stack.pop()
        
        parent_id = stack[-1][0] if stack else -1
        
        hierarchy[heading_id] = {
            'level': level,
            'parent_id': parent_id,
            'children_ids': []
        }
        
        if parent_id != -1:
            hierarchy[parent_id]['children_ids'].append(heading_id)
        
        stack.append((heading_id, level))
    
    return hierarchy, intro_ids

def filter_path_extract_fun(doc_outline, raw_headings):
    """Filter heading extraction results for one document."""
    if not raw_headings:
        return []
    
    hierarchy, intro_ids = parse_heading_hierarchy(doc_outline)
    
    valid_heading_ids = set(hierarchy.keys())
    
    valid_raw_headings = [hid for hid in raw_headings if hid in valid_heading_ids]
    
    # Keep only the innermost selected headings.
    filtered_headings = set(valid_raw_headings)
    
    for heading_id in valid_raw_headings:
        if heading_id not in hierarchy:
            continue

        if heading_id == 0:
            continue
        
        children_ids = hierarchy[heading_id]['children_ids']
        if any(child_id in valid_raw_headings for child_id in children_ids):
            filtered_headings.discard(heading_id)
    
    final_headings = set(filtered_headings)
    
    if 0 not in final_headings:
        for intro_id in intro_ids:
            if intro_id in raw_headings:
                final_headings.add(0)
                break
    
    return sorted(list(final_headings))


def parse_model_output(raw_output):
    """Parse model output and extract titles plus heading ids."""
    if not raw_output or "No relevant headings".lower() in raw_output.lower():
        return []
    
    results = []
    current_title = None
    
    lines = raw_output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('[title]'):
            if len(line.strip().split(':', 1)) == 2:
                current_title = line.strip().split(':', 1)[-1].strip().strip('=').strip()
            else:
                current_title = line[7:].strip("0").strip().strip("=").strip()
        elif line.startswith('[heading]'):
            heading_content = line[9:].strip()
            
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
    """Parse path content into document titles and outlines while preserving order."""
    docs = {}
    doc_order = []
    
    doc_sections = re.split(r'(?:^|\n)\[([^\]]+)\]\n', path_content.strip())
    
    for i in range(1, len(doc_sections), 2):
        if i + 1 < len(doc_sections):
            doc_name = doc_sections[i]
            doc_content = doc_sections[i + 1].strip()
            
            lines = doc_content.split('\n')
            if lines:
                first_line = lines[0].strip()
                title_match = re.match(r'0:\s*=(.+)=', first_line)
                if title_match:
                    doc_title = title_match.group(1).strip()
                    docs[doc_title] = doc_content
                    doc_order.append(doc_title)
    
    return docs, doc_order


def validate_heading_id_in_outline(heading_id, doc_outline):
    """Return whether a heading id exists in the document outline."""
    lines = doc_outline.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith(f"{heading_id}:"):
            return True
    return False


def parse_and_filter_results(input_file, output_file, batch_size=10):
    """Stage 1 substage 3: parse and filter inference results."""
    
    print("Loading input data...")
    df = pd.read_parquet(input_file)
    
    if "asqa" in input_file:
        query_str = 'ambiguous_question'
    elif "eli5" in input_file:
        query_str = 'title'
    elif "qampari" in input_file:
        query_str = 'question_text'
    else:
        query_str = 'question'
    
    existing_df, processed_sample_queries = load_existing_results(output_file)
    
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
    
    batch_processed = []
    
    for idx, row in tqdm(unprocessed_df.iterrows(), total=len(unprocessed_df), desc="Processing samples"):
        try:
            new_row = copy.deepcopy(row)
            path_extract_raw_output = row['path_extract_raw_output']
            
            doc_results = {}  # {doc_title: {"doc_outline": str, "raw_headings": set, "scores": dict}}
            
            for doc_idx, path_output in enumerate(path_extract_raw_output):
                raw_output = path_output.get('raw_output', '')
                path_text = path_output.get('path_text', '')
                
                if not raw_output or not path_text:
                    continue
                
                extracted_results = parse_model_output(raw_output)
                
                if not extracted_results:
                    continue
                
                path_docs, doc_order = parse_path_documents_with_order(path_text)
                
                for result_item in extracted_results:
                    doc_title = result_item.get('doc_title', '').strip()
                    heading_id = result_item.get('heading_id')
                    
                    if not doc_title or heading_id is None:
                        continue
                    
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
                    
                    if doc_title not in doc_results:
                        doc_results[doc_title] = {
                            "doc_outline": doc_outline,
                            "raw_headings": set(),
                        }
                    doc_results[doc_title]["raw_headings"].add(heading_id)
            
            path_extract_results = []
            
            for doc_title, doc_data in doc_results.items():
                doc_outline = doc_data["doc_outline"]
                raw_headings = list(doc_data["raw_headings"])
                
                filtered_headings = filter_path_extract_fun(doc_outline, raw_headings)
                
                if filtered_headings:
                    path_extract_results.append({
                        "title": doc_title,
                        "headings": filtered_headings,
                    })

            new_row['path_extract_results'] = path_extract_results
            
            batch_processed.append(new_row)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    if batch_processed:
        save_batch_results(batch_processed, output_file)

    print(f"Parse and filter complete! Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Path Extract Inference Pipeline')
    
    parser.add_argument('--input_file', '-i', 
                        type=str, 
                        required=True,
                        help='Input Parquet file path.')
    
    parser.add_argument('--output_file', '-o', 
                        type=str, 
                        required=True,
                        help='Output Parquet file path.')
    
    parser.add_argument('--substage', 
                        type=str, 
                        required=True,
                        choices=['extract_paths', 'inference', 'parse_and_filter'],
                        help='Processing substage.')
    
    parser.add_argument('--title_index_file', 
                        type=str,
                        default="data/wiki/v2_title_index_w_links.json",
                        help='Title index file path required for extract_paths.')
    
    parser.add_argument('--document_trees_path', 
                        type=str,
                        default="data/wiki/document_tree_w_links",
                        help='Document tree directory required for extract_paths.')
    
    parser.add_argument('--model_path', 
                        type=str,
                        help='Model path required for inference.')
    
    parser.add_argument('--max_new_tokens', 
                        type=int, 
                        default=512,
                        help='Maximum new tokens for inference.')
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=10,
                        help='Batch size.')

    args = parser.parse_args()

    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    if args.substage == 'extract_paths':
        if not args.title_index_file or not args.document_trees_path:
            raise ValueError("extract_paths requires title_index_file and document_trees_path.")
        
        extract_paths(
            input_file=args.input_file,
            output_file=args.output_file,
            title_index_file=args.title_index_file,
            document_trees_path=args.document_trees_path,
            batch_size=args.batch_size
        )
    
    elif args.substage == 'inference':
        if not args.model_path:
            raise ValueError("inference requires model_path.")
        
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
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
