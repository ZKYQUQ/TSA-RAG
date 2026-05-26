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
        """Build the document tree."""
        self.nodes = {}
        self.leaves = []
        
        for data in raw_data["extracted_nodes"]:
            node = TreeNode(data["id"], data["text"], data["type"], data["span"])
            
            if data["id"] == 0 or (self.with_intro and data["type"] == "content" and data["relation"]["up_id"] == 0):
                node.lighted = True
            
            if len(data["relation"]["down_ids"]) == 0:
                self.leaves.append(node)
                
            self.nodes[node.id] = node
        
        self.nodes[0].text = "=" + self.nodes[0].text + "="

        for data in raw_data["extracted_nodes"]:
            current_node = self.nodes[data["id"]]
            up_id = data["relation"]["up_id"]
            
            if up_id != -1:
                parent_node = self.nodes[up_id]
                parent_node.add_child(current_node)
        
        self.leaves.sort(key=lambda x: x.id)
        
        root = None
        for node in self.nodes.values():
            if node.parent is None:
                root = node
                break
        
        return root
    
    def get_siblings(self, node):
        """Return siblings for a node."""
        if node.parent:
            return node.parent.children
        return [node]
    
    def light_heading_descendants(self, node):
        """Mark all descendants of a heading as lighted."""
        for child in node.children:
            child.lighted = True
            self.light_heading_descendants(child)
    
    def light_nodes_by_heading_ids(self, heading_ids):
        """Mark nodes related to the selected heading ids as lighted."""
        for heading_id in heading_ids:
            if heading_id not in self.nodes:
                continue

            if heading_id == 0:
                self.root.lighted = True
                continue
                
            target_node = self.nodes[heading_id]
            
            target_node.lighted = True
            
            self.light_heading_descendants(target_node)
            
            current_node = target_node
            while current_node.parent is not None:
                current_node = current_node.parent
                current_node.lighted = True
                
                if self.include_parent_siblings:
                    siblings = self.get_siblings(current_node)
                    for sibling in siblings:
                        if sibling.type == "content":
                            sibling.lighted = True
    
    def format_data(self, node, level: int = 0):
        """Format one node."""
        if node.type == "title":
            return f"{node.id}: {node.text}"
        elif node.type == "content":
            return f"{node.id}: {node.text}"
        else:
            return f"{node.id}: {node.text}"
    
    def traverse(self, node=None, level: int = 0, result: str = ""):
        """Traverse and format the document tree."""
        if node is None:
            node = self.root
            
        if node.lighted:
            indent = "    " * level
            result += indent + self.format_data(node, level) + "\n"
            
            for child in node.children:
                if child.lighted:
                    result = self.traverse(child, level + 1, result)
        
        return result

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

def extract_document_subtrees(input_file, output_file, with_intro=True, include_parent_siblings=True, batch_size=100, plain_tree=False):
    """Build document subtrees."""
    
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
            path_extract_results = row.get('path_extract_results', [])
            doc_infos = row.get('doc_infos', [])
            
            doc_info_map = {}
            for doc_info in doc_infos:
                title = doc_info.get('title', '')
                if title:
                    doc_info_map[title] = doc_info
            
            document_subtrees = []
            
            for extract_result in path_extract_results:
                doc_title = extract_result.get('title', '')
                headings = extract_result.get('headings', [])
                
                if not doc_title or len(headings) == 0:
                    continue
                
                if doc_title not in doc_info_map:
                    raise ValueError(f"Warning: Document '{doc_title}' not found in doc_infos")
                    continue
                
                doc_info = doc_info_map[doc_title]
                
                try:
                    if plain_tree:
                        doc_info_for_tree, headings = build_plain_doc_info(doc_info)
                    else:
                        doc_info_for_tree = doc_info
                    
                    doc_tree = DocumentTree(doc_info_for_tree, with_intro=with_intro, include_parent_siblings=include_parent_siblings)
                    
                    doc_tree.light_nodes_by_heading_ids(headings)
                    
                    tree_text = doc_tree.traverse()
                    
                    if tree_text.strip():
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
            
            new_row['document_subtrees'] = document_subtrees
            
            batch_processed.append(new_row)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    if batch_processed:
        save_batch_results(batch_processed, output_file)

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
    Run model inference over document_subtrees and add tree_extract_results.
    
    Args:
        input_file: Input Parquet file path.
        output_file: Output Parquet file path.
        model_path: Model path.
        max_new_tokens: Maximum number of generated tokens.
        batch_size: Batch size.
    """
    
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
            document_subtrees = row.get('document_subtrees', [])
            
            tree_extract_results = []
            
            for subtree in document_subtrees:
                title = subtree.get('title', '')
                doc_info = subtree.get('doc_info', {})
                document_tree = subtree.get('document_tree', '')
                path_extract_headings = subtree.get('path_extract_headings', [])
                
                if not title or not document_tree:
                    continue
                
                prompt = construct_tree_extract_prompt(document_tree, question)
                
                messages = [{"role": "user", "content": prompt}]
                
                preds = pipeline(
                    messages,
                    max_new_tokens=max_new_tokens,
                )
                raw_output = preds[0]["generated_text"][-1]["content"]
                
                tree_extract_results.append({
                    "title": title,
                    "doc_info": doc_info,
                    "raw_output": raw_output,
                    "local_score": doc_info.get("local_score", 0),
                    "global_score": doc_info.get("global_score", 0),
                    "total_score": doc_info.get("total_score", 0)
                })
            
            new_row['tree_extract_results'] = tree_extract_results
            
            batch_processed.append(new_row)
            
            if len(batch_processed) >= batch_size:
                save_batch_results(batch_processed, output_file)
                batch_processed = []

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    if batch_processed:
        save_batch_results(batch_processed, output_file)

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
        
        if line.startswith('[paragraph]'):
            paragraph_content = line[len('[paragraph]'):].strip()
            
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
    Parse raw_output in tree_extract_results, filter paragraphs, and build subtrees.
    
    Args:
        input_file: Input Parquet file path.
        output_file: Output Parquet file path.
        batch_size: Batch size.
    """
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
            tree_extract_results = row.get('tree_extract_results', [])
            
            for i, result in enumerate(tree_extract_results):
                if 'cleaned_output' in result and 'formated_subtree' in result:
                    continue
                
                raw_output = result.get('raw_output', '')
                doc_info = result.get('doc_info', {})
                
                paragraph_ids = parse_model_output(raw_output)
                
                valid_paragraph_ids = []
                
                try:
                    doc_tree = DocumentTree(doc_info, with_intro=True, include_parent_siblings=True)
                    
                    for paragraph_id in paragraph_ids:
                        if paragraph_id in doc_tree.nodes and doc_tree.nodes[paragraph_id].type == 'content':
                            valid_paragraph_ids.append(paragraph_id)
                            
                except Exception as e:
                    print(f"Error validating paragraphs: {e}")
                    valid_paragraph_ids = []
                
                new_row['tree_extract_results'][i]['cleaned_output'] = sorted(valid_paragraph_ids)
                
                if valid_paragraph_ids:
                    try:
                        doc_tree = DocumentTree(doc_info, with_intro=subtree_with_intro, include_parent_siblings=subtree_include_parent_siblings)
                        
                        for node_id in doc_tree.nodes:
                            doc_tree.nodes[node_id].lighted = False
                        
                        for paragraph_id in valid_paragraph_ids:
                            if paragraph_id in doc_tree.nodes:
                                doc_tree.nodes[paragraph_id].lighted = True
                                
                                current_node = doc_tree.nodes[paragraph_id]
                                while current_node.parent is not None:
                                    current_node = current_node.parent
                                    current_node.lighted = True
                        
                        formated_subtree = doc_tree.traverse()
                        new_row['tree_extract_results'][i]['formated_subtree'] = formated_subtree
                    
                    except Exception as e:
                        print(f"Error generating formatted subtree: {e}")
                        new_row['tree_extract_results'][i]['formated_subtree'] = ""
                else:
                    new_row['tree_extract_results'][i]['formated_subtree'] = ""
            
            new_row['tree_extract_results'] = sorted(
                new_row['tree_extract_results'], 
                key=lambda x: x['total_score'], 
                reverse=True
            )
            
            batch_processed.append(new_row)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    if batch_processed:
        save_batch_results(batch_processed, output_file)

    print(f"Parse and filter complete! Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Tree Extract Inference Pipeline')
    
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
                        choices=['extract_document_subtrees', 'inference', 'parse_and_filter'],
                        help='Processing substage.')
    
    parser.add_argument('--subtree_with_intro',
                        type=bool,
                        default=False,
                        help='Whether to include intro content nodes.')

    parser.add_argument('--subtree_include_parent_siblings',
                        type=bool,
                        default=False,
                        help='Whether to include content-type siblings of parent nodes while lighting ancestors.')

    parser.add_argument('--plain_tree',
                        type=bool,
                        default=False,
                        help='Whether to keep only the title node and all content nodes as a two-level tree.')

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
            batch_size=args.batch_size,
            subtree_with_intro=args.subtree_with_intro,
            subtree_include_parent_siblings=args.subtree_include_parent_siblings,
        )
    else:
        raise ValueError(f"Unknown substage: {args.substage}. Supported stages are: 'extract_document_subtrees', 'inference', 'parse_and_filter'.")


if __name__ == "__main__":
    main()
