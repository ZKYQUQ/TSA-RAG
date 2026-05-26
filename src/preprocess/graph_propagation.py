import unicodedata
import torch
import pandas as pd
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from neo4j import GraphDatabase
from tqdm import tqdm
from collections import defaultdict, deque
import torch.nn.functional as F
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import copy
import argparse
import tempfile
import shutil

class AttentionGraphQuery:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, 
                 contriever_model_path='facebook/contriever-msmarco'):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        self.tokenizer = AutoTokenizer.from_pretrained(contriever_model_path)
        self.model = AutoModel.from_pretrained(contriever_model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_query_embedding(self, query):
        """Compute the query embedding."""
        inputs = self.tokenizer([query], padding=True, truncation=True, 
                                return_tensors='pt', max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling.
        mask = inputs['attention_mask']
        token_embeddings = outputs[0].masked_fill(~mask[..., None].bool(), 0.)
        embedding = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return embedding[0].cpu()
    
    def get_document_by_title(self, title):
        title = unicodedata.normalize("NFC", title)  # normalize title
        """Fetch document metadata by title, including precomputed embedding."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document {title: $title})
                RETURN d.title AS title, d.intro AS intro, d.embedding AS embedding, d.degree AS degree
            """, title=title)
            
            record = result.single()
            if record:
                embedding = record["embedding"]
                return {
                    "title": record["title"],
                    "intro": record["intro"] or "",
                    "embedding": torch.tensor(embedding) if embedding else None,
                    "degree": record["degree"]
                }
            return None
    
    def get_neighbors(self, titles, max_degree_threshold=1000):
        """Fetch all neighbors for the given titles."""
        with self.driver.session() as session:
            result = session.run("""
                UNWIND $titles AS title
                MATCH (d:Document {title: title})-[:CONNECTED]->(neighbor:Document)
                RETURN d.title AS source_title, 
                    neighbor.title AS neighbor_title,
                    neighbor.intro AS neighbor_intro,
                    neighbor.embedding AS neighbor_embedding,
                    neighbor.degree AS neighbor_degree
            """, titles=titles)
            
            neighbors = defaultdict(list)
            for record in result:
                source = record["source_title"]
                embedding = record["neighbor_embedding"]
                neighbor = {
                    "title": record["neighbor_title"],
                    "intro": record["neighbor_intro"] or "",
                    "embedding": torch.tensor(embedding) if embedding else None,
                    "degree": record["neighbor_degree"]
                }
                neighbors[source].append(neighbor)
            
            return neighbors
    
    def calculate_path_similarity(self, path_embeddings):
        """Calculate path similarity as the product of adjacent similarities."""
        if len(path_embeddings) <= 1:
            return 1.0
        
        similarities = []
        for i in range(len(path_embeddings) - 1):
            sim = torch.cosine_similarity(
                path_embeddings[i].unsqueeze(0), 
                path_embeddings[i+1].unsqueeze(0)
            ).item()
            similarities.append(sim)
        
        path_score = 1.0
        for sim in similarities:
            path_score *= sim
        return path_score

    def propagate_from_document(self, query_embedding, root_title, max_hops=5, top_k=5, 
                               local_weight=0.5, global_weight=0.5, threshold=0.01, max_degree_threshold=1000):
        """
        Run attention-weighted graph propagation from a root document.
        
        Args:
            query_embedding: Query embedding.
            root_title: Root document title.
            max_hops: Maximum hop count.
            top_k: Number of nodes retained at each hop.
            local_weight: Local score weight.
            global_weight: Global score weight.
            threshold: Score threshold; lower-scoring nodes are filtered.
        """
        root_title = unicodedata.normalize("NFC", root_title)  # normalize title 
        root_doc = self.get_document_by_title(root_title)
        if not root_doc:
            print(f"Root document {root_title} does not exist; skipping graph propagation.")
            return None
        
        result = {
            "title": root_title,
            "intro": root_doc["intro"],
            "degree": root_doc["degree"],
            "embedding": root_doc["embedding"].tolist() if root_doc["embedding"] is not None else None,
            **{f"hop-{i}": [] for i in range(1, max_hops + 1)},
            "hop_nodes": []
        }
        
        if root_doc["degree"] > max_degree_threshold:
            print(f"Root document {root_title} has degree {root_doc['degree']} above the threshold; skipping.")
            current_queue = []
        else:
            current_queue = [root_title]
        
        visited = {root_title}
        
        # Store path embeddings for local score computation.
        node_paths = {root_title: [root_doc["embedding"]]}
        
        for hop in range(1, max_hops + 1):
            if not current_queue:
                result["hop_nodes"].append(0)
                for remaining_hop in range(hop + 1, max_hops + 1):
                    result["hop_nodes"].append(0)
                break
                
            all_neighbors = self.get_neighbors(current_queue, max_degree_threshold)
            
            candidate_neighbors = {}
            neighbor_sources = defaultdict(list)
            
            for source_title in current_queue:
                if source_title in all_neighbors:
                    for neighbor in all_neighbors[source_title]:
                        neighbor_title = neighbor["title"]
                        if neighbor_title not in visited:
                            candidate_neighbors[neighbor_title] = neighbor
                            neighbor_sources[neighbor_title].append(source_title)
            
            candidate_count = len(candidate_neighbors)
            result["hop_nodes"].append(candidate_count)

            if not candidate_neighbors:
                for remaining_hop in range(hop + 1, max_hops + 1):
                    result["hop_nodes"].append(0)
                break
            
            neighbor_scores = []
            
            for neighbor_title, neighbor_info in candidate_neighbors.items():
                neighbor_embedding = neighbor_info["embedding"]
                neighbor_degree = neighbor_info["degree"]
                
                max_local_score = None
                best_adjacent_score = None
                best_source_title = None
                
                for source_title in neighbor_sources[neighbor_title]:
                    source_path = node_paths[source_title]
                    source_embedding = source_path[-1]
                    
                    adjacent_score = torch.cosine_similarity(
                        source_embedding.unsqueeze(0),
                        neighbor_embedding.unsqueeze(0)
                    ).item()
                    
                    extended_path = source_path + [neighbor_embedding]
                    path_score = self.calculate_path_similarity(extended_path)
                    
                    local_score = path_score
                    
                    if max_local_score is None or local_score > max_local_score:
                        max_local_score = local_score
                        best_adjacent_score = adjacent_score
                        best_source_title = source_title
                
                global_score = torch.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    neighbor_embedding.unsqueeze(0)
                ).item()
                
                total_score = local_weight * max_local_score + global_weight * global_score
                
                neighbor_scores.append({
                    "title": neighbor_title,
                    "intro": candidate_neighbors[neighbor_title]["intro"],
                    "embedding": neighbor_embedding,
                    "adjacent_score": best_adjacent_score,
                    "local_score": max_local_score,
                    "global_score": global_score,
                    "total_score": total_score,
                    "source_title": best_source_title,
                    "degree": neighbor_degree
                })
            
            if neighbor_scores:
                scores = torch.tensor([n["total_score"] for n in neighbor_scores])
                softmax_scores = F.softmax(scores, dim=0)
                
                for i, neighbor in enumerate(neighbor_scores):
                    neighbor["softmax_score"] = softmax_scores[i].item()
                
                neighbor_scores.sort(key=lambda x: x["softmax_score"], reverse=True)
                
                for rank, neighbor in enumerate(neighbor_scores, 1):
                    neighbor["rank"] = rank
                
                filtered_neighbors = [
                    n for n in neighbor_scores 
                    if n["total_score"] >= threshold
                ]
                
                top_k_neighbors = filtered_neighbors[:top_k]
                
                hop_results = []
                for neighbor in top_k_neighbors:
                    hop_results.append({
                        "title": neighbor["title"],
                        "intro": neighbor["intro"],
                        "embedding": neighbor["embedding"].tolist() if neighbor["embedding"] is not None else None,
                        "degree": neighbor["degree"],
                        "adjacent_score": float(neighbor["adjacent_score"]),
                        "local_score": float(neighbor["local_score"]),
                        "global_score": float(neighbor["global_score"]),
                        "total_score": float(neighbor["total_score"]),
                        "softmax_score": float(neighbor["softmax_score"]),
                        "source_title": neighbor["source_title"],
                        "rank": neighbor["rank"]
                    })
                
                result[f"hop-{hop}"] = hop_results
                
                current_queue = []
                for neighbor in top_k_neighbors:
                    if neighbor["degree"] > max_degree_threshold:
                        continue

                    neighbor_title = neighbor["title"]
                    current_queue.append(neighbor_title)
                    visited.add(neighbor_title)
                    
                    best_source = None
                    best_local = None
                    for source_title in neighbor_sources[neighbor_title]:
                        source_path = node_paths[source_title]
                        source_embedding = source_path[-1]
                        
                        extended_path = source_path + [neighbor["embedding"]]
                        path_score = self.calculate_path_similarity(extended_path)
                        local_score = path_score

                        if best_local is None or local_score > best_local:
                            best_local = local_score
                            best_source = source_title
                    
                    if not best_source:
                        raise ValueError(f"No valid source found for neighbor {neighbor_title}")
                    node_paths[neighbor_title] = node_paths[best_source] + [neighbor["embedding"]]

        
        return result
    
    def process_single_document_propagation(self, args):
        """Run graph propagation for one document in parallel."""
        query_embedding, passage_title, max_hops, top_k, local_weight, global_weight, threshold, max_degree_threshold = args
        
        result = self.propagate_from_document(
            query_embedding=query_embedding,
            root_title=passage_title,
            max_hops=max_hops,
            top_k=top_k,
            local_weight=local_weight,
            global_weight=global_weight,
            threshold=threshold,
            max_degree_threshold=max_degree_threshold
        )
        return passage_title, result
    def load_existing_results(self, output_path):
        """Load an existing result file."""
        if os.path.exists(output_path):
            print(f"Found existing Parquet result file: {output_path}")
            existing_df = pd.read_parquet(output_path)
            print(f"Processed rows: {len(existing_df)}")
            return existing_df
        return None
    
    def atomic_write_parquet(self, df, output_path):
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
    
    def save_batch_results(self, processed_rows, output_path):
        """Save a batch of processed rows."""
        if processed_rows:
            batch_df = pd.DataFrame(processed_rows)
            
            if os.path.exists(output_path):
                existing_df = pd.read_parquet(output_path)
                combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                self.atomic_write_parquet(combined_df, output_path)
            else:
                self.atomic_write_parquet(batch_df, output_path)
            
            print(f"Saved {len(processed_rows)} records.")

    def process_parquet_file(self, input_path, output_path, max_hops=5, top_k=5,
                           local_weight=0.5, global_weight=0.5, threshold=0.01, 
                           batch_size=10, max_workers=None, max_degree_threshold=1000):
        """Process all rows in a Parquet file with incremental parallel propagation."""
        
        print("Configuration:")
        print(f"  input_path: {input_path}")
        print(f"  output_path: {output_path}")
        print(f"  max_hops: {max_hops}")
        print(f"  top_k: {top_k}")
        print(f"  local_weight: {local_weight}")
        print(f"  global_weight: {global_weight}")
        print(f"  threshold: {threshold}")
        print(f"  max_degree_threshold: {max_degree_threshold}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df = pd.read_parquet(input_path)
        filename = os.path.basename(input_path)
        
        existing_df = self.load_existing_results(output_path)
        
        if existing_df is not None:
            processed_count = len(existing_df)
            remaining_df = df.iloc[processed_count:].copy()
            print(f"Resuming from row {processed_count}.")
        else:
            remaining_df = df.copy()
            processed_count = 0
            print("Processing all rows.")
        
        if len(remaining_df) == 0:
            print("All rows have already been processed.")
            return
        
        processed_rows = []
        
        checkpoint_path = output_path.replace('.parquet', '_checkpoint.json')
        
        for idx, (_, row) in enumerate(tqdm(remaining_df.iterrows(), 
                                          total=len(remaining_df), 
                                          desc="Processing queries")):
            try:
                if "asqa" in filename:
                    query = row['ambiguous_question']
                elif "eli5" in filename:
                    query = row['title']
                elif "qampari" in filename:
                    query = row['question_text']
                else:
                    query = row["question"]
                    
                query_embedding = self.get_query_embedding(query)

                pre_retrieved_passages = row['pre_retrieved_passages']
                doc_titles = set(p['title'] for p in pre_retrieved_passages)

                task_args = []
                for passage in doc_titles:
                    passage_title = passage
                    task_args.append((
                        query_embedding, passage_title, max_hops, top_k, 
                        local_weight, global_weight, threshold, max_degree_threshold
                    ))

                graph_propagated_documents = []

                if max_workers is None:
                    actual_max_workers = min(len(task_args), 5)
                else:
                    actual_max_workers = min(max_workers, len(task_args))
                
                with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
                    future_to_title = {
                        executor.submit(self.process_single_document_propagation, args): args[1] 
                        for args in task_args
                    }
                    
                    for future in concurrent.futures.as_completed(future_to_title):
                        passage_title, result = future.result()
                        graph_propagated_documents.append({
                            "title": passage_title,
                            "result": result
                        })

                
                new_row = copy.deepcopy(row)
                new_row['graph_propagated_documents'] = graph_propagated_documents
                processed_rows.append(new_row)
                
                checkpoint_info = {
                    'processed_count': processed_count + idx + 1,
                    'total_count': len(df),
                    'current_batch_size': len(processed_rows)
                }
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_info, f)
                
                if len(processed_rows) >= batch_size:
                    self.save_batch_results(processed_rows, output_path)
                    processed_rows = []
                    
            except Exception as e:
                print(f"Error processing row {processed_count + idx}: {e}")
                if processed_rows:
                    print("Saving processed rows...")
                    self.save_batch_results(processed_rows, output_path)
                raise e
        
        if processed_rows:
            self.save_batch_results(processed_rows, output_path)
        
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"Removed checkpoint file: {checkpoint_path}")
        
        print(f"Processing complete. Final results saved to {output_path}")
    
    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

def main():
    parser = argparse.ArgumentParser(description='Graph propagation utility')
    
    parser.add_argument('--input_file', type=str, help='Input Parquet file path.')
    parser.add_argument('--output_file', type=str, help='Output Parquet file path.')
    parser.add_argument('--local_weight', type=float, default=0.4, help='path consistency score weight')
    parser.add_argument('--global_weight', type=float, default=0.6, help='query relevance score weight')
    parser.add_argument('--neo4j_uri', default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument('--neo4j_user', default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument('--neo4j_password', default=os.getenv("NEO4J_PASSWORD", "password"))
    parser.add_argument('--contriever_model_path', default=os.getenv("CONTRIEVER_MODEL_PATH", "facebook/contriever-msmarco"))
    parser.add_argument('--max_hops', type=int, default=5)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_workers', type=int, default=5)
    parser.add_argument('--max_degree_threshold', type=int, default=3000)

    args = parser.parse_args()
    
    query_engine = AttentionGraphQuery(
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        contriever_model_path=args.contriever_model_path,
    )
    
    try:
        print("\nProcessing Parquet file...")
        input_file = args.input_file
        output_file = args.output_file
        
        query_engine.process_parquet_file(
            input_path=input_file,
            output_path=output_file,
            max_hops=args.max_hops,
            top_k=args.top_k,
            local_weight=args.local_weight,
            global_weight=args.global_weight,
            threshold=0.1,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            max_degree_threshold=args.max_degree_threshold
        )
        
    finally:
        query_engine.close()

if __name__ == "__main__":
    main()
