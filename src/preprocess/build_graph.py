import os
import json
import torch
import numpy as np
from tqdm import tqdm
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor
import pyarrow as pa
import pyarrow.parquet as pq
from neo4j.exceptions import TransientError
import time
from transformers import AutoTokenizer, AutoModel
import argparse

class WikipediaGraphBuilderWithEmbedding:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, 
                 title_index_path, document_tree_path, 
                 contriever_model_path='facebook/contriever-msmarco',
                 batch_size=1000, embedding_batch_size=2048):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.title_index_path = title_index_path
        self.document_tree_path = document_tree_path
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        
        print("Loading Contriever model...")
        self.tokenizer = AutoTokenizer.from_pretrained(contriever_model_path)
        self.model = AutoModel.from_pretrained(contriever_model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")
    
    def get_contriever_embeddings(self, texts, batch_size=None):
        """Compute text embeddings in batches."""
        if batch_size is None:
            batch_size = self.embedding_batch_size
            
        if not texts:
            return []
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                   return_tensors='pt', max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean pooling.
            mask = inputs['attention_mask']
            token_embeddings = outputs[0].masked_fill(~mask[..., None].bool(), 0.)
            batch_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            embeddings.append(batch_embeddings.cpu())
        
        if embeddings:
            return torch.cat(embeddings, dim=0)
        else:
            raise ValueError("No embeddings computed, check input texts.")
    
    def format_embedded_text(self, title, intro):
        """Format document text for embedding."""
        return f"{title}\n{intro}"
    
    def _get_intro_text(self, extracted_nodes):
        """Extract the intro text from extracted_nodes."""
        intro_parts = []
        for node in extracted_nodes:
            if node["id"] == 0:
                continue
            if node["type"] == "content":
                intro_parts.append(node["text"])
            else:
                break
        return "\n".join(intro_parts)
    
    def _process_batch(self, filename, batch_lines):
        """Process one batch and prepare Neo4j writes."""
        create_nodes = []
        create_rels = []
        
        # Collect document text for batched embedding.
        texts_for_embedding = []
        processed_docs = []
        
        for line_num, line in batch_lines:       
            try:
                data = json.loads(line.strip())
                title = data["title"]
                intro = self._get_intro_text(data["extracted_nodes"])
                
                embedded_text = self.format_embedded_text(title, intro)
                texts_for_embedding.append(embedded_text)
                
                processed_docs.append({
                    "title": title,
                    "intro": intro,
                    "data": data
                })
                        
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num} in {filename}: {e}")
                continue
        
        if texts_for_embedding:
            try:
                embeddings = self.get_contriever_embeddings(texts_for_embedding)
            except Exception as e:
                print(f"Error computing embeddings for {filename}: {e}")
                raise Exception(f"Failed to compute embeddings for {filename}")
            
            for i, doc_info in enumerate(processed_docs):
                title = doc_info["title"]
                intro = doc_info["intro"]
                data = doc_info["data"]
                
                embedding_list = embeddings[i].numpy().tolist()
                
                create_nodes.append({
                    "title": title,
                    "intro": intro,
                    "embedding": embedding_list
                })
                
                for rel_type in ["see_also_docs", "external_links_docs", "intro_docs"]:
                    if rel_type not in data:
                        continue
                        
                    for related_title in data[rel_type]:
                        create_rels.append({
                            "source": title,
                            "target": related_title
                        })
                
        return create_nodes, create_rels
    
    def _write_to_neo4j(self, nodes, rels):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                """Write one batch of nodes and relationships to Neo4j."""
                with self.driver.session() as session:
                    if nodes:
                        session.run("""
                            UNWIND $nodes AS node
                            MERGE (d:Document {title: node.title})
                            SET d.intro = node.intro,
                                d.embedding = node.embedding
                        """, {"nodes": nodes})
                    
                    # Create bidirectional relationships for the undirected graph.
                    # MATCH (a:Document {title: rel.source})
                    # MATCH (b:Document {title: rel.target})
                    # Missing target nodes are created with only a title and can be
                    # enriched by later batches.
                    if rels:
                        session.run("""
                            UNWIND $rels AS rel
                            MERGE (a:Document {title: rel.source})
                            MERGE (b:Document {title: rel.target})   
                            MERGE (a)-[:CONNECTED]->(b)
                            MERGE (b)-[:CONNECTED]->(a)
                        """, {"rels": rels})
                    break
            except TransientError as e:
                if "DeadlockDetected" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                raise

    def _calculate_node_degrees(self):
        """Compute and update node degrees in batches."""
        with self.driver.session(database="neo4j") as session:
            total_result = session.run("MATCH (d:Document) RETURN count(d) as total")
            total_nodes = total_result.single()["total"]
            print(f"Computing degrees for {total_nodes} nodes...")
            
            batch_size = 10000
            processed = 0
            original_batch_size = batch_size
            
            for skip in range(0, total_nodes, batch_size):
                try:
                    session.run("""
                        MATCH (d:Document)
                        WITH d
                        SKIP $skip LIMIT $batch_size
                        OPTIONAL MATCH (d)-[:CONNECTED]->(neighbor)
                        WITH d, count(neighbor) as degree
                        SET d.degree = degree
                    """, skip=skip, batch_size=batch_size, timeout=60000)
                    
                    processed += min(batch_size, total_nodes - skip)
                    print(f"Processed {processed}/{total_nodes} nodes ({processed/total_nodes*100:.1f}%)")
                    
                    batch_size = original_batch_size
                    
                    if (skip // original_batch_size) % 10 == 0 and skip > 0:
                        time.sleep(2)
                        
                except Exception as e:
                    print(f"Error processing batch {skip}-{skip+batch_size}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise ValueError(f"Batch processing failed: {str(e)}")
                    
            
            print("Collecting degree statistics...")
            try:
                result = session.run("""
                    MATCH (d:Document)
                    WHERE d.degree IS NOT NULL
                    RETURN min(d.degree) as min_degree, 
                        max(d.degree) as max_degree, 
                        avg(d.degree) as avg_degree,
                        count(d) as total_nodes
                """)
                
                stats = result.single()
                if stats:
                    print(f"Degree stats - min: {stats['min_degree']}, max: {stats['max_degree']}, "
                        f"avg: {stats['avg_degree']:.2f}, total nodes: {stats['total_nodes']}")
            except Exception as e:
                print(f"Error collecting statistics: {str(e)}")
                import traceback
                traceback.print_exc()
                raise ValueError(f"Failed to collect statistics: {str(e)}")
        
    def build_graph(self):
        """Build the knowledge graph."""
        with self.driver.session() as session:
            session.run("CREATE INDEX document_title_index IF NOT EXISTS FOR (d:Document) ON (d.title)")
            print("Created document title index.")
        
        # Keep the worker count low because embedding uses GPU resources.
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for filename in os.listdir(self.document_tree_path):
                if not '.jsonl' in filename:
                    continue
                    
                filepath = os.path.join(self.document_tree_path, filename)
                futures.append(executor.submit(self._process_file, filepath))
                
            for future in tqdm(futures, desc="Processing files"):
                future.result()
        
        print("Computing node degrees...")
        self._calculate_node_degrees()

        
    def _process_file(self, filepath):
        """Process one jsonl file."""
        filename = os.path.basename(filepath)
        batch_lines = []
        
        print(f"Processing file: {filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                batch_lines.append((index, line))
                
                if len(batch_lines) >= self.batch_size:
                    nodes, rels = self._process_batch(filename, batch_lines)
                    self._write_to_neo4j(nodes, rels)
                    batch_lines = []
                    
                    if (index + 1) % (self.batch_size * 10) == 0:
                        print(f"Processed {index + 1} lines from {filename}")
                                    
            if batch_lines:
                nodes, rels = self._process_batch(filename, batch_lines)
                self._write_to_neo4j(nodes, rels)
                
        print(f"Finished file: {filename}")
    
    def close(self):
        self.driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the global document graph in Neo4j with Contriever embeddings.")
    parser.add_argument("--neo4j_uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j_user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j_password", default=os.getenv("NEO4J_PASSWORD", "password"))
    parser.add_argument("--title_index_path", required=True)
    parser.add_argument("--document_tree_path", required=True)
    parser.add_argument("--contriever_model_path", default=os.getenv("CONTRIEVER_MODEL_PATH", "facebook/contriever-msmarco"))
    parser.add_argument("--batch_size", type=int, default=40000)
    parser.add_argument("--embedding_batch_size", type=int, default=1024)
    parser.add_argument("--degree_only", action="store_true", help="Only recompute node degrees for an existing graph.")
    args = parser.parse_args()
    
    builder = WikipediaGraphBuilderWithEmbedding(
        args.neo4j_uri, args.neo4j_user, args.neo4j_password,
        args.title_index_path, args.document_tree_path,
        contriever_model_path=args.contriever_model_path,
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size
    )
    
    try:
        if args.degree_only:
            builder._calculate_node_degrees()
        else:
            builder.build_graph()
    finally:
        builder.close()
