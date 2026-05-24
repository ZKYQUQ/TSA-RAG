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
        
        # 初始化Contriever模型
        print("正在加载Contriever模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(contriever_model_path)
        self.model = AutoModel.from_pretrained(contriever_model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"模型已加载到设备: {self.device}")
    
    def get_contriever_embeddings(self, texts, batch_size=None):
        """批量计算文本嵌入"""
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
            # 平均池化
            mask = inputs['attention_mask']
            token_embeddings = outputs[0].masked_fill(~mask[..., None].bool(), 0.)
            batch_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            embeddings.append(batch_embeddings.cpu())
        
        if embeddings:
            return torch.cat(embeddings, dim=0)
        else:
            raise ValueError("No embeddings computed, check input texts.")
    
    def format_embedded_text(self, title, intro):
        """格式化文档嵌入文本"""
        return f"{title}\n{intro}"
    
    def _get_intro_text(self, extracted_nodes):
        """从extracted_nodes中提取intro文本"""
        intro_parts = []
        for node in extracted_nodes:
            if node["id"] == 0:  # 跳过标题节点
                continue
            if node["type"] == "content":
                intro_parts.append(node["text"])
            else:
                break
        return "\n".join(intro_parts)
    
    def _process_batch(self, filename, batch_lines):
        """处理一批数据并生成Neo4j查询"""
        create_nodes = []
        create_rels = []
        
        # 收集所有文档的文本用于批量embedding
        texts_for_embedding = []
        processed_docs = []
        
        for line_num, line in batch_lines:       
            try:
                data = json.loads(line.strip())
                title = data["title"]
                intro = self._get_intro_text(data["extracted_nodes"])
                
                # 格式化用于embedding的文本
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
        
        # 批量计算所有文档的embedding
        if texts_for_embedding:
            try:
                embeddings = self.get_contriever_embeddings(texts_for_embedding)
            except Exception as e:
                print(f"Error computing embeddings for {filename}: {e}")
                raise Exception(f"Failed to compute embeddings for {filename}")
            
            # 为每个文档创建节点和关系
            for i, doc_info in enumerate(processed_docs):
                title = doc_info["title"]
                intro = doc_info["intro"]
                data = doc_info["data"]
                
                # 将embedding转换为列表格式存储
                embedding_list = embeddings[i].numpy().tolist()
                
                # 创建节点（包含embedding）
                create_nodes.append({
                    "title": title,
                    "intro": intro,
                    "embedding": embedding_list
                })
                
                # 处理关系
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
                """将一批节点和关系写入Neo4j"""
                with self.driver.session() as session:
                    # 批量创建节点（包含embedding）
                    if nodes:
                        session.run("""
                            UNWIND $nodes AS node
                            MERGE (d:Document {title: node.title})
                            SET d.intro = node.intro,
                                d.embedding = node.embedding
                        """, {"nodes": nodes})
                    
                    # 批量创建关系(无向图，所以创建双向关系)
                    # MATCH (a:Document {title: rel.source})
                    # MATCH (b:Document {title: rel.target})
                    # 对于缺失的目标节点，会创建一个只有title的空节点，后续可以通过其他批次补充intro信息
                    if rels:
                        session.run("""
                            UNWIND $rels AS rel
                            MERGE (a:Document {title: rel.source})
                            MERGE (b:Document {title: rel.target})   
                            MERGE (a)-[:CONNECTED]->(b)
                            MERGE (b)-[:CONNECTED]->(a)
                        """, {"rels": rels})
                    break  # 成功则退出循环
            except TransientError as e:
                if "DeadlockDetected" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))  # 指数退避
                    continue
                raise  # 重试次数用尽后重新抛出异常

    def _calculate_node_degrees(self):
        """分批计算并更新所有节点的度数"""
        with self.driver.session(database="neo4j") as session:  # 统一使用一个会话
            # 获取所有节点总数
            total_result = session.run("MATCH (d:Document) RETURN count(d) as total")
            total_nodes = total_result.single()["total"]
            print(f"开始计算 {total_nodes} 个节点的度数...")
            
            # 分批处理节点度数计算
            batch_size = 10000
            processed = 0
            original_batch_size = batch_size  # 保存原始批次大小
            
            for skip in range(0, total_nodes, batch_size):
                try:
                    # 使用参数化查询增加超时设置
                    session.run("""
                        MATCH (d:Document)
                        WITH d
                        SKIP $skip LIMIT $batch_size
                        OPTIONAL MATCH (d)-[:CONNECTED]->(neighbor)
                        WITH d, count(neighbor) as degree
                        SET d.degree = degree
                    """, skip=skip, batch_size=batch_size, timeout=60000)  # 60秒超时
                    
                    processed += min(batch_size, total_nodes - skip)
                    print(f"已处理 {processed}/{total_nodes} 个节点 ({processed/total_nodes*100:.1f}%)")
                    
                    # 恢复原始批次大小
                    batch_size = original_batch_size
                    
                    # 根据处理时间动态调整暂停时间
                    if (skip // original_batch_size) % 10 == 0 and skip > 0:
                        time.sleep(2)
                        
                except Exception as e:
                    print(f"处理批次 {skip}-{skip+batch_size} 时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise ValueError(f"批次处理失败: {str(e)}")
                    
            
            # 分批获取统计信息（对于大型图）
            print("正在获取度数统计信息...")
            try:
                # 使用apoc库的聚合函数或分批次计算统计信息
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
                    print(f"度数统计 - 最小: {stats['min_degree']}, 最大: {stats['max_degree']}, "
                        f"平均: {stats['avg_degree']:.2f}, 总节点数: {stats['total_nodes']}")
            except Exception as e:
                print(f"获取统计信息时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                raise ValueError(f"获取统计信息失败: {str(e)}")
        
    def build_graph(self):
        """构建知识图谱"""
        # 先创建索引加速后续查询
        with self.driver.session() as session:
            session.run("CREATE INDEX document_title_index IF NOT EXISTS FOR (d:Document) ON (d.title)")
            print("创建了文档标题索引")
        
        # 使用线程池并行处理文件（由于embedding需要GPU，建议使用单线程或较少线程）
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for filename in os.listdir(self.document_tree_path):
                if not '.jsonl' in filename:
                    continue
                    
                filepath = os.path.join(self.document_tree_path, filename)
                futures.append(executor.submit(self._process_file, filepath))
                
            # 等待所有任务完成
            for future in tqdm(futures, desc="Processing files"):
                future.result()
        
        # 计算并更新节点度数
        print("正在计算节点度数...")
        self._calculate_node_degrees()

        
    def _process_file(self, filepath):
        """处理单个jsonl文件"""
        filename = os.path.basename(filepath)
        batch_lines = []
        
        print(f"开始处理文件: {filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                batch_lines.append((index, line))
                
                if len(batch_lines) >= self.batch_size:
                    nodes, rels = self._process_batch(filename, batch_lines)
                    self._write_to_neo4j(nodes, rels)
                    batch_lines = []
                    
                    # 打印进度
                    if (index + 1) % (self.batch_size * 10) == 0:
                        print(f"已处理 {filename} 的 {index + 1} 行")
                                    
            # 处理剩余的行
            if batch_lines:
                nodes, rels = self._process_batch(filename, batch_lines)
                self._write_to_neo4j(nodes, rels)
                
        print(f"完成处理文件: {filename}")
    
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
