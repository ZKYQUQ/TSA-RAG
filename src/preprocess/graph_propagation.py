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
        # Neo4j连接
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # 初始化Contriever模型
        self.tokenizer = AutoTokenizer.from_pretrained(contriever_model_path)
        self.model = AutoModel.from_pretrained(contriever_model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

        # # 添加线程锁来保护GPU资源（用于计算query embedding）
        # self._gpu_lock = threading.Lock()
    
    def get_query_embedding(self, query):
        """计算查询文本的embedding（线程安全）"""
        inputs = self.tokenizer([query], padding=True, truncation=True, 
                                return_tensors='pt', max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 平均池化
        mask = inputs['attention_mask']
        token_embeddings = outputs[0].masked_fill(~mask[..., None].bool(), 0.)
        embedding = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return embedding[0].cpu()
    
    def get_document_by_title(self, title):
        title = unicodedata.normalize("NFC", title)  # normalize title
        """根据title获取文档信息（包含预计算的embedding）"""
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
        """获取给定标题列表的所有邻居（过滤高度节点）"""
        with self.driver.session() as session:
            # result = session.run("""
            #     UNWIND $titles AS title
            #     MATCH (d:Document {title: title})-[:CONNECTED]->(neighbor:Document)
            #     WHERE neighbor.degree <= $max_degree_threshold
            #     RETURN d.title AS source_title, 
            #         neighbor.title AS neighbor_title,
            #         neighbor.intro AS neighbor_intro,
            #         neighbor.embedding AS neighbor_embedding,
            #         neighbor.degree AS neighbor_degree
            # """, titles=titles, max_degree_threshold=max_degree_threshold)

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
        """计算路径相似度得分（路径上相邻节点相似度的乘积）"""
        if len(path_embeddings) <= 1:
            return 1.0
        
        similarities = []
        for i in range(len(path_embeddings) - 1):
            sim = torch.cosine_similarity(
                path_embeddings[i].unsqueeze(0), 
                path_embeddings[i+1].unsqueeze(0)
            ).item()
            # similarities.append(max(sim, 0.0))  # 确保非负
            similarities.append(sim)
        
        # 返回相似度的乘积
        path_score = 1.0
        for sim in similarities:
            path_score *= sim
        return path_score

    def propagate_from_document(self, query_embedding, root_title, max_hops=5, top_k=5, 
                               local_weight=0.5, global_weight=0.5, threshold=0.01, max_degree_threshold=1000):
        """
        从指定文档开始进行基于注意力权重的图传播
        
        Args:
            query: 查询文本
            root_title: 根文档标题
            max_hops: 最大跳数
            top_k: 每层保留的top-k节点数
            local_weight: local得分权重
            global_weight: global得分权重
            threshold: 权重阈值，低于此值的节点将被过滤
        """
        root_title = unicodedata.normalize("NFC", root_title)  # normalize title 
        # 检查根文档是否存在
        root_doc = self.get_document_by_title(root_title)
        if not root_doc:
            # return {
            #     "title": root_title,
            #     "intro": "",
            #     **{f"hop-{i}": [] for i in range(1, max_hops + 1)}
            # }
            print(f"根文档 {root_title} 不存在，无法进行图传播")
            return None
        
        # # 计算query embedding
        # query_embedding = self.get_contriever_embeddings([query])[0]
        
        # 初始化结果结构
        result = {
            "title": root_title,
            "intro": root_doc["intro"],
            "degree": root_doc["degree"],
            "embedding": root_doc["embedding"].tolist() if root_doc["embedding"] is not None else None,
            **{f"hop-{i}": [] for i in range(1, max_hops + 1)},
            "hop_nodes": []
        }
        
        # 初始化队列和访问记录
        if root_doc["degree"] > max_degree_threshold:
            print(f"根文档 {root_title} 的度数: {root_doc['degree']} 超出阈值，跳过")
            current_queue = []
        else:
            current_queue = [root_title]
        
        visited = {root_title}
        
        # 存储每个节点的路径embedding（用于计算local score）
        node_paths = {root_title: [root_doc["embedding"]]}
        
        for hop in range(1, max_hops + 1):
            if not current_queue:
                # 如果没有更多节点可扩展，记录0并跳出
                result["hop_nodes"].append(0)
                # 为剩余的hop也添加0
                for remaining_hop in range(hop + 1, max_hops + 1):
                    result["hop_nodes"].append(0)
                break
                
            # 获取当前层所有节点的邻居
            all_neighbors = self.get_neighbors(current_queue, max_degree_threshold)
            
            # 收集所有候选邻居（去重且未访问过的）
            candidate_neighbors = {}
            neighbor_sources = defaultdict(list)  # 记录每个邻居来自哪些源节点
            
            for source_title in current_queue:
                if source_title in all_neighbors:
                    for neighbor in all_neighbors[source_title]:
                        neighbor_title = neighbor["title"]
                        if neighbor_title not in visited:
                            candidate_neighbors[neighbor_title] = neighbor
                            neighbor_sources[neighbor_title].append(source_title)
            
            # 记录当前hop的候选邻居数量
            candidate_count = len(candidate_neighbors)
            result["hop_nodes"].append(candidate_count)

            if not candidate_neighbors:
                # 为剩余的hop添加0
                for remaining_hop in range(hop + 1, max_hops + 1):
                    result["hop_nodes"].append(0)
                break
            
            # 计算每个邻居的得分
            neighbor_scores = []
            
            for neighbor_title, neighbor_info in candidate_neighbors.items():
                neighbor_embedding = neighbor_info["embedding"]
                neighbor_degree = neighbor_info["degree"]
                
                # 计算该邻居与所有可能源节点的local score，取最大值
                # max_local_score = 0.0
                # best_adjacent_score = 0.0
                max_local_score = None
                best_adjacent_score = None
                best_source_title = None
                
                for source_title in neighbor_sources[neighbor_title]:
                    # 计算与当前源节点的相邻相似度
                    source_path = node_paths[source_title]
                    source_embedding = source_path[-1]  # 源节点的embedding
                    
                    adjacent_score = torch.cosine_similarity(
                        source_embedding.unsqueeze(0),
                        neighbor_embedding.unsqueeze(0)
                    ).item()
                    
                    # 计算路径相似度（从根到当前邻居的路径）
                    extended_path = source_path + [neighbor_embedding]
                    path_score = self.calculate_path_similarity(extended_path)
                    
                    # Local score = 相邻相似度 * 路径相似度
                    # local_score = adjacent_score * path_score
                    local_score = path_score
                    
                    # if local_score > max_local_score:
                    #     max_local_score = local_score
                    #     best_adjacent_score = adjacent_score
                    if max_local_score is None or local_score > max_local_score:
                        max_local_score = local_score
                        best_adjacent_score = adjacent_score
                        best_source_title = source_title
                
                # 计算global score
                global_score = torch.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    neighbor_embedding.unsqueeze(0)
                ).item()
                
                # 计算总得分
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
            
            # 对得分进行softmax归一化
            if neighbor_scores:
                scores = torch.tensor([n["total_score"] for n in neighbor_scores])
                softmax_scores = F.softmax(scores, dim=0)
                
                # 添加softmax得分并排序
                for i, neighbor in enumerate(neighbor_scores):
                    neighbor["softmax_score"] = softmax_scores[i].item()
                
                # 按softmax得分排序
                neighbor_scores.sort(key=lambda x: x["softmax_score"], reverse=True)
                
                # 添加排名
                for rank, neighbor in enumerate(neighbor_scores, 1):
                    neighbor["rank"] = rank
                
                # 过滤低权重节点并选择top-k
                filtered_neighbors = [
                    n for n in neighbor_scores 
                    if n["total_score"] >= threshold
                ]
                
                top_k_neighbors = filtered_neighbors[:top_k]
                
                # 保存当前hop的结果
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
                
                # 更新队列和路径信息
                current_queue = []
                for neighbor in top_k_neighbors:
                    if neighbor["degree"] > max_degree_threshold:
                        continue

                    neighbor_title = neighbor["title"]
                    current_queue.append(neighbor_title)
                    visited.add(neighbor_title)
                    
                    # 找到该邻居的最佳路径（local score最高的那条）
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
                    
                    # 存储最佳路径
                    if not best_source:
                        raise ValueError(f"No valid source found for neighbor {neighbor_title}")
                    node_paths[neighbor_title] = node_paths[best_source] + [neighbor["embedding"]]

        
        return result
    
    def process_single_document_propagation(self, args):
        """处理单个文档的图传播（用于并行执行）"""
        query_embedding, passage_title, max_hops, top_k, local_weight, global_weight, threshold, max_degree_threshold = args
        
        # try:
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
        # except Exception as e:
        #     print(f"处理文档 {passage_title} 时出错: {e}")
        #     return passage_title, None

    def load_existing_results(self, output_path):
        """加载已有的结果文件"""
        if os.path.exists(output_path):
            print(f"发现已有结果parquet文件: {output_path}")
            existing_df = pd.read_parquet(output_path)
            print(f"已处理的数据条数: {len(existing_df)}")
            return existing_df
        return None
    
    def atomic_write_parquet(self, df, output_path):
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
    
    def save_batch_results(self, processed_rows, output_path):
        """保存批次结果"""
        if processed_rows:
            batch_df = pd.DataFrame(processed_rows)
            
            # 如果输出文件已存在，则追加
            if os.path.exists(output_path):
                existing_df = pd.read_parquet(output_path)
                combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                self.atomic_write_parquet(combined_df, output_path)
            else:
                self.atomic_write_parquet(batch_df, output_path)
            
            print(f"保存了 {len(processed_rows)} 条记录")

    def process_parquet_file(self, input_path, output_path, max_hops=5, top_k=5,
                           local_weight=0.5, global_weight=0.5, threshold=0.01, 
                           batch_size=10, max_workers=None, max_degree_threshold=1000):
        """处理parquet文件中的所有数据，支持增量处理和并行图传播"""
        
        print("所有参数设置：")
        print(f"  input_path: {input_path}")
        print(f"  output_path: {output_path}")
        print(f"  max_hops: {max_hops}")
        print(f"  top_k: {top_k}")
        print(f"  local_weight: {local_weight}")
        print(f"  global_weight: {global_weight}")
        print(f"  threshold: {threshold}")
        print(f"  max_degree_threshold: {max_degree_threshold}")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 读取输入数据
        df = pd.read_parquet(input_path)
        filename = os.path.basename(input_path)
        
        # 加载已有结果
        existing_df = self.load_existing_results(output_path)
        
        # 确定需要处理的数据范围
        if existing_df is not None:
            processed_count = len(existing_df)
            remaining_df = df.iloc[processed_count:].copy()
            print(f"继续处理从第 {processed_count} 条开始的数据")
        else:
            remaining_df = df.copy()
            processed_count = 0
            print("开始处理全部数据")
        
        if len(remaining_df) == 0:
            print("所有数据已处理完成！")
            return
        
        # 批量处理
        processed_rows = []
        
        # 创建检查点文件路径
        checkpoint_path = output_path.replace('.parquet', '_checkpoint.json')
        
        for idx, (_, row) in enumerate(tqdm(remaining_df.iterrows(), 
                                          total=len(remaining_df), 
                                          desc="Processing queries")):
            try:
                # 获取查询文本
                if "asqa" in filename:
                    query = row['ambiguous_question']
                elif "eli5" in filename:
                    query = row['title']
                elif "qampari" in filename:
                    query = row['question_text']
                # elif "hotpotqa" in filename:
                #     query = row['question']
                else:
                    query = row["question"]
                    # raise ValueError(f"未知文件格式: {filename}")
                    

                # 计算query embedding（只需要计算一次）
                query_embedding = self.get_query_embedding(query)

                pre_retrieved_passages = row['pre_retrieved_passages']
                doc_titles = set(p['title'] for p in pre_retrieved_passages)

                # 准备并行任务参数
                task_args = []
                for passage in doc_titles:
                    # passage_title = passage['title']
                    passage_title = passage
                    task_args.append((
                        query_embedding, passage_title, max_hops, top_k, 
                        local_weight, global_weight, threshold, max_degree_threshold
                    ))

                # 并行执行图传播
                graph_propagated_documents = []

                # 动态设置并行度：不超过chunk数，也不超过5
                if max_workers is None:
                    actual_max_workers = min(len(task_args), 5)
                else:
                    actual_max_workers = min(max_workers, len(task_args))
                
                with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
                    # 提交所有任务
                    future_to_title = {
                        executor.submit(self.process_single_document_propagation, args): args[1] 
                        for args in task_args
                    }
                    
                    # 收集结果
                    for future in concurrent.futures.as_completed(future_to_title):
                        # try:
                        #     passage_title, result = future.result()
                        #     graph_propagated_documents.append({
                        #         "title": passage_title,
                        #         "result": result
                        #     })
                        # except Exception as e:
                        #     passage_title = future_to_title[future]
                        #     print(f"并行处理文档 {passage_title} 时出错: {e}")
                        #     graph_propagated_documents.append({
                        #         "title": passage_title,
                        #         "result": None
                        #     })
                        passage_title, result = future.result()
                        graph_propagated_documents.append({
                            "title": passage_title,
                            "result": result
                        })

                
                # 创建新行数据
                new_row = copy.deepcopy(row)
                new_row['graph_propagated_documents'] = graph_propagated_documents
                processed_rows.append(new_row)
                
                # 保存检查点信息
                checkpoint_info = {
                    'processed_count': processed_count + idx + 1,
                    'total_count': len(df),
                    'current_batch_size': len(processed_rows)
                }
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_info, f)
                
                # 每处理完batch_size条数据就保存一次
                if len(processed_rows) >= batch_size:
                    self.save_batch_results(processed_rows, output_path)
                    processed_rows = []  # 清空缓存
                    
            except Exception as e:
                print(f"处理第 {processed_count + idx} 条数据时出错: {e}")
                # 保存已处理的数据
                if processed_rows:
                    print("保存已处理的数据...")
                    self.save_batch_results(processed_rows, output_path)
                raise e
        
        # 保存剩余的数据
        if processed_rows:
            self.save_batch_results(processed_rows, output_path)
        
        # 清理检查点文件
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"清理检查点文件: {checkpoint_path}")
        
        print(f"处理完成！最终结果保存到 {output_path}")
    
    def close(self):
        """关闭Neo4j连接"""
        self.driver.close()

def main():
    parser = argparse.ArgumentParser(description='图传播处理工具')
    
    # 必需参数
    parser.add_argument('--input_file', type=str, help='输入的parquet文件路径')
    parser.add_argument('--output_file', type=str, help='输出的parquet文件路径')
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
    
    # 初始化查询器
    query_engine = AttentionGraphQuery(
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        contriever_model_path=args.contriever_model_path,
    )
    
    try:
        # 处理完整的parquet文件
        print("\n处理parquet文件...")
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
