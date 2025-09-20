import json
import os
import csv
from typing import List, Tuple, Optional, Dict, Any
from openai import OpenAI
import chromadb
from dotenv import load_dotenv
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorProcessor:
    """
    向量处理器类，专门负责文本向量化和向量存储功能
    
    主要功能：
    - 文本文件加载和预处理
    - 文本向量化（使用Qwen3-Embedding-8B模型，支持4096维向量）
    - 向量存储到ChromaDB数据库
    
    注意：相似度检索功能已移至vector_search.py文件中的VectorSearcher类
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: str = "https://api.siliconflow.cn/v1", 
                 chroma_db_path: str = "./chroma_db"):
        """
        初始化向量处理器
        
        Args:
            api_key: SiliconFlow的API密钥，如果为None则从环境变量SILICONFLOW_API_KEY获取
            base_url: API基础URL，默认为SiliconFlow的API地址
            chroma_db_path: Chroma数据库存储路径
        """
        # 获取API密钥
        if api_key is None:
            api_key = os.getenv('SILICONFLOW_API_KEY')
            if not api_key:
                raise ValueError("API密钥未提供，请设置SILICONFLOW_API_KEY环境变量或直接传入api_key参数")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = "Qwen/Qwen3-Embedding-8B"  # SiliconFlow的Qwen3向量模型，支持4096维向量
        
        # 初始化Chroma数据库
        try:
            self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
            logging.info(f"Chroma数据库初始化完成，存储路径: {chroma_db_path}")
        except Exception as e:
            logging.error(f"Chroma数据库初始化失败: {e}")
        
        # 速率限制控制
        self._request_times = []
        self._lock = threading.Lock()
        self._rpm_limit = 2000  # 每分钟请求数限制
        self._tpm_limit = 1000000  # 每分钟token数限制
        self._current_tokens = 0
        self._token_reset_time = time.time()
    
    def load_text_file(self, file_path: str, max_chunks: Optional[int] = None) -> List[str]:
        """
        加载文本文件并按回车符分块
        
        Args:
            file_path: 文件路径
            max_chunks: 最大处理的文本块数量，None表示处理全部
            
        Returns:
            文本块列表
            
        Raises:
            FileNotFoundError: 文件不存在
            UnicodeDecodeError: 文件编码错误
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            logging.warning(f"UTF-8解码失败，尝试使用GBK编码读取文件: {file_path}")
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
        
        # 按回车符分块，过滤空行
        chunks = [chunk.strip() for chunk in content.split('\n') if chunk.strip()]
        
        # 限制处理数量
        if max_chunks is not None and len(chunks) > max_chunks:
            total_chunks = len(chunks)
            chunks = chunks[:max_chunks]
            logging.info(f"成功加载文件，限制处理前 {max_chunks} 个文本块（总共 {total_chunks} 个）")
        else:
            logging.info(f"成功加载文件，共分割为 {len(chunks)} 个文本块")
        
        return chunks
    
    def truncate_text(self, text: str, max_length: int = 30000) -> str:
        """
        截断过长的文本，避免超出模型限制
        Qwen3-Embedding-8B模型最大token限制为32K，约30000个中文字符
        
        Args:
            text: 输入文本
            max_length: 最大字符长度
            
        Returns:
            截断后的文本
        """
        if len(text) <= max_length:
            return text
        return text[:max_length]
    

    
    def _check_rate_limit(self, estimated_tokens: int = 2000):
        """
        检查并控制API调用速率
        
        Args:
            estimated_tokens: 预估的token数量
        """
        with self._lock:
            current_time = time.time()
            
            # 清理超过1分钟的请求记录
            self._request_times = [t for t in self._request_times if current_time - t < 60]
            
            # 重置token计数器（每分钟重置）
            if current_time - self._token_reset_time > 60:
                self._current_tokens = 0
                self._token_reset_time = current_time
            
            # 检查RPM限制
            if len(self._request_times) >= self._rpm_limit:
                sleep_time = 60 - (current_time - self._request_times[0])
                if sleep_time > 0:
                    logging.info(f"达到RPM限制，等待 {sleep_time:.2f} 秒")
                    time.sleep(sleep_time)
            
            # 检查TPM限制
            if self._current_tokens + estimated_tokens > self._tpm_limit:
                sleep_time = 60 - (current_time - self._token_reset_time)
                if sleep_time > 0:
                    logging.info(f"达到TPM限制，等待 {sleep_time:.2f} 秒")
                    time.sleep(sleep_time)
                    self._current_tokens = 0
                    self._token_reset_time = time.time()
            
            # 记录请求时间和token使用
            self._request_times.append(current_time)
            self._current_tokens += estimated_tokens
    
    def _process_batch(self, batch_data: Tuple[int, List[str]]) -> Tuple[int, List[List[float]], List[int]]:
        """
        处理单个批次的文本
        
        Args:
            batch_data: (批次索引, 文本列表)
            
        Returns:
            (批次索引, 向量列表, 失败索引列表)
        """
        batch_idx, batch_texts = batch_data
        embeddings = []
        failed_indices = []
        
        # 估算token数量（粗略估计：中文1字符≈1token，英文1词≈1token）
        estimated_tokens = sum(len(text) for text in batch_texts)
        
        # 检查速率限制
        self._check_rate_limit(estimated_tokens)
        
        try:
            # 截断过长的文本
            truncated_texts = [self.truncate_text(text) for text in batch_texts]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=truncated_texts
            )
            
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            
            logging.info(f"批次 {batch_idx} 处理完成，包含 {len(batch_texts)} 个文本")
            
        except Exception as e:
            logging.error(f"批次 {batch_idx} 处理失败: {e}")
            # 逐个处理失败的批次
            for i, text in enumerate(batch_texts):
                try:
                    self._check_rate_limit(len(text))
                    truncated_text = self.truncate_text(text)
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=truncated_text
                    )
                    embedding = response.data[0].embedding
                    embeddings.append(embedding)
                except Exception as single_e:
                    logging.warning(f"文本 {batch_idx * len(batch_texts) + i + 1} 处理失败: {single_e}")
                    embeddings.append([0.0] * 4096)  # Qwen3-Embedding-8B向量维度为4096
                    failed_indices.append(batch_idx * len(batch_texts) + i)
        
        return batch_idx, embeddings, failed_indices
    
    def get_embeddings(self, texts: List[str], batch_size: int = 10, max_workers: int = 10) -> List[List[float]]:
        """
        获取文本的向量表示，支持并行批量处理
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小，避免单次请求过大
            max_workers: 最大并行工作线程数（建议2-4，避免超出API限制）
            
        Returns:
            向量列表
        """
        if not texts:
            return []
        
        # 准备批次数据
        batches = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batches.append((i // batch_size, batch_texts))
        
        # 存储结果
        all_embeddings = [None] * len(texts)
        all_failed_indices = []
        
        # 并行处理批次
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {executor.submit(self._process_batch, batch): batch for batch in batches}
            
            # 收集结果
            completed_batches = 0
            for future in as_completed(future_to_batch):
                try:
                    batch_idx, batch_embeddings, batch_failed_indices = future.result()
                    
                    # 将结果放入正确的位置
                    start_idx = batch_idx * batch_size
                    for i, embedding in enumerate(batch_embeddings):
                        if start_idx + i < len(texts):
                            all_embeddings[start_idx + i] = embedding
                    
                    # 收集失败索引
                    all_failed_indices.extend(batch_failed_indices)
                    
                    completed_batches += 1
                    logging.info(f"已完成 {completed_batches}/{len(batches)} 个批次")
                    
                except Exception as e:
                    batch_data = future_to_batch[future]
                    logging.error(f"批次 {batch_data[0]} 执行异常: {e}")
                    # 为失败的批次填充零向量
                    batch_idx, batch_texts = batch_data
                    start_idx = batch_idx * batch_size
                    for i in range(len(batch_texts)):
                        if start_idx + i < len(texts):
                            all_embeddings[start_idx + i] = [0.0] * 4096
                            all_failed_indices.append(start_idx + i)
        
        # 确保所有位置都有值
        for i in range(len(all_embeddings)):
            if all_embeddings[i] is None:
                all_embeddings[i] = [0.0] * 4096
                all_failed_indices.append(i)
        
        if all_failed_indices:
            logging.warning(f"共有 {len(all_failed_indices)} 个文本块处理失败，已用零向量占位")
        
        logging.info(f"并行向量生成完成，共 {len(all_embeddings)} 个向量")
        return all_embeddings
    
    def save_vectors_to_chroma(self, texts: List[str], embeddings: List[List[float]], 
                              collection_name: str = "text_embeddings", source_file: str = None):
        """
        保存文本和向量到Chroma数据库
        
        Args:
            texts: 原始文本列表
            embeddings: 对应的向量列表
            collection_name: Chroma集合名称
            source_file: 源文件路径，用于生成描述
            
        Returns:
            collection: ChromaDB集合对象
        """
        if len(texts) != len(embeddings):
            raise ValueError(f"文本数量({len(texts)})与向量数量({len(embeddings)})不匹配")
        
        # 获取或创建集合
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            logging.info(f"使用现有集合: {collection_name}")
        except Exception:
            # 动态生成描述
            if source_file:
                description = f"Text embeddings from {os.path.basename(source_file)}"
            else:
                description = "Text embeddings"
            
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": description}
            )
            logging.info(f"创建新集合: {collection_name}")
        
        # 生成唯一文档ID
        existing_count = collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(texts))]
        
        try:
            # 分批保存，避免超过ChromaDB的批次大小限制
            batch_size = 1000  # ChromaDB的安全批次大小
            total_saved = 0
            
            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                batch_texts = texts[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                batch_ids = ids[i:end_idx]
                
                collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    ids=batch_ids
                )
                
                total_saved += len(batch_texts)
                logging.info(f"已保存 {total_saved}/{len(texts)} 个文本块到ChromaDB")
            
            logging.info(f"成功保存 {len(texts)} 个文本块和对应向量到Chroma数据库")
            logging.info(f"集合名称: {collection_name}")
            logging.info(f"集合中总文档数: {collection.count()}")
            
        except Exception as e:
            logging.error(f"保存向量到ChromaDB时出错: {e}")
            raise
        
        return collection
    
    def _get_or_create_collection(self, collection_name: str, source_file: str = None):
        """
        获取或创建ChromaDB集合
        
        Args:
            collection_name: 集合名称
            source_file: 源文件路径，用于生成描述
            
        Returns:
            collection: ChromaDB集合对象
        """
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            logging.info(f"使用现有集合: {collection_name}")
        except Exception:
            # 动态生成描述
            if source_file:
                description = f"Text embeddings from {os.path.basename(source_file)}"
            else:
                description = "Text embeddings"
            
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": description}
            )
            logging.info(f"创建新集合: {collection_name}")
        
        return collection
    
    def _save_batch_to_chroma(self, collection, texts: List[str], embeddings: List[List[float]]):
        """
        保存单个批次的向量到ChromaDB
        
        Args:
            collection: ChromaDB集合对象
            texts: 文本列表
            embeddings: 向量列表
        """
        if len(texts) != len(embeddings):
            raise ValueError(f"文本数量({len(texts)})与向量数量({len(embeddings)})不匹配")
        
        # 生成唯一文档ID
        existing_count = collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(texts))]
        
        try:
            collection.add(
                embeddings=embeddings,
                documents=texts,
                ids=ids
            )
            logging.info(f"成功保存 {len(texts)} 个文本块到ChromaDB")
            
        except Exception as e:
            logging.error(f"保存批次向量到ChromaDB时出错: {e}")
            raise
    
    def search_similar(self, query: str, collection_name: str = "text_embeddings", n_results: int = 10):
        """
        在指定集合中搜索与查询文本相似的文档
        
        Args:
            query: 查询文本
            collection_name: 要搜索的集合名称
            n_results: 返回的结果数量
            
        Returns:
            包含搜索结果的字典，格式与ChromaDB query方法返回格式一致
        """
        try:
            # 获取集合
            collection = self.chroma_client.get_collection(name=collection_name)
            
            # 生成查询向量
            query_embedding = self.get_embeddings([query])[0]
            
            # 执行向量搜索
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            logging.info(f"在集合 {collection_name} 中找到 {len(results['documents'][0])} 个相似文档")
            return results
            
        except Exception as e:
            logging.error(f"向量搜索失败: {e}")
            return {'documents': [[]], 'distances': [[]], 'ids': [[]]}
    
    def load_csv_data(self, csv_file: str, fields: List[str], max_rows: Optional[int] = None) -> List[str]:
        """
        从CSV文件加载指定字段的数据，每行组合成一个文本块
        
        Args:
            csv_file: CSV文件路径
            fields: 要提取的字段列表
            max_rows: 最大处理的行数，None表示处理全部
            
        Returns:
            文本块列表
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV文件不存在: {csv_file}")
        
        texts = []
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # 检查字段是否存在
                missing_fields = [field for field in fields if field not in reader.fieldnames]
                if missing_fields:
                    raise ValueError(f"CSV文件中缺少字段: {missing_fields}")
                
                for i, row in enumerate(reader):
                    if max_rows is not None and i >= max_rows:
                        break
                    
                    # 提取指定字段并组合成文本
                    field_values = []
                    for field in fields:
                        value = row.get(field, '').strip()
                        if value:  # 只添加非空值
                            field_values.append(f"{field}: {value}")
                    
                    if field_values:  # 只有当至少有一个字段有值时才添加
                        text = " | ".join(field_values)
                        texts.append(text)
                
        except UnicodeDecodeError:
            logging.warning(f"UTF-8解码失败，尝试使用GBK编码读取CSV文件: {csv_file}")
            with open(csv_file, 'r', encoding='gbk') as f:
                reader = csv.DictReader(f)
                
                # 检查字段是否存在
                missing_fields = [field for field in fields if field not in reader.fieldnames]
                if missing_fields:
                    raise ValueError(f"CSV文件中缺少字段: {missing_fields}")
                
                for i, row in enumerate(reader):
                    if max_rows is not None and i >= max_rows:
                        break
                    
                    # 提取指定字段并组合成文本
                    field_values = []
                    for field in fields:
                        value = row.get(field, '').strip()
                        if value:  # 只添加非空值
                            field_values.append(f"{field}: {value}")
                    
                    if field_values:  # 只有当至少有一个字段有值时才添加
                        text = " | ".join(field_values)
                        texts.append(text)
        
        logging.info(f"成功从CSV文件加载 {len(texts)} 个文本块")
        return texts
    
    def process_csv_file(self, csv_file: str, fields: List[str], collection_name: str = "csv_embeddings", max_rows: Optional[int] = None, stream_save_size: int = 1000):
        """
        处理CSV文件：加载指定字段 -> 分批生成向量 -> 流式保存到Chroma数据库
        
        Args:
            csv_file: CSV文件路径
            fields: 要提取的字段列表
            collection_name: Chroma集合名称
            max_rows: 最大处理的行数，None表示处理全部
            stream_save_size: 流式保存的批次大小，每生成这么多向量就保存一次
            
        Returns:
            collection: ChromaDB集合对象
        """
        logging.info(f"开始处理CSV文件: {csv_file}")
        logging.info(f"提取字段: {fields}")
        
        try:
            # 1. 加载CSV数据
            texts = self.load_csv_data(csv_file, fields, max_rows)
            
            if not texts:
                logging.warning("CSV文件中没有有效的文本内容")
                return None
            
            logging.info(f"总共需要处理 {len(texts)} 个文本块，将分批处理并流式保存")
            
            # 2. 初始化或获取集合
            collection = self._get_or_create_collection(collection_name, csv_file)
            
            # 3. 分批处理并流式保存
            total_processed = 0
            for i in range(0, len(texts), stream_save_size):
                end_idx = min(i + stream_save_size, len(texts))
                batch_texts = texts[i:end_idx]
                
                logging.info(f"正在处理第 {i//stream_save_size + 1} 批，文本块 {i+1}-{end_idx}")
                
                # 生成当前批次的向量
                batch_embeddings = self.get_embeddings(batch_texts)
                
                # 立即保存当前批次
                self._save_batch_to_chroma(collection, batch_texts, batch_embeddings)
                
                total_processed += len(batch_texts)
                logging.info(f"已完成 {total_processed}/{len(texts)} 个文本块的处理和保存")
            
            logging.info("CSV流式处理完成！")
            logging.info(f"集合中总文档数: {collection.count()}")
            return collection
            
        except Exception as e:
            logging.error(f"处理CSV文件时出错: {e}")
            raise
    
    def process_file(self, input_file: str, collection_name: str = "text_embeddings", max_chunks: Optional[int] = None, stream_save_size: int = 1000):
        """
        处理整个流程：加载文件 -> 分批生成向量 -> 流式保存到Chroma数据库
        
        Args:
            input_file: 输入文件路径
            collection_name: Chroma集合名称
            max_chunks: 最大处理的文本块数量，None表示处理全部
            stream_save_size: 流式保存的批次大小，每生成这么多向量就保存一次
            
        Returns:
            collection: ChromaDB集合对象
        """
        logging.info(f"开始处理文件: {input_file}")
        
        try:
            # 1. 加载文本文件
            texts = self.load_text_file(input_file, max_chunks)
            
            if not texts:
                logging.warning("文件中没有有效的文本内容")
                return None
            
            logging.info(f"总共需要处理 {len(texts)} 个文本块，将分批处理并流式保存")
            
            # 2. 初始化或获取集合
            collection = self._get_or_create_collection(collection_name, input_file)
            
            # 3. 分批处理并流式保存
            total_processed = 0
            for i in range(0, len(texts), stream_save_size):
                end_idx = min(i + stream_save_size, len(texts))
                batch_texts = texts[i:end_idx]
                
                logging.info(f"正在处理第 {i//stream_save_size + 1} 批，文本块 {i+1}-{end_idx}")
                
                # 生成当前批次的向量
                batch_embeddings = self.get_embeddings(batch_texts)
                
                # 立即保存当前批次
                self._save_batch_to_chroma(collection, batch_texts, batch_embeddings)
                
                total_processed += len(batch_texts)
                logging.info(f"已完成 {total_processed}/{len(texts)} 个文本块的处理和保存")
            
            logging.info("流式处理完成！")
            logging.info(f"集合中总文档数: {collection.count()}")
            return collection
            
        except Exception as e:
            logging.error(f"处理文件时出错: {e}")
            raise
        
def main():
    # 配置参数 - 处理CSV文件创建Small_Baby_Item集合
    CSV_FILE = "meta-data/Fusion_Item.csv"
    COLLECTION_NAME = "Small_Baby_Item"
    FIELDS = ["asin", "title", "TCP", "brief_description"]  # 要提取的字段
    MAX_ROWS = None  # 设置为None以处理全部行
    STREAM_SAVE_SIZE = 1000  # 每1000个向量保存一次
    
    # 创建处理器实例
    processor = VectorProcessor()
    
    try:
        # 流式处理CSV文件
        collection = processor.process_csv_file(
            csv_file=CSV_FILE,
            fields=FIELDS,
            collection_name=COLLECTION_NAME,
            max_rows=MAX_ROWS,
            stream_save_size=STREAM_SAVE_SIZE
        )
        
        if collection:
            print(f"\nCSV流式处理完成！集合 '{COLLECTION_NAME}' 中现有 {collection.count()} 个文档")
            print(f"提取的字段: {FIELDS}")
        
    except Exception as e:
        print(f"处理失败: {e}")
        logging.error(f"主程序执行失败: {e}")


if __name__ == "__main__":
    main()