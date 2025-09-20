#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromaDB向量库检查工具

功能描述:
    检查和分析ChromaDB向量数据库的状态和内容。
    主要功能包括:
    1. 连接到指定的ChromaDB数据库
    2. 统计所有集合的向量数量和基本信息
    3. 显示集合的元数据和向量维度
    4. 提供示例文档预览，帮助了解数据内容
    5. 生成详细的统计报告

使用场景:
    - 验证向量库是否正确构建
    - 检查向量数据的完整性
    - 调试向量检索相关问题
    - 监控向量库的状态

输入:
    - ChromaDB数据库路径(默认: ./chroma_db)

输出:
    - 控制台显示详细的统计信息
    - 返回包含统计数据的字典

作者: 张志才
创建时间: 2025
"""

import chromadb
import logging
from typing import Dict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_vector_database(chroma_db_path: str = "./chroma_db"):
    """
    检查ChromaDB向量库中的向量数量
    
    Args:
        chroma_db_path: ChromaDB数据库路径
    
    Returns:
        Dict: 包含各个集合的向量数量信息
    """
    try:
        # 连接到ChromaDB
        client = chromadb.PersistentClient(path=chroma_db_path)
        logging.info(f"成功连接到ChromaDB，路径: {chroma_db_path}")
        
        # 获取所有集合
        collections = client.list_collections()
        
        if not collections:
            print("向量库中没有找到任何集合")
            return {}
        
        total_vectors = 0
        collection_info = {}
        
        print("\n=== 向量库统计信息 ===")
        print(f"数据库路径: {chroma_db_path}")
        print(f"集合数量: {len(collections)}")
        print("\n=== 各集合详细信息 ===")
        
        for collection in collections:
            collection_name = collection.name
            vector_count = collection.count()
            total_vectors += vector_count
            
            collection_info[collection_name] = {
                'vector_count': vector_count,
                'metadata': collection.metadata
            }
            
            print(f"\n集合名称: {collection_name}")
            print(f"向量数量: {vector_count}")
            print(f"集合元数据: {collection.metadata}")
            
            # 如果集合不为空，获取一些示例数据
            if vector_count > 0:
                try:
                    # 获取前3个文档作为示例
                    sample_size = min(3, vector_count)
                    results = collection.get(limit=sample_size)
                    
                    print(f"示例文档ID: {results['ids'][:sample_size]}")
                    if results['documents']:
                        for i, doc in enumerate(results['documents'][:sample_size]):
                            preview = doc[:5100] + "..." if len(doc) > 500 else doc
                            print(f"  文档{i+1}预览: {preview}")
                    
                    # 检查向量维度
                    if results['embeddings'] and len(results['embeddings']) > 0:
                        vector_dim = len(results['embeddings'][0])
                        print(f"向量维度: {vector_dim}")
                        collection_info[collection_name]['vector_dimension'] = vector_dim
                        
                except Exception as e:
                    print(f"获取示例数据时出错: {e}")
        
        print(f"\n=== 总计 ===")
        print(f"总向量数量: {total_vectors}")
        
        return {
            'total_vectors': total_vectors,
            'total_collections': len(collections),
            'collections': collection_info
        }
        
    except Exception as e:
        logging.error(f"检查向量库时出错: {e}")
        print(f"错误: {e}")
        return {}

def main():
    """
    主函数
    """
    print("正在检查向量库...")
    result = check_vector_database()
    
    if result:
        print("\n检查完成！")
    else:
        print("\n检查失败或向量库为空")

if __name__ == "__main__":
    main()