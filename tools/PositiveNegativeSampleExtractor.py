#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正负样本信息提取器 (Positive Negative Sample Extractor)

功能描述:
本模块用于从用户交互数据中提取正负样本信息，为推荐系统提供训练数据。
主要功能包括:
1. 从用户交互CSV文件中读取用户行为数据
2. 提取正样本信息（用户喜欢的商品详细信息）
3. 提取负样本信息（用户不喜欢的商品详细信息）
4. 将商品ASIN与详细商品信息进行映射
5. 清理和标准化文本数据
6. 生成结构化的正负样本数据并输出为CSV格式

输入数据:
- user_interactions.csv: 用户交互数据文件
- Fusion_Item.csv: 商品信息数据文件

输出数据:
- users_features_output.csv: 包含用户ID及其正负样本信息的结构化数据

作者: 张志才
创建时间: 2025
"""

import pandas as pd
import json
import csv
from typing import Dict, Any, List
from pathlib import Path


class PositiveNegativeSampleExtractor:
    """正负样本信息提取器
    
    从CSV文件中提取正负样本信息，包括正样本和负样本的详细商品信息
    """
    
    def __init__(self):
        """初始化正负样本信息提取器"""
        pass
    
    def extract_user_features(self, csv_file_path: str, fusion_item_path: str) -> List[Dict[str, Any]]:
        """提取正负样本数据
        
        Args:
            csv_file_path: 用户交互CSV文件路径
            fusion_item_path: 商品信息CSV文件路径
            
        Returns:
            提取的正负样本列表
        """
        try:
            # 验证输入文件
            csv_path = Path(csv_file_path)
            fusion_path = Path(fusion_item_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV文件不存在: {csv_file_path}")
            if not fusion_path.exists():
                raise FileNotFoundError(f"商品信息文件不存在: {fusion_item_path}")
            
            # 提取正负样本信息
            users_data = self._extract_user_features(csv_file_path, fusion_item_path)
            
            total_users = len(users_data)
            if total_users == 0:
                return []
            
            return users_data
            
        except FileNotFoundError as e:
            return []
        except Exception as e:
            return []
    
    def _extract_user_features(self, csv_file_path: str, fusion_item_path: str) -> List[Dict[str, Any]]:
        """从CSV文件中提取正负样本信息，包括正样本和负样本
        
        Args:
            csv_file_path: 用户交互CSV文件路径
            fusion_item_path: 商品信息CSV文件路径
            
        Returns:
            提取的正负样本列表
        """
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        
        # 读取Fusion_Item.csv文件，建立asin到商品信息的映射
        fusion_df = pd.read_csv(fusion_item_path)
        asin_to_item = {}
        for _, row in fusion_df.iterrows():
            asin = row['asin']
            asin_to_item[asin] = {
                'content': str(row['content']) if pd.notna(row['content']) else "",
                'brief_description': str(row['brief_description']) if pd.notna(row['brief_description']) else "",
                'title': str(row['title']) if pd.notna(row['title']) else ""
            }
        
        # 存储提取结果的列表
        extracted_data = []
        
        # 清理管道符的函数
        def clean_pipe_chars(text):
            """清理文本中的管道符"""
            if isinstance(text, str):
                return text.replace('|', ' ')
            return str(text).replace('|', ' ')
        
        for index, row in df.iterrows():
            user_id = row['UserID']
            
            # 提取正样本特征（按管道符分割）
            pos_content_elements = []
            pos_brief_desc_elements = []
            pos_title_elements = []
            
            if pd.notna(row['content']):
                pos_content_elements = [clean_pipe_chars(part.strip()) for part in str(row['content']).split('|')]
            if pd.notna(row['brief_description']):
                pos_brief_desc_elements = [clean_pipe_chars(part.strip()) for part in str(row['brief_description']).split('|')]
            if pd.notna(row['title']):
                pos_title_elements = [clean_pipe_chars(part.strip()) for part in str(row['title']).split('|')]
            
            # 提取负样本特征
            neg_content_list = []
            neg_brief_desc_list = []
            neg_title_list = []
            
            neg_raw = row['neg']
            neg_asins = []
            if pd.notna(neg_raw):
                neg_asins = [neg.strip() for neg in str(neg_raw).split('|')]
            
            # 去除每个特征的最后一个元素（如果存在的话）
            if len(pos_content_elements) > 0:
                pos_content_elements = pos_content_elements[:-1]
            if len(pos_brief_desc_elements) > 0:
                pos_brief_desc_elements = pos_brief_desc_elements[:-1]
            if len(pos_title_elements) > 0:
                pos_title_elements = pos_title_elements[:-1]
            if len(neg_asins) > 0:
                neg_asins = neg_asins[:-1]
                
            for asin in neg_asins:
                if asin in asin_to_item:
                    item_info = asin_to_item[asin]
                    neg_content_list.append(clean_pipe_chars(item_info['content'] or "No Data"))
                    neg_brief_desc_list.append(clean_pipe_chars(item_info['brief_description'] or "No Data"))
                    neg_title_list.append(clean_pipe_chars(item_info['title'] or "No Data"))
            
            # 构建正负样本数据
            sample_data = {
                'UserID': user_id,
                'pos_title': '|'.join(pos_title_elements),
                'pos_content': '|'.join(pos_content_elements),
                'pos_brief_desc': '|'.join(pos_brief_desc_elements),
                'neg_title': '|'.join(neg_title_list),
                'neg_content': '|'.join(neg_content_list),
                'neg_brief_desc': '|'.join(neg_brief_desc_list)
            }
            
            extracted_data.append(sample_data)
        
        return extracted_data


# 使用示例
if __name__ == "__main__":
    # 创建正负样本信息提取器实例
    extractor = PositiveNegativeSampleExtractor()
    
    # 提取正负样本信息
    csv_file = "meta-data/user_interactions.csv"
    fusion_file = "meta-data/Fusion_Item.csv"
    
    users_data = extractor.extract_user_features(csv_file, fusion_file)
    
    # 保存结果到CSV文件
    output_file = "data/users_features_output.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['UserID', 'pos_title', 'pos_content', 'pos_brief_desc', 'neg_title', 'neg_content', 'neg_brief_desc']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(users_data)
    
    # 数据已保存到CSV文件