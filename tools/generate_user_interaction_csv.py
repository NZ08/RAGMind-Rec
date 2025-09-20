#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户交互数据生成器

功能描述:
    从原始评论数据(5-core.csv)和商品数据(Fusion_Item.csv)生成用户交互CSV文件。
    主要功能包括:
    1. 读取并清理原始数据中的管道符，避免与输出格式冲突
    2. 为每个用户按时间排序评论，限制最多10个交互记录
    3. 提取商品的多维度信息(描述、标题、品牌、价格、TCP、简要描述等)
    4. 为每个正样本生成对应的负样本，确保用户级别无重复
    5. 输出管道符分隔的多元素字段格式，便于后续推荐系统使用

输入文件:
    - meta-data/5-core.csv: 用户评论数据
    - meta-data/Fusion_Item.csv: 商品信息数据

输出文件:
    - data/user_interactions.csv: 用户交互数据(13列，包含正负样本)

作者: 张志才
创建时间: 2025
"""

import pandas as pd
import csv
from datetime import datetime
import random
import json

def parse_review_time(review_time_str):
    """解析reviewTime字符串为datetime对象用于排序"""
    try:
        # 处理格式如 "07 16, 2013"
        return datetime.strptime(review_time_str, "%m %d, %Y")
    except:
        try:
            # 处理其他可能的格式
            return datetime.strptime(review_time_str, "%Y-%m-%d")
        except:
            # 如果解析失败，返回一个默认日期
            return datetime(1900, 1, 1)

def clean_pipe_characters(text):
    """清理文本中的管道符"""
    if pd.isna(text):
        return ''
    return str(text).replace('|', ' ')

def generate_user_interaction_csv():
    """生成用户交互CSV文件"""
    # 读取数据文件
    print("正在读取5-core.csv...")
    reviews_df = pd.read_csv('meta-data/5-core.csv')
    
    print("正在读取Fusion_Item.csv...")
    items_df = pd.read_csv('meta-data/Fusion_Item.csv')
    
    # 获取所有可用的asin列表用于负样本生成
    all_asins = set(items_df['asin'].tolist())
    
    # 清理数据中的管道符
    print("正在清理数据中的管道符...")
    for col in reviews_df.columns:
        if reviews_df[col].dtype == 'object':
            reviews_df[col] = reviews_df[col].apply(clean_pipe_characters)
    
    for col in items_df.columns:
        if items_df[col].dtype == 'object':
            items_df[col] = items_df[col].apply(clean_pipe_characters)
    
    # 创建asin到itemID、content和neg的映射
    asin_to_item = {}
    for _, row in items_df.iterrows():
        neg_list = []
        if pd.notna(row['neg']) and row['neg'].strip():
            try:
                neg_list = json.loads(row['neg'])
            except:
                neg_list = []
        
        asin_to_item[row['asin']] = {
            'itemID': str(row['itemID']),
            'content': str(row['content']),
            'description': str(row.get('description', '')),
            'title': str(row.get('title', '')),
            'brand': str(row.get('brand', '')),
            'price': str(row.get('price', '')),
            'TCP': str(row.get('TCP', '')),
            'brief_description': str(row.get('brief_description', '')),
            'neg': neg_list
        }
    
    # 按用户分组处理数据
    print("正在处理用户数据...")
    user_data = []
    user_id_counter = 1  # 用户ID计数器，从1开始
    
    for reviewer_id, user_reviews in reviews_df.groupby('reviewerID'):
        # 为每个用户的评论按时间排序
        user_reviews_sorted = user_reviews.copy()
        user_reviews_sorted['parsed_time'] = user_reviews_sorted['reviewTime'].apply(parse_review_time)
        user_reviews_sorted = user_reviews_sorted.sort_values('parsed_time')
        
        # 限制每个用户最多10个交互
        user_reviews_sorted = user_reviews_sorted.head(10)
        
        # 收集该用户的所有数据
        asins, item_ids, contents, descriptions, titles, brands, prices, tcps, brief_descriptions, review_texts, review_times, neg_samples = [], [], [], [], [], [], [], [], [], [], [], []
        user_positive_asins = set()  # 记录用户的正样本asin
        
        for _, review in user_reviews_sorted.iterrows():
            asin = str(review['asin'])
            asins.append(asin)
            user_positive_asins.add(review['asin'])
            
            # 从Fusion_Item.csv获取对应的itemID、content和新字段
            item_info = asin_to_item.get(review['asin'], {})
            if not item_info:
                item_info = {'itemID': '', 'content': '', 'description': '', 'title': '', 
                           'brand': '', 'price': '', 'TCP': '', 'brief_description': '', 'neg': []}
            item_ids.append(item_info.get('itemID', ''))
            contents.append(item_info.get('content', ''))
            descriptions.append(item_info.get('description', ''))
            titles.append(item_info.get('title', ''))
            brands.append(item_info.get('brand', ''))
            prices.append(item_info.get('price', ''))
            tcps.append(item_info.get('TCP', ''))
            brief_descriptions.append(item_info.get('brief_description', ''))
            
            review_texts.append(clean_pipe_characters(review['reviewText']))
            review_times.append(clean_pipe_characters(review['reviewTime']))
        
        # 为每个正样本生成对应的负样本（确保用户级别不重复）
        used_neg_samples = set()  # 记录已使用的负样本
        
        for asin in asins:
            neg_candidates = asin_to_item.get(asin, {}).get('neg', [])
            
            # 过滤掉与用户正样本重复和已使用的负样本
            valid_neg_candidates = [neg for neg in neg_candidates 
                                  if neg not in user_positive_asins and neg not in used_neg_samples]
            
            if valid_neg_candidates:
                # 从neg字段中随机选择一个
                neg_sample = random.choice(valid_neg_candidates)
            else:
                # 如果neg字段为空或所有候选都重复，从全部asin中随机选择
                available_asins = list(all_asins - user_positive_asins - used_neg_samples)
                if available_asins:
                    neg_sample = random.choice(available_asins)
                else:
                    neg_sample = ''  # 极端情况下的备选
            
            if neg_sample:  # 只有非空时才添加到已使用集合
                used_neg_samples.add(neg_sample)
            neg_samples.append(str(neg_sample))
        
        # 将列表转换为管道符分隔的字符串格式存储
        user_row = {
            'UserID': user_id_counter,  # 添加UserID作为第一列
            'reviewerID': str(reviewer_id),
            'asin': '|'.join(asins),  # 以管道符分隔格式存储
            'itemID': '|'.join(item_ids),
            'content': '|'.join(contents),
            'description': '|'.join(descriptions),
            'title': '|'.join(titles),
            'brand': '|'.join(brands),
            'price': '|'.join(prices),
            'TCP': '|'.join(tcps),
            'brief_description': '|'.join(brief_descriptions),
            'reviewText': '|'.join(review_texts),
            'reviewTime': '|'.join(review_times),
            'neg': '|'.join(neg_samples)  # 负样本列
        }
        
        user_data.append(user_row)
        user_id_counter += 1  # 递增用户ID
    
    # 写入CSV文件
    output_file = 'data/user_interactions.csv'
    print(f"正在写入{output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['UserID', 'reviewerID', 'asin', 'itemID', 'content', 'description', 'title', 'brand', 'price', 'TCP', 'brief_description', 'reviewText', 'reviewTime', 'neg']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 写入表头
        writer.writeheader()
        
        # 写入数据
        for row in user_data:
            writer.writerow(row)
    
    print(f"完成！共处理了{len(user_data)}个用户的数据")
    print(f"输出文件：{output_file}")

if __name__ == "__main__":
    generate_user_interaction_csv()