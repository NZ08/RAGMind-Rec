#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
有效物品提取工具

功能描述：
1. 从item_list.csv加载有效的ASIN列表作为筛选标准
2. 读取extracted_content1-8.json和Error_Fix.json文件
3. 筛选出ASIN字段在有效列表中的物品数据
4. 将筛选结果保存为文本文件，每行一个JSON对象

主要用途：
- 数据筛选和清洗
- 有效物品数据提取
- 多文件批量处理
- 数据格式转换和整理

作者：张志才
创建时间：2025
"""

import json
import csv
import os

def load_item_list(csv_file):
    """加载item_list.csv中的所有asin"""
    valid_asins = set()
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid_asins.add(row['asin'])
    return valid_asins

def extract_valid_items(json_files, valid_asins, output_file):
    """从JSON文件中提取有效的items并保存到txt文件"""
    valid_items = []
    
    for json_file in json_files:
        if os.path.exists(json_file):
            print(f"正在处理文件: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'asin' in item:
                                if item['asin'] in valid_asins:
                                    valid_items.append(item)
                                    print(f"找到有效item: {item['asin']}")
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误 {json_file}: {e}")
        else:
            print(f"文件不存在: {json_file}")
    
    # 保存到txt文件，每行一个JSON对象
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in valid_items:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write('\n')
    
    print(f"总共提取了 {len(valid_items)} 个有效items，保存到 {output_file}")
    return len(valid_items)

def main():
    # 文件路径
    base_dir = "d:\\WorkSpace\\RAGRec"
    data_dir = os.path.join(base_dir, "data")
    meta_dir = os.path.join(base_dir, "meta-data")
    
    # 输入文件
    item_list_file = os.path.join(meta_dir, "item_list.csv")
    json_files = [
        os.path.join(data_dir, f"extracted_content{i}.json") for i in range(1, 9)
    ] + [os.path.join(data_dir, "Error_Fix.json")]
    
    # 输出文件
    output_file = os.path.join(base_dir, "valid_items_output.txt")
    
    # 加载有效的asin列表
    print("加载item_list.csv...")
    valid_asins = load_item_list(item_list_file)
    print(f"加载了 {len(valid_asins)} 个有效asin")
    
    # 提取有效items
    print("开始提取有效items...")
    total_count = extract_valid_items(json_files, valid_asins, output_file)
    
    print(f"\n处理完成！总共提取了 {total_count} 个有效items")

if __name__ == "__main__":
    main()