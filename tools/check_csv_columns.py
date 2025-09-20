#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV数据质量检查工具

功能描述:
    检查用户交互CSV文件的数据质量和一致性。
    主要功能包括:
    1. 验证asin和neg列的长度是否一致
    2. 检查asin列内部是否存在重复元素
    3. 检查neg列内部是否存在重复元素
    4. 检查asin和neg之间是否存在重复(正负样本冲突)
    5. 提供详细的统计信息和示例数据
    6. 生成数据质量报告

使用场景:
    - 验证用户交互数据的完整性
    - 检查负样本生成的正确性
    - 调试推荐系统训练数据问题
    - 确保数据格式符合预期

输入文件:
    - data/user_interactions.csv: 用户交互数据

输出:
    - 控制台显示详细的检查结果
    - 问题行的具体信息和统计数据

作者: 张志才
创建时间: 2025
"""

import pandas as pd

def check_csv_columns():
    # 读取CSV文件
    df = pd.read_csv('data/user_interactions.csv')
    
    print(f"总行数: {len(df)}")
    
    # 检查每行的asin和neg列长度
    length_mismatches = []
    duplicate_issues = []
    
    for idx, row in df.iterrows():
        # 解析asin列（管道符分隔）
        asin_list = row['asin'].split('|')
        
        # 解析neg列（管道符分隔）
        neg_list = row['neg'].split('|')
        
        # 检查长度是否一致
        if len(asin_list) != len(neg_list):
            length_mismatches.append({
                'row': idx,
                'reviewerID': row['reviewerID'],
                'asin_length': len(asin_list),
                'neg_length': len(neg_list)
            })
        
        # 检查是否有重复元素（asin和neg之间）
        asin_set = set(asin_list)
        neg_set = set(neg_list)
        
        # 检查asin列内部是否有重复
        if len(asin_list) != len(asin_set):
            duplicate_issues.append({
                'row': idx,
                'reviewerID': row['reviewerID'],
                'type': 'asin内部重复',
                'original_length': len(asin_list),
                'unique_length': len(asin_set)
            })
        
        # 检查neg列内部是否有重复
        if len(neg_list) != len(neg_set):
            duplicate_issues.append({
                'row': idx,
                'reviewerID': row['reviewerID'],
                'type': 'neg内部重复',
                'original_length': len(neg_list),
                'unique_length': len(neg_set)
            })
        
        # 检查asin和neg之间是否有重复
        overlap = asin_set.intersection(neg_set)
        if overlap:
            duplicate_issues.append({
                'row': idx,
                'reviewerID': row['reviewerID'],
                'type': 'asin和neg之间重复',
                'overlap_items': list(overlap)
            })
    
    # 输出结果
    print("\n=== 长度检查结果 ===")
    if length_mismatches:
        print(f"发现 {len(length_mismatches)} 行长度不匹配:")
        for mismatch in length_mismatches[:10]:  # 只显示前10个
            print(f"  行 {mismatch['row']}: asin长度={mismatch['asin_length']}, neg长度={mismatch['neg_length']}")
        if len(length_mismatches) > 10:
            print(f"  ... 还有 {len(length_mismatches) - 10} 行")
    else:
        print("✓ 所有行的asin和neg列长度都一致")
    
    print("\n=== 重复性检查结果 ===")
    if duplicate_issues:
        print(f"发现 {len(duplicate_issues)} 个重复问题:")
        for issue in duplicate_issues[:10]:  # 只显示前10个
            if issue['type'] == 'asin和neg之间重复':
                print(f"  行 {issue['row']}: {issue['type']}, 重复元素: {issue['overlap_items']}")
            else:
                print(f"  行 {issue['row']}: {issue['type']}, 原长度={issue['original_length']}, 去重后={issue['unique_length']}")
        if len(duplicate_issues) > 10:
            print(f"  ... 还有 {len(duplicate_issues) - 10} 个问题")
    else:
        print("✓ 没有发现重复问题")
    
    # 统计信息
    print("\n=== 统计信息 ===")
    total_asin_items = sum(len(row['asin'].split('|')) for _, row in df.iterrows())
    total_neg_items = sum(len(row['neg'].split('|')) for _, row in df.iterrows())
    print(f"总asin元素数: {total_asin_items}")
    print(f"总neg元素数: {total_neg_items}")
    
    # 检查前几行的具体内容
    print("\n=== 前3行示例 ===")
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        asin_list = row['asin'].split('|')
        neg_list = row['neg'].split('|')
        print(f"行 {idx}:")
        print(f"  asin数量: {len(asin_list)}")
        print(f"  neg数量: {len(neg_list)}")
        print(f"  asin前3个: {asin_list[:3]}")
        print(f"  neg前3个: {neg_list[:3]}")
        print()

if __name__ == "__main__":
    check_csv_columns()