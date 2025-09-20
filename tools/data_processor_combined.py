#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理组合工具

功能描述：
1. 从item_list.csv加载有效的ASIN列表
2. 从error_records1-8.csv文件中提取有效的ASIN
3. 从meta.csv中提取匹配的元数据信息
4. 将处理结果保存为CSV格式文件

主要用途：
- 数据清洗和筛选
- 错误记录分析
- 元数据提取和匹配

作者：张志才
创建时间：2025
"""

import pandas as pd
import os

def load_valid_asins():
    """从item_list.csv加载有效的asin"""
    csv_path = 'd:\\WorkSpace\\RAGRec\\meta-data\\item_list.csv'
    
    # 读取item_list.csv文件
    df = pd.read_csv(csv_path)
    valid_asins = set(df['asin'].tolist())
    
    print(f"从 {csv_path} 加载了 {len(valid_asins)} 个有效物品")
    
    return valid_asins

def get_valid_asins_from_error_records(valid_asins):
    """从error_records文件中提取有效的asin，不生成中间文件"""
    data_dir = 'd:\\WorkSpace\\RAGRec\\data'
    
    all_valid_asins = set()
    
    # 处理error_records1-8.csv文件
    for i in range(1, 9):
        file_path = os.path.join(data_dir, f'error_records{i}.csv')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"读取 {file_path}，包含 {len(df)} 条记录")
                
                # 检查是否有asin列
                if 'asin' in df.columns:
                    # 筛选出asin存在于item_list中的记录
                    valid_asins_in_file = set(df['asin'].tolist()) & valid_asins
                    all_valid_asins.update(valid_asins_in_file)
                    print(f"  找到 {len(valid_asins_in_file)} 个有效的asin")
                else:
                    print(f"  警告: {file_path} 中没有找到 'asin' 列")
                    
            except Exception as e:
                print(f"读取 {file_path} 时出错: {e}")
        else:
            print(f"文件不存在: {file_path}")
    
    if all_valid_asins:
        print(f"\n总共找到 {len(all_valid_asins)} 个有效的asin")
        return all_valid_asins
    else:
        print("没有找到任何有效的asin")
        return None

def extract_matching_meta_data(valid_asins_set):
    """
    直接使用asin集合从meta.csv中提取匹配的数据
    """
    # 文件路径
    meta_path = 'd:\\WorkSpace\\RAGRec\\meta-data\\meta.csv'
    output_path = 'd:\\WorkSpace\\RAGRec\\data\\extracted_meta_data.csv'
    
    try:
        print(f"使用 {len(valid_asins_set)} 个有效的asin")
        
        # 读取meta.csv
        print("正在读取meta.csv...")
        meta_df = pd.read_csv(meta_path)
        print(f"meta.csv包含 {len(meta_df)} 行数据")
        
        # 筛选匹配的数据
        print("正在筛选匹配的数据...")
        matching_data = meta_df[meta_df['asin'].isin(valid_asins_set)]
        print(f"找到 {len(matching_data)} 行匹配的数据")
        
        # 保存结果
        matching_data.to_csv(output_path, index=False)
        print(f"匹配的数据已保存到: {output_path}")
        
        # 显示统计信息
        print(f"\n统计信息:")
        print(f"- 输入的asin数量: {len(valid_asins_set)}")
        print(f"- meta.csv总行数: {len(meta_df)}")
        print(f"- 匹配的行数: {len(matching_data)}")
        print(f"- 输出文件列数: {len(matching_data.columns)}")
        print(f"- 输出文件列名: {list(matching_data.columns)}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return False
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return False

def main():
    print("开始完整的数据处理流程...")
    
    # 步骤1: 加载有效的asin
    print("\n=== 步骤1: 加载有效的asin ===")
    valid_asins = load_valid_asins()
    
    # 步骤2: 从error_records文件中提取有效的asin
    print("\n=== 步骤2: 从error_records文件中提取有效的asin ===")
    valid_asins_from_errors = get_valid_asins_from_error_records(valid_asins)
    
    if valid_asins_from_errors is not None:
        print("\n=== 步骤3: 提取匹配的meta数据 ===")
        success = extract_matching_meta_data(valid_asins_from_errors)
        
        if success:
            print("\n=== 所有步骤完成 ===")
            print(f"有效物品总数: {len(valid_asins)}")
            print(f"从错误记录中找到的有效asin数量: {len(valid_asins_from_errors)}")
            print("数据提取完成!")
        else:
            print("\n=== 数据提取失败 ===")
    else:
        print("\n=== 处理完成，但没有找到匹配的记录 ===")

if __name__ == "__main__":
    main()