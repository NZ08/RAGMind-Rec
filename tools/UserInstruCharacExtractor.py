#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取results1.jsonl中所有用户的content字段并保存为CSV
"""

import json
import os
import csv
import re

def remove_chinese_characters(text):
    """
    删除文本中的中文字符
    """
    if not text:
        return text
    # 使用正则表达式匹配中文字符（包括中文标点符号）
    chinese_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3000-\u303f\uff00-\uffef]'
    # 删除中文字符
    result = re.sub(chinese_pattern, '', text)
    # 清理多余的空格
    result = re.sub(r'\s+', ' ', result).strip()
    return result

def extract_all_content():
    """
    从results1.jsonl和results2.jsonl中提取所有用户的content字段并保存为CSV
    """
    # 输入文件路径
    input_files = [
        r"D:\WorkSpace\RAGRec\data\product-feature-batch\results1.jsonl",
        r"D:\WorkSpace\RAGRec\data\product-feature-batch\results2.jsonl"
    ]
    # 输出文件路径
    output_file = "all_users_content.csv"
    
    # 存储提取的数据
    extracted_data = []
    
    # 处理每个输入文件
    for input_file in input_files:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"警告：找不到文件 {input_file}，跳过")
            continue
        
        print(f"开始读取文件: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        # 解析JSON行
                        data = json.loads(line)
                        custom_id = data.get('custom_id', '')
                        
                        # 提取content字段
                        if ('response' in data and 
                            data['response'] is not None and
                            'body' in data['response'] and
                            'choices' in data['response']['body'] and
                            len(data['response']['body']['choices']) > 0 and
                            'message' in data['response']['body']['choices'][0] and
                            'content' in data['response']['body']['choices'][0]['message']):
                            
                            content = data['response']['body']['choices'][0]['message']['content']
                            
                            # 尝试解析content中的JSON字符串
                            try:
                                content_json = json.loads(content)
                                user_instructions = content_json.get('user_instructions', '')
                                user_characteristics = content_json.get('user_characteristics', [])
                                
                                # 删除中文字符
                                user_instructions = remove_chinese_characters(user_instructions)
                                
                                # 对user_characteristics删除中文字符后直接转为字符串
                                if isinstance(user_characteristics, list):
                                    user_characteristics = [remove_chinese_characters(char) for char in user_characteristics]
                                    user_characteristics_str = str(user_characteristics)
                                else:
                                    user_characteristics_str = remove_chinese_characters(str(user_characteristics))
                                
                                extracted_data.append({
                                    'UserID': custom_id,
                                    'user_instructions': user_instructions,
                                    'user_characteristics': user_characteristics_str
                                })
                                print(f"找到 custom_id: {custom_id}")
                                
                            except json.JSONDecodeError as e:
                                print(f"警告：custom_id {custom_id} 的content字段JSON解析失败: {e}")
                                # 如果解析失败，添加空记录
                                extracted_data.append({
                                    'UserID': custom_id,
                                    'user_instructions': '',
                                    'user_characteristics': ''
                                })
                        else:
                            print(f"警告：custom_id {custom_id} 的数据结构不完整（第{line_num}行）")
                                
                    except json.JSONDecodeError as e:
                        print(f"警告：第{line_num}行JSON解析失败: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"错误：无法打开文件 {input_file}")
            continue
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            continue
    
    if not extracted_data:
        print("错误：没有提取到任何数据")
        return
    
    # 按UserID从小到大排序
    extracted_data.sort(key=lambda x: int(x['UserID']))
    
    # 保存提取的数据到CSV文件
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if extracted_data:
                fieldnames = ['UserID', 'user_instructions', 'user_characteristics']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(extracted_data)
        
        print(f"\n提取完成！")
        print(f"找到 {len(extracted_data)} 个用户记录")
        print(f"结果已保存到: {output_file}")
            
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

def main():
    """
    主函数
    """
    print("开始提取results1.jsonl和results2.jsonl中所有用户的content字段并保存为CSV...")
    extract_all_content()

if __name__ == "__main__":
    main()