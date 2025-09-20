#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误记录查找和分析工具

功能描述：
1. 根据custom_id在批处理文件中查找对应的ASIN和标题信息
2. 从错误文件中提取错误记录并匹配批处理数据
3. 处理LLM响应结果，识别和分类各种错误类型
4. 将错误记录保存为CSV格式，便于后续分析

主要用途：
- 批处理任务错误分析
- LLM响应质量评估
- 数据处理异常定位
- 错误统计和报告生成

作者：张志才
创建时间：2025
"""

import json
import os
import csv

def find_asin_title_by_custom_id(custom_id, batch_file):
    """
    根据custom_id在batch文件中查找对应的asin和title
    
    Args:
        custom_id (str): 要查找的custom_id
        batch_file (str): 批处理文件路径
    
    Returns:
        tuple: (asin, title) 如果找到，否则返回 ("未知", "未知")
    """
    try:
        with open(batch_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    batch_data = json.loads(line)
                    if 'custom_id' in batch_data and batch_data['custom_id'] == custom_id:
                        # 提取asin和title
                        asin = "未知"
                        title = "未知"
                        
                        # 从body.messages中提取产品信息
                        if 'body' in batch_data and 'messages' in batch_data['body']:
                            for message in batch_data['body']['messages']:
                                if message.get('role') == 'user' and 'content' in message:
                                    content = message['content']
                                    if isinstance(content, list):
                                        for item in content:
                                            if item.get('type') == 'text' and 'text' in item:
                                                text_content = item['text']
                                                # 提取asin
                                                if '"asin":' in text_content:
                                                    asin_start = text_content.find('"asin":') + 8
                                                    asin_end = text_content.find('"', asin_start + 1)
                                                    if asin_end > asin_start:
                                                        asin = text_content[asin_start:asin_end]
                                                
                                                # 提取title
                                                if '"title":' in text_content:
                                                    title_start = text_content.find('"title":') + 9
                                                    title_end = text_content.find('"', title_start + 1)
                                                    if title_end > title_start:
                                                        title = text_content[title_start:title_end]
                                                break
                        return asin, title
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"错误：找不到批处理文件 {batch_file}")
    
    return "未知", "未知"

def find_error_records_from_batch(errors_file, batch_file):
    """
    从errors文件中读取custom_id，然后在batch文件中查找对应的asin和title
    
    Args:
        errors_file (str): 错误文件路径
        batch_file (str): 批处理文件路径
    
    Returns:
        list: 错误记录列表
    """
    # 读取errors文件中的custom_id
    custom_ids = []
    try:
        with open(errors_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    error_data = json.loads(line)
                    if 'custom_id' in error_data:
                        custom_ids.append(error_data['custom_id'])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"错误：找不到错误文件 {errors_file}")
        return []
    
    print(f"从错误文件中读取到 {len(custom_ids)} 个custom_id")
    
    # 使用统一的函数查找每个custom_id对应的asin和title
    error_records = []
    for custom_id in custom_ids:
        asin, title = find_asin_title_by_custom_id(custom_id, batch_file)
        error_records.append({
            'custom_id': custom_id,
            'asin': asin,
            'title': title
        })
    
    print(f"在批处理文件中找到 {len(error_records)} 个匹配记录")
    
    # 打印错误记录信息
    if error_records:
        print(f"\n=== 批处理错误记录信息 ===")
        for i, record in enumerate(error_records, 1):
            print(f"记录 {i}:")
            print(f"  Custom ID: {record['custom_id']}")
            print(f"  ASIN: {record['asin'].strip('\"')}")
            print(f"  Title: {record['title'].strip('\"')[:20]}")
            print()
    else:
        print("未找到匹配的错误记录")
    
    return error_records



def process_llm_responses(input_file, output_file, batch_file):
    """
    从JSONL文件中提取大模型的响应内容并保存为JSON文件，同时计算total_tokens总和
    
    Args:
        input_file (str): 输入的JSONL文件路径
        output_file (str): 输出的JSON文件路径
        batch_file (str): 批处理文件路径，用于查找custom_id对应的asin和title
    """
    extracted_data = []
    total_tokens_sum = 0
    success_count = 0
    error_count = 0
    error_records = []  # 存储错误记录的信息
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # 解析每一行的JSON数据
                    data = json.loads(line)
                    
                    # 提取响应内容
                    if 'response' in data and 'body' in data['response']:
                        body = data['response']['body']
                        if 'choices' in body and len(body['choices']) > 0:
                            message_content = body['choices'][0]['message']['content']
                            
                            # 累加total_tokens
                            usage = body.get('usage', {})
                            total_tokens = usage.get('total_tokens', 0)
                            total_tokens_sum += total_tokens
                            
                            # 尝试解析message content中的JSON
                            try:
                                parsed_content = json.loads(message_content)
                                
                                # 检查解析后的内容是否包含错误信息
                                if isinstance(parsed_content, dict) and 'error' in parsed_content:
                                    # 如果包含错误信息，不保存到extracted_data，而是记录到error_records
                                    error_count += 1
                                    custom_id = data.get('custom_id', '未知')
                                    asin, title = find_asin_title_by_custom_id(custom_id, batch_file)
                                    
                                    error_records.append({
                                        'line_num': line_num,
                                        'custom_id': custom_id,
                                        'asin': asin,
                                        'title': title,
                                        'error': parsed_content['error']
                                    })
                                else:
                                    # 只有干净的数据才保存到extracted_data
                                    extracted_data.append(parsed_content)
                                    success_count += 1
                                
                            except json.JSONDecodeError as e:
                                # 如果无法解析为JSON，不保存原始内容，只记录错误
                                error_count += 1
                                
                                # 从custom_id在batch文件中查找asin和title
                                custom_id = data.get('custom_id', '未知')
                                asin, title = find_asin_title_by_custom_id(custom_id, batch_file)
                                
                                error_records.append({
                                    'line_num': line_num,
                                    'custom_id': custom_id,
                                    'asin': asin,
                                    'title': title,
                                    'error': f'JSON解析失败: {str(e)}'
                                })
                                
                except json.JSONDecodeError as e:
                    print(f"错误：第{line_num}行不是有效的JSON: {e}")
                    # 尝试从行内容中提取custom_id，如果无法提取则使用未知
                    custom_id = "未知"
                    try:
                        # 简单尝试从原始行中提取custom_id
                        if '"custom_id"' in line:
                            start = line.find('"custom_id"') + 13
                            end = line.find('"', start)
                            if end > start:
                                custom_id = line[start:end]
                    except:
                        pass
                    
                    asin, title = find_asin_title_by_custom_id(custom_id, batch_file)
                    error_records.append({
                        'line_num': line_num,
                        'custom_id': custom_id,
                        'asin': asin,
                        'title': title,
                        'error': str(e)
                    })
                    continue
                    
        # 保存提取的数据到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=2)
        
        # 打印JSON解析错误记录信息
        if error_records:
            print(f"\n=== 解析失败的记录信息 ===")
            for i, record in enumerate(error_records, 1):
                print(f"错误 {i}: 第{record['line_num']}行")
                print(f"  ASIN: {record['asin'].strip('\"')}")
                print(f"  Title: {record['title'].strip('\"')[:20]}")
                print(f"  错误信息: {record['error']}")
                print()
        
        return total_tokens_sum, success_count, error_count, error_records
        
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}")
        return 0, 0, 0, []
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return 0, 0, 0, []

def save_all_errors_to_csv(json_error_records, batch_error_records, csv_file):
    """
    保存所有错误记录到CSV文件（覆盖写入）
    
    Args:
        json_error_records (list): JSON解析错误记录列表
        batch_error_records (list): 批处理错误记录列表
        csv_file (str): CSV文件路径
    """
    def clean_record(record):
        """清理记录中的asin和title"""
        return [record['asin'].strip('"'), record['title'].strip('"')[:20]]
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        writer.writerow(['asin', 'title'])
        
        # 写入所有错误记录
        for record in json_error_records + batch_error_records:
            writer.writerow(clean_record(record))
    
    total_records = len(json_error_records) + len(batch_error_records)
    print(f"\n错误记录已保存到: {csv_file}")
    print(f"JSON解析失败: {len(json_error_records)} 条记录")
    print(f"批处理错误: {len(batch_error_records)} 条记录")
    print(f"总错误记录: {total_records} 条记录")

def main():
    # 设置批次号参数
    batch_number = 7
    
    # 设置输入和输出文件路径
    input_file = f"data/results ({batch_number}).jsonl"
    errors_file = f"data/errors ({batch_number}).jsonl"
    batch_file = f"meta-data/batch_00{batch_number}.jsonl"

    output_file = f"data/extracted_content{batch_number}.json"
    csv_file = f"data/error_records{batch_number}.csv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return
    
    print(f"开始处理文件: {input_file}")
    total_tokens_sum, success_count, error_count, json_error_records = process_llm_responses(input_file, output_file, batch_file)
    
    # 显示最终统计信息
    print(f"\n处理完成!")
    print(f"成功解析: {success_count} 条记录")
    print(f"解析失败: {error_count} 条记录")
    print(f"总记录数: {success_count + error_count} 条记录")
    print(f"total_tokens总和: {total_tokens_sum}")
    print(f"结果已保存到: {output_file}")
    
    # 处理批处理错误记录
    print(f"\n开始处理批处理错误记录...")
    batch_error_records = find_error_records_from_batch(errors_file, batch_file)
    
    # 保存所有错误记录到CSV文件
    save_all_errors_to_csv(json_error_records, batch_error_records, csv_file)

if __name__ == "__main__":
    main()