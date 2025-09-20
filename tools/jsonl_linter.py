#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONL文件格式检查工具

功能描述：
1. 检查JSONL文件的格式正确性
2. 验证每行是否为有效的JSON数据
3. 检查custom_id字段的存在性和唯一性
4. 验证body字段的数据类型
5. 统计文件中有效数据的行数

主要用途：
- 批处理输入文件格式验证
- JSONL文件质量检查
- 数据完整性验证
- 预处理阶段的错误检测

作者：张志才
创建时间：2025
"""

import json


def check_jsonl_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        total = 0
        custom_id_set = set()
        for line in file:
            if line.strip() == "":
                continue
            try:
                line_dict = json.loads(line)
            except json.decoder.JSONDecodeError:
                raise Exception(f"批量推理输入文件格式错误，第{total + 1}行非json数据")
            if not line_dict.get("custom_id"):
                raise Exception(f"批量推理输入文件格式错误，第{total + 1}行custom_id不存在")
            if not isinstance(line_dict.get("custom_id"), str):
                raise Exception(f"批量推理输入文件格式错误, 第{total + 1}行custom_id不是string")
            if line_dict.get("custom_id") in custom_id_set:
                raise Exception(
                    f"批量推理输入文件格式错误，custom_id={line_dict.get('custom_id', '')}存在重复"
                )
            else:
                custom_id_set.add(line_dict.get("custom_id"))
            if not isinstance(line_dict.get("body", ""), dict):
                raise Exception(
                    f"批量推理输入文件格式错误，custom_id={line_dict.get('custom_id', '')}的body非json字符串"
                )
            total += 1
    return total


# 替换<YOUR_JSONL_FILE>为你的JSONL文件路径
file_path = "D:\\WorkSpace\\RAGRec\\data\\batch_008.jsonl"
total_lines = check_jsonl_file(file_path)
print(f"文件中有效JSON数据的行数为: {total_lines}")
