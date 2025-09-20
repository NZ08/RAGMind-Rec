import pandas as pd
import requests
import json
import time
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

# 加载环境变量
load_dotenv()

class DoubaoAPIProcessor:
    def __init__(self, max_workers: int = 58):
        # 从环境变量获取API配置
        self.api_key = os.getenv('DOUBAO_API_KEY')
        self.base_url = 'https://ark.cn-beijing.volces.com/api/v3'
        self.model = 'doubao-seed-1-6-250615'  # 替换为实际的endpoint ID
        self.max_workers = max_workers  # 并行处理的最大线程数
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # 线程锁，用于保护共享资源
        self.lock = threading.Lock()
        
        # Token统计相关
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.token_log_file = "Token.txt"
        
        # 初始化Token日志文件
        with open(self.token_log_file, 'w', encoding='utf-8') as f:
            f.write("Token使用统计日志\n")
            f.write("=" * 50 + "\n")
            f.write("格式: [时间] ASIN: {asin} | Prompt Tokens: {prompt} | Completion Tokens: {completion} | Total: {total}\n\n")
    
    def create_prompt(self, product_data: Dict[str, Any]) -> str:
        """创建用于API调用的提示词"""
        prompt = f"""Task: Combine the text information and images of the following goods to extract the core features for RAG retrieval, taking into account the text attributes and image visual details.

Product text information:
"asin": "{product_data.get('asin', '')}",
"description": "{product_data.get('description', '')}",
"title": "{product_data.get('title', '')}",
"price": {product_data.get('price', '')},
"brand": "{product_data.get('brand', '')}

Requirements:
1. Features not specified in the supplementary text but observable in the image;
2. Infer the implied attributes of the text;
3. It is presented in short sentences of "dimension+specific content", without redundancy, highlighting high-frequency keywords;
4. The output JSON format should include asin, brand of the product, description of the product, title of the product, price of the product, target customer portrait, and brief description (such as schoolbag, pacifier, milk powder, etc.). All field contents should not be bracketed. 
5. If the description, title, price and brand fields of the original product information are empty or do not exist, you can omit them from the output JSON.
6. You can add other fields according to the actual situation, but they should be useful for product information.

IMPORTANT JSON FORMAT REQUIREMENTS:
- Output MUST be valid JSON format only
- All string values MUST be properly escaped (use \\" for quotes, \\n for newlines, \\t for tabs)
- Do NOT include any text before or after the JSON
- Ensure all quotes and special characters in string values are properly escaped
- Do NOT use unescaped newlines or control characters in string values
- Example format: {{"asin": "B123", "title": "Product Title", "description": "Clean description without special chars"}}
"""
        return prompt
    
    def create_messages(self, product_data: Dict[str, Any]) -> list:
        """创建消息列表，包含文本和图像"""
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的商品信息分析助手，擅长从商品的文本描述和图像中提取核心特征并使用英文回答。"
            },
            {
                "role": "user",
                "content": [],
            }
        ]
        
        # 添加文本内容
        text_content = {
            "type": "text",
            "text": self.create_prompt(product_data)
        }
        messages[1]["content"].append(text_content)
        
        # 添加图像内容（如果有图像URL）
        image_url = product_data.get('imUrl', '')
        if image_url and image_url.strip():
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "high"  # 高细节模式，更好地理解图像细节
                }
            }
            messages[1]["content"].append(image_content)
        
        return messages
    
    def log_token_usage(self, asin: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        """记录token使用情况"""
        with self.lock:
            # 更新总计
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            
            # 记录到文件
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"[{timestamp}] ASIN: {asin} | Prompt Tokens: {prompt_tokens} | Completion Tokens: {completion_tokens} | Total: {total_tokens}\n"
            
            with open(self.token_log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            # 打印到控制台
            print(f"Token使用 - ASIN: {asin} | Prompt: {prompt_tokens} | Completion: {completion_tokens} | Total: {total_tokens}")
    

    
    def print_total_token_usage(self):
        """输出总的token用量统计"""
        print("\n" + "=" * 60)
        print("总Token使用统计:")
        print(f"总Prompt Tokens: {self.total_prompt_tokens:,}")
        print(f"总Completion Tokens: {self.total_completion_tokens:,}")
        print(f"总Token数: {self.total_tokens:,}")
        print("=" * 60)
        
        # 保存总统计到文件
        with open(self.token_log_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("总Token使用统计:\n")
            f.write(f"总Prompt Tokens: {self.total_prompt_tokens:,}\n")
            f.write(f"总Completion Tokens: {self.total_completion_tokens:,}\n")
            f.write(f"总Token数: {self.total_tokens:,}\n")
            f.write("=" * 60 + "\n")
            f.write(f"统计时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def clean_json_content(self, json_content: str) -> str:
        """清理JSON内容，处理常见的格式问题"""
        import re
        
        # 移除多余的空白字符
        json_content = json_content.strip()
        
        # 处理字符串中的换行符和制表符
        # 将字符串值中的实际换行符转换为\n
        json_content = re.sub(r'"([^"]*?)\n([^"]*?)"', r'"\1\\n\2"', json_content, flags=re.DOTALL)
        json_content = re.sub(r'"([^"]*?)\t([^"]*?)"', r'"\1\\t\2"', json_content, flags=re.DOTALL)
        json_content = re.sub(r'"([^"]*?)\r([^"]*?)"', r'"\1\\r\2"', json_content, flags=re.DOTALL)
        
        # 处理字符串中未转义的引号
        # 查找字符串值中的未转义引号并转义它们
        def escape_quotes_in_strings(match):
            content = match.group(1)
            # 转义内部的引号，但保留已经转义的
            content = re.sub(r'(?<!\\)"', r'\\"', content)
            return f'"{content}"'
        
        # 匹配JSON字符串值并处理其中的引号
        json_content = re.sub(r'"((?:[^"\\]|\\.)*)"', escape_quotes_in_strings, json_content)
        
        # 移除可能的BOM标记
        if json_content.startswith('\ufeff'):
            json_content = json_content[1:]
        
        return json_content
    
    def fix_json_errors(self, json_content: str) -> Optional[str]:
        """尝试修复常见的JSON错误"""
        import re
        
        try:
            # 尝试1: 移除末尾可能多余的逗号
            fixed_content = re.sub(r',\s*}', '}', json_content)
            fixed_content = re.sub(r',\s*]', ']', fixed_content)
            
            # 尝试2: 确保所有字符串都被正确引用
            # 查找可能未被引用的键名
            fixed_content = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_content)
            
            # 尝试3: 处理可能的控制字符
            fixed_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', fixed_content)
            
            # 尝试4: 确保JSON结构完整
            if not fixed_content.strip().startswith('{'):
                # 如果不是以{开始，尝试找到第一个{
                start_idx = fixed_content.find('{')
                if start_idx != -1:
                    fixed_content = fixed_content[start_idx:]
            
            if not fixed_content.strip().endswith('}'):
                # 如果不是以}结束，尝试找到最后一个}
                end_idx = fixed_content.rfind('}')
                if end_idx != -1:
                    fixed_content = fixed_content[:end_idx+1]
            
            return fixed_content
            
        except Exception as e:
            print(f"修复JSON时发生错误: {e}")
            return None
    
    def call_doubao_api(self, product_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """调用豆包API"""
        try:
            messages = self.create_messages(product_data)
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 2000,
                "thinkings": {
                    "type": "disabled"
                },
                "top_p": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # 提取并记录token使用情况
                if 'usage' in result:
                    usage = result['usage']
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)
                    
                    # 记录token使用情况
                    self.log_token_usage(product_data.get('asin', 'unknown'), prompt_tokens, completion_tokens, total_tokens)
                
                # 尝试解析JSON响应
                try:
                    # 提取JSON部分（可能包含在代码块中）
                    if '```json' in content:
                        json_start = content.find('```json') + 7
                        json_end = content.find('```', json_start)
                        json_content = content[json_start:json_end].strip()
                    elif '{' in content and '}' in content:
                        json_start = content.find('{')
                        json_end = content.rfind('}') + 1
                        json_content = content[json_start:json_end]
                    else:
                        json_content = content
                    
                    # 清理JSON内容，处理常见的格式问题
                    json_content = self.clean_json_content(json_content)
                    
                    parsed_result = json.loads(json_content)
                    return parsed_result
                except json.JSONDecodeError as e:
                    asin = product_data.get('asin', 'unknown')
                    print(f"JSON解析错误 for ASIN {asin}: {e}")
                    print(f"错误位置: 第{e.lineno}行，第{e.colno}列")
                    print(f"原始响应: {content}")
                    print(f"清理后的JSON: {json_content[:500]}..." if len(json_content) > 500 else f"清理后的JSON: {json_content}")
                    
                    # 尝试修复常见的JSON错误
                    try:
                        fixed_json = self.fix_json_errors(json_content)
                        if fixed_json:
                            parsed_result = json.loads(fixed_json)
                            print(f"JSON修复成功 for ASIN {asin}")
                            return parsed_result
                    except Exception as fix_error:
                        print(f"JSON修复失败 for ASIN {asin}: {fix_error}")
                    
                    return None
            else:
                print(f"API调用失败 for ASIN {product_data.get('asin', 'unknown')}: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"API调用异常 for ASIN {product_data.get('asin', 'unknown')}: {e}")
            return None
    
    def process_single_row(self, index: int, row: pd.Series) -> Optional[Dict[str, Any]]:
        """处理单行数据的方法，用于并行处理"""
        try:
            print(f"\n处理第{index+1}行数据，ASIN: {row.get('asin', 'unknown')}")
            
            # 准备商品数据，处理空值和NaN
            def safe_get_value(row, key, default=''):
                """安全获取值，处理NaN和空值"""
                value = row.get(key, default)
                if pd.isna(value) or value == '' or str(value).lower() == 'nan':
                    return default
                return str(value)
            
            product_data = {
                'asin': safe_get_value(row, 'asin'),
                'description': safe_get_value(row, 'description'),
                'title': safe_get_value(row, 'title'),
                'price': safe_get_value(row, 'price'),
                'imUrl': safe_get_value(row, 'imUrl'),
                'brand': safe_get_value(row, 'brand')
            }
            
            # 调用API
            api_result = self.call_doubao_api(product_data)
            
            if api_result:
                print(f"成功处理ASIN: {product_data['asin']}")
                return api_result
            else:
                print(f"处理失败ASIN: {product_data['asin']}")
                return None
                
        except Exception as e:
            print(f"处理第{index+1}行数据时发生错误: {e}")
            return None
    
    def process_csv_file(self, csv_file_path: str, output_file_path: str, start_row: int = 0, max_rows: int = None):
        """处理CSV文件中的商品数据，使用并行处理"""
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file_path)
            print(f"成功读取CSV文件，共{len(df)}行数据")
            
            # 确定处理范围
            end_row = min(start_row + max_rows, len(df)) if max_rows else len(df)
            print(f"将处理第{start_row}行到第{end_row-1}行的数据")
            print(f"使用并行处理，最大线程数: {self.max_workers}")
            
            results = []
            processed_count = 0
            error_count = 0
            
            # 使用并行处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有API调用任务
                future_to_index = {}
                for index, row in df.iloc[start_row:end_row].iterrows():
                    future = executor.submit(self.process_single_row, index, row)
                    future_to_index[future] = index
                
                total_tasks = len(future_to_index)
                
                # 使用tqdm显示进度
                with tqdm(total=total_tasks, desc="处理进度", unit="任务") as pbar:
                    # 收集结果并保存
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                                processed_count += 1
                            else:
                                error_count += 1
                                
                            # 更新进度条描述
                            success_rate = (processed_count / (processed_count + error_count)) * 100 if (processed_count + error_count) > 0 else 0
                            pbar.set_postfix({
                                '成功': processed_count,
                                '失败': error_count,
                                '成功率': f'{success_rate:.1f}%'
                            })
                            pbar.update(1)
                            
                            # 每处理10条数据保存一次
                            if len(results) % 5 == 0 and results:
                                self.save_batch_results(results, output_file_path)
                                tqdm.write(f"已批量保存{len(results)}条结果到文件")
                                    
                        except Exception as e:
                            error_count += 1
                            # 更新进度条描述（异常情况）
                            success_rate = (processed_count / (processed_count + error_count)) * 100 if (processed_count + error_count) > 0 else 0
                            pbar.set_postfix({
                                '成功': processed_count,
                                '失败': error_count,
                                '成功率': f'{success_rate:.1f}%'
                            })
                            pbar.update(1)
                            tqdm.write(f"处理第{index+1}行时发生异常: {e}")
            
            # 保存最终结果
            if results:
                self.save_results(results, output_file_path)
            
            # 输出总的token用量统计
            self.print_total_token_usage()
            
            print(f"\n处理完成！")
            print(f"成功处理: {processed_count}条")
            print(f"处理失败: {error_count}条")
            print(f"失败率: {(error_count / (processed_count + error_count) * 100):.2f}%" if (processed_count + error_count) > 0 else "失败率: 0.00%")
            print(f"结果已保存到: {output_file_path}")
            print(f"Token使用统计已保存到: {self.token_log_file}")
            
        except Exception as e:
            print(f"处理CSV文件时发生错误: {e}")
    
    def save_results(self, results: list, output_file_path: str):
        """保存结果到JSON文件"""
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_file_path}")
            
            # 同时保存为txt文件，每行一个商品JSON
            txt_file_path = output_file_path.replace('.json', '.txt')
            self.save_results_as_txt(results, txt_file_path)
            
        except Exception as e:
            print(f"保存结果时发生错误: {e}")
    
    def save_results_as_txt(self, results: list, txt_file_path: str):
        """保存结果到TXT文件，每行一个商品JSON"""
        try:
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                for result in results:
                    # 移除original_row_index和original_data字段，只保留AI生成的内容
                    clean_result = {k: v for k, v in result.items() 
                                  if k not in ['original_row_index', 'original_data']}
                    json_line = json.dumps(clean_result, ensure_ascii=False, separators=(',', ':'))
                    f.write(json_line + '\n')
            print(f"TXT格式结果已保存到: {txt_file_path}")
        except Exception as e:
            print(f"保存TXT结果时发生错误: {e}")
    
    def save_batch_results(self, results: list, output_file_path: str):
        """批量保存结果，用于增量保存"""
        try:
            # 保存JSON格式
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 保存TXT格式
            txt_file_path = output_file_path.replace('.json', '.txt')
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                for result in results:
                    # 移除不需要的字段，只保留AI生成的内容
                    clean_result = {k: v for k, v in result.items() 
                                  if k not in ['original_row_index', 'original_data']}
                    json_line = json.dumps(clean_result, ensure_ascii=False, separators=(',', ':'))
                    f.write(json_line + '\n')
                    
        except Exception as e:
            print(f"批量保存结果时发生错误: {e}")

def main():
    # 配置参数

    # 本代码文件以后要修改成专门的json格式，而不是用提示词告诉他生成json格式
    csv_file_path = "d:\\WorkSpace\\RAGRec\\data\\extracted_meta_data.csv"
    output_file_path = "d:\\WorkSpace\\RAGRec\\doubao_processed_results5.json"
    
    # 创建处理器实例（设置最大线程数为58）
    processor = DoubaoAPIProcessor(max_workers=58)
    
    # 处理数据（可以设置起始行和最大处理行数）
    # 示例：从第0行开始，最多处理10行数据
    processor.process_csv_file(
        csv_file_path=csv_file_path,
        output_file_path=output_file_path,
        start_row=0
    )

if __name__ == "__main__":
    main()