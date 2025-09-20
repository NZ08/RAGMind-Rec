import json
import pandas as pd
import os
import warnings
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
from openai import OpenAI
import openai
import logging
from pathlib import Path
import time
import random
import csv
from dotenv import load_dotenv
import threading

load_dotenv()

# 全局模型配置
# GLOBAL_MODEL = "qwen-turbo-latest"

# GLOBAL_MODEL = "gpt-4o-mini"

GLOBAL_MODEL = "doubao-seed-1-6-250615"

# GLOBAL_MODEL = "qwen3:32b"



class BAP():
    """智能推荐代理类，通过多步骤工作流生成个性化推荐"""
    def __init__(self, task_input, user_data=None):
        self.task_input = task_input
        self.user_data = user_data  # 用户数据行，避免重复读取CSV
        self.messages = []
        # Token统计变量
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.api_call_count = 0
        
        # 从task_input中获取用户指令和特征数据
        self.user_instructions = task_input.get('user_instructions')
        self.user_characteristics = task_input.get('user_characteristics')
        
        # 用于暂存第一步信息的属性
        self.step_one_message = None
        self.recommend_content = None
        
        # 用于记录每一步的Token统计
        self.step_one_tokens = {'input': 0, 'output': 0, 'total': 0, 'calls': 0}
        self.step_two_tokens = {'input': 0, 'output': 0, 'total': 0, 'calls': 0}
        # 初始化豆包API客户端
        # 之前的配置（已注释）
        # self.client = OpenAI(
        #     api_key=os.environ.get("DASHSCOPE_API_KEY"),
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        # )
        # self.client = OpenAI(
        #     base_url="https://s.lconai.com/v1/",
        #     api_key=os.environ.get("LCONAI_API_GPT_KEY"),
        # )

        # 豆包API配置
        self.client = OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.environ.get("DOUBAO_API_KEY"),
        )

    def run(self):
        """执行推荐任务的主要方法"""


        user_data = self.user_data
        
        # 解析管道符分隔的数据
        pos_title = user_data['pos_title'].split('|')
        pos_content = user_data['pos_content'].split('|')
        pos_brief_desc = user_data['pos_brief_desc'].split('|')
        neg_title = user_data['neg_title'].split('|')
        neg_content = user_data['neg_content'].split('|')
        neg_brief_desc = user_data['neg_brief_desc'].split('|') 

        self.messages_initial = []
        

        # 步骤1：初始推荐内容生成
        # 构建候选物品列表字符串（交叉排列正负样本）
        candidate_items_str = ""
        item_counter = 1
        
        # 交叉排列正负样本
        max_samples = max(len(pos_title), len(neg_title))
        for idx in range(max_samples):
            # 添加正样本（如果还有）
            if idx < len(pos_title):
                cross_pos_title, cross_pos_brief_desc, cross_pos_content = pos_title[idx], pos_brief_desc[idx], pos_content[idx]
                # 只对content使用前后截断策略，brief_desc使用原来的后截断
                cleaned_brief_desc = re.sub(u"\\<.*?\\>", "", str(cross_pos_brief_desc)[-200:])
                truncated_content = self._truncate_text(cross_pos_content, 1000, 500, 500)
                cleaned_content = re.sub(u"\\<.*?\\>", "", truncated_content)
                candidate_items_str += f"Item {item_counter} : title: {cross_pos_title}, brief_desc: {cleaned_brief_desc}, content: {cleaned_content}\n"
                item_counter += 1
            
            # 添加负样本（如果还有）
            if idx < len(neg_title):
                cross_neg_title, cross_neg_brief_desc, cross_neg_content = neg_title[idx], neg_brief_desc[idx], neg_content[idx]
                # 只对content使用前后截断策略，brief_desc使用原来的后截断
                cleaned_brief_desc = re.sub(u"\\<.*?\\>", "", str(cross_neg_brief_desc)[-200:])
                truncated_content = self._truncate_text(cross_neg_content, 1000, 500, 500)
                cleaned_content = re.sub(u"\\<.*?\\>", "", truncated_content)
                candidate_items_str += f"Item {item_counter} : title: {cross_neg_title}, brief_desc: {cleaned_brief_desc}, content: {cleaned_content}\n"
                item_counter += 1
        
        total_items = len(pos_title) + len(neg_title)
        recommend_count = total_items // 2  # 推荐总样本数的一半
        
        step_one_message_str = "Here are {} candidate items for recommendation:\n{}\nPlease select {} items from the above candidates that you would recommend to this user. Please return a JSON object with a single key 'recommendations' containing a list of objects, each with 'item_number' and 'title' fields.".format(
            total_items, candidate_items_str, recommend_count
        )
        self.messages_initial.append({
            "role": "user",
            "content": step_one_message_str
        })
        retries = 0
        # API调用重试机制
        while retries < 3:
            try:
                completion = self.client.chat.completions.create(
                    messages=self.messages_initial,
                    model=GLOBAL_MODEL,
                    max_tokens=600,
                    temperature=0.3,
                    extra_body={"thinking": {
                        "type": "disabled",  # 不使用深度思考能力
                        # "type": "enabled", # 使用深度思考能力
                        # "type": "auto", # 模型自行判断是否使用深度思考能力
                    }}
                    # thinking={"type":"disabled"}
                )
                
                # 统计token使用量
                if hasattr(completion, 'usage') and completion.usage:
                    self.total_input_tokens += completion.usage.prompt_tokens
                    self.total_output_tokens += completion.usage.completion_tokens
                    self.total_tokens += completion.usage.total_tokens
                    self.api_call_count += 1
                    
                    # 记录第一步的Token统计
                    self.step_one_tokens['input'] += completion.usage.prompt_tokens
                    self.step_one_tokens['output'] += completion.usage.completion_tokens
                    self.step_one_tokens['total'] += completion.usage.total_tokens
                    self.step_one_tokens['calls'] += 1
                    
                    # 第一步API调用统计（不输出详细信息）
                
                try:
                    response_content = completion.choices[0].message.content
                    # 检查响应是否被markdown代码块包裹（```json...```）
                    if response_content.strip().startswith("```json"):
                        response_content = response_content.strip()[7:-4]
                    recommendations = json.loads(response_content)["recommendations"]
                    
                    # 格式化推荐内容
                    recommend_content = "Selected recommendations:\n"
                    for rec in recommendations:
                        recommend_content += f"Item {rec['item_number']}: title:{rec['title']}\n"
                    
                    # 将生成的推荐内容添加到消息列表
                    self.messages_initial.append({
                        "role": "assistant",
                        "content": "The recommend content: {}. ".format(recommend_content)
                    })
                    
                    # 暂存第一步信息，等第二步完成后一起保存
                    self.step_one_message = step_one_message_str
                    self.recommend_content = recommend_content
                    
                    break
                except Exception as e:
                    recommend_content = f"Extract Error: {str(e)}"
                    print(f"JSON解析错误: {str(e)}")
                    retries += 1    
            
            except Exception as e:
                recommend_content = f"API调用错误: {str(e)}"
                print(f"API调用错误: {str(e)}")
                retries += 1
                time.sleep(5)
        
        # 如果重试3次后仍然失败，也要暂存信息
        if retries >= 3:
            self.step_one_message = step_one_message_str
            self.recommend_content = recommend_content

        
        # 步骤2：用户画像生成
        # 构建正样本信息字符串
        pos_sample = ""
        item_counter = 1
        
        # 交叉排列正负样本
        for idx in range(max_samples):
            # 添加正样本（如果还有）
            if idx < len(pos_title):
                cross_pos_title = pos_title[idx]
                pos_sample += f"Item {item_counter} : title: {cross_pos_title}\n"
                item_counter += 2
        
        second_message_str = f"""Great! Actually, this user has interacted with these items: {pos_sample} 
User's instructions：{self.user_instructions}
User's characteristics：{self.user_characteristics}
Can you generate the profile of this user background based on their preferences shown in these interactions? Please make a detailed profile. Don't use numerical numbering for the generated content; you can use bullet points instead. 
Please return a JSON object with a single key 'profile'.
The key 'profile' is also in JSON format internally, generate a purely structured JSON format user profile, strictly adhering to the following requirements:
Top-level Structure: Use "profile" as the only top-level key.
Subkey Rule: Generate subkeys independently based on the core attributes of the user (e.g., user type, focus points, demand scenarios, etc.). No fixed subkeys are required.
Content Format:
Subkey values are only allowed in 3 forms: ① Concise phrase strings (e.g., "has pets at home"); ② Keyword arrays (each element no more than 5 words, e.g., ["baby safety", "easy cleaning"]); ③ Arrays of "core category + (supplementary demand in parentheses)" (e.g., ["bath safety (water temperature monitoring)", "bedding protection (spit-up management)"]).
Prohibitions:
No paragraph descriptions or logical explanations (words like "because", "so", "reflects" are not allowed);
No content integration or expansion, only core key points are allowed;
No long sentences or complex sentences as subkey values."""

        self.messages_initial.append({
            "role": "user",
            "content": second_message_str,
        })
        retries = 0
        # API调用重试机制
        while retries < 3:
            try:
                completion = self.client.chat.completions.create(
                    messages=self.messages_initial,
                    model=GLOBAL_MODEL,
                    max_tokens=600,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {
                        "type": "disabled",  # 不使用深度思考能力
                    }}
                )
                
                # 统计token使用量
                if hasattr(completion, 'usage') and completion.usage:
                    self.total_input_tokens += completion.usage.prompt_tokens
                    self.total_output_tokens += completion.usage.completion_tokens
                    self.total_tokens += completion.usage.total_tokens
                    self.api_call_count += 1
                    
                    # 记录第二步的Token统计
                    self.step_two_tokens['input'] += completion.usage.prompt_tokens
                    self.step_two_tokens['output'] += completion.usage.completion_tokens
                    self.step_two_tokens['total'] += completion.usage.total_tokens
                    self.step_two_tokens['calls'] += 1
                    
                    # 第二步API调用统计（不输出详细信息）
                
                try:
                    response_content = completion.choices[0].message.content
                    # 检查响应是否被markdown代码块包裹（```json...```）
                    if response_content.strip().startswith("```json"):
                        response_content = response_content.strip()[7:-4]
                    profile = json.loads(response_content)["profile"]
                    break
                except Exception as e:
                    profile = f"Extract Error: {str(e)}"
                    print(f"JSON解析错误: {str(e)}")
                    retries += 1    
            
            except Exception as e:
                profile = f"API调用错误: {str(e)}"
                print(f"API调用错误: {str(e)}")
                retries += 1
                time.sleep(5)
        
        # 返回用户画像结果
        return profile
    
    def _truncate_text(self, text, max_length=1000, front_chars=500, back_chars=500):
        """前后截断文本策略
        
        Args:
            text: 要截断的文本
            max_length: 最大长度阈值
            front_chars: 前面保留的字符数
            back_chars: 后面保留的字符数
        
        Returns:
            截断后的文本
        """
        text_str = str(text)
        if len(text_str) <= max_length:
            return text_str
        
        # 前后截断并用省略号连接
        front_part = text_str[:front_chars]
        back_part = text_str[-back_chars:]
        return front_part + "..." + back_part
    
    def get_token_stats(self):
        """获取token使用统计信息"""
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_tokens,
            'api_call_count': self.api_call_count,
            'average_tokens_per_call': self.total_tokens / self.api_call_count if self.api_call_count > 0 else 0
        }
    
    def print_token_stats(self):
        """打印token使用统计信息"""
        stats = self.get_token_stats()
        user_id = self.task_input.get('user_id', 'unknown')
        print(f"用户{user_id} - 输入:{stats['total_input_tokens']} 输出:{stats['total_output_tokens']} 总计:{stats['total_tokens']} 调用:{stats['api_call_count']}次 平均:{stats['average_tokens_per_call']:.1f}")
    

    
    def save_tokens_to_file(self, tokens_file_path):
        """保存tokens统计信息到文件"""
        try:
            user_id = self.task_input.get('user_id', 'unknown')
            stats = self.get_token_stats()
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            
            with open(tokens_file_path, 'a', encoding='utf-8') as f:
                f.write(f"用户ID: {user_id} | 时间: {timestamp} | ")
                f.write(f"输入tokens: {stats['total_input_tokens']} | ")
                f.write(f"输出tokens: {stats['total_output_tokens']} | ")
                f.write(f"总计tokens: {stats['total_tokens']} | ")
                f.write(f"API调用次数: {stats['api_call_count']} | ")
                f.write(f"平均每次调用tokens: {stats['average_tokens_per_call']:.2f}\n")
                
        except Exception as e:
            print(f"保存tokens统计时出错: {str(e)}")


# 全局锁，用于保护文件写入操作
file_write_lock = threading.Lock()

def process_single_user(user_data, df_profile, tokens_log_file, profile_csv_file):
    """处理单个用户的函数"""
    try:
        index, row = user_data
        user_id = row['UserID']
        
        # 获取用户指令和特征数据
        user_profile = df_profile[df_profile['UserID'] == user_id]
        if user_profile.empty:
            print(f"警告: 用户 {user_id} 在用户指令特征文件中未找到")
            return
            
        user_instructions = user_profile.iloc[0]['user_instructions']
        user_characteristics = user_profile.iloc[0]['user_characteristics']
        
        # 创建任务输入
        task_input = {
            'user_id': user_id,
            'user_instructions': user_instructions,
            'user_characteristics': user_characteristics
        }
        
        # 创建BAP实例并运行，传递用户数据行
        bap = BAP(task_input, user_data=row)
        profile = bap.run()
        
        # 打印token统计
        bap.print_token_stats()
        
        # 使用锁保护文件写入操作
        with file_write_lock:
            # 保存tokens统计到文件
            bap.save_tokens_to_file(tokens_log_file)
            
            # 保存用户画像到CSV文件
            with open(profile_csv_file, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                # 处理错误情况，将错误信息统一替换
                if isinstance(profile, str):
                    if profile.startswith("API调用错误:"):
                        profile_str = "api调用出错"
                    elif profile.startswith("Extract Error:"):
                        profile_str = "json解析出错"
                    else:
                        profile_str = profile
                else:
                    # 将profile转换为JSON字符串保存
                    profile_str = json.dumps(profile, ensure_ascii=False)
                writer.writerow([user_id, profile_str])
                
    except Exception as e:
        print(f"处理用户 {user_id if 'user_id' in locals() else 'unknown'} 时出错: {str(e)}")

def main():
    # 读取用户特征数据
    csv_file_path = 'data/PositiveNegativeSample.csv'
    user_profile_path = 'data/UserInstruCharac.csv'
    
    df = pd.read_csv(csv_file_path)
    print(f"成功读取用户特征CSV文件，共有 {len(df)} 个用户")
    
    # 读取用户指令和特征数据
    df_profile = pd.read_csv(user_profile_path)
    print(f"成功读取用户指令和特征CSV文件，共有 {len(df_profile)} 个用户")
    
    # 创建tokens日志文件（以时间命名）
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tokens_log_file = f"tokens_log_{timestamp}.txt"
    print(f"Tokens统计将保存到: {tokens_log_file}")
    
    # 创建用户画像CSV文件（以时间命名）
    profile_csv_file = f"user_profiles_{timestamp}.csv"
    # 创建CSV文件并写入表头
    with open(profile_csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['UserID', 'user_profile'])
    print(f"用户画像将保存到: {profile_csv_file}")
    
    # 使用多线程并行处理用户
    max_workers = 58  # 设置最大线程数，避免API调用过于频繁
    user_data_list = list(df.iterrows())  # 处理所有用户
    
    print(f"\n开始并行处理 {len(user_data_list)} 个用户，使用 {max_workers} 个线程...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_user = {
            executor.submit(process_single_user, user_data, df_profile, tokens_log_file, profile_csv_file): user_data[1]['UserID']
            for user_data in user_data_list
        }
        
        # 等待所有任务完成
        completed_count = 0
        for future in as_completed(future_to_user):
            user_id = future_to_user[future]
            completed_count += 1
            try:
                future.result()  # 获取结果，如果有异常会在这里抛出
                print(f"✓ 用户 {user_id} 处理完成 ({completed_count}/{len(user_data_list)})")
            except Exception as exc:
                print(f"✗ 用户 {user_id} 处理失败: {exc}")
    
    print(f"\n所有用户处理完成！结果已保存到:")
    print(f"- 用户画像: {profile_csv_file}")
    print(f"- Tokens统计: {tokens_log_file}")


if __name__ == "__main__":
    main()
