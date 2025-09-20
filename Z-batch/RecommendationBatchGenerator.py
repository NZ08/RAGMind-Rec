import pandas as pd
import json
from typing import Dict, Any, List
import os
from pathlib import Path
import asyncio
import aiohttp
import time
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class RecommendationBatchGenerator:
    """商品推荐批处理生成器
    
    基于用户交叉正负样本数据生成批处理请求，用于商品推荐：
    基于交叉样本推荐原数量一半的商品
    """
    
    def __init__(self, temperature: float = 0.5):
        """初始化生成器
        
        Args:
            temperature: AI模型的温度参数，控制输出的随机性
        """
        self.temperature = temperature
        self.api_key = os.getenv('DOUBAO_API_KEY')
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3"
        self.model_endpoint = os.getenv('DOUBAO_MODEL_ENDPOINT')
        self.recommendation_system_prompt = (
            "You are a professional recommendation system assistant, "
            "skilled at analyzing user preference patterns from positive and negative samples "
            "to recommend suitable products."
        )
    
    def _create_recommendation_prompt(self, user_data: Dict[str, Any]) -> str:
        """创建商品推荐的提示词模板
        
        Args:
            user_data: 用户数据字典
            
        Returns:
            格式化的提示词字符串
        """
        cross_arranged_items = user_data.get('cross_arranged_items', [])
        total_items = len(cross_arranged_items)
        recommend_count = max(1, total_items // 2)  # 推荐原数量一半的商品
        
        # 构建交叉样本数据
        items_text = "\n".join([f"{i+1}. {item}" for i, item in enumerate(cross_arranged_items)])
        
        prompt = f"""Task: Based on the user's historical product interaction data, analyze user preferences and recommend {recommend_count} products that best match the user's interests.

User's Historical Product Interactions:
{items_text}

Analysis Guidelines:
1. Analyze the user's interaction patterns with different products
2. Identify what types of products, features, categories, and characteristics the user prefers
3. Look for patterns in product attributes that indicate user preferences
4. Consider factors like product categories, features, brands, price ranges, etc.
5. Recommend {recommend_count} products that align with identified user preferences

Requirements:
1. Carefully analyze the user's interaction history to understand preferences
2. Extract key preference indicators from the interaction data
3. Recommend {recommend_count} products with detailed reasoning based on identified patterns
4. Strictly output results in JSON object format, containing the following fields:
   - analysis: Brief analysis of user preferences based on interaction patterns
   - recommended_items: Array of {recommend_count} recommended products, each containing:
     - title: Product title
     - category: Product category
     - key_features: Key features that match user preferences
     - reasoning: Why this product is recommended based on user's interaction patterns

Please ensure the output is in valid JSON format."""
        return prompt
    
    async def _execute_volcengine_batch_inference(self, session: aiohttp.ClientSession, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行火山方舟批量推理（异步版本）
        
        Args:
            session: aiohttp客户端会话
            user_data: 用户数据字典
            
        Returns:
            推理结果字典
        """
        try:
            # 创建推荐提示词
            recommendation_prompt = self._create_recommendation_prompt(user_data)
            
            # 构建请求消息
            messages = [
                {
                    "role": "system",
                    "content": self.recommendation_system_prompt
                },
                {
                    "role": "user",
                    "content": recommendation_prompt
                }
            ]
            
            # 构建批量推理API请求
            batch_url = f"{self.base_url.rstrip('/')}/batch/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_endpoint,
                "messages": messages,
                "response_format": {
                    "type": "json_object"
                },
                "temperature": self.temperature,
                "max_tokens": 1500,
                "thinking": {
                    "type": "disabled"
                },
            }
            
            # 发送异步请求
            async with session.post(batch_url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=1800)) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "user_id": user_data.get('UserID', 'Unknown'),
                        "result": result,
                        "model": result.get('model', 'N/A'),
                        "usage": result.get('usage', {})
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "user_id": user_data.get('UserID', 'Unknown'),
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                
        except asyncio.TimeoutError:
            return {
                "success": False,
                "user_id": user_data.get('UserID', 'Unknown'),
                "error": "Request timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "user_id": user_data.get('UserID', 'Unknown'),
                "error": str(e)
            }
    
    async def execute_realtime_batch_inference_async(self, 
                                                    csv_file_path: str,
                                                    fusion_item_path: str,
                                                    output_file: str = "./data/doubao_results.txt",
                                                    max_users: int = None,
                                                    max_concurrent: int = 10) -> bool:
        """执行异步实时推理
        
        Args:
            csv_file_path: 用户交互CSV文件路径
            fusion_item_path: 商品信息CSV文件路径
            output_file: 结果输出文件路径（txt格式）
            max_users: 最大处理用户数量，None表示处理所有用户
            max_concurrent: 最大并发数量
            
        Returns:
            是否成功完成所有推理
        """
        try:
            # 验证输入文件
            csv_path = Path(csv_file_path)
            fusion_path = Path(fusion_item_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV文件不存在: {csv_file_path}")
            if not fusion_path.exists():
                raise FileNotFoundError(f"商品信息文件不存在: {fusion_item_path}")
            
            # 创建输出目录
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 提取用户特征
            print(f"正在从CSV文件提取用户特征: {csv_file_path}")
            print(f"正在读取商品信息文件: {fusion_item_path}")
            users_data = self._extract_user_features(csv_file_path, fusion_item_path)
            
            total_users = len(users_data)
            if total_users == 0:
                print("警告: 没有提取到用户数据")
                return True
            
            # 限制处理的用户数量
            if max_users is not None and max_users > 0:
                users_data = users_data[:max_users]
                process_users = min(max_users, total_users)
                print(f"成功提取用户特征，共{total_users:,}个用户，将处理前{process_users}个用户")
            else:
                process_users = total_users
                print(f"成功提取用户特征，共{total_users:,}个用户")
            
            print(f"使用火山方舟推理接入点: {self.base_url}")
            print(f"最大并发数: {max_concurrent}")
            print("-" * 60)
            
            # 创建信号量控制并发
            semaphore = asyncio.Semaphore(max_concurrent)
            
            # 创建aiohttp会话
            connector = aiohttp.TCPConnector(limit=max_concurrent * 2, limit_per_host=max_concurrent)
            timeout = aiohttp.ClientTimeout(total=1800)  # 30分钟超时
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                # 创建异步任务列表
                tasks = []
                for i, user_data in enumerate(users_data):
                    task = self._process_single_user_async(session, semaphore, user_data, i+1, process_users)
                    tasks.append(task)
                
                # 并发执行所有任务
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 写入结果到文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    for result in results:
                        if isinstance(result, Exception):
                            error_result = {
                                "success": False,
                                "user_id": "Unknown",
                                "error": str(result)
                            }
                            f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                        else:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"\n🎉 异步推理完成！结果已保存到: {output_path.absolute()}")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ 文件错误: {e}")
            return False
        except Exception as e:
            print(f"❌ 执行异步推理时发生错误: {e}")
            return False
    
    async def _process_single_user_async(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, 
                                       user_data: Dict[str, Any], index: int, total: int) -> Dict[str, Any]:
        """异步处理单个用户
        
        Args:
            session: aiohttp客户端会话
            semaphore: 并发控制信号量
            user_data: 用户数据
            index: 当前用户索引
            total: 总用户数
            
        Returns:
            处理结果
        """
        async with semaphore:
            user_id = user_data.get('UserID', f'User_{index}')
            print(f"处理用户 {user_id} ({index}/{total})... ", end="")
            
            # 执行推理
            result = await self._execute_volcengine_batch_inference(session, user_data)
            
            if result['success']:
                print("✓ 成功")
            else:
                print(f"✗ 失败: {result['error']}")
            
            return result
    
    def execute_realtime_batch_inference(self, 
                                       csv_file_path: str,
                                       fusion_item_path: str,
                                       output_file: str = "./data/doubao_results.txt",
                                       max_users: int = None) -> bool:
        """执行实时推理
        
        Args:
            csv_file_path: 用户交互CSV文件路径
            fusion_item_path: 商品信息CSV文件路径
            output_file: 结果输出文件路径（txt格式）
            max_users: 最大处理用户数量，None表示处理所有用户
            
        Returns:
            是否成功完成所有推理
        """
        try:
            # 验证输入文件
            csv_path = Path(csv_file_path)
            fusion_path = Path(fusion_item_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV文件不存在: {csv_file_path}")
            if not fusion_path.exists():
                raise FileNotFoundError(f"商品信息文件不存在: {fusion_item_path}")
            
            # 创建输出目录
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 提取用户特征
            print(f"正在从CSV文件提取用户特征: {csv_file_path}")
            print(f"正在读取商品信息文件: {fusion_item_path}")
            users_data = self._extract_user_features(csv_file_path, fusion_item_path)
            
            total_users = len(users_data)
            if total_users == 0:
                print("警告: 没有提取到用户数据")
                return True
            
            # 限制处理的用户数量
            if max_users is not None and max_users > 0:
                users_data = users_data[:max_users]
                process_users = min(max_users, total_users)
                print(f"成功提取用户特征，共{total_users:,}个用户，将处理前{process_users}个用户")
            else:
                process_users = total_users
                print(f"成功提取用户特征，共{total_users:,}个用户")
            
            print(f"使用火山方舟推理接入点: {self.base_url}")

            print("-" * 60)
            
            # 打开输出文件
            with open(output_file, 'w', encoding='utf-8') as f:
                # 处理每个用户
                for i, user_data in enumerate(users_data):
                    user_id = user_data.get('UserID', f'User_{i+1}')
                    print(f"处理用户 {user_id} ({i+1}/{process_users})... ", end="")
                    
                    # 执行推理
                    result = self._execute_volcengine_batch_inference(user_data)
                    
                    # 直接将结果写入txt文件，每行一个结果
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
                    if result['success']:
                        print("✓ 成功")
                    else:
                        print(f"✗ 失败: {result['error']}")
                    
                    # 小延迟避免请求过快
                    time.sleep(0.1)
            
            print(f"\n🎉 推理完成！结果已保存到: {output_path.absolute()}")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ 文件错误: {e}")
            return False
        except Exception as e:
            print(f"❌ 执行推理时发生错误: {e}")
            return False
    
    def _extract_user_features(self, csv_file_path: str, fusion_item_path: str) -> List[Dict[str, Any]]:
        """从CSV文件中提取用户特征，包括正样本和负样本
        
        Args:
            csv_file_path: 用户交互CSV文件路径
            fusion_item_path: 商品信息CSV文件路径
            
        Returns:
            提取的用户特征列表
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
        
        for index, row in df.iterrows():
            user_id = row['UserID']
            
            # 提取正样本content字段（用管道符分隔的JSON字符串）
            pos_content_raw = row['content']
            pos_content_elements = []
            if pd.notna(pos_content_raw):
                # 按管道符分割content
                content_parts = pos_content_raw.split('|')
                for part in content_parts:
                    try:
                        # 尝试解析JSON
                        parsed_content = json.loads(part)
                        pos_content_elements.append(parsed_content)
                    except json.JSONDecodeError:
                        # 如果不是有效JSON，直接保存原始字符串
                        pos_content_elements.append(part.strip())
            
            # 提取正样本brief_description字段（用管道符分隔）
            pos_brief_desc_raw = row['brief_description']
            pos_brief_desc_elements = []
            if pd.notna(pos_brief_desc_raw):
                # 按管道符分割brief_description
                pos_brief_desc_elements = [desc.strip() for desc in pos_brief_desc_raw.split('|')]
            
            # 提取正样本title字段（用管道符分隔）
            pos_title_raw = row['title']
            pos_title_elements = []
            if pd.notna(pos_title_raw):
                # 按管道符分割title
                pos_title_elements = [title.strip() for title in pos_title_raw.split('|')]
            
            # 提取neg字段中的asin（用管道符分隔）
            neg_raw = row['neg']
            neg_asins = []
            if pd.notna(neg_raw):
                # 按管道符分割neg
                neg_asins = [neg.strip() for neg in neg_raw.split('|')]
            
            # 去除每个特征的最后一个元素（如果存在的话）
            if len(pos_content_elements) > 0:
                pos_content_elements = pos_content_elements[:-1]
            if len(pos_brief_desc_elements) > 0:
                pos_brief_desc_elements = pos_brief_desc_elements[:-1]
            if len(pos_title_elements) > 0:
                pos_title_elements = pos_title_elements[:-1]
            if len(neg_asins) > 0:
                neg_asins = neg_asins[:-1]
            
            # 根据neg_asins查找负样本特征
            neg_content_elements = []
            neg_brief_desc_elements = []
            neg_title_elements = []
            
            for asin in neg_asins:
                if asin in asin_to_item:
                    item_info = asin_to_item[asin]
                    
                    # 负样本content特征
                    if item_info['content']:
                        try:
                            parsed_content = json.loads(item_info['content'])
                            neg_content_elements.append(parsed_content)
                        except json.JSONDecodeError:
                            neg_content_elements.append(item_info['content'])
                    else:
                        neg_content_elements.append("No Data")
                    
                    # 负样本brief_description特征
                    if item_info['brief_description']:
                        neg_brief_desc_elements.append(item_info['brief_description'])
                    else:
                        neg_brief_desc_elements.append("No Data")
                    
                    # 负样本title特征
                    if item_info['title']:
                        neg_title_elements.append(item_info['title'])
                    else:
                        neg_title_elements.append("No Data")
            
            # 直接构建交叉排列的字符串
            cross_arranged_items = []
            min_length = min(len(pos_content_elements), len(neg_content_elements))
            for i in range(min_length):
                # 正样本项
                pos_item = f"title: {pos_title_elements[i]}, Brief Description: {pos_brief_desc_elements[i]}, content: {pos_content_elements[i]}"
                # 负样本项
                neg_item = f"title: {neg_title_elements[i]}, Brief Description: {neg_brief_desc_elements[i]}, content: {neg_content_elements[i]}"
                # 交叉排列
                cross_arranged_items.append(f"Item {i*2+1} {pos_item}")
                cross_arranged_items.append(f"Item {i*2+2} {neg_item}")
            
            # 构建用户特征数据（只保留必要的交叉排列数据）
            user_features = {
                'UserID': user_id,
                'cross_arranged_items': cross_arranged_items
            }
            
            extracted_data.append(user_features)
        
        return extracted_data

async def main_async():
    """异步主函数 - 执行豆包异步推理"""
    # 配置参数
    config = {
        'csv_file_path': "meta-data/user_interactions.csv",
        'fusion_item_path': "meta-data/Fusion_Item.csv",
        'temperature': 0.7,
        'output_file': "./data/doubao_results_async.txt",
        'max_concurrent': 10
    }
    
    print("=" * 60)
    print("🚀 商品推荐豆包异步推理")
    print("=" * 60)
    print(f"用户交互文件: {config['csv_file_path']}")
    print(f"商品信息文件: {config['fusion_item_path']}")
    print(f"温度参数: {config['temperature']}")
    print(f"最大并发数: {config['max_concurrent']}")
    print(f"结果输出: {config['output_file']}")
    print("-" * 60)
    
    try:
        generator = RecommendationBatchGenerator(temperature=config['temperature'])
        
        # 检查API配置
        if not generator.api_key or not generator.model_endpoint:
            print("\n❌ API配置不完整，请检查.env文件")
            return
        
        print("\n✓ API配置已验证")
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行异步推理
        success = await generator.execute_realtime_batch_inference_async(
            csv_file_path=config['csv_file_path'],
            fusion_item_path=config['fusion_item_path'],
            output_file=config['output_file'],
            max_users=50,
            max_concurrent=config['max_concurrent']
        )
        
        # 记录结束时间
        end_time = time.time()
        async_duration = end_time - start_time
        
        if success:
            print(f"\n✅ 异步推理完成！耗时: {async_duration:.2f}秒")
        else:
            print("\n❌ 异步推理失败，请检查错误信息")
            
        # 性能对比说明
        print("\n" + "=" * 60)
        print("📊 性能优化说明:")
        print(f"• 异步并发处理可显著减少总耗时")
        print(f"• 当前并发数设置: {config['max_concurrent']}")
        print(f"• 可通过调整max_concurrent参数控制并发数量")
        print(f"• 建议根据API限制和网络条件选择合适的并发数")
        print("=" * 60)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 程序执行错误: {e}")

def main():
    """主函数 - 执行豆包实时推理（同步版本，保留用于对比）"""
    # 配置参数
    config = {
        'csv_file_path': "meta-data/user_interactions.csv",
        'fusion_item_path': "meta-data/Fusion_Item.csv",
        'temperature': 0.7,
        'output_file': "./data/doubao_results_sync.txt"
    }
    
    print("=" * 60)
    print("🚀 商品推荐豆包同步推理（对比版本）")
    print("=" * 60)
    print(f"用户交互文件: {config['csv_file_path']}")
    print(f"商品信息文件: {config['fusion_item_path']}")
    print(f"温度参数: {config['temperature']}")
    print(f"结果输出: {config['output_file']}")
    print("-" * 60)
    
    try:
        generator = RecommendationBatchGenerator(temperature=config['temperature'])
        
        # 检查API配置
        if not generator.api_key or not generator.model_endpoint:
            print("\n❌ API配置不完整，请检查.env文件")
            return
        
        print("\n✓ API配置已验证")
        
        # 记录开始时间
        start_time = time.time()
        
        success = generator.execute_realtime_batch_inference(
            csv_file_path=config['csv_file_path'],
            fusion_item_path=config['fusion_item_path'],
            output_file=config['output_file'],
            max_users=5
        )
        
        # 记录结束时间
        end_time = time.time()
        sync_duration = end_time - start_time
        
        if success:
            print(f"\n✅ 同步推理完成！耗时: {sync_duration:.2f}秒")
        else:
            print("\n❌ 同步推理失败，请检查错误信息")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 程序执行错误: {e}")


if __name__ == "__main__":
    # 运行异步版本
    asyncio.run(main_async())