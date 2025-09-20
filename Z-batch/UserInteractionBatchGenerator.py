import pandas as pd
import json
from typing import Dict, Any, List
import os
from pathlib import Path


class UserInteractionBatchGenerator:
    """用户交互数据批处理生成器
    
    用于将CSV格式的用户交互数据转换为AI批处理请求的JSONL格式文件。
    专门处理reviewText和brief_description字段中的管道符分隔数据。
    """
    
    def __init__(self, temperature: float = 0.5):
        """初始化生成器
        
        Args:
            temperature: AI模型的温度参数，控制输出的随机性
        """
        self.temperature = temperature
        self.system_prompt = (
            "You are a professional user review and product information analysis assistant, "
            "skilled at extracting key information from user reviews and product descriptions for structured analysis."
        )
    
    def _format_pipe_separated_content(self, content: str, field_name: str) -> str:
        """将管道符分隔的内容转换为带序号的回车符分隔格式，去掉最后一个元素用于预测
        
        Args:
            content: 管道符分隔的内容字符串
            field_name: 字段名称
            
        Returns:
            格式化后的字符串
        """
        if pd.isna(content) or not content.strip():
            return f"{field_name}: No content"
        
        items = content.split('|')
        
        # 直接去掉最后一个元素用于预测
        items = items[:-1]
        
        # 检查总字符长度，如果超过5000则进行截断处理
        total_length = sum(len(item.strip()) for item in items if item.strip())
        if total_length > 5000:
            items = self._truncate_items(items)
        
        formatted_items = []
        
        for i, item in enumerate(items, 1):
            if item.strip():  # 只处理非空项
                formatted_items.append(f"{i}. {item.strip()}")
        
        if not formatted_items:
            return f"{field_name}: No valid content"
        
        return f"{field_name}:\n" + "\n".join(formatted_items)
    
    def _truncate_items(self, items: List[str]) -> List[str]:
        """截断评论项目，每个项目保留前750字符和后750字符，中间用句号连接
        
        Args:
            items: 原始评论项目列表
            
        Returns:
            截断后的评论项目列表
        """
        truncated_items = []
        
        for item in items:
            item = item.strip()
            if not item:
                continue
                
            if len(item) <= 1000:  # 如果长度不超过1000，不需要截断
                truncated_items.append(item)
            else:
                # 截断：前500字符 + 句号 + 后500字符
                front_part = item[:500]
                back_part = item[-500:]
                truncated_item = f"{front_part}.{back_part}"
                truncated_items.append(truncated_item)
        
        return truncated_items
    
    def _create_prompt_template(self, interaction_data: Dict[str, Any]) -> str:
        """创建用于API调用的提示词模板
        
        Args:
            interaction_data: 用户交互数据字典
            
        Returns:
            格式化的提示词字符串
        """
        # 格式化reviewText和brief_description（自动去掉最后一个元素用于预测）
        formatted_reviews = self._format_pipe_separated_content(
            interaction_data.get('reviewText', ''), 'User Reviews'
        )
        formatted_descriptions = self._format_pipe_separated_content(
            interaction_data.get('brief_description', ''), 'Product Descriptions'
        )
        
        prompt = f"""Task: Based on user reviews and product descriptions, analyze user purchase instructions and user characteristic keywords.

{formatted_reviews}

{formatted_descriptions}

Requirements:
1. Extract user purchase instructions and demand intentions from user reviews
2. Identify user key characteristics and preference keywords
3. Strictly output results in JSON object format, containing the following fields:
   - user_instructions: User instructions (only two sentences, indicating what kind of products the user is looking for)
   - user_characteristics: User characteristic keywords (only a few keywords, no more than ten keywords, for example: ["keyword1", "keyword2", "keyword3"])

Please ensure the output is in valid JSON format."""
        return prompt
    
    def _create_batch_request(self, interaction_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """创建单个批处理请求
        
        Args:
            interaction_data: 用户交互数据字典
            request_id: 请求ID
            
        Returns:
            批处理请求字典
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self._create_prompt_template(interaction_data)
            }
        ]
        
        return {
            "custom_id": request_id,
            "body": {
                "messages": messages,
                "response_format": {
                    "type": "json_object"
                },
                "thinking": {
                    "type": "disabled"
                },
                "temperature": self.temperature,
                "max_tokens": 500
            }
        }
    
    @staticmethod
    def _safe_get_value(row: pd.Series, key: str) -> str:
        """安全获取DataFrame行中的值
        
        Args:
            row: pandas Series对象
            key: 字段名
            
        Returns:
            字符串值，如果为空或NaN则返回空字符串
        """
        value = row.get(key, '')
        return '' if pd.isna(value) else str(value)
    
    def _extract_interaction_data(self, row: pd.Series) -> Dict[str, str]:
        """从DataFrame行中提取用户交互数据
        
        Args:
            row: pandas Series对象
            
        Returns:
            用户交互数据字典
        """
        return {
            'brief_description': self._safe_get_value(row, 'brief_description'),
            'reviewText': self._safe_get_value(row, 'reviewText')
        }
    
    def generate_batch_files(self, 
                           csv_file_path: str, 
                           output_dir: str = "./data/user-interaction-batch", 
                           batch_size: int = 1000,
                           file_prefix: str = "user_batch") -> bool:
        """将CSV数据按批次转换为JSONL格式文件
        
        Args:
            csv_file_path: CSV文件路径
            output_dir: 输出目录路径
            batch_size: 每个批次的数据量
            file_prefix: 输出文件前缀
            
        Returns:
            是否成功生成所有文件
        """
        try:
            # 验证输入文件
            csv_path = Path(csv_file_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV文件不存在: {csv_file_path}")
            
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 读取CSV文件
            print(f"正在读取CSV文件: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            
            total_rows = len(df)
            
            if total_rows == 0:
                print("警告: CSV文件为空")
                return True
            
            print(f"成功读取CSV文件，共{total_rows:,}行数据")
            
            # 计算批次数
            num_batches = (total_rows + batch_size - 1) // batch_size
            print(f"将分成{num_batches}个批次，每批次最多{batch_size:,}条数据")
            
            # 处理每个批次
            for batch_num in range(num_batches):
                start_row = batch_num * batch_size
                end_row = min(start_row + batch_size, total_rows)
                current_batch_size = end_row - start_row
                
                # 生成输出文件路径
                output_file = output_path / f"{file_prefix}_{batch_num + 1:03d}.jsonl"
                
                print(f"正在处理批次 {batch_num + 1}/{num_batches}，"
                      f"数据范围: {start_row:,}-{end_row-1:,} ({current_batch_size:,}条)")
                
                # 写入批次文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    for index, row in df.iloc[start_row:end_row].iterrows():
                        # 提取用户交互数据
                        interaction_data = self._extract_interaction_data(row)
                        
                        # 创建批处理请求
                        user_id = self._safe_get_value(row, 'UserID')
                        request_id = f"{user_id}"
                        batch_request = self._create_batch_request(interaction_data, request_id)
                        
                        # 写入JSONL文件
                        f.write(json.dumps(batch_request, ensure_ascii=False) + '\n')
                
                print(f"✓ 成功生成批次文件: {output_file} ({current_batch_size:,}条记录)")
            
            print(f"\n🎉 所有批次文件生成完成！")
            print(f"   - 总计: {num_batches}个文件")
            print(f"   - 数据量: {total_rows:,}条记录")
            print(f"   - 保存位置: {output_path.absolute()}")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ 文件错误: {e}")
            return False
        except pd.errors.EmptyDataError:
            print(f"❌ CSV文件为空或格式错误: {csv_file_path}")
            return False
        except Exception as e:
            print(f"❌ 生成JSONL文件时发生错误: {e}")
            return False


def main():
    """主函数 - 配置参数并执行批处理生成"""
    # 配置参数
    config = {
        'csv_file_path': "d:\\WorkSpace\\RAGRec\\meta-data\\user_interactions.csv",
        'output_dir': "./data/user-interaction-batch",
        'batch_size': 10000,  # 每批次10000个样本
        'temperature': 0.5,
        'file_prefix': "user_batch"
    }
    
    print("=" * 60)
    print("🚀 用户交互数据批处理生成器")
    print("=" * 60)
    print(f"输入文件: {config['csv_file_path']}")
    print(f"输出目录: {config['output_dir']}")
    print(f"批次大小: {config['batch_size']:,}")
    print(f"温度参数: {config['temperature']}")
    print("-" * 60)
    
    # 创建生成器实例
    generator = UserInteractionBatchGenerator(temperature=config['temperature'])
    
    # 执行批处理生成
    success = generator.generate_batch_files(
        csv_file_path=config['csv_file_path'],
        output_dir=config['output_dir'],
        batch_size=config['batch_size'],
        file_prefix=config['file_prefix']
    )
    
    if success:
        print("\n✅ 所有JSONL文件生成成功！")
    else:
        print("\n❌ JSONL文件生成失败，请检查错误信息")


if __name__ == "__main__":
    main()