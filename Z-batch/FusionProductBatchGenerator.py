import pandas as pd
import json
from typing import Dict, Any, Optional
import os
from pathlib import Path


class ProductBatchGenerator:
    """产品数据批处理生成器
    
    用于将CSV格式的产品数据转换为AI批处理请求的JSONL格式文件。
    支持多模态数据处理（文本+图像），适用于RAG检索系统。
    """
    
    def __init__(self, temperature: float = 0.5):
        """初始化生成器
        
        Args:
            temperature: AI模型的温度参数，控制输出的随机性
        """
        self.temperature = temperature
        self.system_prompt = (
            "你是一个专业的商品信息分析助手，"
            "擅长从商品的文本描述和图像中提取核心特征并使用英文回答。"
        )
    
    def _create_prompt_template(self, product_data: Dict[str, Any]) -> str:
        """创建用于API调用的提示词模板
        
        Args:
            product_data: 产品数据字典
            
        Returns:
            格式化的提示词字符串
        """
        prompt = f"""Task: Combine the text information and images of the following goods to extract the core features for RAG retrieval, taking into account the text attributes and image visual details.

Product text information:
"asin": "{product_data.get('asin', '')}",
"description": "{product_data.get('description', '')}",
"title": "{product_data.get('title', '')}",
"price": {product_data.get('price', '')},
"brand": "{product_data.get('brand', '')}"

Requirements:
1. Features not specified in the supplementary text but observable in the image;
2. Infer the implied attributes of the text;
3. It is presented in short sentences of "dimension+specific content", without redundancy, highlighting high-frequency keywords;
4. The output JSON format should include asin, brand of the product, description of the product, title of the product, price of the product, target customer portrait, and brief description (such as schoolbag, pacifier, milk powder, etc.). All field contents should not be bracketed.
5. If the description, title, price and brand fields of the original product information are empty or do not exist, you can omit them from the output JSON.
6. You can add other fields according to the actual situation, but they should be useful for product information."""
        return prompt
    
    def _create_message_content(self, product_data: Dict[str, Any]) -> list:
        """创建消息内容列表，包含文本和图像
        
        Args:
            product_data: 产品数据字典
            
        Returns:
            消息内容列表
        """
        content = []
        
        # 添加文本内容
        text_content = {
            "type": "text",
            "text": self._create_prompt_template(product_data)
        }
        content.append(text_content)
        
        # 添加图像内容（如果存在有效的图像URL）
        image_url = product_data.get('imUrl', '').strip()
        if image_url:
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "high"  # 高细节模式，更好地理解图像细节
                }
            }
            content.append(image_content)
        
        return content
    
    def _create_batch_request(self, product_data: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """创建单个批处理请求
        
        Args:
            product_data: 产品数据字典
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
                "content": self._create_message_content(product_data)
            }
        ]
        
        return {
            "custom_id": request_id,
            "body": {
                "messages": messages,
                "thinking": {
                    "type": "disabled"
                },
                "temperature": self.temperature
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
    
    def _extract_product_data(self, row: pd.Series) -> Dict[str, str]:
        """从DataFrame行中提取产品数据
        
        Args:
            row: pandas Series对象
            
        Returns:
            产品数据字典
        """
        return {
            'asin': self._safe_get_value(row, 'asin'),
            'description': self._safe_get_value(row, 'description'),
            'title': self._safe_get_value(row, 'title'),
            'price': self._safe_get_value(row, 'price'),
            'imUrl': self._safe_get_value(row, 'imUrl'),
            'brand': self._safe_get_value(row, 'brand')
        }
    
    def generate_batch_files(self, 
                           csv_file_path: str, 
                           output_dir: str = "./data", 
                           batch_size: int = 10000,
                           file_prefix: str = "batch") -> bool:
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
                        # 提取产品数据
                        product_data = self._extract_product_data(row)
                        
                        # 创建批处理请求
                        request_id = f"request-{index + 1}"
                        batch_request = self._create_batch_request(product_data, request_id)
                        
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

    # 本代码文件以后要修改成专门的json格式，而不是用提示词告诉他生成json格式
    config = {
        'csv_file_path': "d:\\WorkSpace\\RAGRec\\meta.csv",
        'output_dir': "./data",
        'batch_size': 10000,
        'temperature': 0.5,
        'file_prefix': "batch"
    }
    
    print("=" * 60)
    print("🚀 产品数据批处理生成器")
    print("=" * 60)
    print(f"输入文件: {config['csv_file_path']}")
    print(f"输出目录: {config['output_dir']}")
    print(f"批次大小: {config['batch_size']:,}")
    print(f"温度参数: {config['temperature']}")
    print("-" * 60)
    
    # 创建生成器实例
    generator = ProductBatchGenerator(temperature=config['temperature'])
    
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