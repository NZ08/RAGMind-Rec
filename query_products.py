import pandas as pd
import json
import numpy as np
from VectorProcessor import VectorProcessor
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProductQuerySystem:
    """
    商品查询系统，基于用户特征和商品特征进行向量查询
    """
    
    def __init__(self):
        """初始化查询系统"""
        self.vector_processor = VectorProcessor()
        
    def load_user_data(self):
        """加载用户相关数据"""
        try:
            # 加载用户指令和特征
            self.user_instruc_charac = pd.read_csv('data/UserInstruCharac.csv')
            # 加载商品特征
            self.product_features = pd.read_csv('data/ProductFeature.csv')
            # 加载用户画像
            self.user_profiles = pd.read_csv('data/UserProfiles.csv')
            
            logging.info(f"加载用户指令特征数据: {len(self.user_instruc_charac)} 条")
            logging.info(f"加载商品特征数据: {len(self.product_features)} 条")
            logging.info(f"加载用户画像数据: {len(self.user_profiles)} 条")
            
        except Exception as e:
            logging.error(f"加载数据失败: {e}")
            raise
    
    def construct_query_text(self, user_id):
        """构建查询文本，结合用户指令和偏好商品特征"""
        query_parts = []
        
        # 获取用户指令
        user_instruc_row = self.user_instruc_charac[self.user_instruc_charac['UserID'] == user_id]
        if not user_instruc_row.empty:
            instructions = user_instruc_row.iloc[0]['user_instructions']
            # characteristics = user_instruc_row.iloc[0]['user_characteristics']
            query_parts.append(f"用户指令: {instructions}")
            # query_parts.append(f"用户特征: {characteristics}")
        
        # 获取商品特征偏好
        product_row = self.product_features[self.product_features['UserID'] == user_id]
        if not product_row.empty:
            features = product_row.iloc[0]['product_features']
            query_parts.append(f"偏好商品特征: {features}")
        
        # 获取用户画像
        # profile_row = self.user_profiles[self.user_profiles['UserID'] == user_id]
        # if not profile_row.empty:
        #     profile = profile_row.iloc[0]['user_profile']
        #     query_parts.append(f"用户画像: {profile}")
        
        return " ".join(query_parts)
    
    def query_products_for_user(self, user_id, n_results=20, collection_name="Small_Baby_Item"):
        """为指定用户查询商品"""
        try:
            # 构建查询文本
            query_text = self.construct_query_text(user_id)
            
            if not query_text.strip():
                logging.warning(f"用户 {user_id} 没有找到相关数据")
                return None
            
            logging.info(f"为用户 {user_id} 查询商品...")
            logging.info(f"查询文本长度: {len(query_text)} 字符")
            
            # 执行向量查询
            results = self.vector_processor.search_similar(
                query=query_text,
                collection_name=collection_name,
                n_results=n_results
            )
            
            return {
                'user_id': user_id,
                'query_text': query_text,
                'results': results
            }
            
        except Exception as e:
            logging.error(f"为用户 {user_id} 查询商品失败: {e}")
            return None
    
    def query_all_users(self, products_per_user=20, collection_name="Small_Baby_Item"):
        """为所有用户查询商品"""
        self.load_user_data()
        
        # 获取所有用户ID
        unique_user_ids = self.user_instruc_charac['UserID'].unique()
        
        results = []
        
        for user_id in unique_user_ids:
            logging.info(f"\n处理用户 {user_id}...")
            
            user_result = self.query_products_for_user(
                user_id=user_id,
                n_results=products_per_user,
                collection_name=collection_name
            )
            
            if user_result:
                results.append(user_result)
                
                # 显示查询结果摘要
                if user_result['results']['documents'][0]:
                    logging.info(f"用户 {user_id} 找到 {len(user_result['results']['documents'][0])} 个相关商品")
                    # 显示前3个结果的ID
                    top_3_ids = user_result['results']['ids'][0][:3]
                    logging.info(f"前3个商品ID: {top_3_ids}")
                else:
                    logging.warning(f"用户 {user_id} 没有找到相关商品")
        
        return results
    
    def save_results(self, results, output_file="query_results.json"):
        """保存查询结果到文件"""
        try:
            # 转换numpy类型为Python原生类型
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            converted_results = convert_numpy_types(results)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, ensure_ascii=False, indent=2)
            logging.info(f"查询结果已保存到 {output_file}")
        except Exception as e:
            logging.error(f"保存结果失败: {e}")
    
    def extract_user_asins_to_csv(self, results, output_file="user_recommendations.csv"):
        """提取每个用户的UserID和推荐商品的ASIN，保存到CSV文件"""
        try:
            data = []
            
            for result in results:
                user_id = result['user_id']
                documents = result['results']['documents'][0] if result['results']['documents'][0] else []
                
                asins = []
                for doc_str in documents:
                    try:
                        # 解析JSON字符串
                        doc = json.loads(doc_str) if isinstance(doc_str, str) else doc_str
                        if isinstance(doc, dict) and 'asin' in doc:
                            asins.append(doc['asin'])
                        elif isinstance(doc_str, str) and doc_str.startswith('asin:'):
                            # 处理新格式：直接从字符串中提取asin
                            asin_part = doc_str.split('|')[0].strip()
                            if asin_part.startswith('asin:'):
                                asin = asin_part.split(':', 1)[1].strip()
                                asins.append(asin)
                    except json.JSONDecodeError as e:
                        # 如果不是JSON格式，尝试直接解析字符串格式
                        if isinstance(doc_str, str) and doc_str.startswith('asin:'):
                            asin_part = doc_str.split('|')[0].strip()
                            if asin_part.startswith('asin:'):
                                asin = asin_part.split(':', 1)[1].strip()
                                asins.append(asin)
                        else:
                            logging.warning(f"解析文档失败: {e}")
                        continue
                
                # 将ASIN列表用管道符连接
                asin_string = '|'.join(asins)
                data.append({
                    'UserID': user_id,
                    'asin': asin_string
                })
            
            # 创建DataFrame并按UserID排序
            df = pd.DataFrame(data)
            df = df.sort_values('UserID')
            df.to_csv(output_file, index=False, encoding='utf-8')
            logging.info(f"用户推荐结果已保存到 {output_file}，共 {len(data)} 个用户（按UserID排序）")
            
        except Exception as e:
            logging.error(f"提取用户ASIN失败: {e}")
    
    def query_user_parallel(self, user_id, products_per_user=20, collection_name="Small_Baby_Item"):
        """并行查询单个用户的商品（线程安全版本）"""
        try:
            # 为每个线程创建独立的VectorProcessor实例
            vector_processor = VectorProcessor()
            
            # 构建查询文本
            query_text = self.construct_query_text(user_id)
            
            if not query_text.strip():
                logging.warning(f"用户 {user_id} 没有找到相关数据")
                return None
            
            logging.info(f"为用户 {user_id} 查询商品...")
            
            # 执行向量查询
            results = vector_processor.search_similar(
                query=query_text,
                collection_name=collection_name,
                n_results=products_per_user
            )
            
            return {
                'user_id': user_id,
                'query_text': query_text,
                'results': results
            }
            
        except Exception as e:
            logging.error(f"为用户 {user_id} 查询商品失败: {e}")
            return None
    
    def query_all_users_parallel(self, products_per_user=20, collection_name="Small_Baby_Item", max_workers=58, max_users=10):
        """使用并行计算为指定数量的用户查询商品"""
        self.load_user_data()
        
        # 获取前max_users个用户ID
        unique_user_ids = self.user_instruc_charac['UserID'].unique()[:max_users]
        
        results = []
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_user = {
                executor.submit(self.query_user_parallel, user_id, products_per_user, collection_name): user_id 
                for user_id in unique_user_ids
            }
            
            # 收集结果
            for future in as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    user_result = future.result()
                    if user_result:
                        results.append(user_result)
                        
                        # 显示查询结果摘要
                        if user_result['results']['documents'][0]:
                            logging.info(f"用户 {user_id} 找到 {len(user_result['results']['documents'][0])} 个相关商品")
                        else:
                            logging.warning(f"用户 {user_id} 没有找到相关商品")
                            
                except Exception as e:
                    logging.error(f"用户 {user_id} 处理失败: {e}")
        
        return results

def main():
    """主函数"""
    try:
        # 创建查询系统
        query_system = ProductQuerySystem()
        
        # 使用并行计算为前10000个用户查询商品，每个用户查询20个商品
        logging.info("开始使用并行计算为前10000个用户查询商品...")
        results = query_system.query_all_users_parallel(
            products_per_user=50,
            max_workers=58,  # 使用58个线程并行处理
            max_users=1000  # 限制处理前10000个用户
        )
        
        # 保存JSON结果
        query_system.save_results(results)
        
        # 提取UserID和ASIN到CSV文件
        query_system.extract_user_asins_to_csv(results, "user_recommendations.csv")
        
        # 显示总结
        logging.info(f"\n查询完成！")
        logging.info(f"成功处理 {len(results)} 个用户")
        
        total_recommendations = 0
        for result in results:
            user_id = result['user_id']
            num_products = len(result['results']['documents'][0]) if result['results']['documents'][0] else 0
            total_recommendations += num_products
            logging.info(f"用户 {user_id}: {num_products} 个商品")
        
        logging.info(f"总共生成 {total_recommendations} 条推荐记录")
        logging.info("结果已保存到 query_results.json 和 user_recommendations.csv")
            
    except Exception as e:
        logging.error(f"程序执行失败: {e}")

if __name__ == "__main__":
    main()