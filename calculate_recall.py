import pandas as pd
import json
import logging
from query_products import ProductQuerySystem

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RecallCalculator:
    """
    计算推荐系统的Recall@20指标
    """
    
    def __init__(self):
        """初始化计算器"""
        self.query_system = ProductQuerySystem()
        self.target_data = None
        
    def load_target_data(self):
        """加载目标数据"""
        try:
            self.target_data = pd.read_csv('data/Target.csv')
            logging.info(f"加载目标数据: {len(self.target_data)} 条")
            return True
        except Exception as e:
            logging.error(f"加载目标数据失败: {e}")
            return False
    
    def get_user_target(self, user_id):
        """获取指定用户的目标商品"""
        user_target = self.target_data[self.target_data['UserID'] == user_id]
        if not user_target.empty:
            return user_target.iloc[0]['Target']
        return None
    
    def calculate_recall_for_user(self, user_id, recommended_items, target_item):
        """计算单个用户的recall@20"""
        if target_item is None:
            return 0
        
        # 检查目标商品是否在推荐列表中
        if target_item in recommended_items:
            return 1
        else:
            return 0
    
    def load_recommendations_from_file(self, file_path="query_results.json"):
        """从文件中加载推荐结果"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                recommendations = json.load(f)
            logging.info(f"从{file_path}加载了{len(recommendations)}个用户的推荐结果")
            return recommendations
        except Exception as e:
            logging.error(f"加载推荐结果失败: {e}")
            return None
    
    def load_recommendations_from_csv(self, file_path="user_recommendations.csv"):
        """从CSV文件中加载推荐结果"""
        try:
            df = pd.read_csv(file_path)
            logging.info(f"从{file_path}加载了{len(df)}个用户的推荐结果")
            return df
        except Exception as e:
            logging.error(f"加载CSV推荐结果失败: {e}")
            return None
    
    def extract_asins_from_documents(self, documents):
        """从documents字段中提取ASIN列表"""
        asins = []
        for doc in documents:
            try:
                # 解析JSON字符串
                doc_data = json.loads(doc)
                if 'asin' in doc_data:
                    asins.append(doc_data['asin'])
            except Exception as e:
                logging.warning(f"解析文档失败: {e}")
                continue
        return asins
    
    def calculate_recall_from_csv(self, file_path="user_recommendations.csv"):
        """从CSV文件中读取推荐结果并计算recall@20"""
        # 加载目标数据
        if not self.load_target_data():
            return None
        
        logging.info(f"开始从CSV文件中读取推荐结果并计算recall@20...")
        
        # 从CSV文件加载推荐结果
        recommendations_df = self.load_recommendations_from_csv(file_path)
        
        if recommendations_df is None or recommendations_df.empty:
            logging.error("加载推荐结果失败")
            return None
        
        # 计算recall@20
        total_users = 0
        total_hits = 0
        user_results = []
        
        for _, row in recommendations_df.iterrows():
            user_id = row['UserID']
            
            # 从asin列中提取ASIN列表（用管道符分割）
            asin_str = row['asin']
            if pd.isna(asin_str) or asin_str == '':
                recommended_items = []
            else:
                recommended_items = asin_str.split('|')
            
            # 获取用户的目标商品
            target_item = self.get_user_target(user_id)
            
            # 计算该用户的recall
            user_recall = self.calculate_recall_for_user(user_id, recommended_items, target_item)
            
            user_results.append({
                'user_id': user_id,
                'target_item': target_item,
                'recommended_items': recommended_items,
                'recall': user_recall,
                'hit': user_recall == 1
            })
            
            total_users += 1
            total_hits += user_recall
            
            logging.info(f"用户 {user_id}: 目标={target_item}, 推荐数量={len(recommended_items)}, 命中={'是' if user_recall else '否'}")
        
        # 计算总体recall@20
        overall_recall = total_hits / total_users if total_users > 0 else 0
        
        results = {
            'total_users': total_users,
            'total_hits': total_hits,
            'overall_recall_20': overall_recall,
            'user_results': user_results
        }
        
        return results
    
    def calculate_recall_from_file(self, top_n_users=100, file_path="query_results.json"):
        """从文件中读取推荐结果并计算recall@20"""
        # 加载目标数据
        if not self.load_target_data():
            return None
        
        logging.info(f"开始从文件中读取前{top_n_users}个用户的推荐结果并计算recall@20...")
        
        # 从文件加载推荐结果
        recommendations = self.load_recommendations_from_file(file_path)
        
        if not recommendations:
            logging.error("加载推荐结果失败")
            return None
        
        # 只处理前top_n_users个用户
        recommendations = recommendations[:top_n_users]
        
        # 计算recall@20
        total_users = 0
        total_hits = 0
        user_results = []
        
        for rec in recommendations:
            user_id = rec['user_id']
            
            # 从documents字段中提取ASIN列表
            if 'results' in rec and 'documents' in rec['results'] and rec['results']['documents']:
                documents = rec['results']['documents'][0]  # 取第一个文档列表
                recommended_items = self.extract_asins_from_documents(documents)
            else:
                recommended_items = []
            
            # 获取用户的目标商品
            target_item = self.get_user_target(user_id)
            
            # 计算该用户的recall
            user_recall = self.calculate_recall_for_user(user_id, recommended_items, target_item)
            
            user_results.append({
                'user_id': user_id,
                'target_item': target_item,
                'recommended_items': recommended_items,
                'recall': user_recall,
                'hit': user_recall == 1
            })
            
            total_users += 1
            total_hits += user_recall
            
            logging.info(f"用户 {user_id}: 目标={target_item}, 推荐数量={len(recommended_items)}, 命中={'是' if user_recall else '否'}")
        
        # 计算总体recall@20
        overall_recall = total_hits / total_users if total_users > 0 else 0
        
        results = {
            'total_users': total_users,
            'total_hits': total_hits,
            'overall_recall_20': overall_recall,
            'user_results': user_results
        }
        
        return results
    
    def save_recall_results(self, results, output_file="recall_results.json"):
        """保存recall计算结果"""
        try:
            # 转换numpy类型为Python原生类型
            def convert_types(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {key: convert_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                return obj
            
            converted_results = convert_types(results)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, ensure_ascii=False, indent=2)
            logging.info(f"Recall结果已保存到 {output_file}")
        except Exception as e:
            logging.error(f"保存recall结果失败: {e}")
    
    def print_summary(self, results):
        """打印结果摘要"""
        if not results:
            logging.error("没有结果可显示")
            return
        
        print("\n" + "="*50)
        print("Recall@20 计算结果摘要")
        print("="*50)
        print(f"总用户数: {results['total_users']}")
        print(f"命中用户数: {results['total_hits']}")
        print(f"总体 Recall@20: {results['overall_recall_20']:.4f} ({results['overall_recall_20']*100:.2f}%)")
        print("="*50)
        
        # 显示命中的用户详情
        hit_users = [ur for ur in results['user_results'] if ur['hit']]
        if hit_users:
            print(f"\n命中的用户详情 (共{len(hit_users)}个):")
            for ur in hit_users[:10]:  # 只显示前10个
                print(f"  用户 {ur['user_id']}: 目标商品 {ur['target_item']}")
            if len(hit_users) > 10:
                print(f"  ... 还有 {len(hit_users)-10} 个用户命中")
        
        # 显示未命中的用户示例
        miss_users = [ur for ur in results['user_results'] if not ur['hit']]
        if miss_users:
            print(f"\n未命中的用户示例 (共{len(miss_users)}个):")
            for ur in miss_users[:5]:  # 只显示前5个
                print(f"  用户 {ur['user_id']}: 目标商品 {ur['target_item']}, 推荐了 {len(ur['recommended_items'])} 个商品")
            if len(miss_users) > 5:
                print(f"  ... 还有 {len(miss_users)-5} 个用户未命中")

def main():
    """主函数"""
    try:
        # 创建recall计算器
        calculator = RecallCalculator()
        
        # 从CSV文件中读取推荐结果并计算recall@20
        results = calculator.calculate_recall_from_csv("user_recommendations.csv")
        
        if results:
            # 保存结果
            calculator.save_recall_results(results)
            
            # 打印摘要
            calculator.print_summary(results)
        else:
            logging.error("计算失败")
            
    except Exception as e:
        logging.error(f"程序执行失败: {e}")

if __name__ == "__main__":
    main()