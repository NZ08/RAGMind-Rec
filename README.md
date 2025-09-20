# RAG LLM 推荐系统

基于多模态融合和向量检索的商品推荐系统，结合文本和图像信息进行智能推荐。

## 主要功能

### 1. 多模态特征提取 (MultimodalFusion.py)
- 调用豆包API处理商品的文本和图像信息
- 提取商品的核心特征用于RAG检索
- 支持批量处理和并发调用

### 2. 向量处理 (VectorProcessor.py)
- 文本向量化处理（使用Qwen3-Embedding-8B模型）
- 向量存储到ChromaDB数据库
- 支持CSV文件批量处理

### 3. 商品查询系统 (query_products.py)
- 基于用户特征和商品特征进行向量查询
- 结合用户指令和偏好进行个性化推荐
- 支持并行查询处理

### 4. 推荐效果评估 (calculate_recall.py)
- 计算推荐系统的Recall@20指标
- 评估推荐准确性和系统性能

### 5. 批量数据生成 (Z-batch/)
- 用户交互数据生成
- 商品特征批量提取
- 推荐结果批量生成

## 环境配置

在项目根目录创建 `.env` 文件，配置以下API密钥：

```
DOUBAO_API_KEY=your_doubao_api_key
SILICONFLOW_API_KEY=your_siliconflow_api_key
```

## 安装依赖

```bash
pip install pandas requests openai chromadb python-dotenv tqdm
```

## 启动项目

### 1. 多模态特征提取
```bash
python MultimodalFusion.py
```

### 2. 向量处理和存储
```bash
python VectorProcessor.py
```

### 3. 商品查询推荐
```bash
python query_products.py
```

### 4. 计算推荐效果
```bash
python calculate_recall.py
```

## 数据文件说明

- `data/ProductFeature.csv` - 商品特征数据
- `data/UserProfiles.csv` - 用户画像数据
- `data/UserInstruCharac.csv` - 用户指令和特征
- `data/Target.csv` - 推荐目标数据
- `data/user_interactions.csv` - 用户交互数据

## 注意事项

1. 确保API密钥配置正确
2. 首次运行需要创建ChromaDB数据库
3. 建议按顺序执行：特征提取 → 向量处理 → 查询推荐 → 效果评估