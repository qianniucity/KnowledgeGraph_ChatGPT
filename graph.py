import json
import logging
import matplotlib.pyplot as plt
import networkx as nx
from networkx import connected_components
import pandas as pd
from sentence_transformers import SentenceTransformer, util

import service

s = service.Service()

# 从CSV文件中读取数据
data = pd.read_csv("./data/amazon_products.csv")

# 将多个列的内容合并为一列
data['text'] = data['TITLE'] + data['BULLET_POINTS'] + data['DESCRIPTION']

# 遍历前 100 条数据的 text 字段，调用 simple_graph 函数 返回的结果添加到 kg 列表
kg = []
for content in data['text'].values[:100]:
  try:
    extracted_relations = s.extract_information(content,"")
    extracted_relations = json.loads(extracted_relations)
    kg.extend(extracted_relations)
  except Exception as e:
    logging.error(e)

# 将kg列表转换为pandas的DataFrame对象
kg_relations = pd.DataFrame(kg)

# 从kg_relations DataFrame中获取名为head的列的值，即关系中的头实体
heads = kg_relations['head'].values
# 创建一个名为embedding_model的SentenceTransformer对象，用于对文本进行嵌入。
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(heads)
similarity = util.cos_sim(embeddings[0], embeddings[1])


G = nx.Graph()
for _, row in kg_relations.iterrows():
  G.add_edge(row['head'], row['tail'], label=row['relation'])

# 使用Spring布局算法计算节点的位置，设置随机种子为47，k参数为0.9
pos = nx.spring_layout(G, seed=47, k=0.9)
# 获取图中边的属性'label'
labels = nx.get_edge_attributes(G, 'label')

# 创建一个图形窗口，大小为15x15
plt.figure(figsize=(15, 15))

# 绘制图形，显示节点和边，设置节点和边的样式
nx.draw(G, pos, with_labels=True, font_size=10, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
# 绘制边的标签，设置标签的样式和位置
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8, label_pos=0.3, verticalalignment='baseline')
# 设置图形的标题
plt.title('Product Knowledge Graph')
# 显示图形
plt.show()