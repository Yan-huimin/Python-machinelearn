from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 生成数据
X, _ = make_blobs(
    n_samples=300, # 样本数
    centers=5, # 聚类数(分类的类别数)
    random_state=42 # 随机数种子
)

# 创建模型
model = KMeans(
    n_clusters=5, # 聚类数(分类的类别数)
    random_state=42 # 随机数种子
)

model.fit(X)

plt.scatter(
    X[:,0],
    X[:,1],
    c=model.labels_
)

plt.show()