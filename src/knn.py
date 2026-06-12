from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 加载数据
iris = load_iris()

X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 创建模型
model = KNeighborsClassifier(n_neighbors=3)

# 训练
model.fit(X_train, y_train)

# 测试
score = model.score(X_test, y_test)

print(score)