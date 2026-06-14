import numpy as np
from typing import Tuple, Optional

def generate_kmeans_data(
    n_samples: int = 300,
    n_features: int = 2,
    n_clusters: int = 3,
    cluster_std: float = 1.0,
    center_box: Tuple[float, float] = (-10.0, 10.0),
    random_state: Optional[int] = 42,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 K-Means 测试数据集，并可保存为空格分隔的 .txt 文件。

    参数
    ----------
    n_samples : int, 默认 300
        总样本数（近似均匀分配到各簇）。
    n_features : int, 默认 2
        数据维度。
    n_clusters : int, 默认 3
        簇数。
    cluster_std : float, 默认 1.0
        各簇的标准差（各向同性高斯）。
    center_box : (min, max), 默认 (-10, 10)
        中心点的均匀采样范围。
    random_state : int 或 None, 默认 42
        随机种子。
    save_path : str 或 None, 可选
        保存文件的完整路径，推荐以 .txt 结尾。
        若未提供则不保存。

    返回
    ----------
    X : ndarray (n_samples, n_features)
        特征矩阵。
    y : ndarray (n_samples,)
        真实簇标签（0 ~ n_clusters-1）。
    """
    rng = np.random.default_rng(random_state)

    # 分配每个簇的样本数
    samples_per_cluster = np.full(n_clusters, n_samples // n_clusters, dtype=int)
    samples_per_cluster[:n_samples % n_clusters] += 1

    # 随机簇中心
    centers = rng.uniform(center_box[0], center_box[1], size=(n_clusters, n_features))

    X_list, y_list = [], []
    for i in range(n_clusters):
        points = rng.normal(loc=centers[i], scale=cluster_std,
                            size=(samples_per_cluster[i], n_features))
        X_list.append(points)
        y_list.append(np.full(samples_per_cluster[i], i, dtype=int))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # 打乱样本顺序
    shuffle_idx = rng.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    if save_path is not None:
        # 保存为空格分隔的文本文件
        np.savetxt(save_path, X, delimiter=' ', fmt='%.6f')
        print(f"数据集已保存至 {save_path}")

    return X, y


if __name__ == "__main__":
    X, y = generate_kmeans_data(
        n_samples=500,
        n_features=3,
        n_clusters=6,
        cluster_std=1.5,
        center_box=(-10, 10),
        random_state=123,
        save_path="kmeans_data.txt"   # 输出为 .txt
    )
    print(f"数据形状: {X.shape}, 标签分布: {np.bincount(y)}")