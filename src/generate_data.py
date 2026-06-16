#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================
聚类测试数据生成器
====================================================

功能：
    - K-Means测试数据
    - DBSCAN测试数据
    - 支持噪声点
    - 支持不同簇密度
    - 支持二维/三维/高维
    - 输出txt文件（空格分隔）

----------------------------------------------------
使用示例
----------------------------------------------------

1. 查看帮助

python generate_data.py --help


----------------------------------------------------
2. 生成K-Means测试集
----------------------------------------------------

python generate_data.py ^
    --samples 5000 ^
    --features 2 ^
    --clusters 5 ^
    --std 1.0 ^
    --noise 0 ^
    --output kmeans.txt


PowerShell单行：

python generate_data.py --samples 5000 --features 2 --clusters 5 --std 1.0 --noise 0 --output kmeans.txt


----------------------------------------------------
3. 生成DBSCAN测试集
----------------------------------------------------

python generate_data.py ^
    --samples 10000 ^
    --features 2 ^
    --clusters 5 ^
    --std 0.5 1.0 2.0 3.0 4.0 ^
    --noise 0.15 ^
    --output dbscan.txt

说明：

簇0标准差 = 0.5
簇1标准差 = 1.0
簇2标准差 = 2.0
簇3标准差 = 3.0
簇4标准差 = 4.0

并添加15%噪声点


----------------------------------------------------
4. 生成三维点云
----------------------------------------------------

python generate_data.py ^
    --samples 100000 ^
    --features 3 ^
    --clusters 10 ^
    --std 2 ^
    --noise 0.1 ^
    --output pointcloud.txt


----------------------------------------------------
5. 固定随机种子
----------------------------------------------------

python generate_data.py ^
    --seed 123 ^
    --output test.txt

相同seed会生成完全相同的数据


----------------------------------------------------
输出格式
----------------------------------------------------

1.234567 2.345678
3.456789 4.567890
...

三维数据：

1.234567 2.345678 3.456789
4.567890 5.678901 6.789012

仅保存坐标，不保存标签

====================================================
"""

import argparse
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np


def generate_cluster_data(
    n_samples: int = 10000,
    n_features: int = 2,
    n_clusters: int = 5,
    cluster_std: Union[float, list] = 1.0,
    noise_ratio: float = 0.05,
    center_box: Tuple[float, float] = (-20, 20),
    random_state: Optional[int] = 42,
    save_path: Optional[str] = None,
):
    """
    生成聚类测试数据

    Returns
    -------
    X : ndarray
        特征矩阵

    y : ndarray
        真实标签

        0 ~ n_clusters-1:
            聚类标签

        -1:
            噪声点
    """

    rng = np.random.default_rng(random_state)

    n_noise = int(n_samples * noise_ratio)
    n_cluster_samples = n_samples - n_noise

    if isinstance(cluster_std, (float, int)):
        cluster_std = [cluster_std] * n_clusters

    if len(cluster_std) != n_clusters:
        raise ValueError(
            f"cluster_std长度({len(cluster_std)}) "
            f"必须等于n_clusters({n_clusters})"
        )

    samples_per_cluster = np.full(
        n_clusters,
        n_cluster_samples // n_clusters,
        dtype=int,
    )

    samples_per_cluster[: n_cluster_samples % n_clusters] += 1

    centers = rng.uniform(
        center_box[0],
        center_box[1],
        size=(n_clusters, n_features),
    )

    X_list = []
    y_list = []

    for i in range(n_clusters):

        points = rng.normal(
            loc=centers[i],
            scale=cluster_std[i],
            size=(samples_per_cluster[i], n_features),
        )

        X_list.append(points)

        y_list.append(
            np.full(samples_per_cluster[i], i)
        )

    # 噪声点
    if n_noise > 0:

        noise = rng.uniform(
            center_box[0],
            center_box[1],
            size=(n_noise, n_features),
        )

        X_list.append(noise)

        y_list.append(
            np.full(n_noise, -1)
        )

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # 打乱顺序
    idx = rng.permutation(len(X))

    X = X[idx]
    y = y[idx]

    # 保存文件
    if save_path is not None:

        np.savetxt(
            save_path,
            X,
            fmt="%.6f",
            delimiter=" ",
        )

        print(f"\n数据已保存至: {save_path}")

    return X, y


def main():

    parser = argparse.ArgumentParser(
        description="聚类测试数据生成器(K-Means / DBSCAN)"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="总点数(包含噪声)",
    )

    parser.add_argument(
        "--features",
        type=int,
        default=2,
        help="数据维度",
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="聚类数量",
    )

    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=[1.0],
        help=(
            "簇标准差\n"
            "单个值: 所有簇相同\n"
            "多个值: 每个簇不同"
        ),
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.05,
        help="噪声比例(0~1)",
    )

    parser.add_argument(
        "--min-center",
        type=float,
        default=-20,
        help="聚类中心最小值",
    )

    parser.add_argument(
        "--max-center",
        type=float,
        default=20,
        help="聚类中心最大值",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="cluster_data.txt",
        help="输出文件",
    )

    args = parser.parse_args()

    cluster_std = (
        args.std[0]
        if len(args.std) == 1
        else args.std
    )

    X, y = generate_cluster_data(
        n_samples=args.samples,
        n_features=args.features,
        n_clusters=args.clusters,
        cluster_std=cluster_std,
        noise_ratio=args.noise,
        center_box=(
            args.min_center,
            args.max_center,
        ),
        random_state=args.seed,
        save_path=args.output,
    )

    print("\n========== 数据集信息 ==========")
    print(f"总点数      : {len(X)}")
    print(f"数据维度    : {args.features}")
    print(f"聚类数量    : {args.clusters}")
    print(f"噪声比例    : {args.noise:.2%}")
    print(f"随机种子    : {args.seed}")
    print(f"输出文件    : {args.output}")

    unique, counts = np.unique(y, return_counts=True)

    print("\n标签统计：")

    for label, count in zip(unique, counts):

        if label == -1:
            print(f"噪声点(-1) : {count}")
        else:
            print(f"簇{label:<8}: {count}")

    print("\n生成完成")


if __name__ == "__main__":
    main()