#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
====================================================
sklearn DBSCAN 聚类 + 可视化
====================================================

用法:
    python dbscan_sklearn.py

    python dbscan_sklearn.py --eps 0.5 --min-samples 5

    python dbscan_sklearn.py --data ../../data/dbscan.txt --eps 0.8 --min-samples 10

参数:
    --data          数据文件路径 (默认: ../../data/dbscan.txt)
    --eps           邻域半径 ε
    --min-samples   核心点最小邻居数 (sklearn 的 min_samples)
    --no-show       仅保存图像不弹出窗口
    --output        输出图像路径 (默认: dbscan_sklearn_result.png)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


def load_data(filepath: str) -> np.ndarray:
    """从空格分隔的 txt 文件加载点云数据"""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def print_statistics(labels: np.ndarray, n_points: int):
    """打印聚类统计信息"""
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))

    print("=" * 50)
    print(f"  总点数      : {n_points}")
    print(f"  簇数量      : {n_clusters}")
    print(f"  噪声点      : {n_noise} ({100 * n_noise / n_points:.1f}%)")
    print("-" * 50)

    for label in sorted(unique_labels):
        count = int(np.sum(labels == label))
        if label == -1:
            print(f"  噪声 (  -1): {count}")
        else:
            print(f"  簇   ({label:3d}): {count}")
    print("=" * 50)


def visualize(data: np.ndarray, labels: np.ndarray,
              eps: float, min_samples: int,
              data_path: str, output_path: str, show: bool):
    """可视化聚类结果，支持 2D / 3D"""

    n_dims = data.shape[1]
    unique_labels = set(labels)
    cluster_labels = sorted(unique_labels - {-1})  # 排除噪声
    n_clusters = len(cluster_labels)
    n_noise = int(np.sum(labels == -1))

    # ---- 调色：噪声用灰色，簇用 tab10 ----
    cmap = plt.cm.tab10
    noise_color = (0.6, 0.6, 0.6)  # 灰色

    if n_dims == 2:
        fig, ax = plt.subplots(figsize=(10, 7))

        # 先画噪声（最底层）
        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(data[noise_mask, 0], data[noise_mask, 1],
                       s=6, alpha=0.35, c=[noise_color], label=f'Noise ({n_noise})',
                       edgecolors='none')

        # 再画各簇
        for idx, label in enumerate(cluster_labels):
            mask = labels == label
            ax.scatter(data[mask, 0], data[mask, 1],
                       s=12, alpha=0.75, color=cmap(idx % 10),
                       label=f'Cluster {label}', edgecolors='none')

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    elif n_dims == 3:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(data[noise_mask, 0], data[noise_mask, 1], data[noise_mask, 2],
                       s=6, alpha=0.35, c=[noise_color], label=f'Noise ({n_noise})',
                       edgecolors='none')

        for idx, label in enumerate(cluster_labels):
            mask = labels == label
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2],
                       s=12, alpha=0.75, color=cmap(idx % 10),
                       label=f'Cluster {label}', edgecolors='none')

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")

    else:
        # >3 维：仅显示前 3 维
        print(f"⚠  数据为 {n_dims} 维，仅显示前三维。")
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(data[noise_mask, 0], data[noise_mask, 1], data[noise_mask, 2],
                       s=6, alpha=0.35, c=[noise_color], label=f'Noise ({n_noise})',
                       edgecolors='none')

        for idx, label in enumerate(cluster_labels):
            mask = labels == label
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2],
                       s=12, alpha=0.75, color=cmap(idx % 10),
                       label=f'Cluster {label}', edgecolors='none')

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")

    # ---- 标题与图例 ----
    ax.set_title(
        f"sklearn DBSCAN  —  eps={eps}, min_samples={min_samples}\n"
        f"Source: {data_path}\n"
        f"({data.shape[0]} points, {n_clusters} clusters, {n_noise} noise)",
        fontsize=11
    )
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=2, fontsize='small', loc='best',
              ncol=max(1, (n_clusters + 1) // 15))
    fig.tight_layout()

    # ---- 保存 ----
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图像已保存至: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="sklearn DBSCAN 聚类 + 可视化"
    )
    parser.add_argument(
        "--data", type=str, default="../../data/dbscan.txt",
        help="数据文件路径 (默认: ../../data/dbscan.txt)"
    )
    parser.add_argument(
        "--eps", type=float, default=0.5,
        help="邻域半径 ε (默认: 0.5)"
    )
    parser.add_argument(
        "--min-samples", type=int, default=5,
        help="核心点最小邻居数 (默认: 5)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="仅保存图像，不弹出窗口"
    )
    parser.add_argument(
        "--output", type=str, default="dbscan_sklearn_result.png",
        help="输出图像路径 (默认: dbscan_sklearn_result.png)"
    )
    args = parser.parse_args()

    # ---- 加载数据 ----
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path.resolve()}")
        sys.exit(1)

    print(f"加载数据: {data_path.resolve()}")
    data = load_data(str(data_path))
    n_points, n_dims = data.shape
    print(f"  点数: {n_points}, 维度: {n_dims}")

    # ---- sklearn DBSCAN ----
    print(f"\n运行 DBSCAN: eps={args.eps}, min_samples={args.min_samples}")
    db = DBSCAN(eps=args.eps, min_samples=args.min_samples, n_jobs=-1)
    labels = db.fit_predict(data)

    # ---- 统计 + 可视化 ----
    print_statistics(labels, n_points)
    visualize(data, labels,
              eps=args.eps, min_samples=args.min_samples,
              data_path=str(data_path.resolve()),
              output_path=args.output,
              show=not args.no_show)


if __name__ == "__main__":
    main()
