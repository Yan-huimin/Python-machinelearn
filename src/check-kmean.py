#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import warnings


def parse_clusters(filepath):
    """
    解析格式:

    Cluster 0:
    x y
    x y

    Cluster 1:
    x y
    x y
    """

    clusters = {}
    current_label = None
    n_dims = None

    cluster_pattern = re.compile(
        r'^Cluster\s+(\d+):\s*$'
    )

    with open(filepath, 'r', encoding='utf-8') as f:

        for raw_line in f:

            line = raw_line.strip()

            if not line:
                continue

            match = cluster_pattern.match(line)

            if match:

                current_label = (
                    f"Cluster {match.group(1)}"
                )

                clusters.setdefault(
                    current_label,
                    []
                )

                continue

            parts = line.split()

            try:
                coords = [
                    float(p)
                    for p in parts
                ]
            except ValueError:
                continue

            if n_dims is None and coords:
                n_dims = len(coords)

            if (
                n_dims is not None
                and len(coords) != n_dims
            ):
                warnings.warn(
                    f"忽略维度不一致的点: "
                    f"{coords} "
                    f"(期望 {n_dims} 维)"
                )
                continue

            if current_label is not None:
                clusters[current_label].append(
                    tuple(coords)
                )

    return clusters, n_dims


def main():

    if len(sys.argv) < 2:

        print(
            "用法: python check.py <簇文件路径>"
        )

        sys.exit(1)

    filepath = sys.argv[1]

    try:

        clusters, n_dims = parse_clusters(
            filepath
        )

    except Exception as e:

        print(f"读取文件失败: {e}")

        sys.exit(1)

    if not clusters or n_dims is None:

        print("未找到有效的簇数据")

        sys.exit(1)

    total_points = sum(
        len(points)
        for points in clusters.values()
    )

    cluster_names = sorted(
        clusters.keys(),
        key=lambda x: int(x.split()[1])
    )

    print(
        f"读取到 {len(cluster_names)} 个簇，"
        f"共计 {total_points} 个点，"
        f"维度：{n_dims}D"
    )

    colors = plt.cm.tab10

    # =====================
    # 2D
    # =====================
    if n_dims == 2:

        plt.figure(figsize=(8, 6))

        for idx, cname in enumerate(cluster_names):

            points = clusters[cname]

            if not points:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            if cname == "Cluster 0":
                color = "white"
            else:
                color = colors(idx % 10)

            plt.scatter(
                xs,
                ys,
                s=10,
                alpha=0.7,
                color=color
            )

        plt.grid(True, alpha=0.3)

    # =====================
    # 3D
    # =====================
    elif n_dims == 3:

        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        for idx, cname in enumerate(cluster_names):

            points = clusters[cname]

            if not points:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            zs = [p[2] for p in points]

            if cname == "Cluster 0":
                color = "black"
            else:
                color = colors(idx % 10)

            ax.scatter(
                xs,
                ys,
                zs,
                s=10,
                alpha=0.7,
                color=color
            )

    # =====================
    # >3D
    # =====================
    else:

        print(
            f"警告：数据为 {n_dims} 维，仅显示前三维。"
        )

        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        for idx, cname in enumerate(cluster_names):

            points = clusters[cname]

            if not points:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            zs = [
                p[2]
                if len(p) > 2
                else 0
                for p in points
            ]

            if cname == "Cluster 0":
                color = "black"
            else:
                color = colors(idx % 10)

            ax.scatter(
                xs,
                ys,
                zs,
                s=10,
                alpha=0.7,
                color=color
            )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()