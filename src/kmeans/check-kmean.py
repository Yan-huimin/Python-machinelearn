#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于 3D 投影
import re
import warnings

def parse_clusters(filepath):
    """
    解析格式为：
    Cluster 1:
    -7.20328 6.58626
    -7.53198 7.8988
    ...
    返回 (clusters_dict, n_dims)：
      - clusters_dict: {簇名: [(坐标元组), ...]}
      - n_dims: 数据维度 (2 或 3)
    """
    clusters = {}
    current_label = None
    n_dims = None
    cluster_pattern = re.compile(r'^Cluster\s+(\d+):\s*$')

    with open(filepath, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            match = cluster_pattern.match(line)
            if match:
                current_label = f'Cluster {match.group(1)}'
                clusters.setdefault(current_label, [])
                continue

            # 解析浮点数行
            parts = line.split()
            try:
                coords = [float(p) for p in parts]
            except ValueError:
                continue  # 忽略无效行

            if n_dims is None and coords:
                n_dims = len(coords)   # 以第一个有效点的维度为准
            if n_dims is not None and len(coords) != n_dims:
                warnings.warn(f"忽略维度不一致的点: {coords} (期望{n_dims}维)")
                continue

            if current_label is not None:
                clusters[current_label].append(tuple(coords))

    return clusters, n_dims

def main():
    if len(sys.argv) < 2:
        print("用法: python check.py <簇文件路径>")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        clusters, n_dims = parse_clusters(filepath)
    except Exception as e:
        print(f"读取文件失败: {e}")
        sys.exit(1)

    if not clusters or n_dims is None:
        print("未找到有效的簇数据")
        sys.exit(1)

    total_points = sum(len(points) for points in clusters.values())
    cluster_names = sorted(clusters.keys())
    print(f"读取到 {len(cluster_names)} 个簇，共计 {total_points} 个点，维度：{n_dims}D")

    # 根据维度创建图形
    if n_dims == 2:
        plt.figure(figsize=(8, 6))
        colors = plt.cm.tab10
        for idx, cname in enumerate(cluster_names):
            points = clusters[cname]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.scatter(xs, ys, s=10, alpha=0.7, color=colors(idx % 10), label=cname)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
    elif n_dims == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.tab10
        for idx, cname in enumerate(cluster_names):
            points = clusters[cname]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            zs = [p[2] for p in points]
            ax.scatter(xs, ys, zs, s=10, alpha=0.7, color=colors(idx % 10), label=cname)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
    else:
        # 维度 >3：降维显示前三维，并提示
        print(f"警告：数据为{n_dims}维，仅显示前三维。")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.tab10
        for idx, cname in enumerate(cluster_names):
            points = clusters[cname]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            zs = [p[2] if len(p) > 2 else 0 for p in points]  # 保证第三维存在
            ax.scatter(xs, ys, zs, s=10, alpha=0.7, color=colors(idx % 10), label=cname)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")

    plt.title(f"Cluster Visualization from {filepath}\n"
              f"({total_points} points in {len(cluster_names)} clusters, {n_dims}D)")
    plt.grid(True, alpha=0.3)
    plt.legend(markerscale=2, fontsize='small', loc='best')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()