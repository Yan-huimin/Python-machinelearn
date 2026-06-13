#!/usr/bin/env python
import sys
import matplotlib.pyplot as plt
import re

def parse_clusters(filepath):
    """
    解析格式为：
    Cluster 1:
    -7.20328 6.58626 
    -7.53198 7.8988 
    ...
    Cluster 2:
    5.6469 -8.62495 
    ...
    的文件，返回字典 {簇名: [(x1,y1), (x2,y2), ...]}。
    """
    clusters = {}
    current_label = None
    cluster_pattern = re.compile(r'^Cluster\s+(\d+):\s*$')  # 匹配 "Cluster 1:" 等

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

            # 否则尝试解析为两个浮点数
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    if current_label is not None:
                        clusters[current_label].append((x, y))
                except ValueError:
                    # 忽略无法解析的行（比如可能是其他文本）
                    pass

    return clusters

def main():
    if len(sys.argv) < 2:
        print("用法: python check.py <簇文件路径>")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        clusters = parse_clusters(filepath)
    except Exception as e:
        print(f"读取文件失败: {e}")
        sys.exit(1)

    if not clusters:
        print("未找到有效的簇数据")
        sys.exit(1)

    # 统计总样本数
    total_points = sum(len(points) for points in clusters.values())
    cluster_names = sorted(clusters.keys())  # 按簇编号排序
    print(f"读取到 {len(cluster_names)} 个簇，共计 {total_points} 个点")

    # 创建绘图
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10  # 使用颜色映射，确保不同颜色

    for idx, cname in enumerate(cluster_names):
        points = clusters[cname]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = colors(idx % 10)  # 最多10种颜色循环
        plt.scatter(xs, ys, s=10, alpha=0.7, color=color, label=cname)

    plt.title(f"Cluster Visualization from {filepath}\n({total_points} points in {len(cluster_names)} clusters)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, alpha=0.3)
    plt.legend(markerscale=2, fontsize='small', loc='best')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()