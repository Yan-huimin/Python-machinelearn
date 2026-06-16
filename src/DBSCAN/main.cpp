#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <deque>
#include <cmath>

struct Point {
    std::vector<double> coordinates;
    int clusterId = -1; // 用于标记所属的簇
    int status = 0;    // 0: 未访问, 1: 已访问
};

// [FIX 1] 用命名常量区分"未处理"与"噪声"，避免共用 -1 导致状态混淆
//         原来：未处理 = -1, 噪声 = -1（共用，逻辑隐含依赖）
//         现在：未处理 = -1, 噪声 = -2（显式区分）
const int UNCLASSIFIED = -1;
const int NOISE       = -2;

void showHelp(char* programName) {
    std::cout << "Usage: " << programName << " [filename]" << " [radius]" << " [minPts]" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  filename      The input data file (e.g., data.txt)" << std::endl;
    std::cout << "  radius        > 0" << std::endl;
    std::cout << "  minPts        > 1" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help    Show this help message and exit" << std::endl;
}

/// @brief 距离计算函数
/// @param a 点a
/// @param b 点b
/// @return 返回点a和点b之间的欧氏距离
double distance(const Point& a, const Point& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.coordinates.size(); i++) {
        sum += (a.coordinates[i] - b.coordinates[i]) * (a.coordinates[i] - b.coordinates[i]);
    }
    return std::sqrt(sum);
}

std::string writeResults(const std::vector<Point>& points, const std::vector<std::vector<size_t>>& clusters) {

    const std::string path = "dbscan_result.txt";
    size_t clusterCount = clusters.size();
    std::ofstream outFile(path);

    // [FIX 3] 使用 NOISE 常量替代裸 -1 判断噪声点
    outFile << "Cluster " << 0 << ":" << std::endl;
    for(const auto& p : points){
        if(p.clusterId == NOISE){
            for(const auto& val : p.coordinates)
                outFile << val << " ";
            outFile << std::endl;
        }
    }

    // 写入一般点
    for(size_t i = 0; i < clusterCount; i++){
        outFile << "Cluster " << i + 1 << ":" << std::endl;
        for(size_t idx : clusters[i]){
            const auto& point = points[idx];
            for(const auto& val : point.coordinates){
                outFile << val << " ";
            }
            outFile << std::endl;
        }
    }

    outFile.close();

    return path;
}

std::vector<Point> readPoints(const std::string& filename){
    std::vector<Point> points;

    std::ifstream file(filename);
    std::string line;
    while(std::getline(file, line)){
        if(line.empty()) continue; // 跳过空行
        std::istringstream iss(line);
        // [FIX 2] 使用 UNCLASSIFIED 替代裸 -1
        Point p;    p.clusterId = UNCLASSIFIED; p.status = 0; // 初始化簇ID和状态
        double value;
        while(iss >> value){
            p.coordinates.push_back(value);
        }
        points.push_back(std::move(p));
    }
    return points;
}

std::vector<size_t> regionQuery(const std::vector<Point>& points, size_t index, double radius){
    std::vector<size_t> neighbors;
    const auto& point = points[index];

    for(size_t i = 0; i < points.size(); i++){
        if(distance(point, points[i]) <= radius){
            neighbors.push_back(i);
        }
    }

    return neighbors;
}

// [FIX 5a] currentId 类型从 size_t 改为 int，与 Point::clusterId 类型一致，避免溢出
void expandCluster(std::vector<Point>& points, size_t index,
                   std::vector<size_t>& newCluster,
                   int currentId, std::deque<size_t>& seeds,
                   double radius, int minPts)
{
    newCluster.push_back(index); // 将核心点加入簇
    points[index].clusterId = currentId; // 将核心点分配到当前簇

    while(!seeds.empty()){
        size_t currentIndex = seeds.front(); // 获取种子点的索引
        seeds.pop_front();
        Point& p = points[currentIndex];

        if(p.status == 0)   
        {
            p.status = 1; // 标记为已访问
            auto neighborsIndex = regionQuery(points, currentIndex, radius); // 获取邻域内的点索引
            // [FIX 7] 使用 static_cast 消除 signed/unsigned 比较警告
            if(neighborsIndex.size() >= static_cast<size_t>(minPts))
                for(size_t idx : neighborsIndex)
                    if(points[idx].status == 0) // 仅将未访问过的点加入种子集合
                        seeds.push_back(idx);
        }

        // [FIX 5d] 按伪代码第226-227行：最终检查 — 尚未分配到任何簇的点归入当前簇
        //         原来缺少此检查，依赖 UNCLASSIFIED == NOISE == -1 的巧合
        if(p.clusterId == UNCLASSIFIED || p.clusterId == NOISE){
            p.clusterId = currentId;
            newCluster.push_back(currentIndex);
        }
    }
}

void DBSCAN(const std::string& filename, double radius, int minPts){
    std::vector<Point> points = readPoints(filename);
    std::vector<std::vector<size_t>> clusters; // 用于存储簇的点索引

    int clusterId = 0;
    // DBSCAN算法的实现
    for(size_t i = 0; i < points.size(); i++){
        if(points[i].status) continue; // 已访问过的点跳过
        points[i].status = 1; // 标记为已访问

        const auto& neighbors = regionQuery(points, i, radius); // 获取邻域内的点
        if(neighbors.size() < static_cast<size_t>(minPts)){
            // [FIX 4] 使用 NOISE 常量替代裸 -1，与 UNCLASSIFIED 显式区分
            points[i].clusterId = NOISE; // 标记为噪声
        }else{
            std::vector<size_t> newCluster; // 用于存储当前簇的点索引
            std::deque<size_t> seeds(neighbors.begin(), neighbors.end()); // 初始化种子集合
            expandCluster(points, i, newCluster, clusterId, seeds, radius, minPts); // 扩展簇
            // [FIX 6] 删除冗余的 clusterId 重赋值循环
            //         原代码：for(size_t idx : newCluster) points[idx].clusterId = clusterId;
            //         expandCluster 内部已逐个设置了 clusterId，无需重复
            clusters.push_back(std::move(newCluster)); // 将当前簇加入簇列表
            clusterId++; // 增加簇ID
        }
    }

    const auto& path = writeResults(points, clusters); // 将结果写入文件或输出
    std::string cmd = "python ../check-kmean.py " + path;
    system(cmd.c_str());
}

int main(int argc, char** argv){
    if(argc < 2 || argc > 4){
        showHelp(argv[0]);
        return (1);
    }

    // [FIX 8] 使用 std::strcmp 而非 ::strcmp，符合 C++ <cstring> 标准
    if(std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0){
        showHelp(argv[0]);
        return (0);
    }

    std::string filename = argv[1];
    double radius = std::stod(argv[2]);
    int minPts =    std::stoi(argv[3]);

    if(radius <= 0 || minPts <= 1){
        showHelp(argv[0]);
        return (1);
    }

    DBSCAN(filename, radius, minPts);


    return (0);
}

/** DBSCAN算法实现 @copyright(https://zh.wikipedia.org/wiki/DBSCAN)
DBSCAN(D, eps, MinPts) {
   C = 0
   for each point P in dataset D {
      if P is visited
         continue next point
      mark P as visited
      NeighborPts = regionQuery(P, eps)
      if sizeof(NeighborPts) < MinPts
         mark P as NOISE
      else {
         C = next cluster
         expandCluster(P, NeighborPts, C, eps, MinPts)
      }
   }
}

expandCluster(P, NeighborPts, C, eps, MinPts) {
   add P to cluster C
   for each point P' in NeighborPts { 
      if P' is not visited {
         mark P' as visited
         NeighborPts' = regionQuery(P', eps)
         if sizeof(NeighborPts') >= MinPts
            NeighborPts = NeighborPts joined with NeighborPts'
      }
      if P' is not yet member of any cluster
         add P' to cluster C
   }
}

regionQuery(P, eps)
   return all points within P's eps-neighborhood (including P)
 */