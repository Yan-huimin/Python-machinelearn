#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>   // for std::iota
#include <algorithm> // for std::random_shuffle
#include <random>    // for std::default_random_engine

#define LIMIT 1e-6
#define SEED 42

/** K-Means算法步骤：
 * 
 * 
 *  1. 选取K个点做为初始聚集的簇心（也可选择非样本点）;
 *  2. 分别计算每个样本点到 K个簇核心的距离（这里的距离一般取欧氏距离或余弦距离），找到离该点最近的簇核心，将它归属到对应的簇；
 *  3. 所有点都归属到簇之后， M个点就分为了 K个簇。之后重新计算每个簇的重心（平均距离中心），将其定为新的“簇核心”；
 *  4. 反复迭代 2 - 3 步骤，直到达到某个中止条件。
 */

/**
 * K-Means++算法步骤：
 * 1. 从数据点中随机选择一个中心。
 * 2. 对于每个数据点x，计算D(x)，即x与已经选择的最接近中心之间的距离。
 * 3. 使用加权概率分布随机选择一个新的数据点作为新的中心，其中选择点 x 的概率与D(x)^2成正比。
 * 4. 重复步骤2和3，直到选择了K个中心。
 */

void showHelp(char * programName)
{
    std::cout << "Usage: " << programName << " [k] [option(plus/normal)]" << " [filename]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help    Show this help message and exit" << std::endl;
}

double distance(const std::vector<double>& a, const std::vector<double>& b){
    double sum = 0.0;
    for(size_t i = 0; i < a.size(); i++){
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

double squareDistance(const std::vector<double>& a, const std::vector<double>& b){
    double sum = 0.0;
    for(size_t i = 0; i < a.size(); i++){
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}

std::vector<std::vector<double>> getData(const std::string& filename){
    std::vector<std::vector<double>> data;

    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        if(line.empty()) continue;
        std::istringstream iss(line);
        std::vector<double> point;
        double value;
        while (iss >> value) {
            point.push_back(value);
        }
        if(!point.empty()) {
            data.push_back(point);
        }
    }

    return data;
}

std::vector<std::vector<double>> generateCentroids(const std::vector<std::vector<double>>& data, int k){
    std::vector<std::vector<double>> centroids;
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(std::random_device{}()));

    for(int i = 0; i < k; i++){
        centroids.push_back(data[indices[i]]);
    }

    return centroids;
}


std::vector<double> randomCenterPoint(const std::vector<std::vector<double>>& data, std::vector<std::vector<double>>& rpt, std::mt19937& rng){
    size_t n = data.size(); // 点的个数

    if(rpt.size() == 0){
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        return data[static_cast<size_t>(dist(rng))];
    }

    std::vector<double> min_dist_sq(n, std::numeric_limits<double>::max());
    for(size_t i = 0; i < n; i++){
        for(size_t j = 0; j < rpt.size(); j++){
            double d2 = squareDistance(data[i], rpt[j]);
            min_dist_sq[i] = std::min(d2, min_dist_sq[i]);
        }
    }

    std::discrete_distribution<size_t> dist(min_dist_sq.begin(), min_dist_sq.end());

    return data[static_cast<size_t>(dist(rng))];
}

std::vector<std::vector<double>> generateCentroidsPulsPlus(const std::vector<std::vector<double>>& data, int k){
    std::mt19937 rng(SEED);
    std::vector<std::vector<double>> res;
    res.reserve(k); // 中心点动态生成 预分配内存
    for(int i = 0; i < k; i++){
        auto ptc = randomCenterPoint(data, res, rng);
        res.push_back(ptc);
    }

    return res;
}

std::string writeClusters(const std::vector<std::vector<std::vector<double>>>& clustersRes, int k, const std::string& option){
    const std::string& filename = "..\\..\\data\\clusters_" + option + std::to_string(k) + ".txt";

    std::ofstream outFile(filename);
    for(size_t i = 0; i < clustersRes.size(); i++){
        outFile << "Cluster " << i + 1 << ":" << std::endl;
        for(const auto& point : clustersRes[i]){
            for(const auto& value : point){
                outFile << value << " ";
            }
            outFile << std::endl;
        }
    }
    outFile.close();

    return filename;
}

void KMeans(int k, const std::string& filename = "data.txt", const std::string& option = "plus")
{
    std::cout << "KMeans with k = " << k << std::endl;

    std::vector<std::vector<double>> data = getData(filename);

    // 初始化中心点
    // auto centroids = generateCentroids(data, k);
    // auto centroids = generateCentroidsPulsPlus(data, k);

    auto centroids = option == "plus" ? generateCentroidsPulsPlus(data, k) : generateCentroids(data, k);
    auto oldCentroids = centroids;

    std::vector<std::vector<std::vector<double>>> resultClusters;

    while(true){
        std::vector<std::vector<std::vector<double>>> clusters(k);

        // 分配簇
        for(const auto& point : data){
            double minDist = std::numeric_limits<double>::max();
            int clusterIndex = 0;
            for(int i = 0; i < k; i++){
                double dist = distance(point, centroids[i]);
                if(dist < minDist){
                    minDist = dist;
                    clusterIndex = i;
                }
            }
            clusters[clusterIndex].push_back(point);
        }

        // 更新中心点
        for(int i = 0; i < k; i++){
            std::vector<double> newCentroid(centroids[i].size(), 0.0);
            for(const auto& point : clusters[i]){
                for(size_t j = 0; j < point.size(); j++){
                    newCentroid[j] += point[j];
                }
            }
            for(size_t j = 0; j < newCentroid.size(); j++){
                newCentroid[j] /= clusters[i].size();
            }
            centroids[i] = newCentroid;
        }

        // 检查收敛
        double maxShift = 0.0;
        for(int i = 0; i < k; i++)  maxShift = std::max(maxShift, distance(oldCentroids[i], centroids[i]));

        if(maxShift < LIMIT){
            resultClusters = std::move(clusters);
            break;
        }

        oldCentroids = centroids;
    }

    // 输出结果
    const std::string path = writeClusters(resultClusters, k, option);
    std::string cmd = "python ../check-kmean.py " + path;
    system(cmd.c_str()); 
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        showHelp(argv[0]);
        return (1);
    }

    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
    {
        showHelp(argv[0]);
        return (0);
    }

    std::string option = argv[2];
    int k = std::stoi(argv[1]);

    if(option != "plus" && option != "normal"){
        showHelp(argv[0]);
        return (1);
    }

    std::string filename = argv[3];
    KMeans(k, filename, option);

    return (0);
}