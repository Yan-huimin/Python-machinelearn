/**
 * @file main.cpp
 * @brief 逻辑回归（Logistic Regression）的 C++ 从零实现
 *
 * 本文件实现了二分类逻辑回归模型，包含：
 * - Sigmoid 激活函数
 * - 梯度下降训练（批量梯度下降）
 * - 二分类交叉熵损失
 * - 概率预测与类别预测
 *
 * 在 main() 中以简单一维特征数据演示训练与预测流程。
 */

#include <iostream>   // 标准输入输出流：cout, endl
#include <vector>     // 动态数组容器：vector
#include <cmath>      // 数学函数库：exp, log
#include <algorithm>  // 算法库：max

using namespace std;

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief Sigmoid / Logistic 激活函数，将实数映射到 (0, 1) 区间
 *
 * 公式: σ(z) = 1 / (1 + e^(-z))
 *
 * @param z 输入值（可以是线性组合的结果）
 * @return double 输出概率值，范围 (0, 1)
 */
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

/**
 * @brief 两个向量的点积（内积）
 *
 * 公式: a · b = Σ(a[i] * b[i])
 *
 * @param a 第一个向量
 * @param b 第二个向量（长度需与 a 相同）
 * @return double 点积结果
 */
double dot(const vector<double>& a, const vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    return s;
}

// ============================================================================
// LogisticRegression 类
// ============================================================================

/**
 * @class LogisticRegression
 * @brief 二分类逻辑回归模型
 *
 * 使用批量梯度下降（Batch Gradient Descent）优化二分类交叉熵损失函数。
 * 模型形式: P(y=1|x) = σ(w0 + w1*x1 + w2*x2 + ...)
 * 其中 w0 为偏置项（bias），对应特征矩阵添加的常数列 1。
 */
class LogisticRegression {
private:
    vector<double> weights;  ///< 模型权重（含偏置项 w0），weights[0] 为偏置
    double lr;               ///< 学习率（Learning Rate），控制每步更新的步长
    int epochs;              ///< 最大迭代次数

    /**
     * @brief 为特征矩阵添加偏置列（全 1 列）
     *
     * 将 X (n×d) 扩展为 X_b (n×(d+1))，在第 0 列位置插入全 1，
     * 使模型能够学习偏置项 w0。
     *
     * @param X 原始特征矩阵，大小为 n(样本数) × d(特征数)
     * @return vector<vector<double>> 扩展后的矩阵，大小为 n × (d+1)
     */
    vector<vector<double>> addOnes(const vector<vector<double>>& X) {
        size_t n = X.size(), d = X[0].size();               // n: 样本数, d: 特征维度
        vector<vector<double>> X_ones(n, vector<double>(d + 1, 1.0));  // 初始化为全 1
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j)
                X_ones[i][j + 1] = X[i][j];                 // 原特征从第 1 列开始填入
        return X_ones;
    }

public:
    /**
     * @brief 构造函数，初始化学习率和迭代次数
     *
     * @param learning_rate 学习率，默认 0.1
     * @param iters 最大迭代次数，默认 1000
     */
    LogisticRegression(double learning_rate = 0.1, int iters = 1000)
        : lr(learning_rate), epochs(iters) {}

    /**
     * @brief 使用训练数据拟合逻辑回归模型
     *
     * 采用批量梯度下降法（BGD）最小化二分类交叉熵损失：
     *   L = -(1/n) * Σ[y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i)]
     *
     * 每 iter % 100 == 0 时输出当前损失值，用于监控训练过程。
     *
     * @param X 训练特征矩阵，每行一个样本
     * @param y 训练标签向量，取值 0 或 1
     */
    void fit(const vector<vector<double>>& X, const vector<double>& y) {
        auto X_b = addOnes(X);                               // 添加偏置列
        size_t n = X_b.size(), m = X_b[0].size();            // n: 样本数, m: 特征数+1
        weights.assign(m, 0.0);                              // 权重初始化为 0

        for (int iter = 0; iter < epochs; ++iter) {
            // ----------------------------------------------------------------
            // 前向传播：计算所有样本的预测概率 ŷ = σ(X_b · w)
            // ----------------------------------------------------------------
            vector<double> preds(n);                         // 存储 n 个样本的预测值
            for (size_t i = 0; i < n; ++i)
                preds[i] = sigmoid(dot(X_b[i], weights));    // ŷ_i = σ(w · x_i)

            // ----------------------------------------------------------------
            // 反向传播：计算梯度 ∇L = (1/n) * X_b^T · (preds - y)
            // ----------------------------------------------------------------
            vector<double> grad(m, 0.0);                     // 梯度向量，长度 = 特征数+1
            for (size_t i = 0; i < n; ++i) {
                double error = preds[i] - y[i];              // 预测误差
                for (size_t j = 0; j < m; ++j)
                    grad[j] += error * X_b[i][j];            // 累积每个权重的梯度分量
            }
            for (size_t j = 0; j < m; ++j) {
                grad[j] /= n;                                // 取平均（批量梯度下降）
                weights[j] -= lr * grad[j];                  // 梯度下降更新: w = w - lr * ∇L
            }

            // ----------------------------------------------------------------
            // 定期打印损失值，用于监控训练收敛情况
            // ----------------------------------------------------------------
            if (iter % 100 == 0) {
                double loss = 0.0;                           // 累计交叉熵损失
                for (size_t i = 0; i < n; ++i) {
                    double p = preds[i];
                    double eps = 1e-15;                      // 极小值，防止 log(0) 导致数值溢出
                    // 二分类交叉熵: -y·log(p) - (1-y)·log(1-p)
                    loss += -y[i] * log(max(p, eps))
                            - (1 - y[i]) * log(max(1 - p, eps));
                }
                loss /= n;                                   // 平均损失
                cout << "Iter " << iter << ", Loss: " << loss << endl;
            }
        }
    }

    /**
     * @brief 预测单个样本属于正类的概率
     *
     * @param x 单个样本的特征向量（不含偏置项）
     * @return double 预测概率 P(y=1|x)，范围 (0, 1)
     */
    double predict_prob(const vector<double>& x) const {
        vector<double> xb = {1.0};                           // 在第 0 位插入偏置项 1
        xb.insert(xb.end(), x.begin(), x.end());             // 追加原始特征
        return sigmoid(dot(xb, weights));                    // ŷ = σ(w · x_b)
    }

    /**
     * @brief 预测单个样本的类别标签（0 或 1）
     *
     * 当预测概率 >= 阈值时判为正类（1），否则判为负类（0）。
     *
     * @param x 单个样本的特征向量（不含偏置项）
     * @param threshold 分类阈值，默认 0.5
     * @return int 预测类别，0 或 1
     */
    int predict(const vector<double>& x, double threshold = 0.5) const {
        return predict_prob(x) >= threshold ? 1 : 0;
    }

    /**
     * @brief 获取训练后的模型权重
     *
     * @return vector<double> 权重向量，weights[0] 为偏置项，后续为各特征权重
     */
    vector<double> getWeights() const { return weights; }
};

// ============================================================================
// 主函数：演示逻辑回归的训练与预测
// ============================================================================

int main() {
    // ------------------------------------------------------------------------
    // 构造简单二分类数据集
    // 规则: 当 x1 >= 3 时标签为 1（正类），否则为 0（负类）
    // ------------------------------------------------------------------------
    vector<vector<double>> X = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};  // 5 个样本，1 维特征
    vector<double> y = {0, 0, 1, 1, 1};                               // 对应的二分类标签

    // 创建逻辑回归模型: 学习率 0.3，训练 100000 轮
    LogisticRegression model(0.3, 100000);
    model.fit(X, y);                                                   // 训练模型

    // 输出训练好的权重
    cout << "\nTrained weights: ";
    for (double w : model.getWeights()) cout << w << " ";
    cout << endl;

    // 验证：对新数据点进行预测
    cout << "Predict prob for x=2.5: " << model.predict_prob({2.5}) << endl;  // 应接近 0
    cout << "Predict class for x=2.5: " << model.predict({2.5}) << endl;      // 应为 0
    cout << "Predict class for x=3.5: " << model.predict({3.5}) << endl;      // 应为 1

    return 0;
}
