---
title: 模糊C均值聚类
english_name: Fuzzy C-Means Clustering, FCM
slug: fuzzy-c-means-clustering-fcm
aliases: [FCM, fuzzy c-means, fuzzy c-means clustering, 模糊C均值, "模糊C均值聚类（Fuzzy C-Means Clustering, FCM）"]
category: 聚类与无监督学习
subcategory: 软聚类
tags: [医学统计, 数据科学, 聚类, 无监督学习, 软聚类]
status: 已建
difficulty: intermediate
question_type: 模糊隶属度分群与边界样本识别
data_type: [表格数据, 高维特征矩阵]
outcome_type: [无监督分群, 隶属度]
python_packages: [scikit-fuzzy]
r_packages: [e1071, ppclust]
---

# 模糊C均值聚类（Fuzzy C-Means Clustering, FCM）

## 1. 方法概览

### 1.1 定义

模糊 C 均值聚类是一种软聚类方法。它不是把每个样本硬分配到单一簇，而是给出样本属于每个簇的隶属度。

### 1.2 它主要解决什么问题

- 研究问题：当样本可能处在多个亚群之间的过渡状态时，如何表达不确定归属。
- 适用任务：软分群、边界样本识别、连续表型过渡分析。
- 常见医学场景：疾病谱系中介状态识别，患者亚型不确定性刻画，影像区域软分割。

### 1.3 直觉理解

K-means 会说“这个患者属于第 1 类”。FCM 会说“这个患者 70% 像第 1 类，25% 像第 2 类，5% 像第 3 类”。这对临床连续谱或混合表型尤其有用。

## 2. 数学形式

### 2.1 核心公式

设 $u_{ik}$ 为样本 $i$ 属于簇 $k$ 的隶属度，满足：

$$
\sum_{k=1}^{C}u_{ik}=1,\quad 0\le u_{ik}\le 1
$$

FCM 最小化目标函数：

$$
J_m=\sum_{i=1}^{n}\sum_{k=1}^{C}u_{ik}^m\|x_i-c_k\|^2
$$

其中 $m>1$ 为模糊系数。簇中心更新为：

$$
c_k=\frac{\sum_{i=1}^{n}u_{ik}^m x_i}{\sum_{i=1}^{n}u_{ik}^m}
$$

隶属度更新为：

$$
u_{ik}=
\frac{1}
{\sum_{j=1}^{C}
\left(\frac{\|x_i-c_k\|}{\|x_i-c_j\|}\right)^{2/(m-1)}}
$$

### 2.2 参数或统计量含义

- $C$：簇数。
- $u_{ik}$：样本 $i$ 对簇 $k$ 的隶属度。
- $m$：模糊系数，越大隶属度越平滑。
- $c_k$：簇中心。
- partition coefficient：隶属度清晰程度指标。

### 2.3 关键假设

- 簇可由中心原型表达。
- 样本可能存在混合归属或过渡状态。
- 距离越近，隶属度越高。
- 簇数和模糊系数选择合理。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续型数值特征。
- 因变量类型：无监督方法，不需要结局变量。
- 数据结构：样本乘以特征矩阵。
- 是否适合高维数据：可用，但需注意距离稳定性，常先降维或筛选特征。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：需先构造样本级特征。

### 3.2 示例表格

以炎症-代谢混合表型为例：

| CRP | IL6 | BMI | TG | HDL |
| --- | --- | --- | --- | --- |
| 5.2 | 8.1 | 31.2 | 2.4 | 0.9 |
| 1.1 | 2.0 | 24.8 | 1.1 | 1.5 |
| 3.8 | 5.5 | 29.5 | 2.0 | 1.0 |
| 0.8 | 1.5 | 22.9 | 0.9 | 1.7 |

### 3.3 输入与产出

#### 输入

- 输入数据：连续型特征矩阵。
- 关键变量：簇数、模糊系数、停止阈值、最大迭代次数。
- 需要预处理的内容：缺失处理、标准化、异常值检查、变量选择。

#### 产出

- 模型对象/统计结果：隶属度矩阵、硬标签、簇中心、目标函数收敛曲线。
- 参数估计：簇中心和隶属度。
- 预测结果：新样本可计算对各簇的隶属度。
- 不确定性指标：最大隶属度、隶属度熵、不同初始化稳定性。

## 4. 适用场景

- 适合：亚群边界模糊、存在过渡样本、希望量化归属不确定性的场景。
- 不适合：需要清晰互斥分组、簇形状复杂、离群点很多或类别变量为主的场景。
- 使用前需要特别检查的点：模糊系数、簇数、低最大隶属度样本的临床含义。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-fuzzy`

```python
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("inflammation_metabolic_profiles.csv")
X = df[["CRP", "IL6", "BMI", "TG", "HDL"]]
X_scaled = StandardScaler().fit_transform(X).T

centers, membership, _, _, objective, _, fpc = fuzz.cluster.cmeans(
    X_scaled,
    c=3,
    m=2.0,
    error=0.005,
    maxiter=1000,
    seed=42
)

cluster = np.argmax(membership, axis=0)
print("Fuzzy partition coefficient:", fpc)
print(pd.Series(cluster).value_counts().sort_index())
```

### 5.2 R

常用包：

- `e1071`

```r
library(e1071)

x <- scale(df[, c("CRP", "IL6", "BMI", "TG", "HDL")])
fit <- cmeans(x, centers = 3, m = 2, iter.max = 100)

cluster <- fit$cluster
membership <- fit$membership
head(membership)
```

## 6. 结果如何解释

- 核心结果看什么：隶属度矩阵、最大隶属度分布、簇中心、低确定性样本。
- 每个主要参数如何解释：$m=2$ 是常用模糊系数，增大 $m$ 会让样本对各簇的隶属度更平均。
- 临床或医学意义如何表达：可描述患者对不同表型的隶属程度，而不是只给单一亚型。
- 常见误读：隶属度不是严格后验概率，不能直接当作概率风险解释。

## 7. 推荐可视化

- 隶属度热图。
- 最大隶属度或隶属度熵分布图。
- PCA/UMAP 空间中按硬标签着色、按不确定性调透明度。
- 各簇中心特征热图。

## 8. 优势、局限与常见坑

### 优势

- 能表达边界样本和过渡状态。
- 输出比硬聚类更丰富。
- 适合疾病谱系和混合表型分析。

### 局限

- 仍需预设簇数。
- 对初始化、距离尺度和离群点敏感。
- 隶属度解释需要谨慎。

### 常见坑

- 把隶属度当作真实概率。
- 不检查低最大隶属度样本。
- 用硬标签丢掉软聚类最有价值的信息。

## 9. 与相近方法的区别

- 和 [[K-means聚类（K-means Clustering）]] 的区别：K-means 给硬标签，FCM 给每个样本对各簇的隶属度。
- 和 [[高斯混合模型（Gaussian Mixture Model, GMM）]] 的区别：GMM 的软归属来自概率分布模型，FCM 的隶属度来自距离和模糊目标函数。
- 和 [[期望最大化算法（Expectation-Maximization Algorithm, EM）]] 的区别：EM 是估计含隐变量模型的通用算法，FCM 是一种模糊原型聚类方法。

## 10. 医学研究中的典型应用

- 疾病亚型之间过渡患者的识别。
- 影像区域软分割。
- 炎症、代谢、免疫等连续谱表型的模糊归属分析。

## 11. 相关方法

- [[K-means聚类（K-means Clustering）]]
- [[高斯混合模型（Gaussian Mixture Model, GMM）]]
- [[期望最大化算法（Expectation-Maximization Algorithm, EM）]]
- [[主成分分析（Principal Component Analysis, PCA）]]

## 12. 参考资料

- Bezdek JC. *Pattern Recognition with Fuzzy Objective Function Algorithms*. Springer; 1981.
- Dunn JC. A fuzzy relative of the ISODATA process and its use in detecting compact well-separated clusters. *Journal of Cybernetics*. 1973;3(3):32-57.
- Ross TJ. *Fuzzy Logic with Engineering Applications*. 4th ed. Wiley; 2016.
