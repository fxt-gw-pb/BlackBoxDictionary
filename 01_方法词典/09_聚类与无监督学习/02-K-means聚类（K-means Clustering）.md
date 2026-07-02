---
title: K-means聚类
english_name: K-means Clustering
slug: k-means-clustering
aliases: [K-means, k-means clustering, K均值聚类, "K-means聚类（K-means Clustering）"]
category: 聚类与无监督学习
subcategory: 原型聚类
tags: [医学统计, 数据科学, 聚类, 无监督学习, 原型方法]
status: 已建
difficulty: basic
question_type: 硬聚类与患者亚群探索
data_type: [表格数据, 高维特征矩阵]
outcome_type: [无监督分群]
python_packages: [scikit-learn]
r_packages: [stats, factoextra]
---

# K-means聚类（K-means Clustering）

## 1. 方法概览

### 1.1 定义

K-means 聚类是一种基于质心的硬聚类方法。它把样本划分为预先指定的 $K$ 个簇，使每个样本尽量接近自己所属簇的中心。

### 1.2 它主要解决什么问题

- 研究问题：在没有标签的情况下，样本能否按数值特征被划分为若干相对紧凑的亚群。
- 适用任务：患者分群、原型发现、样本压缩、聚类结果作为后续描述分析的分组变量。
- 常见医学场景：基于代谢指标识别代谢表型，按多项实验室指标探索患者亚群，基于影像组学特征进行初步分型。

### 1.3 直觉理解

K-means 会先放置 $K$ 个中心点，然后反复做两件事：把每个样本分给最近的中心，再用每个簇内样本的平均位置更新中心。迭代稳定后，同一簇内样本通常彼此更相似。

## 2. 数学形式

### 2.1 核心公式

设样本为 $x_i\in\mathbb{R}^p$，簇标签为 $c_i\in\{1,\dots,K\}$，第 $k$ 个簇的质心为 $\mu_k$。K-means 最小化簇内平方和：

$$
\min_{c_1,\dots,c_n,\mu_1,\dots,\mu_K}
\sum_{i=1}^{n}\left\|x_i-\mu_{c_i}\right\|_2^2
$$

给定标签时，质心更新为：

$$
\mu_k=\frac{1}{|C_k|}\sum_{i:c_i=k}x_i
$$

给定质心时，标签更新为：

$$
c_i=\arg\min_k\|x_i-\mu_k\|_2^2
$$

### 2.2 参数或统计量含义

- $K$：预先指定的簇数。
- $\mu_k$：第 $k$ 个簇的中心。
- inertia / within-cluster sum of squares：簇内平方和，越小表示簇内越紧凑。
- `n_init`：不同随机初始化重复运行次数。
- `max_iter`：每次初始化允许的最大迭代次数。

### 2.3 关键假设

- 簇大致呈球形或凸形。
- 各变量尺度可比，通常需要标准化。
- 预先指定的 $K$ 接近真实或有解释意义的分群数量。
- 离群点不会严重扭曲质心。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续型数值特征。
- 因变量类型：无监督方法，不需要结局变量。
- 数据结构：每行一个样本，每列一个特征。
- 是否适合高维数据：可用，但高维时距离可能变得不稳定，常先降维或筛选特征。
- 是否适合缺失较多数据：不适合直接处理，需先插补或剔除缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：需先构造个体级特征，普通 K-means 不建模相关结构。

### 3.2 示例表格

以代谢表型聚类为例：

| BMI | TG | HDL | FastingGlucose | SBP |
| --- | --- | --- | --- | --- |
| 31.2 | 2.4 | 0.9 | 7.8 | 146 |
| 24.8 | 1.1 | 1.5 | 5.2 | 122 |
| 29.5 | 2.0 | 1.0 | 6.9 | 138 |
| 22.9 | 0.9 | 1.7 | 4.9 | 116 |

### 3.3 输入与产出

#### 输入

- 输入数据：数值型特征矩阵。
- 关键变量：簇数 $K$、初始化方法、重复初始化次数。
- 需要预处理的内容：缺失处理、标准化、异常值检查、变量选择。

#### 产出

- 模型对象/统计结果：簇标签、簇中心、簇内平方和。
- 参数估计：每个簇的质心。
- 预测结果：新样本可分配到最近质心。
- 不确定性指标：不同随机种子或重采样下的聚类稳定性，轮廓系数。

## 4. 适用场景

- 适合：希望快速得到互斥分组、簇较紧凑且形状接近球形、样本量较大的场景。
- 不适合：簇形状弯曲、密度差异明显、噪声点很多、类别变量为主的场景。
- 使用前需要特别检查的点：变量标准化、$K$ 的选择、离群值、聚类稳定性和临床可解释性。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("metabolic_profiles.csv")
X = df[["BMI", "TG", "HDL", "FastingGlucose", "SBP"]]

model = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=3, n_init=20, random_state=42)
)
cluster = model.fit_predict(X)

score = silhouette_score(model.named_steps["standardscaler"].transform(X), cluster)
print("Silhouette:", score)
print(pd.Series(cluster).value_counts().sort_index())
```

### 5.2 R

常用包：

- `stats`

```r
x <- scale(df[, c("BMI", "TG", "HDL", "FastingGlucose", "SBP")])
fit <- kmeans(x, centers = 3, nstart = 20)

cluster <- fit$cluster
fit$centers
fit$tot.withinss
```

## 6. 结果如何解释

- 核心结果看什么：簇中心、各簇样本量、簇内紧凑度、不同簇的临床特征差异。
- 每个主要参数如何解释：`n_clusters=3` 表示强制把样本划分成 3 个互斥亚群。
- 临床或医学意义如何表达：应先描述每个簇的特征模式，再谨慎讨论是否对应潜在疾病表型。
- 常见误读：K-means 总会给出分组，但分组不一定有真实生物学意义。

## 7. 推荐可视化

- 肘部法曲线。
- 轮廓系数随 $K$ 变化图。
- PCA/UMAP 二维空间中的聚类散点图。
- 各簇特征均值热图或雷达图。

## 8. 优势、局限与常见坑

### 优势

- 简单、快速、可扩展。
- 簇中心易于解释为典型原型。
- 适合做聚类分析的基线方法。

### 局限

- 需要预先指定 $K$。
- 对异常值和初始质心敏感。
- 难以识别非球形簇和不同密度簇。

### 常见坑

- 未标准化变量就直接聚类。
- 只凭肘部法机械选择 $K$。
- 把聚类标签当成真实诊断标签使用。

## 9. 与相近方法的区别

- 和 [[Mini-Batch K-means聚类（Mini-Batch K-means Clustering）]] 的区别：Mini-Batch K-means 用小批量样本更新质心，更适合大数据。
- 和 [[高斯混合模型（Gaussian Mixture Model, GMM）]] 的区别：K-means 是硬聚类且偏球形簇，GMM 是概率软聚类且可拟合椭圆簇。
- 和 [[模糊C均值聚类（Fuzzy C-Means Clustering, FCM）]] 的区别：K-means 给硬标签，FCM 给每个样本属于各簇的隶属度。

## 10. 医学研究中的典型应用

- 代谢综合征相关指标的患者亚群探索。
- 影像组学特征的初步无监督分型。
- 多个连续实验室指标的临床表型聚类。

## 11. 相关方法

- [[Mini-Batch K-means聚类（Mini-Batch K-means Clustering）]]
- [[高斯混合模型（Gaussian Mixture Model, GMM）]]
- [[模糊C均值聚类（Fuzzy C-Means Clustering, FCM）]]
- [[主成分分析（Principal Component Analysis, PCA）]]

## 12. 参考资料

- MacQueen J. Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*. 1967;1:281-297.
- Lloyd S. Least squares quantization in PCM. *IEEE Transactions on Information Theory*. 1982;28(2):129-137.
- scikit-learn Developers. `sklearn.cluster.KMeans`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) （访问日期：2026-07-02）
