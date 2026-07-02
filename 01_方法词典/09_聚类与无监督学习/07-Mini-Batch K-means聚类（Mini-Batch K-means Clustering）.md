---
title: Mini-Batch K-means聚类
english_name: Mini-Batch K-means Clustering
slug: mini-batch-k-means-clustering
aliases: [Mini-Batch K-means, mini-batch k-means clustering, 小批量K均值聚类, "Mini-Batch K-means聚类（Mini-Batch K-means Clustering）"]
category: 聚类与无监督学习
subcategory: 原型聚类
tags: [医学统计, 数据科学, 聚类, 无监督学习, 大规模学习]
status: 已建
difficulty: basic
question_type: 大规模硬聚类与快速质心学习
data_type: [表格数据, 高维特征矩阵, 图像像素矩阵]
outcome_type: [无监督分群]
python_packages: [scikit-learn]
r_packages: [ClusterR]
---

# Mini-Batch K-means聚类（Mini-Batch K-means Clustering）

## 1. 方法概览

### 1.1 定义

Mini-Batch K-means 是 [[K-means聚类（K-means Clustering）]] 的大规模近似版本。它每次只用一小批样本更新质心，从而显著降低计算成本。

### 1.2 它主要解决什么问题

- 研究问题：样本量很大时，如何更快地近似 K-means 聚类结果。
- 适用任务：大规模患者分群、图像颜色量化、在线或近在线聚类、快速原型学习。
- 常见医学场景：大规模影像像素压缩，海量监测记录分组，大样本 EHR 表型聚类初筛。

### 1.3 直觉理解

标准 K-means 每一轮都看完整数据，Mini-Batch K-means 则每次抽一小批数据来微调质心。它牺牲一点精确度，换取更快速度和更低内存占用。

## 2. 数学形式

### 2.1 核心公式

标准 K-means 目标仍是最小化：

$$
\sum_{i=1}^{n}\|x_i-\mu_{c_i}\|_2^2
$$

Mini-batch 版本在第 $t$ 次迭代抽取小批量 $B_t$，对 $x_i\in B_t$ 分配最近质心：

$$
c_i=\arg\min_k\|x_i-\mu_k^{(t)}\|_2^2
$$

若样本 $x_i$ 分给簇 $k$，质心可用递推均值更新：

$$
\mu_k^{(t+1)}=(1-\eta_{k,t})\mu_k^{(t)}+\eta_{k,t}x_i
$$

其中 $\eta_{k,t}$ 通常与该簇已经接收的样本数有关。

### 2.2 参数或统计量含义

- `n_clusters`：簇数。
- `batch_size`：每次用于更新的小批量样本数。
- `n_init`：不同初始化重复次数。
- `max_iter`：最大迭代轮数。
- inertia：近似簇内平方和。
- reassignment ratio：控制低计数簇中心是否被重新分配。

### 2.3 关键假设

- 与 K-means 相同，簇大致紧凑且接近球形。
- 小批量样本能代表整体数据分布。
- 随机更新带来的近似误差在任务中可接受。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续型数值特征或像素/向量特征。
- 因变量类型：无监督方法，不需要结局变量。
- 数据结构：大样本特征矩阵。
- 是否适合高维数据：可用于稀疏或高维数据，但需注意距离稳定性。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：需先构造样本级输入。

### 3.2 示例表格

以大规模 EHR 表型特征为例：

| Age | BMI | VisitCount | MedicationCount | LabAbnormalCount |
| --- | --- | --- | --- | --- |
| 71 | 31.2 | 14 | 8 | 5 |
| 54 | 24.8 | 3 | 2 | 1 |
| 67 | 29.5 | 10 | 6 | 4 |
| 42 | 22.9 | 2 | 1 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：大规模数值特征矩阵。
- 关键变量：簇数、batch size、初始化次数。
- 需要预处理的内容：缺失处理、标准化、异常值检查、必要时降维。

#### 产出

- 模型对象/统计结果：簇标签、近似质心、inertia。
- 参数估计：每个簇的质心。
- 预测结果：新样本可分配到最近质心。
- 不确定性指标：不同 batch size、随机种子和初始化下的稳定性。

## 4. 适用场景

- 适合：样本量大、需要快速近似 K-means、可接受少量随机误差的场景。
- 不适合：样本量较小且需要最稳定的 K-means 结果、簇形状复杂、离群点很多的场景。
- 使用前需要特别检查的点：batch size 是否足够、结果是否接近标准 K-means、簇中心是否稳定。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("ehr_features_large.csv")
X = df[["Age", "BMI", "VisitCount", "MedicationCount", "LabAbnormalCount"]]

model = make_pipeline(
    StandardScaler(),
    MiniBatchKMeans(
        n_clusters=4,
        batch_size=512,
        n_init=10,
        random_state=42
    )
)
cluster = model.fit_predict(X)
print(pd.Series(cluster).value_counts().sort_index())
```

### 5.2 R

常用包：

- `ClusterR`

```r
library(ClusterR)

x <- scale(df[, c("Age", "BMI", "VisitCount", "MedicationCount", "LabAbnormalCount")])
fit <- MiniBatchKmeans(x, clusters = 4, batch_size = 512, num_init = 10)

cluster <- predict_MBatchKMeans(x, fit$centroids)
table(cluster)
```

## 6. 结果如何解释

- 核心结果看什么：近似质心、各簇样本量、inertia、与标准 K-means 的一致性。
- 每个主要参数如何解释：`batch_size` 越大，更新越接近标准 K-means，但速度优势降低。
- 临床或医学意义如何表达：更适合作为大规模数据的快速分群工具，聚类解释仍需回到簇中心和临床变量差异。
- 常见误读：速度更快不代表结果更好；它是 K-means 的近似优化策略。

## 7. 推荐可视化

- 不同 batch size 下 inertia 和运行时间对比。
- 聚类标签在 PCA/UMAP 空间中的分布。
- 各簇中心热图。
- Mini-Batch 与标准 K-means 标签一致性矩阵。

## 8. 优势、局限与常见坑

### 优势

- 适合大规模数据。
- 速度快、内存压力小。
- 可作为 K-means 的快速近似替代。

### 局限

- 结果有随机近似误差。
- 仍继承 K-means 对球形簇和离群点敏感的局限。
- 小批量过小可能导致质心不稳定。

### 常见坑

- 为追求速度把 batch size 设得过小。
- 不设置随机种子和重复初始化。
- 用它解决明显非球形或密度不均的聚类问题。

## 9. 与相近方法的区别

- 和 [[K-means聚类（K-means Clustering）]] 的区别：Mini-Batch K-means 用小批量近似更新，更快但略有近似误差。
- 和 [[BIRCH聚类（Balanced Iterative Reducing and Clustering using Hierarchies, BIRCH）]] 的区别：BIRCH 先构建 CF 树压缩数据，Mini-Batch K-means 直接随机小批量更新质心。
- 和 [[高斯混合模型（Gaussian Mixture Model, GMM）]] 的区别：GMM 给概率归属，Mini-Batch K-means 给硬标签和质心。

## 10. 医学研究中的典型应用

- 大规模 EHR 患者表型初步分群。
- 医学影像像素或 patch 特征快速聚类。
- 大样本多指标监测数据的原型学习。

## 11. 相关方法

- [[K-means聚类（K-means Clustering）]]
- [[BIRCH聚类（Balanced Iterative Reducing and Clustering using Hierarchies, BIRCH）]]
- [[高斯混合模型（Gaussian Mixture Model, GMM）]]
- [[主成分分析（Principal Component Analysis, PCA）]]

## 12. 参考资料

- Sculley D. Web-scale k-means clustering. *WWW*. 2010:1177-1178.
- scikit-learn Developers. `sklearn.cluster.MiniBatchKMeans`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html) （访问日期：2026-07-02）
- Lloyd S. Least squares quantization in PCM. *IEEE Transactions on Information Theory*. 1982;28(2):129-137.
