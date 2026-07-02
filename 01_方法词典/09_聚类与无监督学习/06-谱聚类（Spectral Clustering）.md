---
title: 谱聚类
english_name: Spectral Clustering
slug: spectral-clustering
aliases: [spectral clustering, 谱聚类, "谱聚类（Spectral Clustering）"]
category: 聚类与无监督学习
subcategory: 图聚类
tags: [医学统计, 数据科学, 聚类, 无监督学习, 图方法]
status: 已建
difficulty: advanced
question_type: 图相似性分割与非凸簇发现
data_type: [相似度矩阵, 图数据, 高维特征矩阵]
outcome_type: [无监督分群]
python_packages: [scikit-learn]
r_packages: [kernlab]
---

# 谱聚类（Spectral Clustering）

## 1. 方法概览

### 1.1 定义

谱聚类是一种基于图的聚类方法。它先把样本表示成相似度图，再利用图拉普拉斯矩阵的特征向量把样本嵌入到低维空间，最后在该空间中进行聚类。

### 1.2 它主要解决什么问题

- 研究问题：当簇形状非凸或由复杂相似关系定义时，如何进行分组。
- 适用任务：图分割、非球形簇识别、相似度矩阵聚类、复杂结构样本分群。
- 常见医学场景：患者相似性网络分群，单细胞相似图聚类，影像区域图分割。

### 1.3 直觉理解

谱聚类把样本看作图上的节点，相似样本之间连边更强。算法要做的是把图切成几块，使块内连接强、块间连接弱。

## 2. 数学形式

### 2.1 核心公式

设相似度矩阵为 $W$，度矩阵为：

$$
D_{ii}=\sum_j W_{ij}
$$

未归一化图拉普拉斯矩阵为：

$$
L=D-W
$$

常见归一化拉普拉斯包括：

$$
L_{\text{sym}}=I-D^{-1/2}WD^{-1/2}
$$

或：

$$
L_{\text{rw}}=I-D^{-1}W
$$

谱聚类取拉普拉斯矩阵前 $K$ 个特征向量组成嵌入矩阵 $U$，再对 $U$ 的行进行 [[K-means聚类（K-means Clustering）]]。

### 2.2 参数或统计量含义

- $W$：样本相似度矩阵。
- $D$：度矩阵，表示每个节点总连接强度。
- $L$：图拉普拉斯矩阵。
- `n_clusters`：目标簇数。
- `affinity`：相似度构造方式，如 RBF 核、近邻图或预计算矩阵。
- `gamma`：RBF 相似度的尺度参数。

### 2.3 关键假设

- 样本间相似度图能表达真实结构。
- 图分割目标与研究问题一致。
- 簇内连接强、簇间连接弱。
- 相似度参数设置合理，不会造成图过稀或过密。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续特征、图相似度矩阵、预计算邻接矩阵。
- 因变量类型：无监督方法，不需要结局变量。
- 数据结构：样本乘以特征矩阵，或样本乘以样本相似度矩阵。
- 是否适合高维数据：可用，但相似度构造应谨慎，常先降维。
- 是否适合缺失较多数据：需先处理缺失或定义能处理缺失的相似度。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：可通过自定义相似度纳入，但普通谱聚类不建模相关结构。

### 3.2 示例表格

以患者相似度矩阵为例：

| Patient | P001 | P002 | P003 | P004 |
| --- | --- | --- | --- | --- |
| P001 | 1.00 | 0.82 | 0.12 | 0.20 |
| P002 | 0.82 | 1.00 | 0.18 | 0.22 |
| P003 | 0.12 | 0.18 | 1.00 | 0.79 |
| P004 | 0.20 | 0.22 | 0.79 | 1.00 |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵或相似度矩阵。
- 关键变量：簇数、相似度函数、近邻数或核参数。
- 需要预处理的内容：缺失处理、标准化、相似度矩阵检查、图连通性检查。

#### 产出

- 模型对象/统计结果：簇标签、谱嵌入、相似度图。
- 参数估计：拉普拉斯特征向量。
- 预测结果：标准谱聚类不自然支持新样本预测。
- 不确定性指标：不同相似度参数下的聚类稳定性、图割质量指标。

## 4. 适用场景

- 适合：非凸簇、样本关系适合图表达、已有相似度矩阵的场景。
- 不适合：样本量非常大、相似度矩阵难以存储、需要直接解释簇中心的场景。
- 使用前需要特别检查的点：相似度构造、图是否连通、簇数选择、谱嵌入稳定性。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("patient_features.csv")
X = df[["Marker_1", "Marker_2", "Marker_3", "Marker_4"]]
X_scaled = StandardScaler().fit_transform(X)

fit = SpectralClustering(
    n_clusters=3,
    affinity="nearest_neighbors",
    n_neighbors=10,
    assign_labels="kmeans",
    random_state=42
)
cluster = fit.fit_predict(X_scaled)

print(pd.Series(cluster).value_counts().sort_index())
```

### 5.2 R

常用包：

- `kernlab`

```r
library(kernlab)

x <- scale(df[, c("Marker_1", "Marker_2", "Marker_3", "Marker_4")])
fit <- specc(x, centers = 3, kernel = "rbfdot")

cluster <- as.integer(fit)
table(cluster)
```

## 6. 结果如何解释

- 核心结果看什么：相似图中的分割、各簇样本量、谱嵌入中的分离情况、临床变量在簇间的差异。
- 每个主要参数如何解释：近邻数越小，图越局部；近邻数越大，图更平滑但可能模糊边界。
- 临床或医学意义如何表达：谱聚类发现的是相似性图上的社区结构，应说明相似度定义。
- 常见误读：谱聚类没有天然质心，不能像 K-means 那样直接解释“中心患者”。

## 7. 推荐可视化

- 相似度矩阵热图。
- 谱嵌入二维散点图。
- 图网络分割可视化。
- 聚类标签在 PCA/UMAP 空间中的分布。

## 8. 优势、局限与常见坑

### 优势

- 能处理非凸形状簇。
- 可直接利用相似度矩阵或图结构。
- 与图分割和网络社区发现关系紧密。

### 局限

- 相似度矩阵计算和存储成本高。
- 参数选择影响很大。
- 新样本预测不如质心模型方便。

### 常见坑

- 使用默认 RBF 参数而不检查图结构。
- 图不连通或过密时仍直接解释结果。
- 把谱聚类标签当作稳定真实亚型而不做敏感性分析。

## 9. 与相近方法的区别

- 和 [[K-means聚类（K-means Clustering）]] 的区别：谱聚类先做图谱嵌入，能处理更复杂形状；K-means 直接在原特征空间按质心分组。
- 和 [[DBSCAN（Density-Based Spatial Clustering of Applications with Noise）]] 的区别：DBSCAN 基于密度连通，谱聚类基于相似图切分。
- 和 [[层次聚类（Hierarchical Clustering）]] 的区别：层次聚类输出树状结构，谱聚类输出图分割结果。

## 10. 医学研究中的典型应用

- 患者相似性网络中的亚群发现。
- 单细胞近邻图或相似图聚类。
- 医学影像区域分割和复杂形状病灶结构探索。

## 11. 相关方法

- [[K-means聚类（K-means Clustering）]]
- [[DBSCAN（Density-Based Spatial Clustering of Applications with Noise）]]
- [[层次聚类（Hierarchical Clustering）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]

## 12. 参考资料

- von Luxburg U. A tutorial on spectral clustering. *Statistics and Computing*. 2007;17:395-416.
- Ng AY, Jordan MI, Weiss Y. On spectral clustering: analysis and an algorithm. *NeurIPS*. 2001.
- scikit-learn Developers. `sklearn.cluster.SpectralClustering`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html) （访问日期：2026-07-02）
