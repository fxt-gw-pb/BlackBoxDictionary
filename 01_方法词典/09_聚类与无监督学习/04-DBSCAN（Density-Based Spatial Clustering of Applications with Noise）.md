---
title: DBSCAN
english_name: Density-Based Spatial Clustering of Applications with Noise
slug: dbscan-density-based-spatial-clustering-of-applications-with-noise
aliases: [DBSCAN, density-based spatial clustering, "DBSCAN（Density-Based Spatial Clustering of Applications with Noise）"]
category: 聚类与无监督学习
subcategory: 密度聚类
tags: [医学统计, 数据科学, 聚类, 无监督学习, 密度方法, 异常点]
status: 已建
difficulty: intermediate
question_type: 任意形状簇发现与噪声识别
data_type: [表格数据, 空间数据, 高维特征矩阵]
outcome_type: [无监督分群, 异常点]
python_packages: [scikit-learn]
r_packages: [dbscan]
---

# DBSCAN（Density-Based Spatial Clustering of Applications with Noise）

## 1. 方法概览

### 1.1 定义

DBSCAN 是一种基于密度的聚类方法。它把高密度区域连接成簇，并把低密度区域中的点标记为噪声或离群点。

### 1.2 它主要解决什么问题

- 研究问题：数据中是否存在任意形状的高密度样本群，同时伴随噪声点。
- 适用任务：非球形簇识别、噪声识别、空间聚类、异常样本探索。
- 常见医学场景：地理流行病学热点探索，影像特征异常区域识别，患者相似性空间中的离群样本发现。

### 1.3 直觉理解

DBSCAN 的核心想法是“人多的地方连成片”。如果一个点周围半径 $\varepsilon$ 内有足够多邻居，它就是核心点；核心点之间能互相连通时，就形成一个簇。

## 2. 数学形式

### 2.1 核心公式

给定半径 $\varepsilon$，样本 $x_i$ 的邻域为：

$$
N_\varepsilon(x_i)=\{x_j:\ d(x_i,x_j)\le \varepsilon\}
$$

若：

$$
|N_\varepsilon(x_i)|\ge \mathrm{MinPts}
$$

则 $x_i$ 是核心点。若 $x_j\in N_\varepsilon(x_i)$ 且 $x_i$ 为核心点，则 $x_j$ 从 $x_i$ 密度可达。由密度可达关系连接起来的最大样本集合构成一个簇。

### 2.2 参数或统计量含义

- `eps`：邻域半径 $\varepsilon$。
- `min_samples`：成为核心点所需的最小邻域样本数。
- 核心点：邻域内样本数足够多的点。
- 边界点：不满足核心点条件，但位于某个核心点邻域内。
- 噪声点：既不是核心点也不能从核心点密度可达的点。

### 2.3 关键假设

- 簇可由密度连通区域定义。
- 同一数据集中各簇密度差异不宜过大。
- 所选距离度量能反映样本相似性。
- `eps` 和 `min_samples` 设置合理。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续特征、空间坐标或可定义距离的特征。
- 因变量类型：无监督方法，不需要结局变量。
- 数据结构：样本乘以特征矩阵。
- 是否适合高维数据：高维距离可能失效，常需降维或特征筛选。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：需先构造样本级特征或使用合适距离。

### 3.2 示例表格

以社区病例空间聚类为例：

| Longitude | Latitude | PM25 | PopulationDensity | CaseRate |
| --- | --- | --- | --- | --- |
| 116.31 | 39.98 | 48 | 9300 | 0.12 |
| 116.33 | 39.97 | 52 | 9100 | 0.15 |
| 116.80 | 40.20 | 31 | 2500 | 0.03 |
| 116.82 | 40.22 | 29 | 2400 | 0.02 |

### 3.3 输入与产出

#### 输入

- 输入数据：数值特征矩阵或预计算距离矩阵。
- 关键变量：`eps`、`min_samples`、距离度量。
- 需要预处理的内容：缺失处理、标准化、距离尺度确认、K 距离图辅助选参。

#### 产出

- 模型对象/统计结果：簇标签、核心样本索引、噪声点标签。
- 参数估计：DBSCAN 无传统参数估计，输出密度连通结构。
- 预测结果：标准 DBSCAN 不自然支持新样本预测。
- 不确定性指标：参数敏感性、噪声比例、轮廓系数或领域指标。

## 4. 适用场景

- 适合：簇形状不规则、存在噪声点、不想预先指定簇数的场景。
- 不适合：不同簇密度差异很大、样本维度很高、所有簇都接近球形且需要质心解释的场景。
- 使用前需要特别检查的点：`eps` 选择、噪声比例、标准化和距离度量。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("spatial_cases.csv")
X = df[["Longitude", "Latitude", "PM25", "PopulationDensity"]]

model = make_pipeline(
    StandardScaler(),
    DBSCAN(eps=0.6, min_samples=8)
)
cluster = model.fit_predict(X)

df["cluster"] = cluster
print(df["cluster"].value_counts().sort_index())
```

### 5.2 R

常用包：

- `dbscan`

```r
library(dbscan)

x <- scale(df[, c("Longitude", "Latitude", "PM25", "PopulationDensity")])
fit <- dbscan(x, eps = 0.6, minPts = 8)

cluster <- fit$cluster
table(cluster)
```

## 6. 结果如何解释

- 核心结果看什么：簇数量、每簇样本量、噪声点比例、核心点分布。
- 每个主要参数如何解释：`eps` 决定邻域半径，过小会产生很多噪声，过大可能把多个簇连成一片。
- 临床或医学意义如何表达：可说“在给定距离尺度下识别出若干密度连通的高风险区域/患者群”。
- 常见误读：噪声点不一定是错误数据，也可能代表罕见或异质临床个体。

## 7. 推荐可视化

- 聚类散点图，噪声点单独标色。
- K 距离曲线辅助选择 `eps`。
- 参数网格下簇数和噪声比例热图。
- 空间数据中的地图叠加图。

## 8. 优势、局限与常见坑

### 优势

- 不需要预先指定簇数。
- 能识别任意形状簇。
- 能显式标记噪声点。

### 局限

- 对 `eps` 敏感。
- 不适合密度差异很大的簇。
- 高维数据中距离度量容易失效。

### 常见坑

- 直接使用未经标准化的多量纲特征。
- 只试一组参数就解释结果。
- 把所有噪声点都当作无意义异常删除。

## 9. 与相近方法的区别

- 和 [[K-means聚类（K-means Clustering）]] 的区别：K-means 偏球形簇且需指定 $K$，DBSCAN 可发现任意形状簇并识别噪声。
- 和 [[均值漂移聚类（Mean Shift Clustering）]] 的区别：两者都与密度有关，均值漂移寻找密度峰，DBSCAN 基于密度连通。
- 和 [[谱聚类（Spectral Clustering）]] 的区别：谱聚类通过图分割处理复杂结构，DBSCAN 通过局部密度和连通性定义簇。

## 10. 医学研究中的典型应用

- 疾病地理热点和环境暴露聚集区识别。
- 高维患者相似性空间中的异常个体发现。
- 图像或空间组学数据中不规则区域分割。

## 11. 相关方法

- [[K-means聚类（K-means Clustering）]]
- [[均值漂移聚类（Mean Shift Clustering）]]
- [[谱聚类（Spectral Clustering）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]

## 12. 参考资料

- Ester M, Kriegel HP, Sander J, Xu X. A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD*. 1996:226-231.
- Schubert E, Sander J, Ester M, Kriegel HP, Xu X. DBSCAN revisited, revisited. *ACM TODS*. 2017;42(3):19.
- scikit-learn Developers. `sklearn.cluster.DBSCAN`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) （访问日期：2026-07-02）
