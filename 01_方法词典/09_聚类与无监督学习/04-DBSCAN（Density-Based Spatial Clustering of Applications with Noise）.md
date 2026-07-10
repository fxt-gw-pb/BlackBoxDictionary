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

### 1.1 一句话本质

DBSCAN 从局部邻居足够多的核心点向外扩展密度连通区域，并把无法连接到任何核心区域的点标记为噪声。

### 1.2 定义

DBSCAN 是基于密度连通的聚类方法。它由邻域半径 $\varepsilon$ 和最小邻居数 `MinPts` 定义核心点，再将可由核心点链连接的样本归为同一簇。

### 1.3 它主要解决什么问题

- 不预先指定簇数。
- 发现弯曲、环形等非球形簇。
- 在聚类过程中显式区分核心点、边界点和噪声点。

### 1.4 直觉与类比

把每个样本周围画一个半径为 $\varepsilon$ 的圆。圆内人足够多的是“人群中心”；相互重叠的人群中心连成社区，贴着社区但邻居少的是边界，孤零零的点是噪声。

## 2. 核心思想与原理

### 2.1 簇由连通而非质心定义

DBSCAN 不需要一个中心代表整簇。只要存在核心点链，两个距离很远的点也可属于同一弯曲簇。

### 2.2 三类点

- 核心点：$\varepsilon$ 邻域样本数至少为 `MinPts`，计数包含自身。
- 边界点：自身不是核心点，但落在核心点邻域内。
- 噪声点：既非核心点，也无法由核心点扩展到达。

### 2.3 参数共同定义“密”

$\varepsilon$ 决定观察半径，`MinPts` 决定最少人数。参数只有结合特征缩放、维度与距离度量才有意义。

## 3. 数学形式

### 3.1 邻域与核心点

$$
N_\varepsilon(x_i)=
\{x_j:d(x_i,x_j)\le\varepsilon\}
$$

$$
|N_\varepsilon(x_i)|\ge\operatorname{MinPts}
\quad\Longrightarrow\quad
x_i\text{ 是核心点}
$$

### 3.2 直接密度可达

若 $x_i$ 为核心点且 $x_j\in N_\varepsilon(x_i)$，则 $x_j$ 从 $x_i$ 直接密度可达。多步核心点链给出密度可达和密度连通。

### 3.3 K-distance 选参

对每点计算到第 `MinPts` 个近邻的距离并排序，曲线拐点常作为 $\varepsilon$ 候选，但不是自动真值。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 单一密度尺度大致适用 | 稀簇被当噪声、密簇合并 | 参数网格、OPTICS |
| 距离在当前维度有效 | 邻居距离趋同 | 降维与距离诊断 |
| 特征尺度合理 | 大量纲变量主导邻域 | 标准化敏感性 |
| 噪声有领域解释 | 误删罕见临床亚型 | 逐例审查噪声点 |

## 4. 手把手算例

一维样本为：

$$
x=(0,\ 0.4,\ 0.8,\ 3.0,\ 3.3,\ 3.6,\ 8.0)
$$

取 $\varepsilon=0.5$，`MinPts=3`，邻域计入自身。

**Step 1：第一片区域。**

- $N(0)=\{0,0.4\}$，只有 2 点，0 不是核心点。
- $N(0.4)=\{0,0.4,0.8\}$，有 3 点，0.4 是核心点。
- $N(0.8)=\{0.4,0.8\}$，只有 2 点。

因此 0 和 0.8 虽不是核心点，却都在核心点 0.4 的邻域内，是边界点；三者组成一簇。

**Step 2：第二片区域。** 同理，3.3 是核心点，3.0 与 3.6 是边界点，三者组成第二簇。

**Step 3：噪声点。**

$$
N(8.0)=\{8.0\}
$$

它不在任何核心点邻域内，因此标记为噪声。

**结论：** DBSCAN 得到两个簇和一个噪声点，且无需提前指定“两个簇”。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 连续特征、空间坐标或预计算距离。
- 多量纲特征通常需标准化，但经纬度等空间坐标需选择合适距离和单位。
- 高维时应先评估距离集中问题。

### 5.2 输入与产出

输入为 $\varepsilon$、`MinPts` 和距离度量。输出为簇标签、核心样本索引与噪声标签。标准 DBSCAN 不自然支持新样本预测。

## 6. 适用场景

- 簇形状不规则，且希望同时识别噪声。
- 空间热点、图像区域或低维嵌入探索。
- 不适合同一数据中密度差异悬殊或维度很高的情况。

## 7. 实现

### 7.1 Python

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

X_s = StandardScaler().fit_transform(X)

neighbors = NearestNeighbors(n_neighbors=5).fit(X_s)
distances, _ = neighbors.kneighbors(X_s)
k_distance = sorted(distances[:, -1])

model = DBSCAN(
    eps=0.6,
    min_samples=5,
    metric="euclidean",
    n_jobs=-1,
)
cluster = model.fit_predict(X_s)
core_index = model.core_sample_indices_
noise_rate = (cluster == -1).mean()
print(noise_rate, core_index)
```

### 7.2 R

```r
library(dbscan)

x <- scale(df[, c("Longitude", "Latitude", "PM25", "CaseRate")])
kNNdistplot(x, k = 4)
abline(h = 0.6, col = "red", lty = 2)

fit <- dbscan(x, eps = 0.6, minPts = 5)
cluster <- fit$cluster
table(cluster)
```

R 包中噪声标签通常为 0；scikit-learn 中为 -1。

## 8. 结果如何解释

- 簇编号无顺序意义，噪声也不等同于错误数据。
- `eps` 过小会产生许多小簇与噪声；过大可能通过“桥”连接不同簇。
- 边界点可能因遍历顺序在相邻簇间存在分配歧义，不宜过度解释。
- 若用于空间流行病学，必须考虑人口基数、空间尺度和多重探索。

## 9. 诊断与稳健性

1. 绘制 K-distance 曲线，提出多个 $\varepsilon$ 候选。
2. 网格比较簇数、噪声率和样本共同聚类矩阵。
3. 改变标准化和距离度量。
4. 检查核心、边界与噪声病例的临床特征。
5. 密度差异明显时与 OPTICS 或 HDBSCAN 思路比较。

## 10. 推荐可视化

- K-distance 曲线。
- 核心点、边界点、噪声点分层散点图。
- $\varepsilon$ 与 `MinPts` 参数稳定性热图。
- 空间地图或二维嵌入上的簇与噪声。

## 11. 优势、局限与常见坑

**优势：** 无需指定簇数，可识别非凸簇并显式标记噪声。

**局限：** 参数敏感，单一密度尺度难兼顾稀密簇，高维距离失效，预测新样本不自然。

**常见坑：** 未标准化；把噪声全删掉；把 `eps` 当簇内最大距离；只试一组参数；在二维可视化嵌入上聚类后当作原空间真相。

## 12. 与相近方法的区别

- [[K-means聚类（K-means Clustering）]]：质心硬聚类，需给 $K$，不标噪声。
- [[均值漂移聚类（Mean Shift Clustering）]]：寻找密度峰，不使用密度可达定义。
- [[谱聚类（Spectral Clustering）]]：通过图切分处理复杂形状，但需指定簇数。
- 选择经验：低维、噪声明显、簇形状不规则时优先考虑 DBSCAN。

## 13. 医学研究中的典型应用

- 地理病例热点和环境暴露聚集区探索。
- 空间组学或图像中的不规则高密度区域。
- 患者嵌入空间中的罕见模式与异常病例发现。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| core point | 邻域样本数达到阈值的核心点 |
| border point | 位于核心点邻域内但自身不够密的点 |
| noise point | 不属于任何密度连通区域的点 |
| density reachable | 可沿核心点邻域链到达 |
| K-distance plot | 每点第 K 近邻距离的排序曲线 |

## 15. 相关方法

- [[K-means聚类（K-means Clustering）]]
- [[均值漂移聚类（Mean Shift Clustering）]]
- [[谱聚类（Spectral Clustering）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]

## 16. 参考资料

- Ester M, Kriegel HP, Sander J, Xu X. A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD*. 1996:226-231.
- Schubert E, Sander J, Ester M, Kriegel HP, Xu X. DBSCAN revisited, revisited. *ACM Trans Database Syst*. 2017;42(3):19.
- scikit-learn Developers. `DBSCAN` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) （访问日期：2026-07-09）
