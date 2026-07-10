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

### 1.1 一句话本质

谱聚类把样本变成相似图，利用图拉普拉斯的低频特征向量寻找“内部连接强、彼此连接弱”的节点组。

### 1.2 定义

谱聚类是一类图分割方法。它先构造相似度矩阵，再从图拉普拉斯矩阵的特征向量得到低维谱嵌入，最后对嵌入行进行 K-means 或直接离散化。

### 1.3 它主要解决什么问题

- 原空间中的簇非凸，质心距离无法正确切分。
- 数据天然以邻接图或患者相似矩阵表示。
- 希望按照图连接模式而非全局欧氏中心分群。

### 1.4 直觉与类比

把患者看成节点，相似患者之间连边。理想切法像剪断少数弱桥，把网络分成内部紧密的社区；拉普拉斯特征向量提供了连续近似的切图坐标。

## 2. 核心思想与原理

### 2.1 图比坐标更重要

谱聚类先决定谁与谁相似，再做聚类。通过近邻图，两个月牙形簇可各自沿曲线连通，即使它们没有球形质心结构。

### 2.2 拉普拉斯编码平滑性

对向量 $f$：

$$
f^\top Lf=
\frac12\sum_{i,j}W_{ij}(f_i-f_j)^2
$$

若强连接节点取值接近，该量较小。因此小特征值对应图上变化缓慢、可揭示社区的方向。

### 2.3 图构造决定答案

RBF 尺度、近邻数、互近邻规则和特征标准化会改变边。谱算法只能忠实切分给定图，无法弥补错误相似度。

## 3. 数学形式

### 3.1 图拉普拉斯

$$
D_{ii}=\sum_jW_{ij},\qquad L=D-W
$$

常用对称归一化形式：

$$
L_{\mathrm{sym}}=
I-D^{-1/2}WD^{-1/2}
$$

### 3.2 谱嵌入

取 $L_{\mathrm{sym}}$ 最小的 $K$ 个特征值对应特征向量组成 $U$，按行归一化后对 $U$ 的行聚类。

### 3.3 RBF 相似度

$$
W_{ij}=
\exp[-\gamma\|x_i-x_j\|^2]
$$

也可只保留 K 近邻边，得到稀疏图。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 相似图表达真实邻近 | 切分没有临床意义 | 图与邻居审查 |
| 图不过稀或过密 | 断裂或所有点连成一团 | 连通分量与度分布 |
| 簇数和特征间隙稳定 | 分割对参数敏感 | eigengap 与重采样 |
| 样本量可计算 | 相似矩阵内存过大 | 稀疏近邻图 |

## 4. 手把手算例

四名患者构成两个互不连接的二节点小组：

$$
W=
\begin{pmatrix}
0&1&0&0\\
1&0&0&0\\
0&0&0&1\\
0&0&1&0
\end{pmatrix}
$$

每个节点度为 1，所以 $D=I$，未归一化与归一化拉普拉斯相同：

$$
L=D-W=
\begin{pmatrix}
1&-1&0&0\\
-1&1&0&0\\
0&0&1&-1\\
0&0&-1&1
\end{pmatrix}
$$

**Step 1：找零特征向量。**

$$
u_1=\frac{1}{\sqrt2}(1,1,0,0)^\top,\qquad
u_2=\frac{1}{\sqrt2}(0,0,1,1)^\top
$$

均满足 $Lu=0$。零特征值的重数为 2，正好等于图的连通分量数。

**Step 2：组成谱坐标。** $U=(u_1,u_2)$ 的四行是：

$$
\left(\frac1{\sqrt2},0\right),
\left(\frac1{\sqrt2},0\right),
\left(0,\frac1{\sqrt2}\right),
\left(0,\frac1{\sqrt2}\right)
$$

**Step 3：聚类。** 前两行完全相同，后两行完全相同，K-means 自然得到 $\{1,2\}$ 与 $\{3,4\}$。

**结论：** 谱嵌入把图连接结构变成容易分开的坐标；一般图不是完全断开时，小非零特征值提供近似分割。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 可输入数值特征、相似矩阵或邻接图。
- 相似矩阵应非负、通常对称，且需检查图连通性。
- 缺失需在构图前处理。

### 5.2 输入与产出

输入为相似度规则、近邻数或 $\gamma$、簇数与标签离散化方法。输出为谱嵌入和簇标签；标准实现不自然支持新样本外推。

## 6. 适用场景

- 非凸簇、患者相似网络、单细胞近邻图和图像分割。
- 数据规模允许构造稀疏图和求特征向量。
- 不适合超大稠密图、相似度定义不可靠或需在线预测的任务。

## 7. 实现

### 7.1 Python

```python
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

X_s = StandardScaler().fit_transform(X)
model = SpectralClustering(
    n_clusters=3,
    affinity="nearest_neighbors",
    n_neighbors=15,
    assign_labels="kmeans",
    n_init=50,
    random_state=42,
    n_jobs=-1,
)
cluster = model.fit_predict(X_s)
affinity = model.affinity_matrix_
print(cluster)
```

### 7.2 R

```r
library(kernlab)

x <- scale(df[, c("Marker_1", "Marker_2", "Marker_3", "Marker_4")])
fit <- specc(
  as.matrix(x),
  centers = 3,
  kernel = "rbfdot",
  kpar = "automatic"
)

cluster <- as.integer(fit)
table(cluster)
```

## 8. 结果如何解释

- 簇是所选相似图上的社区，不一定有原空间质心。
- 小特征值和 eigengap 可辅助理解图结构，但不自动确定唯一簇数。
- 簇编号无顺序意义。
- 必须报告 affinity、近邻数、核参数和标准化方式。

## 9. 诊断与稳健性

1. 检查图的连通分量、节点度与孤立点。
2. 改变近邻数、$\gamma$ 和相似度定义。
3. 比较前若干拉普拉斯特征值与 eigengap。
4. 重采样后比较共同聚类矩阵。
5. 与 K-means、DBSCAN 和层次聚类对照。

## 10. 推荐可视化

- 排序后的相似度矩阵热图。
- 近邻图按簇着色。
- 谱嵌入二维散点图。
- 拉普拉斯特征值 scree/eigengap 图。

## 11. 优势、局限与常见坑

**优势：** 能处理非凸簇，直接利用图或相似矩阵，图切分解释清晰。

**局限：** 构图与特征分解昂贵，参数敏感，新样本预测困难。

**常见坑：** 默认相似度直接使用；图断裂却未检查；把 UMAP 图上的邻近当原始图；只展示标签不报告构图规则。

## 12. 与相近方法的区别

- [[K-means聚类（K-means Clustering）]]：原空间质心分组；谱聚类先把图结构展开。
- [[DBSCAN（Density-Based Spatial Clustering of Applications with Noise）]]：按密度可达分簇并标噪声。
- [[层次聚类（Hierarchical Clustering）]]：产生树状嵌套结构，谱聚类产生给定 $K$ 的图分割。
- 选择经验：复杂非凸结构且能构造可信相似图时使用谱聚类。

## 13. 医学研究中的典型应用

- 患者相似网络亚群。
- 单细胞或空间组学近邻图分群。
- 医学图像区域与病灶轮廓分割。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| affinity matrix | 节点两两相似度矩阵 |
| degree matrix | 对角线上为节点总边权的矩阵 |
| graph Laplacian | 编码图连接和平滑性的矩阵 |
| spectral embedding | 由拉普拉斯低频特征向量形成的坐标 |
| eigengap | 相邻特征值差距，用于观察潜在分割尺度 |

## 15. 相关方法

- [[K-means聚类（K-means Clustering）]]
- [[DBSCAN（Density-Based Spatial Clustering of Applications with Noise）]]
- [[层次聚类（Hierarchical Clustering）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]

## 16. 参考资料

- von Luxburg U. A tutorial on spectral clustering. *Stat Comput*. 2007;17:395-416.
- Ng AY, Jordan MI, Weiss Y. On spectral clustering: analysis and an algorithm. *NeurIPS*. 2001.
- scikit-learn Developers. `SpectralClustering` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html) （访问日期：2026-07-09）
