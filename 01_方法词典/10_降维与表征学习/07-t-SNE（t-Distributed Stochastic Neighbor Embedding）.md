---
title: t-SNE
english_name: t-Distributed Stochastic Neighbor Embedding
slug: t-distributed-stochastic-neighbor-embedding
aliases: [t-SNE, TSNE, "t-SNE（t-Distributed Stochastic Neighbor Embedding）"]
category: 降维与表征学习
subcategory: 非线性可视化
tags: [医学统计, 数据科学, 降维, 可视化, 流形学习]
status: 已建
difficulty: advanced
question_type: 高维样本邻域可视化
data_type: [高维特征矩阵, 图像向量, 组学数据]
outcome_type: [无监督表征, 多种]
python_packages: [scikit-learn, openTSNE]
r_packages: [Rtsne]
---

# t-SNE（t-Distributed Stochastic Neighbor Embedding）

## 1. 方法概览

### 1.1 定义

t-SNE 是一种用于高维数据可视化的非线性降维方法。它把高维空间和低维空间中的邻近关系都表示为概率分布，并让两个分布尽可能接近。

### 1.2 它主要解决什么问题

- 研究问题：高维样本的局部邻域结构如何在二维或三维图中展示。
- 适用任务：探索性可视化、样本分布观察、潜在亚群展示。
- 常见医学场景：单细胞聚类可视化、影像组学样本分布展示、组学或表型数据的探索性分群。

### 1.3 直觉理解

t-SNE 关心“谁是谁的邻居”。如果两个点在高维空间中很相似，算法会尽量让它们在二维图中也靠近；为了避免所有点挤在一起，低维空间使用 t 分布产生更重的尾部。

## 2. 数学形式

### 2.1 核心公式

高维空间中，点 $x_i$ 与 $x_j$ 的相似度定义为条件概率：

$$
p_{j|i}=\frac{\exp(-\|x_i-x_j\|^2/2\sigma_i^2)}
{\sum_{k\neq i}\exp(-\|x_i-x_k\|^2/2\sigma_i^2)}
$$

对称化后得到：

$$
p_{ij}=\frac{p_{j|i}+p_{i|j}}{2n}
$$

低维空间中，使用 t 分布定义相似度：

$$
q_{ij}=\frac{(1+\|y_i-y_j\|^2)^{-1}}
{\sum_{k\neq l}(1+\|y_k-y_l\|^2)^{-1}}
$$

目标是最小化 KL 散度：

$$
C=KL(P\|Q)=\sum_i\sum_j p_{ij}\log\frac{p_{ij}}{q_{ij}}
$$

### 2.2 参数或统计量含义

- $p_{ij}$：高维空间中的样本相似概率。
- $q_{ij}$：低维空间中的样本相似概率。
- `perplexity`：近似控制每个点关注的邻居数量。
- `learning_rate`：优化步长。
- `early_exaggeration`：早期优化阶段放大簇间分离的参数。
- `n_iter` 或 `max_iter`：优化迭代次数。

### 2.3 关键假设

- 局部邻域结构比全局距离更重要。
- 高维距离度量能反映样本相似性。
- 低维图主要用于探索和展示，而不是严格保距。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续型高维特征、嵌入向量或标准化后的矩阵。
- 因变量类型：t-SNE 本身不需要结局变量。
- 数据结构：样本乘以特征矩阵。
- 是否适合高维数据：适合可视化，但常先用 PCA/SVD 降到中等维度再运行。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：需明确样本单位，普通 t-SNE 不处理相关结构。

### 3.2 示例表格

以单细胞表达嵌入为例：

| PC1 | PC2 | PC3 | PC4 | CellType |
| --- | --- | --- | --- | --- |
| 2.1 | -0.4 | 0.3 | 1.2 | T cell |
| 1.9 | -0.2 | 0.5 | 1.0 | T cell |
| -1.2 | 1.5 | 0.1 | -0.7 | B cell |
| -1.4 | 1.2 | 0.0 | -0.8 | B cell |

### 3.3 输入与产出

#### 输入

- 输入数据：高维特征矩阵或预降维后的特征。
- 关键变量：困惑度、学习率、迭代次数、初始化方式、距离度量。
- 需要预处理的内容：缺失处理、标准化、异常值检查、必要时先用 PCA/SVD 预降维。

#### 产出

- 模型对象/统计结果：二维或三维嵌入坐标。
- 参数估计：优化得到的低维坐标。
- 预测结果：不直接预测，主要用于可视化。
- 不确定性指标：不同随机种子、参数和重采样下的嵌入稳定性。

## 4. 适用场景

- 适合：高维数据的局部结构展示、潜在亚群探索、复杂嵌入可视化。
- 不适合：需要解释坐标轴、保持全局距离、比较簇间距离大小或做正式统计检验的场景。
- 使用前需要特别检查的点：perplexity 是否与样本量匹配，随机种子是否影响结论，是否把图形分离过度解释为真实聚类。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("omics_features.csv")
X = df.drop(columns=["Group"])

X_pca = make_pipeline(
    StandardScaler(),
    PCA(n_components=30, random_state=42)
).fit_transform(X)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)
embedding = tsne.fit_transform(X_pca)

embedding_df = pd.DataFrame(embedding, columns=["tSNE1", "tSNE2"])
print(embedding_df.head())
```

### 5.2 R

常用包：

- `Rtsne`

```r
library(Rtsne)

x <- scale(df[, setdiff(names(df), "Group")])
pca <- prcomp(x, center = FALSE, scale. = FALSE)
x_pca <- pca$x[, 1:30]

fit <- Rtsne(x_pca, dims = 2, perplexity = 30, pca = FALSE, seed = 42)
embedding <- fit$Y
head(embedding)
```

## 6. 结果如何解释

- 核心结果看什么：局部邻域、样本是否形成稳定的近邻群、已知分组是否在图上呈现连续或分离结构。
- 每个主要参数如何解释：`perplexity=30` 表示每个样本的有效邻域规模大致在几十个样本量级。
- 临床或医学意义如何表达：t-SNE 图适合提出探索性假设，不应单独作为分类、预后或机制证据。
- 常见误读：簇与簇之间的距离、簇大小和坐标轴方向通常没有直接定量解释。

## 7. 推荐可视化

- t-SNE 二维散点图，按疾病分组、细胞类型或连续临床变量着色。
- 不同 perplexity 的嵌入对比。
- 不同随机种子的稳定性对比。
- t-SNE 图上叠加关键变量表达或风险评分。

## 8. 优势、局限与常见坑

### 优势

- 高维局部结构可视化效果强。
- 常能揭示潜在亚群或局部连续谱。
- 适合与聚类结果、细胞类型标注等结合展示。

### 局限

- 不保留全局距离。
- 对参数和随机初始化敏感。
- 主要用于可视化，不适合直接作为可解释建模结果。

### 常见坑

- 根据 t-SNE 图上两个簇的距离判断生物学距离。
- 不做参数敏感性分析。
- 把 t-SNE 坐标作为普通连续变量进行过度解释。

## 9. 与相近方法的区别

- 和 [[UMAP（Uniform Manifold Approximation and Projection）]] 的区别：UMAP 通常更快、更强调拓扑结构，t-SNE 更经典地用于局部邻域可视化。
- 和 [[主成分分析（Principal Component Analysis, PCA）]] 的区别：PCA 是线性、可解释方差的降维；t-SNE 是非线性、概率邻域可视化。
- 和 [[局部线性嵌入（Locally Linear Embedding, LLE）]] 的区别：LLE 保持局部线性重构权重，t-SNE 匹配邻域概率分布。

## 10. 医学研究中的典型应用

- 单细胞转录组细胞群可视化。
- 医学影像或组学样本的探索性分群展示。
- 临床高维表型数据中异常样本和局部亚群观察。

## 11. 相关方法

- [[UMAP（Uniform Manifold Approximation and Projection）]]
- [[主成分分析（Principal Component Analysis, PCA）]]
- [[局部线性嵌入（Locally Linear Embedding, LLE）]]
- [[多维尺度分析（Multidimensional Scaling, MDS）]]

## 12. 参考资料

- van der Maaten L, Hinton G. Visualizing data using t-SNE. *Journal of Machine Learning Research*. 2008;9:2579-2605.
- van der Maaten L. Accelerating t-SNE using tree-based algorithms. *Journal of Machine Learning Research*. 2014;15:3221-3245.
- scikit-learn Developers. `sklearn.manifold.TSNE`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) （访问日期：2026-07-02）
