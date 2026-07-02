---
title: UMAP
english_name: Uniform Manifold Approximation and Projection
slug: uniform-manifold-approximation-and-projection
aliases: [UMAP, "UMAP（Uniform Manifold Approximation and Projection）"]
category: 降维与表征学习
subcategory: 非线性流形学习
tags: [医学统计, 数据科学, 降维, 可视化, 流形学习]
status: 已建
difficulty: advanced
question_type: 高维邻域结构与拓扑表征可视化
data_type: [高维特征矩阵, 图像向量, 组学数据]
outcome_type: [无监督表征, 多种]
python_packages: [umap-learn, scikit-learn]
r_packages: [uwot]
---

# UMAP（Uniform Manifold Approximation and Projection）

## 1. 方法概览

### 1.1 定义

UMAP 是一种非线性降维和可视化方法，基于流形假设和近邻图构建高维数据的模糊拓扑结构，再在低维空间中寻找相似的结构。

### 1.2 它主要解决什么问题

- 研究问题：如何把高维样本嵌入二维或三维，同时尽量保留局部邻域和部分全局结构。
- 适用任务：高维数据可视化、非线性嵌入、样本相似性探索、下游聚类前表征。
- 常见医学场景：单细胞数据可视化、影像组学分布展示、临床表型亚群探索。

### 1.3 直觉理解

UMAP 先在高维空间中为每个样本建立近邻关系，并把这些关系看作带权图。然后它在低维空间中调整样本位置，让低维图尽量复现高维图的邻近关系。

## 2. 数学形式

### 2.1 核心公式

UMAP 在高维空间中构造模糊单纯复形，可简化理解为近邻边权：

$$
p_{ij}=\exp\left(-\frac{\max(0,d(x_i,x_j)-\rho_i)}{\sigma_i}\right)
$$

其中 $\rho_i$ 为局部连通性距离，$\sigma_i$ 控制局部尺度。低维空间中边权常写为：

$$
q_{ij}=\frac{1}{1+a\|y_i-y_j\|^{2b}}
$$

优化目标是让高维边权 $p_{ij}$ 与低维边权 $q_{ij}$ 的交叉熵尽量小：

$$
C=\sum_{i,j}\left[
p_{ij}\log\frac{p_{ij}}{q_{ij}}+
(1-p_{ij})\log\frac{1-p_{ij}}{1-q_{ij}}
\right]
$$

### 2.2 参数或统计量含义

- `n_neighbors`：平衡局部与全局结构的近邻数量。
- `min_dist`：控制低维嵌入中点之间可压缩的最小距离。
- `metric`：高维空间中的距离度量。
- `n_components`：低维嵌入维度。
- $\rho_i$：保证局部连通性的距离校正。
- $\sigma_i$：样本局部尺度参数。

### 2.3 关键假设

- 数据位于或接近某个低维流形。
- 近邻图能反映样本间有意义的局部关系。
- 所选距离度量与医学或生物学相似性相符。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续特征、嵌入向量、稀疏高维特征。
- 因变量类型：UMAP 本身不需要结局变量；也可使用监督式 UMAP 变体。
- 数据结构：样本乘以特征矩阵。
- 是否适合高维数据：适合，常先 PCA/SVD 预降维以降噪和加速。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：需先明确样本单位或自定义距离。

### 3.2 示例表格

以多组学综合特征为例：

| PC1 | PC2 | PC3 | PC4 | Subtype |
| --- | --- | --- | --- | --- |
| 1.8 | -0.4 | 0.2 | 0.6 | A |
| 1.6 | -0.3 | 0.4 | 0.5 | A |
| -1.1 | 1.2 | -0.2 | -0.7 | B |
| -1.4 | 1.0 | -0.1 | -0.6 | B |

### 3.3 输入与产出

#### 输入

- 输入数据：高维特征矩阵或预降维表示。
- 关键变量：近邻数、最小距离、距离度量、目标维度、随机种子。
- 需要预处理的内容：缺失处理、标准化、异常值检查、必要时预降维。

#### 产出

- 模型对象/统计结果：低维嵌入坐标、近邻图结构。
- 参数估计：低维坐标和图结构权重。
- 预测结果：可通过已拟合模型 transform 新样本，但主要用途仍是表征和可视化。
- 不确定性指标：参数敏感性、随机种子稳定性、邻域保持指标。

## 4. 适用场景

- 适合：大规模高维数据可视化、局部亚群探索、需要比 t-SNE 更快或更易投影新样本的场景。
- 不适合：需要严格统计推断、解释坐标轴、保持精确全局距离的场景。
- 使用前需要特别检查的点：`n_neighbors` 和 `min_dist` 是否改变主要结论，颜色标注是否造成视觉误导，是否用独立证据支持亚群解释。

## 5. 实现

### 5.1 Python

常用包：

- `umap-learn`
- `scikit-learn`

```python
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("multiomics_features.csv")
X = df.drop(columns=["Subtype"])

X_pca = make_pipeline(
    StandardScaler(),
    PCA(n_components=30, random_state=42)
).fit_transform(X)

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="euclidean",
    random_state=42
)
embedding = reducer.fit_transform(X_pca)

embedding_df = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
print(embedding_df.head())
```

### 5.2 R

常用包：

- `uwot`

```r
library(uwot)

x <- scale(df[, setdiff(names(df), "Subtype")])
pca <- prcomp(x, center = FALSE, scale. = FALSE)
x_pca <- pca$x[, 1:30]

embedding <- umap(
  x_pca,
  n_neighbors = 15,
  min_dist = 0.1,
  n_components = 2,
  ret_model = FALSE
)
head(embedding)
```

## 6. 结果如何解释

- 核心结果看什么：局部邻域、亚群分离、连续梯度、已知标注在嵌入中的分布。
- 每个主要参数如何解释：`n_neighbors` 越小越强调局部结构，越大越强调更宽范围的全局结构。
- 临床或医学意义如何表达：UMAP 是探索性表征，亚群或梯度应结合原始变量、外部标签和独立验证解释。
- 常见误读：UMAP 上的簇不自动等于真实疾病亚型；坐标轴本身通常没有直接含义。

## 7. 推荐可视化

- UMAP 二维散点图，按疾病亚型、细胞类型、风险评分或关键基因表达着色。
- 参数敏感性对比图。
- 与 t-SNE/PCA 的嵌入对比图。
- UMAP 上叠加连续临床结局或生物标志物热度。

## 8. 优势、局限与常见坑

### 优势

- 在大规模高维数据上通常较快。
- 可兼顾局部结构与部分全局关系。
- 已拟合模型可用于新样本投影。

### 局限

- 对参数、距离度量和预处理敏感。
- 坐标轴缺乏直接可解释性。
- 可视化结果容易被过度解释。

### 常见坑

- 根据 UMAP 图直接宣布发现新亚型。
- 不说明预处理和参数选择。
- 用全数据 fit UMAP 后再评估监督模型，造成信息泄露。

## 9. 与相近方法的区别

- 和 [[t-SNE（t-Distributed Stochastic Neighbor Embedding）]] 的区别：UMAP 通常更快，可投影新样本，并更强调拓扑图结构。
- 和 [[Isomap（Isometric Mapping）]] 的区别：Isomap 保持图上测地距离，UMAP 优化模糊近邻图的低维表示。
- 和 [[主成分分析（Principal Component Analysis, PCA）]] 的区别：PCA 是线性、可解释方差的降维，UMAP 是非线性邻域图嵌入。

## 10. 医学研究中的典型应用

- 单细胞转录组细胞状态和亚群可视化。
- 多组学患者嵌入和潜在亚型探索。
- 影像组学或临床高维特征的样本相似性展示。

## 11. 相关方法

- [[t-SNE（t-Distributed Stochastic Neighbor Embedding）]]
- [[Isomap（Isometric Mapping）]]
- [[主成分分析（Principal Component Analysis, PCA）]]
- [[多维尺度分析（Multidimensional Scaling, MDS）]]

## 12. 参考资料

- McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426; 2018.
- McInnes L, Healy J, Saul N, Grossberger L. UMAP: Uniform Manifold Approximation and Projection. *Journal of Open Source Software*. 2018;3(29):861.
- umap-learn Developers. UMAP documentation. [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/) （访问日期：2026-07-02）
