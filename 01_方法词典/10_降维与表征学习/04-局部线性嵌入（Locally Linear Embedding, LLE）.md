---
title: 局部线性嵌入
english_name: Locally Linear Embedding, LLE
slug: locally-linear-embedding-lle
aliases: [LLE, locally linear embedding, "局部线性嵌入（Locally Linear Embedding, LLE）"]
category: 降维与表征学习
subcategory: 非线性流形学习
tags: [医学统计, 数据科学, 降维, 流形学习, 无监督学习]
status: 已建
difficulty: advanced
question_type: 非线性流形降维与局部结构保持
data_type: [高维特征矩阵, 图像向量, 组学数据]
outcome_type: [无监督表征, 多种]
python_packages: [scikit-learn]
r_packages: [lle]
---

# 局部线性嵌入（Locally Linear Embedding, LLE）

## 1. 方法概览

### 1.1 定义

局部线性嵌入是一种非线性流形学习方法。它假设高维数据虽然整体弯曲，但每个样本附近的小邻域可以近似看作线性结构，并在低维空间中尽量保持这种局部重构关系。

### 1.2 它主要解决什么问题

- 研究问题：高维样本可能位于低维非线性流形上，如何展开并可视化这个流形。
- 适用任务：非线性降维、局部邻域结构保持、探索性可视化。
- 常见医学场景：影像特征可视化、单细胞或组学样本局部状态探索、复杂表型连续谱展示。

### 1.3 直觉理解

LLE 不直接保留样本之间的全局距离，而是先用每个点周围的邻居重构这个点。如果高维空间中某个点可以由邻居按一组权重拼出来，低维空间中也应保持同样的重构权重。

## 2. 数学形式

### 2.1 核心公式

第一步，对每个样本 $x_i$ 找到邻居集合 $N(i)$，并求重构权重：

$$
\min_{w_{ij}} \sum_i \left\|x_i-\sum_{j\in N(i)}w_{ij}x_j\right\|^2
$$

约束为：

$$
\sum_{j\in N(i)} w_{ij}=1
$$

第二步，在低维空间中寻找 $y_i$，保持这些权重：

$$
\min_{y_i} \sum_i \left\|y_i-\sum_{j\in N(i)}w_{ij}y_j\right\|^2
$$

该问题可写为特征分解：

$$
M=(I-W)^\top(I-W)
$$

取 $M$ 的最小非零特征值对应的特征向量作为低维嵌入。

### 2.2 参数或统计量含义

- $N(i)$：第 $i$ 个样本的近邻集合。
- $w_{ij}$：用邻居 $x_j$ 重构 $x_i$ 的权重。
- $y_i$：第 $i$ 个样本的低维坐标。
- `n_neighbors`：每个点使用的近邻数量。
- `n_components`：目标低维空间维度。
- `method`：标准 LLE、modified LLE、Hessian LLE 等变体。

### 2.3 关键假设

- 数据位于或接近一个低维流形。
- 局部邻域近似线性。
- 邻域图能正确反映流形的局部结构。
- 样本覆盖流形足够密集，噪声不会主导邻域关系。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续型高维特征。
- 因变量类型：LLE 本身不需要结局变量。
- 数据结构：样本乘以特征矩阵。
- 是否适合高维数据：可用于高维数据探索，但近邻搜索和特征分解成本较高。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：需先定义样本单位，或把时间结构纳入距离设计。

### 3.2 示例表格

以影像组学特征为例：

| Texture_1 | Texture_2 | Shape_1 | Shape_2 | Intensity_1 | TumorType |
| --- | --- | --- | --- | --- | --- |
| 0.42 | 1.21 | 0.33 | 0.71 | 2.4 | A |
| 0.39 | 1.18 | 0.35 | 0.75 | 2.2 | A |
| -0.21 | 0.30 | 0.80 | 1.10 | 1.1 | B |
| -0.28 | 0.25 | 0.77 | 1.06 | 1.0 | B |

### 3.3 输入与产出

#### 输入

- 输入数据：高维连续特征矩阵。
- 关键变量：近邻数、目标维度、LLE 变体、正则化参数。
- 需要预处理的内容：缺失处理、标准化、异常值检查、近邻图连通性检查。

#### 产出

- 模型对象/统计结果：低维嵌入坐标、重构误差。
- 参数估计：邻域重构权重和嵌入坐标。
- 预测结果：通常不直接预测，主要用于探索和可视化。
- 不确定性指标：可用参数敏感性、重采样嵌入一致性或邻域保持指标评估。

## 4. 适用场景

- 适合：数据存在连续非线性流形、希望保持局部邻域关系、目标以可视化和探索为主的场景。
- 不适合：需要清晰可解释线性载荷、样本量很小、噪声很高或邻域结构不稳定的场景。
- 使用前需要特别检查的点：近邻数是否过小或过大，邻域图是否连通，嵌入是否对随机扰动敏感。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("radiomics_features.csv")
features = ["Texture_1", "Texture_2", "Shape_1", "Shape_2", "Intensity_1"]
X = df[features]

lle_pipe = make_pipeline(
    StandardScaler(),
    LocallyLinearEmbedding(
        n_neighbors=12,
        n_components=2,
        method="standard",
        random_state=42
    )
)

embedding = lle_pipe.fit_transform(X)
embedding_df = pd.DataFrame(embedding, columns=["LLE1", "LLE2"])
print(embedding_df.head())
```

### 5.2 R

常用包：

- `lle`

```r
library(lle)

x <- scale(df[, c("Texture_1", "Texture_2", "Shape_1", "Shape_2", "Intensity_1")])
fit <- lle(x, m = 2, k = 12)

embedding <- fit$Y
head(embedding)
```

## 6. 结果如何解释

- 核心结果看什么：二维或三维嵌入中样本的局部分布、邻近样本是否保留、不同临床标签是否形成连续梯度或局部簇。
- 每个主要参数如何解释：`n_neighbors` 控制“局部”的范围，过小易断裂，过大则可能抹平流形弯曲。
- 临床或医学意义如何表达：嵌入轴通常不是可直接命名的医学维度，解释应聚焦样本相对位置和局部结构。
- 常见误读：二维图上两团样本分开，不一定说明原始高维空间中存在稳定分类边界。

## 7. 推荐可视化

- LLE1-LLE2 嵌入散点图，按疾病分型或连续指标着色。
- 不同 `n_neighbors` 下的嵌入对比图。
- 邻域保持指标随参数变化曲线。
- 嵌入坐标与关键临床变量的相关图。

## 8. 优势、局限与常见坑

### 优势

- 能捕捉 PCA 难以表达的非线性流形结构。
- 明确强调局部邻域关系。
- 对连续流形可视化直观。

### 局限

- 对近邻数敏感。
- 缺少像 PCA 载荷那样直观的变量解释。
- 对噪声、离群值和采样密度不均较敏感。

### 常见坑

- 只展示一个参数下的漂亮二维图，不做敏感性分析。
- 把低维坐标轴解释为真实医学量表。
- 在训练测试划分前对全数据 fit LLE 后再建模，造成信息泄露。

## 9. 与相近方法的区别

- 和 [[主成分分析（Principal Component Analysis, PCA）]] 的区别：PCA 是线性投影，LLE 是非线性局部结构保持。
- 和 [[Isomap（Isometric Mapping）]] 的区别：Isomap 试图保持流形上的全局测地距离，LLE 更强调局部重构权重。
- 和 [[t-SNE（t-Distributed Stochastic Neighbor Embedding）]] 的区别：t-SNE 更偏向邻域概率可视化，LLE 有明确的局部线性重构目标。

## 10. 医学研究中的典型应用

- 影像组学样本的非线性表型谱可视化。
- 单细胞数据中细胞状态连续变化的探索性展示。
- 多指标临床表型中潜在连续结构识别。

## 11. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[Isomap（Isometric Mapping）]]
- [[t-SNE（t-Distributed Stochastic Neighbor Embedding）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]

## 12. 参考资料

- Roweis ST, Saul LK. Nonlinear dimensionality reduction by locally linear embedding. *Science*. 2000;290(5500):2323-2326.
- VanderPlas J. *Python Data Science Handbook*. O'Reilly Media; 2016.
- scikit-learn Developers. `sklearn.manifold.LocallyLinearEmbedding`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html) （访问日期：2026-07-02）
