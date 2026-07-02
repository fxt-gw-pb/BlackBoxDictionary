---
title: Isomap
english_name: Isometric Mapping
slug: isometric-mapping
aliases: [Isomap, isometric mapping, "Isomap（Isometric Mapping）"]
category: 降维与表征学习
subcategory: 非线性流形学习
tags: [医学统计, 数据科学, 降维, 流形学习, 无监督学习]
status: 已建
difficulty: advanced
question_type: 流形测地距离保持降维
data_type: [高维特征矩阵, 图像向量, 组学数据]
outcome_type: [无监督表征, 多种]
python_packages: [scikit-learn]
r_packages: [vegan, RDRToolbox]
---

# Isomap（Isometric Mapping）

## 1. 方法概览

### 1.1 定义

Isomap 是一种非线性流形学习方法。它通过近邻图上的最短路径近似流形测地距离，再用经典多维尺度分析把样本嵌入低维空间。

### 1.2 它主要解决什么问题

- 研究问题：高维数据在弯曲流形上分布时，如何保持沿流形的全局距离关系。
- 适用任务：非线性降维、流形展开、探索性可视化。
- 常见医学场景：影像表型连续谱展示，单细胞发育轨迹探索，复杂临床表型的非线性结构观察。

### 1.3 直觉理解

在弯曲表面上，两点的直线距离可能穿过“空气”，并不代表沿数据流形真正要走的距离。Isomap 先用近邻图把样本连成道路网，再用最短路径近似沿流形行走的距离。

## 2. 数学形式

### 2.1 核心公式

先为每个样本 $x_i$ 构建近邻图 $G$，边权为欧氏距离：

$$
d_G(i,j)=\|x_i-x_j\|_2,\quad j\in N(i)
$$

再用最短路径得到测地距离近似：

$$
d_{\mathcal{M}}(i,j)\approx \operatorname{shortest\_path}_G(i,j)
$$

令 $D^{(2)}$ 为平方距离矩阵，经典 MDS 的双中心化矩阵为：

$$
B=-\frac{1}{2} H D^{(2)} H
$$

其中：

$$
H=I-\frac{1}{n}\mathbf{1}\mathbf{1}^\top
$$

对 $B$ 做特征分解，取前 $k$ 个正特征值和特征向量得到低维坐标：

$$
Y=V_k \Lambda_k^{1/2}
$$

### 2.2 参数或统计量含义

- $G$：近邻图。
- $d_G(i,j)$：近邻图边权。
- $d_{\mathcal{M}}(i,j)$：流形上的测地距离近似。
- `n_neighbors`：构图时每个样本连接的近邻数。
- `n_components`：低维嵌入维度。
- residual variance：原始测地距离与低维距离不一致的程度。

### 2.3 关键假设

- 数据位于低维流形附近。
- 近邻图足以近似流形局部结构。
- 图连通且没有由噪声边造成的严重短路。
- 样本密度足够覆盖流形。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续型高维特征。
- 因变量类型：Isomap 本身不需要结局变量。
- 数据结构：样本乘以特征矩阵。
- 是否适合高维数据：适合探索，但距离计算和最短路径在大样本下成本较高。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：需先定义距离或样本单位，普通 Isomap 不建模时间相关性。

### 3.2 示例表格

以单细胞表达特征为例：

| GeneSet_1 | GeneSet_2 | GeneSet_3 | GeneSet_4 | CellType |
| --- | --- | --- | --- | --- |
| 1.20 | 0.33 | -0.42 | 0.91 | Progenitor |
| 1.05 | 0.40 | -0.35 | 0.82 | Progenitor |
| -0.21 | 1.44 | 0.18 | 0.12 | Mature |
| -0.33 | 1.30 | 0.22 | 0.20 | Mature |

### 3.3 输入与产出

#### 输入

- 输入数据：连续型特征矩阵。
- 关键变量：近邻数、嵌入维度、距离度量。
- 需要预处理的内容：缺失处理、标准化、异常值检查、近邻图连通性检查。

#### 产出

- 模型对象/统计结果：低维嵌入坐标、近邻图、重构误差。
- 参数估计：测地距离矩阵和 MDS 嵌入。
- 预测结果：通常不直接预测，可用于探索或后续建模输入。
- 不确定性指标：参数敏感性、邻域保持指标、重采样嵌入稳定性。

## 4. 适用场景

- 适合：全局流形距离有意义、数据可能呈弯曲低维结构、样本覆盖较均匀的场景。
- 不适合：样本量很大且距离计算昂贵、近邻图容易断裂、噪声造成跨流形捷径的场景。
- 使用前需要特别检查的点：近邻图是否连通，近邻数选择是否稳定，二维嵌入是否过度解释。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.manifold import Isomap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("single_cell_features.csv")
features = ["GeneSet_1", "GeneSet_2", "GeneSet_3", "GeneSet_4"]
X = df[features]

isomap_pipe = make_pipeline(
    StandardScaler(),
    Isomap(n_neighbors=10, n_components=2)
)

embedding = isomap_pipe.fit_transform(X)
embedding_df = pd.DataFrame(embedding, columns=["Iso1", "Iso2"])
print(embedding_df.head())
```

### 5.2 R

常用包：

- `vegan`

```r
library(vegan)

x <- scale(df[, c("GeneSet_1", "GeneSet_2", "GeneSet_3", "GeneSet_4")])
dist_mat <- dist(x)
fit <- isomap(dist_mat, ndim = 2, k = 10)

embedding <- scores(fit)
head(embedding)
```

## 6. 结果如何解释

- 核心结果看什么：低维嵌入中样本沿流形的排列、全局距离是否被合理保留、分组或连续指标是否沿嵌入变化。
- 每个主要参数如何解释：`n_neighbors` 决定测地距离估计的局部尺度。
- 临床或医学意义如何表达：Isomap 坐标适合描述样本相对关系和潜在连续谱，不宜直接命名为临床指标。
- 常见误读：图上距离近不一定代表所有原始特征都相似，而是算法定义的流形距离近。

## 7. 推荐可视化

- Isomap 二维嵌入散点图。
- 不同近邻数下的嵌入对比图。
- 残差方差随嵌入维度变化曲线。
- 近邻图连通性或最短路径距离诊断图。

## 8. 优势、局限与常见坑

### 优势

- 能捕捉非线性流形上的全局距离结构。
- 算法思路清晰，连接近邻图和经典 MDS。
- 对某些弯曲流形的展开效果直观。

### 局限

- 对近邻数和噪声边敏感。
- 大样本下计算测地距离矩阵成本高。
- 离群点可能破坏近邻图和最短路径。

### 常见坑

- 近邻图不连通时仍直接解释嵌入。
- 把 Isomap 二维图当作稳定聚类证据。
- 忽略标准化导致距离被量纲较大的变量主导。

## 9. 与相近方法的区别

- 和 [[局部线性嵌入（Locally Linear Embedding, LLE）]] 的区别：LLE 保持局部重构权重，Isomap 保持图上测地距离。
- 和 [[多维尺度分析（Multidimensional Scaling, MDS）]] 的区别：MDS 可直接对距离矩阵降维，Isomap 先用近邻图把欧氏距离转换为测地距离。
- 和 [[UMAP（Uniform Manifold Approximation and Projection）]] 的区别：UMAP 更强调近邻图的模糊拓扑结构，通常更适合大规模可视化。

## 10. 医学研究中的典型应用

- 单细胞或组学数据中潜在状态轨迹的探索。
- 医学影像表型中连续病变谱的可视化。
- 多维临床表型中非线性患者相似性展示。

## 11. 相关方法

- [[局部线性嵌入（Locally Linear Embedding, LLE）]]
- [[多维尺度分析（Multidimensional Scaling, MDS）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]
- [[核主成分分析（Kernel Principal Component Analysis, KPCA）]]

## 12. 参考资料

- Tenenbaum JB, de Silva V, Langford JC. A global geometric framework for nonlinear dimensionality reduction. *Science*. 2000;290(5500):2319-2323.
- Cox TF, Cox MAA. *Multidimensional Scaling*. 2nd ed. Chapman and Hall/CRC; 2000.
- scikit-learn Developers. `sklearn.manifold.Isomap`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html) （访问日期：2026-07-02）
