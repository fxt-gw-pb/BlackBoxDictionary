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

### 1.1 一句话本质

LLE 用每个样本由近邻线性重构的权重描述局部几何，再到低维空间寻找一组仍能被同样权重重构的坐标。

### 1.2 定义

局部线性嵌入是一种无监督非线性流形学习方法。它不直接保持所有成对距离，而是先估计局部重构权重，再求解一个特征值问题得到低维嵌入。

### 1.3 它主要解决什么问题

- 高维数据沿弯曲流形分布，线性 PCA 难以展开。
- 研究重点是局部邻域关系，而非原空间的全局欧氏距离。
- 需要以较少维度表示影像、组学或连续疾病谱。

### 1.4 直觉与类比

把一张弯曲但未撕裂的网格纸看成高维流形。每个交点都可由附近交点按固定比例“配出来”；LLE 展平网格时保留这些配方，因此局部形状得以延续。

## 2. 核心思想与原理

### 2.1 根本困难

全局直线不能贴合弯曲流形，但足够小的局部区域通常近似平面。困难在于怎样把许多局部平面拼成一致的全局低维坐标。

### 2.2 关键洞察

仿射重构权重对平移、旋转和统一缩放不敏感。先固定每个点与近邻之间的相对配方，再让所有低维坐标共同满足这些配方，就能把局部信息拼接起来。

### 2.3 三步算法

1. 为每个样本寻找 $k$ 个近邻。
2. 在和为 1 的约束下，求近邻重构原样本的权重。
3. 固定权重，求使低维重构误差最小且已中心化、定标的坐标。

## 3. 数学形式

### 3.1 高维局部重构

令 $\mathcal N_i$ 为样本 $x_i$ 的近邻集合：

$$
\operatorname*{minimize}_{W}
\sum_{i=1}^{n}
\left\|
x_i-\sum_{j\in\mathcal N_i}w_{ij}x_j
\right\|^2
$$

约束为：

$$
\sum_{j\in\mathcal N_i}w_{ij}=1,
\qquad
w_{ij}=0\quad(j\notin\mathcal N_i)
$$

和为 1 使重构关系对整体平移保持不变。

### 3.2 低维嵌入

固定 $W$ 后求坐标 $y_i\in\mathbb R^d$：

$$
\operatorname*{minimize}_{Y}
\sum_{i=1}^{n}
\left\|
y_i-\sum_jw_{ij}y_j
\right\|^2
$$

通常加入 $\sum_i y_i=0$ 与 $Y^\top Y=I$，排除全部坐标相同、任意平移和任意缩放等退化解。

### 3.3 特征值问题

记：

$$
M=(I-W)^\top(I-W)
$$

目标可写为 $\operatorname{tr}(Y^\top MY)$。$M$ 的最小特征值对应常数向量，应丢弃；随后 $d$ 个最小非零特征向量构成嵌入。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 流形局部近似线性 | 重构权重不稳定 | 局部重构误差 |
| 近邻图连通 | 各连通块位置任意 | 连通分量数量 |
| $k$ 与局部维数匹配 | 太小易断裂，太大跨越弯折 | 多组 $k$ 敏感性分析 |
| 距离与预处理合理 | 邻居关系被量纲主导 | 标准化与距离比较 |
| 样本覆盖流形较充分 | 稀疏区域被错误拉伸 | 邻域半径与密度图 |

## 4. 手把手算例

考虑一维局部的三个点：

$$
x_1=0,\qquad x_2=1,\qquad x_3=2
$$

用 $x_1,x_3$ 重构中间点 $x_2$。

**Step 1：写出约束。**

$$
w_{21}+w_{23}=1
$$

令 $w_{23}=w$，则 $w_{21}=1-w$。

**Step 2：代入重构误差。**

$$
\left[
1-(1-w)\times0-w\times2
\right]^2
=(1-2w)^2
$$

误差在 $w=0.5$ 时为 0，所以：

$$
w_{21}=w_{23}=0.5
$$

**Step 3：在低维复现同一配方。**

若低维端点为 $y_1=-1,y_3=1$，则：

$$
y_2=0.5(-1)+0.5(1)=0
$$

因此 $(-1,0,1)$ 完全保留了中间点的局部重构关系。实际 LLE 会同时处理所有点，并用中心化、定标约束确定整体嵌入。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 输入为样本乘特征矩阵，连续特征最常见。
- 高维影像向量或组学数据常先用 PCA 去噪。
- 缺失值需先插补；LLE 本身不处理删失或结局变量。
- 重复测量若被逐次作为独立样本，邻域可能主要反映个体身份，应先明确分析单位。

### 5.2 示例表格

| Patient | Marker_1 | Marker_2 | Marker_3 | Stage |
| --- | ---: | ---: | ---: | --- |
| P01 | 0.2 | 1.1 | 2.0 | early |
| P02 | 0.5 | 1.4 | 1.7 | early |
| P03 | 1.3 | 0.8 | 0.9 | late |
| P04 | 1.7 | 0.3 | 0.4 | late |

`Stage` 仅用于嵌入后着色，不参与无监督 LLE 拟合。

### 5.3 输入与产出

输入包括特征矩阵、近邻数、目标维数、LLE 变体与特征求解器。输出包括低维坐标、重构误差及可用于新样本近似投影的拟合对象；坐标轴本身通常没有直接变量含义。

## 6. 适用场景

- 适合采样较密、局部连续且预期位于弯曲低维流形的数据。
- 可用于影像姿态、连续细胞状态和疾病进展轨迹的探索。
- 不适合强噪声、离散簇为主、样本稀疏或需要严格统计推断的场景。
- 若目标是预测，LLE 必须在每个训练折内拟合，不能先对全数据降维。

## 7. 实现

### 7.1 Python

```python
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding, trustworthiness
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = df[["Marker_1", "Marker_2", "Marker_3"]]

model = make_pipeline(
    StandardScaler(),
    LocallyLinearEmbedding(
        n_neighbors=12,
        n_components=2,
        method="standard",
        eigen_solver="auto",
        random_state=42,
        n_jobs=-1,
    ),
)
embedding = model.fit_transform(X)

X_scaled = model.named_steps["standardscaler"].transform(X)
score = trustworthiness(X_scaled, embedding, n_neighbors=12)
result = pd.DataFrame(embedding, columns=["LLE1", "LLE2"])
print(score, result.head())
```

### 7.2 R

```r
library(lle)

x <- scale(df[, c("Marker_1", "Marker_2", "Marker_3")])
fit <- lle(
  x,
  m = 2,
  k = 12
)

embedding <- fit$Y
head(embedding)
```

## 8. 结果如何解读

- 重点解释谁与谁相邻、连续轨迹是否保留，不解释 LLE1 的单位或正负。
- 图形可任意旋转、镜像或平移；不同运行之间应先对齐再比较。
- 分开的点群可能源于近邻图断裂，不自动等于疾病亚型。
- 若用临床标签着色，应说明标签未参与无监督拟合。

## 9. 假设诊断与稳健性

1. 检查近邻图连通分量和孤立点。
2. 比较多组 `n_neighbors`、目标维数与随机种子。
3. 报告局部重构误差和 trustworthiness。
4. 观察每个样本第 $k$ 近邻距离，识别稀疏区与异常点。
5. 与 PCA、Isomap、UMAP 及原空间近邻重叠率比较。
6. 下游监督任务用嵌套交叉验证，所有预处理只在训练折拟合。

## 10. 推荐可视化

- LLE1-LLE2 散点图，按独立临床变量着色。
- 近邻图叠加图，标出断裂和跨越边。
- `n_neighbors` 敏感性小多图。
- 原空间与嵌入空间近邻重叠率分布。

## 11. 优势、局限与常见坑

**优势：** 非参数地利用局部线性结构；优化可归结为特征分解；对某些平滑流形有清晰几何解释。

**局限：** 对近邻选择、噪声和采样密度敏感；标准 LLE 不强调全局距离；大样本近邻搜索和特征分解成本较高。

**常见坑：** 用过大的 $k$ 连上流形两侧；忽略断开的近邻图；把轴当生物学变量；只展示最好看的一组参数；全数据拟合后再交叉验证。

## 12. 与相近方法的区别

- [[Isomap（Isometric Mapping）]]：保持近邻图上的测地距离；LLE 保持局部线性重构权重。
- [[主成分分析（Principal Component Analysis, PCA）]]：PCA 用一个全局线性子空间；LLE 拼接许多局部线性关系。
- [[UMAP（Uniform Manifold Approximation and Projection）]]：UMAP 优化模糊近邻图交叉熵，通常更适合大规模可视化。
- **如何选择：** 相信局部仿射结构且样本覆盖密集时考虑 LLE；重视全局测地关系时考虑 Isomap；先以 PCA 作为线性基线。

## 13. 医学研究中的典型应用

- 影像表型随姿态或病变进展的连续流形。
- 单细胞或组学数据中的连续状态探索。
- 多指标临床表型的非线性二维展示。

这些应用均应把嵌入视为探索性表征，并用原始变量、外部队列或实验信息验证。

## 14. 关键术语

| 术语 | 含义 |
| --- | --- |
| manifold 流形 | 高维空间中局部近似低维平面的结构 |
| neighborhood 近邻 | 按给定距离与某样本最接近的一组样本 |
| reconstruction weight 重构权重 | 用近邻线性组合还原中心点的系数 |
| affine invariance 仿射不变性 | 局部配方对平移、旋转和统一缩放保持稳定 |
| embedding 嵌入 | 把对象映射到低维坐标的表示 |
| trustworthiness 可信度 | 低维近邻在原空间中仍为近邻的程度 |

## 15. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[Isomap（Isometric Mapping）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]
- [[t-SNE（t-Distributed Stochastic Neighbor Embedding）]]
- [[多维尺度分析（Multidimensional Scaling, MDS）]]

## 16. 参考资料

- Roweis ST, Saul LK. Nonlinear dimensionality reduction by locally linear embedding. *Science*. 2000;290(5500):2323-2326.
- Saul LK, Roweis ST. Think globally, fit locally: unsupervised learning of low dimensional manifolds. *Journal of Machine Learning Research*. 2003;4:119-155.
- scikit-learn Developers. `LocallyLinearEmbedding` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html) （访问日期：2026-07-09）
