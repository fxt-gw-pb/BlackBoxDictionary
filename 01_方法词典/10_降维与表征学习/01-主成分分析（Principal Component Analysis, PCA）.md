---
title: 主成分分析
english_name: Principal Component Analysis, PCA
slug: principal-component-analysis-pca
aliases: [PCA, principal component analysis, "主成分分析（Principal Component Analysis, PCA）"]
category: 降维与表征学习
subcategory: 线性降维
tags: [医学统计, 数据科学, 降维, 特征工程, 无监督学习]
status: 已建
difficulty: intermediate
question_type: 高维连续特征线性降维
data_type: [表格数据, 高维特征矩阵, 组学数据]
outcome_type: [无监督表征, 多种]
python_packages: [scikit-learn]
r_packages: [stats, FactoMineR]
---

# 主成分分析（Principal Component Analysis, PCA）

## 1. 方法概览

### 1.1 定义

主成分分析是一种线性降维方法。它把原始相关特征转换为一组互相正交的新变量，即主成分，并按解释方差从大到小排序，用少数主成分保留数据中的主要变异。

### 1.2 它主要解决什么问题

- 研究问题：如何用更少的综合变量表示高维连续特征。
- 适用任务：降维、可视化、去冗余、建模前特征压缩。
- 常见医学场景：组学数据探索、影像组学特征压缩、多个相关实验室指标综合表征。

### 1.3 直觉理解

PCA 不直接挑选原始变量，而是把多个相关变量重新组合成新的坐标轴。第一主成分是数据变化最大的方向，第二主成分是在与第一主成分正交的条件下变化第二大的方向。

## 2. 数学形式

### 2.1 核心公式

设中心化后的数据矩阵为 $\tilde X\in\mathbb{R}^{n\times p}$，协方差矩阵为：

$$
S=\frac{1}{n-1}\tilde X^\top \tilde X
$$

PCA 寻找单位向量 $w$，使投影方差最大：

$$
\max_w w^\top S w,\quad \text{s.t. } w^\top w=1
$$

其解满足特征值方程：

$$
S w_j=\lambda_j w_j
$$

选择前 $k$ 个特征向量组成 $W_k$，降维后的表示为：

$$
Z=\tilde X W_k
$$

累计解释方差比例为：

$$
\frac{\sum_{j=1}^{k}\lambda_j}{\sum_{j=1}^{p}\lambda_j}
$$

### 2.2 参数或统计量含义

- 主成分：原始变量的线性组合。
- 载荷：每个原始变量在主成分中的权重。
- 特征值 $\lambda_j$：第 $j$ 个主成分解释的方差。
- `n_components`：保留的主成分数量。
- 解释方差比例：每个主成分保留的信息比例。

### 2.3 关键假设

- 主要结构可由线性组合表示。
- 方差较大的方向被视为更重要的信息来源。
- 特征尺度会影响结果，通常需要标准化。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续变量或标准化后的数值特征。
- 因变量类型：PCA 本身不需要结局变量。
- 数据结构：样本乘以特征矩阵。
- 是否适合高维数据：适合，尤其适合相关特征较多的数据。
- 是否适合缺失较多数据：需先插补或使用专门 PCA 变体。
- 是否适合删失数据：PCA 不处理结局删失。
- 是否适合重复测量数据：需先定义样本单位，或使用纵向降维方法。

### 3.2 示例表格

以代谢组学特征降维为例：

| Metab_1 | Metab_2 | Metab_3 | Metab_4 | Disease |
| --- | --- | --- | --- | --- |
| 0.84 | 1.22 | -0.14 | 0.31 | 1 |
| -0.35 | -0.41 | 0.28 | -0.19 | 0 |
| 1.10 | 1.45 | -0.33 | 0.52 | 1 |
| -0.72 | -0.60 | 0.18 | -0.42 | 0 |
| 0.41 | 0.80 | -0.09 | 0.20 | 1 |

### 3.3 输入与产出

#### 输入

- 输入数据：数值型特征矩阵。
- 关键变量：保留主成分数量、标准化方式。
- 需要预处理的内容：缺失处理、异常值检查、中心化和标准化。

#### 产出

- 模型对象/统计结果：主成分载荷、解释方差比例、主成分得分。
- 参数估计：载荷矩阵和特征值。
- 预测结果：PCA 本身不预测，可把主成分作为后续模型输入。
- 不确定性指标：可用重采样评估主成分稳定性。

## 4. 适用场景

- 适合：连续特征多、特征之间相关性强、需要可视化或降维的场景。
- 不适合：希望保留原始变量解释、主要结构非线性、特征尺度不可比且不宜标准化的场景。
- 使用前需要特别检查的点：是否标准化、主成分是否可解释、保留方差比例是否足够。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("metabolomics.csv")
X = df[["Metab_1", "Metab_2", "Metab_3", "Metab_4"]]

pca_pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.90)
)

Z = pca_pipe.fit_transform(X)
pca = pca_pipe.named_steps["pca"]

print("Number of components:", pca.n_components_)
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

### 5.2 R

常用包：

- `stats`

```r
x <- df[, c("Metab_1", "Metab_2", "Metab_3", "Metab_4")]

fit <- prcomp(x, center = TRUE, scale. = TRUE)
summary(fit)

scores <- fit$x[, 1:2]
loadings <- fit$rotation[, 1:2]
```

## 6. 结果如何解释

- 核心结果看什么：前几个主成分解释方差比例、载荷矩阵、样本在主成分空间的分布。
- 每个主要参数如何解释：`n_components=0.90` 表示保留能解释 90% 方差的主成分。
- 临床或医学意义如何表达：主成分是综合特征，不是单个原始变量；解释时需结合载荷较大的变量。
- 常见误读：第一主成分解释方差最大，不代表它一定最能预测结局。

## 7. 推荐可视化

- Scree plot 碎石图。
- 累计解释方差曲线。
- PC1-PC2 样本散点图。
- 主成分载荷热图。

## 8. 优势、局限与常见坑

### 优势

- 能压缩相关连续特征。
- 主成分之间正交，缓解共线性。
- 有助于高维数据可视化。

### 局限

- 新特征可解释性弱于原始变量。
- 只捕捉线性结构。
- 对尺度和异常值敏感。

### 常见坑

- 在训练测试划分之外先 fit PCA，造成信息泄露。
- 不标准化就对量纲差异很大的变量做 PCA。
- 把 PCA 当成“选择原始变量”的方法。

## 9. 与相近方法的区别

- 和 [[相关系数特征选择（Correlation-based Feature Selection）]] 的区别：相关系数法删除冗余原始变量；PCA 构造新的综合变量。
- 和 [[Lasso回归（Lasso Regression）]] 的区别：Lasso 做监督式稀疏选择；PCA 是无监督降维。
- 和 [[高斯混合模型（Gaussian Mixture Model, GMM）]] 的区别：PCA 用于降维和表征；GMM 用于软聚类和密度建模。

## 10. 医学研究中的典型应用

- 组学数据主结构探索和样本可视化。
- 影像组学特征压缩后进入预测模型。
- 多个相关实验室指标合成为少数综合表征。

## 11. 相关方法

- [[相关系数特征选择（Correlation-based Feature Selection）]]
- [[Lasso回归（Lasso Regression）]]
- [[Ridge回归（Ridge Regression）]]
- [[高斯混合模型（Gaussian Mixture Model, GMM）]]

## 12. 参考资料

- Jolliffe IT. *Principal Component Analysis*. 2nd ed. Springer; 2002.
- Pearson K. On lines and planes of closest fit to systems of points in space. *Philos Mag*. 1901;2(11):559-572.
- scikit-learn Developers. `sklearn.decomposition.PCA`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) （访问日期：2026-07-02）
