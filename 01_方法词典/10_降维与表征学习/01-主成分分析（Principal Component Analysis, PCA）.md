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

### 1.1 一句话本质

PCA 旋转坐标轴，依次寻找样本投影方差最大的正交方向，用少数线性组合保留主要总变异。

### 1.2 定义

主成分分析是无监督线性降维方法。它将中心化数据投影到一组两两正交的主成分方向，并按解释方差从大到小排序；前若干方向提供低维表示和最小平方重构误差。

### 1.3 它主要解决什么问题

- 多个连续特征高度相关，信息重复。
- 高维数据需要二维可视化、压缩或去噪。
- 后续模型受共线性或计算成本影响。

### 1.4 直觉与类比

一团斜着延伸的点云用横纵坐标描述很浪费。PCA 把第一根新轴沿点云最长方向放置，第二根轴与它垂直；若点几乎都在第一轴上，第二轴可以舍弃。

## 2. 核心思想与原理

### 2.1 最大方差与最小重构误差

在单位方向中，第一主成分使投影方差最大。保留前 $k$ 个主成分，也等价于寻找重构平方误差最小的 $k$ 维线性子空间。

### 2.2 无监督意味着什么

PCA 不看结局。方差最大的方向可能是批次效应、测量噪声或个体差异，而不一定最能预测疾病。

### 2.3 中心化与标准化

中心化是 PCA 的基本步骤。是否标准化取决于变量单位与研究问题：单位不同或方差不可比时通常标准化；同单位且方差大小有实际意义时可只中心化。

## 3. 数学形式

### 3.1 协方差矩阵

对中心化矩阵 $\widetilde X\in\mathbb R^{n\times p}$：

$$
S=\frac{1}{n-1}\widetilde X^\top\widetilde X
$$

### 3.2 最大方差问题

$$
w_1=
\operatorname*{arg\,max}_{\|w\|=1}
w^\top Sw
$$

解满足：

$$
Sw_j=\lambda_jw_j
$$

### 3.3 得分与解释方差

$$
Z=\widetilde XW_k
$$

$$
\operatorname{EVR}_j=
\frac{\lambda_j}{\sum_{\ell=1}^{p}\lambda_\ell}
$$

载荷方向的符号可整体翻转而不改变解。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 主要结构近似线性 | 弯曲流形被压坏 | 与非线性方法比较 |
| 高方差代表有用信息 | 批次或噪声占主成分 | 载荷与协变量检查 |
| 尺度处理符合问题 | 大方差变量支配 | 标准化敏感性 |
| 训练流程无泄漏 | 测试信息进入变换 | 在训练折内拟合 PCA |

## 4. 手把手算例

四个已中心化的二维点：

$$
(-2,-2),\ (-1,-1),\ (1,1),\ (2,2)
$$

**Step 1：协方差矩阵。** 两列平方和与乘积和均为 10：

$$
S=
\frac13
\begin{pmatrix}
10&10\\
10&10
\end{pmatrix}
$$

**Step 2：特征值与方向。**

$$
\lambda_1=\frac{20}{3},\qquad
w_1=\frac{1}{\sqrt2}(1,1)^\top
$$

$$
\lambda_2=0,\qquad
w_2=\frac{1}{\sqrt2}(-1,1)^\top
$$

**Step 3：主成分得分。**

$$
z_1=\widetilde Xw_1=
(-2\sqrt2,-\sqrt2,\sqrt2,2\sqrt2)^\top
$$

第二主成分得分全部为 0。

**Step 4：解释方差。**

$$
\operatorname{EVR}_1=
\frac{20/3}{20/3+0}=1
$$

**结论：** 所有点都在直线 $x_1=x_2$ 上，一个主成分即可无损表示二维数据。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 连续数值特征或合理变换后的高维矩阵。
- 缺失需预先处理；异常值会明显影响均值和协方差。
- 重复测量应明确以患者还是访视为样本单位。

### 5.2 输入与产出

输入为矩阵、中心化/标准化方案和保留维度。输出为主成分得分、方向/载荷、特征值、解释方差比例及可选重构。

## 6. 适用场景

- 高相关连续特征压缩、可视化和去噪。
- 组学、影像组学和多指标综合表征。
- 不适合强非线性、需要原变量直接解释或低方差信号很重要的任务。

## 7. 实现

### 7.1 Python

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.90, svd_solver="full"),
)
Z_train = pipeline.fit_transform(X_train)
Z_test = pipeline.transform(X_test)

pca = pipeline.named_steps["pca"]
print(pca.n_components_)
print(pca.explained_variance_ratio_)
print(pca.components_)
```

### 7.2 R

```r
x_train <- train[, feature_names]
x_test <- test[, feature_names]

fit <- prcomp(
  x_train,
  center = TRUE,
  scale. = TRUE
)

scores_train <- fit$x
scores_test <- predict(fit, newdata = x_test)
loadings <- fit$rotation
summary(fit)
```

## 8. 结果如何解释

- 得分是患者在新轴上的坐标；载荷说明原变量如何组成该轴。
- 解释方差大不等于临床预测能力强。
- 载荷符号可整体翻转，解释应看相对方向与大小。
- 二维图上的重叠或分离只是探索证据，不是显著性检验。

## 9. 诊断与稳健性

1. 画 scree 和累计解释方差曲线。
2. 检查载荷是否由单一变量、批次或异常样本主导。
3. 比较中心化与标准化方案。
4. bootstrap 后对齐符号，评估载荷稳定性。
5. 在交叉验证每个训练折内拟合 PCA，再评价下游任务。

## 10. 推荐可视化

- scree plot 与累计解释方差曲线。
- PC1-PC2 得分图。
- 载荷热图或 biplot。
- 重构误差随维度变化图。

## 11. 优势、局限与常见坑

**优势：** 理论清晰、计算高效、正交去冗余，并给出最优线性低秩近似。

**局限：** 只捕捉线性总方差，对尺度和异常值敏感，成分可能难命名。

**常见坑：** 全数据先拟合造成泄漏；默认不标准化；把 PCA 当变量选择；把 PC1 叫疾病严重度却不看载荷；只凭二维图判定亚型。

## 12. 与相近方法的区别

- [[奇异值分解（Singular Value Decomposition, SVD）]]：PCA 常通过中心化矩阵 SVD 实现，SVD 本身不要求中心化。
- [[因子分析（Factor Analysis）]]：PCA 解释总方差，因子分析建模共同方差与特异误差。
- [[独立成分分析（Independent Component Analysis, ICA）]]：PCA 要不相关正交方向，ICA 要统计独立来源。
- 选择经验：压缩和可视化先用 PCA；潜变量测量结构应考虑因子分析。

## 13. 医学研究中的典型应用

- 组学样本结构、批次效应和主要变异探索。
- 影像组学特征压缩后进入预测模型。
- 多个相关实验室指标的综合表征。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| score | 样本在主成分轴上的坐标 |
| loading | 原变量构成主成分方向的权重 |
| explained variance | 主成分承载的样本方差 |
| scree plot | 特征值或解释方差按成分排序图 |
| whitening | 将主成分进一步缩放到单位方差 |

## 15. 相关方法

- [[奇异值分解（Singular Value Decomposition, SVD）]]
- [[独立成分分析（Independent Component Analysis, ICA）]]
- [[因子分析（Factor Analysis）]]
- [[核主成分分析（Kernel Principal Component Analysis, KPCA）]]

## 16. 参考资料

- Pearson K. On lines and planes of closest fit to systems of points in space. *Philos Mag*. 1901;2(11):559-572.
- Jolliffe IT, Cadima J. Principal component analysis: a review and recent developments. *Philos Trans A*. 2016;374:20150202.
- scikit-learn Developers. `PCA` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) （访问日期：2026-07-09）
