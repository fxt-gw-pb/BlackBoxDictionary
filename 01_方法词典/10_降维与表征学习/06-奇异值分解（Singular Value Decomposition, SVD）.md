---
title: 奇异值分解
english_name: Singular Value Decomposition, SVD
slug: singular-value-decomposition-svd
aliases: [SVD, singular value decomposition, truncated SVD, "奇异值分解（Singular Value Decomposition, SVD）"]
category: 降维与表征学习
subcategory: 矩阵分解
tags: [医学统计, 数据科学, 降维, 矩阵分解, 特征压缩]
status: 已建
difficulty: intermediate
question_type: 矩阵低秩近似与特征压缩
data_type: [高维特征矩阵, 稀疏矩阵, 图像矩阵]
outcome_type: [无监督表征, 多种]
python_packages: [numpy, scikit-learn]
r_packages: [base, irlba]
---

# 奇异值分解（Singular Value Decomposition, SVD）

## 1. 方法概览

### 1.1 定义

奇异值分解是一种基础矩阵分解方法，可以把任意矩阵分解为左奇异向量、奇异值和右奇异向量三部分。截断 SVD 使用最大的若干奇异值构造低秩近似，从而实现降维、压缩和去噪。

### 1.2 它主要解决什么问题

- 研究问题：如何用低秩结构近似一个高维数据矩阵。
- 适用任务：特征压缩、矩阵降噪、潜在语义分析、图像压缩、稀疏矩阵降维。
- 常见医学场景：影像矩阵压缩，组学矩阵低秩表示，临床文本词频矩阵潜在主题探索。

### 1.3 直觉理解

SVD 把数据矩阵拆成一组按重要性排序的“基础模式”。最大的奇异值对应最主要的变化模式，只保留前几个模式就能近似原矩阵，并丢掉一部分噪声和冗余。

## 2. 数学形式

### 2.1 核心公式

任意矩阵 $X\in\mathbb{R}^{n\times p}$ 可分解为：

$$
X=U\Sigma V^\top
$$

其中 $U^\top U=I$，$V^\top V=I$，$\Sigma$ 为非负奇异值构成的对角矩阵。截断到前 $k$ 个奇异值：

$$
X_k=U_k\Sigma_k V_k^\top
$$

根据 Eckart-Young 定理，$X_k$ 是 Frobenius 范数意义下最优的秩 $k$ 近似：

$$
X_k=\arg\min_{\operatorname{rank}(B)=k}\|X-B\|_F
$$

累计能量比例常写为：

$$
\frac{\sum_{j=1}^{k}\sigma_j^2}{\sum_{j=1}^{r}\sigma_j^2}
$$

### 2.2 参数或统计量含义

- $U$：左奇异向量，表示样本方向的正交基。
- $\Sigma$：奇异值，表示每个模式的重要性。
- $V$：右奇异向量，表示特征方向的正交基。
- $k$：保留的低秩维度。
- 截断误差：原矩阵与低秩近似之间的差异。

### 2.3 关键假设

- 数据矩阵存在可利用的低秩结构。
- 主要信息集中在较大的奇异值对应的方向上。
- 输入尺度会影响分解结果，通常需根据场景决定是否中心化或标准化。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续特征矩阵、稀疏计数矩阵、图像像素矩阵。
- 因变量类型：SVD 本身不需要结局变量。
- 数据结构：样本乘以特征矩阵，或任意二维矩阵。
- 是否适合高维数据：适合，尤其适合稀疏高维矩阵的截断分解。
- 是否适合缺失较多数据：普通 SVD 不直接处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：可分解整理后的矩阵，但不建模相关结构。

### 3.2 示例表格

以临床文本词频矩阵为例：

| PatientID | fever | cough | pain | tumor | treatment |
| --- | --- | --- | --- | --- | --- |
| P001 | 2 | 4 | 0 | 0 | 1 |
| P002 | 0 | 1 | 3 | 2 | 4 |
| P003 | 3 | 5 | 0 | 0 | 0 |
| P004 | 0 | 0 | 4 | 3 | 5 |

### 3.3 输入与产出

#### 输入

- 输入数据：数值矩阵或稀疏矩阵。
- 关键变量：保留维度 $k$、是否中心化、是否使用随机化 SVD。
- 需要预处理的内容：缺失处理、标准化或归一化、稀疏矩阵构造。

#### 产出

- 模型对象/统计结果：奇异值、左右奇异向量、低维表示、低秩重构矩阵。
- 参数估计：$U_k$、$\Sigma_k$、$V_k$。
- 预测结果：不直接预测，可作为后续模型输入。
- 不确定性指标：可用重采样、重构误差或交叉验证选择 $k$。

## 4. 适用场景

- 适合：矩阵存在低秩结构、需要压缩或去噪、输入为稀疏高维矩阵的场景。
- 不适合：主要结构强非线性、缺失值大量存在且未建模、需要直接解释原始变量效应的场景。
- 使用前需要特别检查的点：是否中心化，奇异值衰减是否明显，低秩近似是否保留任务相关信息。

## 5. 实现

### 5.1 Python

常用包：

- `numpy`
- `scikit-learn`

```python
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

df = pd.read_csv("clinical_term_matrix.csv")
X = df[["fever", "cough", "pain", "tumor", "treatment"]]

svd_pipe = make_pipeline(
    TruncatedSVD(n_components=2, random_state=42),
    Normalizer(copy=False)
)

Z = svd_pipe.fit_transform(X)
svd = svd_pipe.named_steps["truncatedsvd"]

print("Explained variance ratio:", svd.explained_variance_ratio_)
print(pd.DataFrame(Z, columns=["SVD1", "SVD2"]).head())
```

### 5.2 R

常用包：

- `irlba`

```r
library(irlba)

x <- as.matrix(df[, c("fever", "cough", "pain", "tumor", "treatment")])
fit <- irlba(x, nv = 2, nu = 2)

scores <- fit$u %*% diag(fit$d)
loadings <- fit$v
head(scores)
fit$d
```

## 6. 结果如何解释

- 核心结果看什么：奇异值谱、累计能量比例、低维得分、右奇异向量中的主要变量贡献。
- 每个主要参数如何解释：`n_components=2` 表示只保留前两个主导低秩模式。
- 临床或医学意义如何表达：低维成分是原始变量的线性组合，需要结合载荷较大的变量和业务背景命名。
- 常见误读：奇异值最大只表示矩阵能量最大，不一定表示最能预测临床结局。

## 7. 推荐可视化

- 奇异值谱图。
- 累计能量比例曲线。
- 前两个 SVD 维度散点图。
- 重构误差随 $k$ 变化曲线。

## 8. 优势、局限与常见坑

### 优势

- 适用于任意矩阵，理论基础清晰。
- 截断 SVD 能提供最优低秩近似。
- 对稀疏高维矩阵特别实用。

### 局限

- 成分可解释性有限。
- 线性低秩结构不能表达复杂非线性流形。
- 对尺度和中心化策略敏感。

### 常见坑

- 把 SVD 和 PCA 完全等同，忽略中心化和应用场景差异。
- 在全数据上做 SVD 后再划分训练测试集。
- 只看二维图，不检查保留能量和重构误差。

## 9. 与相近方法的区别

- 和 [[主成分分析（Principal Component Analysis, PCA）]] 的区别：PCA 通常对中心化数据的协方差结构做降维，SVD 是更一般的矩阵分解工具。
- 和 [[因子分析（Factor Analysis）]] 的区别：因子分析是带误差结构和潜变量解释的统计模型，SVD 更偏代数低秩近似。
- 和 [[核主成分分析（Kernel Principal Component Analysis, KPCA）]] 的区别：KPCA 通过核矩阵处理非线性结构，SVD 本身是线性分解。

## 10. 医学研究中的典型应用

- 影像矩阵或影像组学特征压缩。
- 临床文本词频矩阵的潜在语义降维。
- 组学表达矩阵的低秩去噪和主要模式提取。

## 11. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[因子分析（Factor Analysis）]]
- [[核主成分分析（Kernel Principal Component Analysis, KPCA）]]
- [[多维尺度分析（Multidimensional Scaling, MDS）]]

## 12. 参考资料

- Golub GH, Van Loan CF. *Matrix Computations*. 4th ed. Johns Hopkins University Press; 2013.
- Eckart C, Young G. The approximation of one matrix by another of lower rank. *Psychometrika*. 1936;1:211-218.
- scikit-learn Developers. `sklearn.decomposition.TruncatedSVD`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) （访问日期：2026-07-02）
