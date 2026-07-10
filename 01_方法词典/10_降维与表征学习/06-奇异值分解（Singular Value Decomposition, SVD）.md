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

### 1.1 一句话本质

SVD 把任意矩阵拆成样本方向、按强弱排序的奇异值和特征方向；保留前几项可得到最佳低秩近似。

### 1.2 定义

奇异值分解是基础矩阵分解。任意实矩阵都可写为 $U\Sigma V^\top$；截断 SVD 只保留最大的 $k$ 个奇异值和对应向量，用于压缩、去噪和潜在表示。

### 1.3 它主要解决什么问题

- 高维矩阵是否近似由少数模式生成。
- 如何获得最优平方误差低秩近似。
- 如何高效处理稀疏词频、图像和组学矩阵。

### 1.4 直觉与类比

一张复杂影像可看成多层基础图案叠加。奇异值是各层“能量”，先保留最强图案，再逐层补充细节；小奇异值常对应细节或噪声。

## 2. 核心思想与原理

### 2.1 左右两个空间

$U$ 描述样本/行方向，$V$ 描述特征/列方向，$\Sigma$ 将两者按一一对应的强度连接。SVD 不要求矩阵方阵或对称。

### 2.2 最优低秩近似

Eckart-Young-Mirsky 定理说明，截断前 $k$ 项在 Frobenius 范数和谱范数下给出最佳秩 $k$ 近似。

### 2.3 与 PCA 的联系

对中心化矩阵做 SVD，右奇异向量就是 PCA 方向，$\sigma_j^2/(n-1)$ 是协方差特征值。`TruncatedSVD` 默认不中心化，特别适合保持稀疏性。

## 3. 数学形式

### 3.1 完整分解

$$
X=U\Sigma V^\top
$$

$$
U^\top U=I,\qquad V^\top V=I
$$

### 3.2 截断分解

$$
X_k=
U_k\Sigma_kV_k^\top
=\sum_{j=1}^{k}\sigma_ju_jv_j^\top
$$

### 3.3 最优性与误差

$$
X_k=
\operatorname*{arg\,min}_{\operatorname{rank}(B)\le k}
\|X-B\|_F
$$

$$
\|X-X_k\|_F^2=
\sum_{j\gt k}\sigma_j^2
$$

保留能量比例为：

$$
\frac{\sum_{j=1}^{k}\sigma_j^2}
{\sum_{j=1}^{r}\sigma_j^2}
$$

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 矩阵近似低秩 | 奇异值衰减慢，压缩损失大 | 奇异值谱 |
| 大奇异值承载有用信息 | 批次或背景主导 | 方向与协变量检查 |
| 中心化策略合适 | SVD 与 PCA 含义混淆 | 对照中心化结果 |
| 训练流程无泄漏 | 测试结构进入表示 | 训练折内拟合 |

## 4. 手把手算例

考虑矩阵：

$$
X=
\begin{pmatrix}
3&0\\
0&1
\end{pmatrix}
$$

它已经是对角形式，因此：

$$
U=I,\quad
\Sigma=
\begin{pmatrix}
3&0\\
0&1
\end{pmatrix},
\quad V=I
$$

奇异值为 $\sigma_1=3,\sigma_2=1$。

**Step 1：秩 1 近似。**

$$
X_1=
\sigma_1u_1v_1^\top=
\begin{pmatrix}
3&0\\
0&0
\end{pmatrix}
$$

**Step 2：保留能量。**

$$
\frac{\sigma_1^2}{\sigma_1^2+\sigma_2^2}
=\frac9{10}=0.90
$$

**Step 3：重构误差。**

$$
\|X-X_1\|_F
=
\left\|
\begin{pmatrix}
0&0\\
0&1
\end{pmatrix}
\right\|_F
=1
$$

**结论：** 用一个秩 1 模式保留 90% 矩阵能量，丢失的恰是第二奇异值对应的强度 1。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 任意数值矩阵，尤其是高维稀疏矩阵、图像或样本-特征矩阵。
- 普通 SVD 不直接处理缺失。
- 中心化会破坏稀疏性，应按应用选择 PCA 或 Truncated SVD。

### 5.2 输入与产出

输入为矩阵、保留秩和求解器。输出为奇异值、左右奇异向量、低维得分、重构和误差。

## 6. 适用场景

- 低秩压缩、去噪、潜在语义分析和矩阵近似。
- 稀疏文本、组学和影像数据。
- 不适合主要结构非线性或必须直接解释原始变量效应的任务。

## 7. 实现

### 7.1 Python

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(
    n_components=50,
    algorithm="randomized",
    n_iter=7,
    random_state=42,
)
Z_train = svd.fit_transform(X_train)
Z_test = svd.transform(X_test)

energy = svd.explained_variance_ratio_.sum()
print(energy)
print(svd.singular_values_)
```

### 7.2 R

```r
library(irlba)

fit <- irlba(
  X_train,
  nv = 50,
  nu = 50
)

scores_train <- fit$u %*% diag(fit$d)
scores_test <- X_test %*% fit$v
singular_values <- fit$d
```

## 8. 结果如何解释

- 奇异值越大，该秩 1 模式对矩阵能量贡献越大。
- 右奇异向量是特征组合，左奇异向量是样本模式。
- 符号可整体翻转，不改变重构。
- 大能量模式不一定最能预测临床结局。

## 9. 诊断与稳健性

1. 画奇异值谱和累计能量。
2. 用重构误差或下游交叉验证选择 $k$。
3. 检查前几个方向与批次、测序深度等技术变量的关联。
4. 比较中心化、归一化和不同随机种子。
5. 在训练折拟合，再转换验证与测试数据。

## 10. 推荐可视化

- 奇异值 scree 图。
- 累计能量与重构误差曲线。
- 前两个低维得分散点图。
- 低秩重构前后图像或矩阵热图。

## 11. 优势、局限与常见坑

**优势：** 适用任意矩阵，最优低秩近似有严格理论，稀疏大矩阵可高效计算。

**局限：** 线性、方向难解释、尺度与中心化敏感，缺失需专门处理。

**常见坑：** 把未中心化 Truncated SVD 等同 PCA；全数据先分解；只看二维图；按能量选 $k$ 却忽略下游任务。

## 12. 与相近方法的区别

- [[主成分分析（Principal Component Analysis, PCA）]]：中心化 SVD 对应 PCA，Truncated SVD 通常不中心化。
- [[因子分析（Factor Analysis）]]：因子分析包含特异误差与潜变量模型。
- [[核主成分分析（Kernel Principal Component Analysis, KPCA）]]：在核特征空间处理非线性。
- 选择经验：稀疏矩阵保稀疏降维用 Truncated SVD；连续密集数据方差分析用 PCA。

## 13. 医学研究中的典型应用

- 临床文本词频和诊断编码矩阵的潜在语义。
- 影像矩阵压缩与低秩去噪。
- 组学表达矩阵的主要模式提取。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| singular value | 每个秩 1 模式的强度 |
| left singular vector | 样本/行空间中的正交方向 |
| right singular vector | 特征/列空间中的正交方向 |
| rank-k approximation | 只保留前 $k$ 个奇异模式的矩阵 |
| Frobenius norm | 矩阵所有元素平方和开根号 |

## 15. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[因子分析（Factor Analysis）]]
- [[核主成分分析（Kernel Principal Component Analysis, KPCA）]]
- [[多维尺度分析（Multidimensional Scaling, MDS）]]

## 16. 参考资料

- Eckart C, Young G. The approximation of one matrix by another of lower rank. *Psychometrika*. 1936;1:211-218.
- Golub GH, Van Loan CF. *Matrix Computations*. 4th ed. Johns Hopkins University Press; 2013.
- scikit-learn Developers. `TruncatedSVD` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) （访问日期：2026-07-09）
