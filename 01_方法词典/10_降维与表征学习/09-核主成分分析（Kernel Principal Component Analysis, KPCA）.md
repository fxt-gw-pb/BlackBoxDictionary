---
title: 核主成分分析
english_name: Kernel Principal Component Analysis, KPCA
slug: kernel-principal-component-analysis-kpca
aliases: [Kernel PCA, KPCA, 核PCA, "核主成分分析（Kernel Principal Component Analysis, KPCA）"]
category: 降维与表征学习
subcategory: 非线性核降维
tags: [医学统计, 数据科学, 降维, 核方法, 非线性]
status: 已建
difficulty: advanced
question_type: 非线性主成分表征学习
data_type: [表格数据, 高维特征矩阵, 组学数据]
outcome_type: [无监督表征, 多种]
python_packages: [scikit-learn]
r_packages: [kernlab]
---

# 核主成分分析（Kernel Principal Component Analysis, KPCA）

## 1. 方法概览

### 1.1 定义

核主成分分析是 [[主成分分析（Principal Component Analysis, PCA）]] 的非线性扩展。它通过核函数隐式地把样本映射到高维特征空间，再在该空间中执行 PCA，从而捕捉原始空间中的非线性结构。

### 1.2 它主要解决什么问题

- 研究问题：数据主要结构不是线性方向时，如何提取非线性主成分。
- 适用任务：非线性降维、非线性特征提取、可视化、后续分类或回归的特征构造。
- 常见医学场景：复杂生物标志物模式提取，影像组学非线性特征压缩，非线性疾病表型结构展示。

### 1.3 直觉理解

普通 PCA 只能寻找原始空间中的直线方向。核 PCA 借助核函数在“看不见的高维空间”里做 PCA，使原始空间中弯曲的结构也可能被展开成更容易表示的方向。

## 2. 数学形式

### 2.1 核心公式

设非线性映射为 $\phi(x)$，核函数为：

$$
K(x_i,x_j)=\langle \phi(x_i),\phi(x_j)\rangle
$$

核矩阵 $K$ 需要中心化：

$$
K_c = K - \mathbf{1}_n K - K\mathbf{1}_n + \mathbf{1}_n K \mathbf{1}_n
$$

其中 $\mathbf{1}_n$ 为元素均为 $1/n$ 的矩阵。核 PCA 通过求解：

$$
K_c \alpha = n\lambda \alpha
$$

得到特征空间中的主成分方向。新样本 $x$ 在第 $m$ 个核主成分上的坐标为：

$$
z_m(x)=\sum_{i=1}^{n}\alpha_{mi}K(x_i,x)
$$

### 2.2 参数或统计量含义

- $K$：核矩阵，表示样本在隐式特征空间中的内积。
- $\phi(x)$：隐式非线性映射。
- $\alpha$：核矩阵特征向量。
- `kernel`：核函数类型，如 RBF、多项式、sigmoid。
- `gamma`：RBF 核的尺度参数。
- `n_components`：保留的核主成分数量。

### 2.3 关键假设

- 所选核函数能表达数据中的非线性结构。
- 样本相似性可由核矩阵合理描述。
- 参数如 `gamma` 不会造成过度局部化或过度平滑。
- 主要目标是表征或预测辅助，而不是直接解释原始变量载荷。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续型特征或可定义核函数的对象。
- 因变量类型：KPCA 本身不需要结局变量。
- 数据结构：样本乘以特征矩阵，或预计算核矩阵。
- 是否适合高维数据：可用于高维数据，但核矩阵规模为 $n\times n$，大样本成本较高。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不直接处理删失结局。
- 是否适合重复测量数据：需先定义样本单位或使用合适核函数。

### 3.2 示例表格

以非线性生物标志物模式为例：

| Marker_1 | Marker_2 | Marker_3 | Marker_4 | Disease |
| --- | --- | --- | --- | --- |
| 0.8 | 1.4 | -0.2 | 0.3 | 1 |
| 0.6 | 1.1 | -0.1 | 0.1 | 1 |
| -0.4 | 0.2 | 0.7 | -0.5 | 0 |
| -0.6 | 0.1 | 0.5 | -0.4 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：数值特征矩阵或核矩阵。
- 关键变量：核函数、核参数、保留维度、是否拟合逆变换。
- 需要预处理的内容：缺失处理、标准化、核参数调优、训练测试划分内 fit。

#### 产出

- 模型对象/统计结果：核主成分得分、核矩阵特征值、低维表示。
- 参数估计：核矩阵特征向量和特征值。
- 预测结果：KPCA 本身不预测，可作为后续模型输入。
- 不确定性指标：参数敏感性、交叉验证性能、重采样稳定性。

## 4. 适用场景

- 适合：线性 PCA 难以展开结构、非线性关系明显、样本量中等且核矩阵可计算的场景。
- 不适合：需要清晰变量载荷解释、大样本核矩阵计算不可承受、核参数难以稳定选择的场景。
- 使用前需要特别检查的点：核函数是否合适，`gamma` 是否过大或过小，降维结果是否改善下游任务或可视化。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("biomarker_features.csv")
X = df[["Marker_1", "Marker_2", "Marker_3", "Marker_4"]]

kpca_pipe = make_pipeline(
    StandardScaler(),
    KernelPCA(
        n_components=2,
        kernel="rbf",
        gamma=0.5,
        random_state=42
    )
)

Z = kpca_pipe.fit_transform(X)
embedding = pd.DataFrame(Z, columns=["KPCA1", "KPCA2"])
print(embedding.head())
```

### 5.2 R

常用包：

- `kernlab`

```r
library(kernlab)

x <- scale(df[, c("Marker_1", "Marker_2", "Marker_3", "Marker_4")])
fit <- kpca(x, kernel = "rbfdot", kpar = list(sigma = 0.5), features = 2)

embedding <- rotated(fit)
head(embedding)
```

## 6. 结果如何解释

- 核心结果看什么：核主成分空间中的样本分布、特征值衰减、不同核参数下结构是否稳定。
- 每个主要参数如何解释：RBF 核中的 `gamma` 越大，相似性越局部，嵌入可能更复杂也更容易过拟合。
- 临床或医学意义如何表达：核主成分是隐式特征空间中的方向，解释应围绕样本相似性和下游任务表现，而不是原始变量载荷。
- 常见误读：KPCA 的二维图比 PCA 好看，不代表它一定更适合预测或机制解释。

## 7. 推荐可视化

- KPCA1-KPCA2 散点图。
- 核参数敏感性对比图。
- 下游模型交叉验证性能随核参数变化曲线。
- 与 PCA、UMAP 或 t-SNE 的嵌入对比图。

## 8. 优势、局限与常见坑

### 优势

- 能捕捉非线性结构。
- 可复用核方法的灵活相似性定义。
- 可作为后续模型的非线性特征工程步骤。

### 局限

- 核矩阵计算和存储成本随样本量平方增长。
- 结果解释不如 PCA 载荷直观。
- 核函数和参数选择影响很大。

### 常见坑

- 在全数据上 fit KPCA 后再交叉验证监督模型。
- 不调核参数就比较算法优劣。
- 把隐式核主成分强行解释为原始变量线性组合。

## 9. 与相近方法的区别

- 和 [[主成分分析（Principal Component Analysis, PCA）]] 的区别：PCA 是线性降维，KPCA 通过核技巧做非线性降维。
- 和 [[多项式核回归（Polynomial Kernel Regression）]] 的区别：多项式核回归是监督回归，KPCA 是无监督表征学习。
- 和 [[UMAP（Uniform Manifold Approximation and Projection）]] 的区别：UMAP 面向邻域图嵌入和可视化，KPCA 基于核矩阵特征分解。

## 10. 医学研究中的典型应用

- 多生物标志物非线性综合表征。
- 影像组学非线性结构压缩后进入分类模型。
- 复杂临床表型数据的非线性探索性可视化。

## 11. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[多项式核回归（Polynomial Kernel Regression）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]
- [[支持向量机（Support Vector Machine, SVM）]]

## 12. 参考资料

- Scholkopf B, Smola A, Muller KR. Nonlinear component analysis as a kernel eigenvalue problem. *Neural Computation*. 1998;10(5):1299-1319.
- Scholkopf B, Smola AJ. *Learning with Kernels*. MIT Press; 2002.
- scikit-learn Developers. `sklearn.decomposition.KernelPCA`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html) （访问日期：2026-07-02）
