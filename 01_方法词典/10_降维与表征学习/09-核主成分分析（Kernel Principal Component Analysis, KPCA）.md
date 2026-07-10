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

### 1.1 一句话本质

核 PCA 不显式生成大量非线性特征，而是只计算样本在隐式特征空间中的两两内积，再对中心化核矩阵做 PCA。

### 1.2 定义

核主成分分析是 PCA 的非线性扩展。核函数 $k(x_i,x_j)$ 代表映射后特征向量的内积；对核矩阵中心化并特征分解，即可得到隐式特征空间中的主成分得分。

### 1.3 它主要解决什么问题

- 数据结构不是一个线性子空间，普通 PCA 难以展开。
- 想用 RBF、多项式或领域特定核表达非线性相似性。
- 需要为可视化或下游模型构造非线性无监督特征。

### 1.4 直觉与类比

普通 PCA 只能转动一张平面相机；核 PCA 相当于先用一套非线性镜头把关系展开，再在镜头后的空间找方差方向。核技巧让我们只算样本之间的相似度，不必真的造出那个高维空间。

## 2. 核心思想与原理

### 2.1 根本困难

显式加入所有平方项、交互项或更复杂基函数会迅速膨胀维度，而且许多有用的特征空间可能是无限维的。

### 2.2 关键洞察

PCA 的样本得分可以仅通过内积计算。只要核函数对应一个合法的半正定 Gram 矩阵，就能用 $K_{ij}=k(x_i,x_j)$ 替代显式内积。

### 2.3 中心化不可省

原始数据即使已在输入空间中心化，非线性映射后的 $\phi(x_i)$ 通常仍不中心。核矩阵必须在特征空间意义下双中心化，否则得到的是绕错误原点的方向。

## 3. 数学形式

### 3.1 核矩阵

$$
K_{ij}=k(x_i,x_j)
=\langle\phi(x_i),\phi(x_j)\rangle
$$

常见 RBF 核为：

$$
k(x_i,x_j)
=\exp\left(-\gamma\|x_i-x_j\|^2\right)
$$

### 3.2 核中心化

令：

$$
H=I-\frac1n\mathbf1\mathbf1^\top
$$

则：

$$
K_c=HKH
$$

这等价于在隐式特征空间中减去样本均值。

### 3.3 特征分解与得分

$$
K_c\alpha_m=\lambda_m\alpha_m
$$

若 $\alpha_m^\top\alpha_m=1$，训练样本第 $m$ 个核主成分得分可写为：

$$
z_m=\sqrt{\lambda_m}\alpha_m
$$

新样本投影需要先按训练集均值中心化其核向量，再与归一化特征向量组合。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 核函数表达有意义相似性 | 成分难以泛化 | 核与参数敏感性 |
| 核矩阵近似半正定 | 出现异常负特征值 | 核谱检查 |
| 特征空间正确中心化 | 主方向偏移 | 使用双中心化实现 |
| 样本量可承受核矩阵 | 内存和时间过大 | 估算 $n^2$ 存储 |
| 调参不使用测试结局 | 乐观偏倚 | 嵌套交叉验证 |

## 4. 手把手算例

取两个一维样本：

$$
x_1=1,\qquad x_2=2
$$

使用二次多项式核：

$$
k(x,z)=(xz+1)^2
$$

**Step 1：计算核矩阵。**

$$
K=
\begin{pmatrix}
(1\times1+1)^2&(1\times2+1)^2\\
(2\times1+1)^2&(2\times2+1)^2
\end{pmatrix}
=
\begin{pmatrix}
4&9\\
9&25
\end{pmatrix}
$$

**Step 2：核中心化。**

对两个样本：

$$
H=
\begin{pmatrix}
0.5&-0.5\\
-0.5&0.5
\end{pmatrix}
$$

于是：

$$
K_c=HKH
=
\begin{pmatrix}
2.75&-2.75\\
-2.75&2.75
\end{pmatrix}
$$

每行每列之和均为 0，说明隐式特征已中心化。

**Step 3：特征分解。**

$K_c$ 的特征值为：

$$
\lambda_1=5.5,\qquad\lambda_2=0
$$

第一单位特征向量可取：

$$
\alpha_1=\frac1{\sqrt2}(-1,1)^\top
$$

**Step 4：得到核主成分得分。**

$$
z_1=\sqrt{5.5}\alpha_1
=(-\sqrt{2.75},\sqrt{2.75})^\top
\approx(-1.658,1.658)^\top
$$

核 PCA 在不显式写出二次特征的情况下，已沿隐式特征空间的主要差异方向分开两个样本。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 数值特征矩阵或预计算核矩阵。
- 连续特征通常需在训练集内标准化。
- 序列、图或其他对象可使用领域核，但应验证半正定性。
- 缺失值需先处理；删失结局和类别标签不是无监督 KPCA 的直接输入。
- 重复测量需考虑个体结构，避免同一患者多次记录主导核相似性。

### 5.2 示例表格

| Patient | Marker_1 | Marker_2 | Marker_3 | Disease |
| --- | ---: | ---: | ---: | --- |
| P01 | 0.8 | 1.4 | -0.2 | 1 |
| P02 | 0.6 | 1.1 | -0.1 | 1 |
| P03 | -0.4 | 0.2 | 0.7 | 0 |
| P04 | -0.6 | 0.1 | 0.5 | 0 |

`Disease` 不参与无监督 KPCA 拟合，只用于外部验证或后续监督任务。

### 5.3 输入与产出

输入包括特征或核矩阵、核类型、核参数、成分数和逆变换设置。输出包括核主成分得分、特征值、特征向量及可选近似逆变换；KPCA 本身不产生结局预测。

## 6. 适用场景

- 存在可信非线性相似性且样本量中等。
- 希望以固定核构造下游预测特征。
- 需要比较 PCA 与非线性核表征。
- 不适合超大样本、要求原变量载荷解释或核参数无合理选择依据的任务。
- 若目标主要是二维局部可视化，UMAP 或 t-SNE 可能更直接；若目标是可解释线性压缩，优先 PCA。

## 7. 实现

### 7.1 Python

```python
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

features = ["Marker_1", "Marker_2", "Marker_3"]
X = df[features]

model = make_pipeline(
    StandardScaler(),
    KernelPCA(
        n_components=2,
        kernel="rbf",
        gamma=0.5,
        eigen_solver="auto",
        remove_zero_eig=True,
        fit_inverse_transform=False,
        random_state=42,
        n_jobs=-1,
    ),
)
embedding = model.fit_transform(X)

kpca = model.named_steps["kernelpca"]
result = pd.DataFrame(embedding, columns=["KPCA1", "KPCA2"])
print(kpca.eigenvalues_, result.head())
```

### 7.2 R

```r
library(kernlab)

x <- scale(df[, c("Marker_1", "Marker_2", "Marker_3")])
fit <- kpca(
  x,
  kernel = "rbfdot",
  kpar = list(sigma = 0.5),
  features = 2,
  th = 1e-8
)

embedding <- rotated(fit)
eig(fit)
head(embedding)
```

## 8. 结果如何解读

- 核主成分表示隐式特征空间的方差方向，不是原变量的线性载荷。
- 坐标正负可整体翻转，单独的方向没有固定临床含义。
- 特征值描述所选核空间中的变化，不宜与原空间 PCA 的方差比例直接混为一谈。
- 好看的分离图不证明核参数最优，也不证明存在真实亚型。

## 9. 假设诊断与稳健性

1. 比较线性、RBF 和多项式核，以及一组预先规定的参数。
2. 检查核矩阵特征值谱与异常负值。
3. 观察成分数和核参数对嵌入、重构或下游性能的影响。
4. 用 bootstrap 或子采样评估核主成分子空间稳定性。
5. 检查批次、中心、缺失模式和异常值是否主导得分。
6. 监督任务用嵌套交叉验证调核参数，标准化和 KPCA 均在训练折内拟合。
7. 大样本评估 Nyström 或随机特征近似，并与精确结果核对。

## 10. 推荐可视化

- KPCA1-KPCA2 散点图，按独立变量着色。
- 核矩阵热图与特征值谱。
- `gamma` 或多项式阶数敏感性小多图。
- PCA、KPCA 与下游交叉验证性能对比图。

## 11. 优势、局限与常见坑

**优势：** 能捕捉非线性结构；复用多种核和领域相似性；可作为模块化特征变换进入流水线。

**局限：** 核矩阵需要二次存储；结果解释弱于 PCA；核与参数影响大；逆映射通常只是近似。

**常见坑：** 忘记核中心化；把 KPCA 得分当原变量载荷；全数据拟合后再交叉验证；仅凭二维分离调 `gamma`；把 RBF 参数在不同软件中的符号约定视为完全相同。

## 12. 与相近方法的区别

- [[主成分分析（Principal Component Analysis, PCA）]]：PCA 在原特征空间做线性分解；KPCA 在核定义的隐式空间做分解。
- [[UMAP（Uniform Manifold Approximation and Projection）]]：UMAP 优化近邻图用于可视化；KPCA 是核矩阵的谱分解。
- [[多项式核回归（Polynomial Kernel Regression）]]：核回归利用结局做监督预测；KPCA 不使用结局。
- [[支持向量机（Support Vector Machine, SVM）]]：SVM 用核寻找监督决策边界；KPCA 用核寻找无监督方差方向。
- **如何选择：** 需要非线性、可流水线化的谱特征且样本规模可承受核矩阵时考虑 KPCA；需要原变量解释时用 PCA；需要局部二维展示时考虑 UMAP。

## 13. 医学研究中的典型应用

- 多生物标志物的非线性综合表征。
- 影像组学特征压缩后进入分类或生存模型。
- 复杂临床表型的探索性非线性坐标。

报告应注明核函数、参数、中心化、标准化、成分数、调参数据范围及稳定性分析。

## 14. 关键术语

| 术语 | 含义 |
| --- | --- |
| kernel function 核函数 | 直接计算隐式特征空间内积的函数 |
| kernel trick 核技巧 | 不显式构造高维映射而只计算内积 |
| Gram matrix Gram 矩阵 | 所有样本两两核内积组成的矩阵 |
| positive semidefinite 半正定 | 保证核矩阵可解释为某特征空间内积 |
| kernel centering 核中心化 | 在隐式特征空间减去样本均值 |
| pre-image 前像 | 与某个隐式特征表示对应的原空间近似对象 |

## 15. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]
- [[多项式核回归（Polynomial Kernel Regression）]]
- [[支持向量机（Support Vector Machine, SVM）]]
- [[多维尺度分析（Multidimensional Scaling, MDS）]]

## 16. 参考资料

- Scholkopf B, Smola A, Muller KR. Nonlinear component analysis as a kernel eigenvalue problem. *Neural Computation*. 1998;10(5):1299-1319.
- Scholkopf B, Smola AJ. *Learning with Kernels*. MIT Press; 2002.
- scikit-learn Developers. `KernelPCA` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html) （访问日期：2026-07-09）
