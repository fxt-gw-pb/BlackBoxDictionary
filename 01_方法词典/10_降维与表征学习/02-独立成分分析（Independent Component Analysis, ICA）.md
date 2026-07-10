---
title: 独立成分分析
english_name: Independent Component Analysis, ICA
slug: independent-component-analysis-ica
aliases: [ICA, independent component analysis, FastICA, "独立成分分析（Independent Component Analysis, ICA）"]
category: 降维与表征学习
subcategory: 盲源分离与表征学习
tags: [医学统计, 数据科学, 降维, 无监督学习, 信号分离]
status: 已建
difficulty: intermediate
question_type: 混合信号分解与独立表征提取
data_type: [信号数据, 高维特征矩阵, 组学数据]
outcome_type: [无监督表征, 多种]
python_packages: [scikit-learn]
r_packages: [fastICA]
---

# 独立成分分析（Independent Component Analysis, ICA）

## 1. 方法概览

### 1.1 一句话本质

ICA 假设观测信号是若干非高斯独立源的线性混合，并寻找反混合矩阵把这些来源尽可能分离。

### 1.2 定义

独立成分分析是盲源分离与无监督表征方法。它不只让成分不相关，还利用高阶统计量最大化独立性或非高斯性，以恢复潜在来源。

### 1.3 它主要解决什么问题

- 多个传感器记录的是潜在信号的不同混合。
- 需要分离脑活动、眼动、肌电或设备噪声。
- 希望提取比 PCA 更接近独立生成机制的模式。

### 1.4 直觉与类比

房间里两人同时说话，两支麦克风各录到不同混合。ICA 只凭混合录音，寻找一组线性组合，让输出尽可能像两条彼此独立的原始声源。

## 2. 核心思想与原理

### 2.1 不相关不等于独立

PCA 消除二阶相关，但非高斯变量仍可能存在高阶依赖。ICA 追求更强的统计独立性，因此适合源分离。

### 2.2 非高斯性提供可识别信息

多个独立变量的和通常比各源更接近高斯。反过来寻找最非高斯的投影，有助于恢复独立源；因此至多允许一个源为高斯。

### 2.3 白化与旋转

常先中心化和白化，使协方差为单位阵，去掉尺度与相关结构；随后只需在白化空间寻找使成分最独立的旋转。

## 3. 数学形式

### 3.1 混合与分离

按每行一个观测时：

$$
X=SA^\top
$$

$$
\widehat S=XW^\top
$$

理想情况下 $W\approx A^{-1}$ 的相应转置表示。

### 3.2 非高斯目标

FastICA 常最大化近似负熵：

$$
J(y)\approx
\left[
E\{G(y)\}-E\{G(v)\}
\right]^2
$$

$v$ 为标准正态变量，$G$ 为非二次函数。

### 3.3 固定点更新示意

对白化数据 $x$，一个常见更新为：

$$
w^{new}=
E[xg(w^\top x)]
-E[g'(w^\top x)]w
$$

随后归一化并与已提取方向正交化。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 源近似独立 | 成分仍混合或不稳定 | 独立性和领域检查 |
| 混合近似线性 | 无法正确反混合 | 重构与残差 |
| 至多一个高斯源 | 高斯子空间不可唯一旋转 | 分布与峰度检查 |
| 样本量足够且收敛 | 结果依赖初始化 | 多种子稳定性 |

## 4. 手把手算例

两个独立二值源在四个时点为：

$$
s_1=(1,-1,1,-1)
$$

$$
s_2=(1,1,-1,-1)
$$

四种 $(s_1,s_2)$ 组合各出现一次，因此在这个小样本中它们独立且均值为 0。

设混合矩阵：

$$
A=
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix}
$$

观测通道为：

$$
x_1=s_1+s_2=(2,0,0,-2)
$$

$$
x_2=s_1-s_2=(0,-2,2,0)
$$

矩阵的逆为：

$$
A^{-1}=
\frac12
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix}
$$

因此：

$$
\widehat s_1=\frac{x_1+x_2}{2}=s_1
$$

$$
\widehat s_2=\frac{x_1-x_2}{2}=s_2
$$

**结论：** 若找到正确反混合矩阵，就能从两个混合通道恢复两条源；ICA 的任务正是在不知道 $A$ 时依靠独立性估计它。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 时间点/样本乘以连续通道或特征矩阵。
- 缺失需处理，信号常需去趋势、滤波和伪迹初查。
- 成分数不能超过可识别的观测维数。

### 5.2 输入与产出

输入为观测矩阵、成分数、白化和独立性函数。输出为源得分、混合矩阵、反混合矩阵、均值和收敛迭代数。

## 6. 适用场景

- EEG/MEG/fMRI 盲源分离和伪迹去除。
- 非高斯独立过程混合的组学或信号数据。
- 不适合来源高度相关、非线性混合或需要唯一顺序尺度的任务。

## 7. 实现

### 7.1 Python

```python
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

X_s = StandardScaler().fit_transform(X)
ica = FastICA(
    n_components=3,
    whiten="unit-variance",
    fun="logcosh",
    max_iter=1000,
    tol=1e-5,
    random_state=42,
)
sources = ica.fit_transform(X_s)
mixing = ica.mixing_
reconstructed = ica.inverse_transform(sources)
print(ica.n_iter_)
```

### 7.2 R

```r
library(fastICA)

x <- scale(df[, c("Channel_1", "Channel_2", "Channel_3", "Channel_4")])
set.seed(42)
fit <- fastICA(
  x,
  n.comp = 3,
  alg.typ = "parallel",
  fun = "logcosh",
  maxit = 1000,
  tol = 1e-5
)

sources <- fit$S
mixing <- fit$A
```

## 8. 结果如何解释

- 成分顺序、符号和尺度不可识别；跨运行比较前需匹配和对齐。
- 混合矩阵列描述某独立成分如何投射到观测通道。
- “统计独立”不等于因果独立或生物过程完全独立。
- 删除伪迹成分前应结合空间、频谱、时间和外部通道证据。

## 9. 诊断与稳健性

1. 检查收敛迭代和重构误差。
2. 多随机种子重复并匹配成分。
3. 比较不同成分数、非线性函数和白化方案。
4. 检查成分互信息、峰度及领域特征。
5. 用保留数据或跨受试者评估成分复现。

## 10. 推荐可视化

- 原始混合信号与独立成分时间序列。
- 混合矩阵热图或脑空间图。
- 成分频谱、峰度与稳定性图。
- 去除某成分前后的重构信号对比。

## 11. 优势、局限与常见坑

**优势：** 能利用高阶统计分离非高斯来源，适合伪迹去除与信号解释。

**局限：** 假设强，顺序符号不唯一，初始化和成分数敏感。

**常见坑：** 把 ICA 当普通降维；只运行一次；将独立称为因果；自动删除成分而无领域核查；混淆 `components_` 与 `mixing_`。

## 12. 与相近方法的区别

- [[主成分分析（Principal Component Analysis, PCA）]]：最大化方差并正交，ICA 最大化独立性。
- [[因子分析（Factor Analysis）]]：解释共同协方差并显式建模特异误差。
- [[奇异值分解（Singular Value Decomposition, SVD）]]：代数分解，不包含独立生成假设。
- 选择经验：压缩总方差用 PCA，盲源分离且非高斯独立假设可信时用 ICA。

## 13. 医学研究中的典型应用

- EEG/MEG 眼动、心电和肌电伪迹分离。
- fMRI 独立空间网络提取。
- 多组学中的相对独立变化模式探索。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| source | 未直接观察的独立生成信号 |
| mixing matrix | 将独立源映射为观测通道的矩阵 |
| unmixing matrix | 从观测恢复源的线性变换 |
| whitening | 使数据不相关且方差标准化 |
| negentropy | 衡量偏离高斯程度的独立性代理 |

## 15. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[因子分析（Factor Analysis）]]
- [[奇异值分解（Singular Value Decomposition, SVD）]]
- [[t-SNE（t-Distributed Stochastic Neighbor Embedding）]]

## 16. 参考资料

- Comon P. Independent component analysis, a new concept? *Signal Process*. 1994;36(3):287-314.
- Hyvarinen A, Oja E. Independent component analysis: algorithms and applications. *Neural Netw*. 2000;13(4-5):411-430.
- scikit-learn Developers. `FastICA` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html) （访问日期：2026-07-09）
