---
title: 高斯过程回归
english_name: Gaussian Process Regression
slug: gaussian-process-regression
aliases: [GPR, gaussian process regression, 高斯过程回归, "高斯过程回归（Gaussian Process Regression）"]
category: 支持向量机与核方法
subcategory: 核方法与概率回归
tags: [医学统计, 数据科学, 高斯过程, 核方法, 贝叶斯方法]
status: 已建
difficulty: intermediate
question_type: 连续结局非线性概率回归
data_type: [表格数据]
outcome_type: [连续型]
python_packages: [scikit-learn]
r_packages: [kernlab]
---

# 高斯过程回归（Gaussian Process Regression）

## 1. 方法概览

### 1.1 一句话本质

高斯过程回归给未知函数本身设置概率分布，用核函数规定任意两点的函数值应有多相似，并在观测数据后得到预测分布。

### 1.2 定义

高斯过程回归（GPR）是一种贝叶斯非参数回归方法。任意有限组输入点对应的函数值被假定联合服从多元正态分布；条件于训练数据后，新点仍有高斯后验预测分布。

### 1.3 它主要解决什么问题

- 小到中等样本的平滑非线性回归。
- 在点预测之外量化随输入位置变化的不确定性。
- 用核组合表达平滑、周期、线性或多尺度结构。

### 1.4 直觉与类比

不是先选一条固定公式，而是先想象许多可能的曲线。靠近观测点且符合核结构的曲线留下，冲突曲线被淘汰；曲线意见一致处区间窄，数据稀疏处区间宽。

## 2. 核心思想与原理

### 2.1 函数分布

高斯过程由均值函数和协方差核完全描述。核不仅是计算技巧，还编码“距离多远仍应相似”“函数多平滑”等先验结构。

### 2.2 条件正态分布

训练函数值与新点函数值联合正态。利用多元正态条件分布公式，即可得到后验均值与方差；均值负责预测，方差反映数据覆盖与噪声下的不确定性。

### 2.3 不确定性的边界

GPR 区间依赖核、噪声模型和超参数是否合理。它是模型条件下的不确定性，不自动覆盖分布漂移、测量偏倚或错误结局定义。

## 3. 数学形式

### 3.1 先验与观测模型

$$
f(x)\sim\mathcal{GP}[m(x),k(x,x')]
$$

$$
y_i=f(x_i)+\varepsilon_i,\qquad
\varepsilon_i\sim N(0,\sigma_n^2)
$$

### 3.2 后验预测

令 $K=K(X,X)$、$k_*=K(X,x_*)$、$k_{**}=K(x_*,x_*)$，零均值时：

$$
\mu_*=k_*^\top(K+\sigma_n^2I)^{-1}y
$$

$$
\sigma_{f,*}^2=
k_{**}-k_*^\top(K+\sigma_n^2I)^{-1}k_*
$$

若预测未来观测 $y_*$，还需加噪声方差：

$$
\sigma_{y,*}^2=\sigma_{f,*}^2+\sigma_n^2
$$

### 3.3 RBF 核

$$
k(x,z)=\sigma_f^2
\exp\left[-\frac{\|x-z\|^2}{2\ell^2}\right]
$$

$\ell$ 是长度尺度：越大，函数变化越平缓。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 核结构合理 | 均值与区间形状失真 | 比较候选核与残差 |
| 噪声模型合理 | 区间覆盖不足或过宽 | 覆盖率与标准化残差 |
| 样本规模适中 | 矩阵分解成本过高 | 记录时间、考虑稀疏近似 |
| 输入维度可控 | 距离与长度尺度难估 | 特征筛选和领域先验 |

## 4. 手把手算例

只有一个训练点 $x=0,y=2$。采用零均值、单位方差 RBF 核：

$$
k(x,z)=\exp[-(x-z)^2/2]
$$

噪声方差 $\sigma_n^2=0.25$。

**Step 1：在训练位置 $x_*=0$。** 此时 $K=1,k_*=1,k_{**}=1$：

$$
\mu_*(0)=\frac{1}{1+0.25}\times2=1.6
$$

$$
\sigma_{f,*}^2(0)=1-\frac{1}{1.25}=0.2
$$

潜在函数标准差约为 $\sqrt{0.2}=0.447$。

**Step 2：在较远位置 $x_*=1$。**

$$
k_*=\exp(-1/2)\approx0.607
$$

$$
\mu_*(1)=\frac{0.607}{1.25}\times2\approx0.971
$$

$$
\sigma_{f,*}^2(1)=
1-\frac{0.607^2}{1.25}
\approx0.706
$$

标准差约为 $0.840$。

**结论：** 离训练点更远时，预测向先验均值 0 回归，潜在函数不确定性由 0.447 增至 0.840。若预测带噪声的新观测，两处方差还要各加 0.25。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 连续结局和通常为数值型的输入。
- 缺失需预先处理，特征尺度影响长度尺度解释。
- 标准 GPR 不直接处理删失或纵向相关性，但可通过专门核与似然扩展。

### 5.2 输入与产出

输入为均值函数、核、噪声和训练数据。输出为每个新点的后验均值、方差或样本曲线，以及拟合后的核超参数和边际似然。

## 6. 适用场景

- 小样本、低到中维、需要位置特异性不确定性的连续预测。
- 剂量反应、校准曲线、生理函数和贝叶斯优化。
- 不适合未经近似的大样本、高维稀疏表格或强外推任务。

## 7. 实现

### 7.1 Python

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
)
model = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=5,
    random_state=42,
)
model.fit(X_train_s, y_train)
mean, std = model.predict(X_test_s, return_std=True)
print(model.kernel_)
```

### 7.2 R

```r
library(kernlab)

fit <- gausspr(
  x = as.matrix(X_train),
  y = y_train,
  kernel = "rbfdot",
  kpar = "automatic",
  var = 0.01,
  scaled = TRUE
)

mean_pred <- predict(fit, as.matrix(X_test))
```

`kernlab::gausspr()` 上述代码主要返回预测均值；若研究必须获得完整后验方差，应选择明确支持预测协方差输出的高斯过程实现。

## 8. 结果如何解释

- 后验均值是给定核与数据的点预测。
- 后验标准差通常在观测附近较小，在稀疏区域较大。
- 长度尺度表示函数变化距离，须结合标准化后的特征单位解释。
- 报告区间时区分潜在函数区间与未来观测预测区间。

## 9. 诊断与稳健性

1. 比较 RBF、Matérn、线性或组合核。
2. 检查边际似然优化是否卡在参数边界。
3. 画标准化残差并检验预测区间经验覆盖率。
4. 进行留一或交叉验证，比较简单回归和 SVR 基线。
5. 审查远离训练数据区域的均值回归与区间行为。

## 10. 推荐可视化

- 观测点、后验均值和 95% 区间。
- 从后验抽取的若干函数曲线。
- 预测标准差随输入变化图。
- 不同核函数的拟合与区间对比。

下图展示 GPR 的预测均值与不确定性带：

![](../../04_示例图像/gaussian_process_regression_curve.png)

## 11. 优势、局限与常见坑

**优势：** 非线性灵活，自然量化不确定性，小样本中可融入结构先验。

**局限：** 标准算法计算和存储成本高，高维核设计困难，结果依赖核与噪声模型。

**常见坑：** 把后验区间称为无条件保证；混淆潜在函数和新观测区间；只看拟合均值；大样本直接使用精确 GPR。

## 12. 与相近方法的区别

- [[支持向量回归（Support Vector Regression, SVR）]]：SVR 以间隔损失优化，GPR 给出概率预测分布。
- [[贝叶斯回归（Bayesian Regression）]]：参数回归对有限参数设先验，GPR 对函数设先验。
- 核 Ridge：其预测均值与特定 GPR 有紧密联系，但通常不输出完整概率不确定性。
- 选择经验：需要可靠模型条件不确定性且样本不大时优先考虑 GPR。

## 13. 医学研究中的典型应用

- 小样本剂量-反应和生理曲线。
- 连续生物标志物的平滑预测与不确定性量化。
- 个体化采样、试验设计和需要主动选择下一个测量点的任务。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| Gaussian process | 任意有限函数值联合正态的随机过程 |
| covariance kernel | 定义不同输入函数值协方差的函数 |
| length scale | 控制相关性随输入距离衰减的尺度 |
| posterior mean | 条件于数据后的平均预测函数 |
| marginal likelihood | 综合拟合度与复杂度的超参数目标 |

## 15. 相关方法

- [[支持向量回归（Support Vector Regression, SVR）]]
- [[贝叶斯回归（Bayesian Regression）]]
- [[多项式核回归（Polynomial Kernel Regression）]]
- [[局部加权回归（Locally Weighted Regression）]]

## 16. 参考资料

- Rasmussen CE, Williams CKI. *Gaussian Processes for Machine Learning*. MIT Press; 2006.
- scikit-learn Developers. Gaussian Processes User Guide. [https://scikit-learn.org/stable/modules/gaussian_process.html](https://scikit-learn.org/stable/modules/gaussian_process.html) （访问日期：2026-07-09）
- CRAN. Package `kernlab`. [https://cran.r-project.org/package=kernlab](https://cran.r-project.org/package=kernlab) （访问日期：2026-07-09）
