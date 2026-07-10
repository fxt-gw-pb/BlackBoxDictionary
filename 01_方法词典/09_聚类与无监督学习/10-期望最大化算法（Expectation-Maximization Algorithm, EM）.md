---
title: 期望最大化算法
english_name: Expectation-Maximization Algorithm, EM
slug: expectation-maximization-algorithm-em
aliases: [EM, EM算法, expectation-maximization, expectation-maximization algorithm, "期望最大化算法（Expectation-Maximization Algorithm, EM）"]
category: 聚类与无监督学习
subcategory: 隐变量模型估计
tags: [医学统计, 数据科学, 无监督学习, 隐变量, 混合模型, 最大似然]
status: 已建
difficulty: advanced
question_type: 含隐变量模型的最大似然估计
data_type: [表格数据, 高维特征矩阵, 混合模型数据]
outcome_type: [无监督分群, 隐变量]
python_packages: [scikit-learn, scipy]
r_packages: [mclust, mixtools]
---

# 期望最大化算法（Expectation-Maximization Algorithm, EM）

## 1. 方法概览

### 1.1 一句话本质

EM 在“根据当前参数推断隐藏信息”和“把推断出的软信息当作完整数据更新参数”之间交替，逐步提高观测数据似然。

### 1.2 定义

期望最大化算法是含隐变量、潜在类别或部分缺失数据模型的迭代最大似然方法。E 步计算完整数据对数似然关于隐变量后验的期望，M 步最大化该期望以更新参数。

### 1.3 它主要解决什么问题

- 直接似然中含有“对隐藏状态求和后再取对数”，难以优化。
- 成分标签、潜在状态或部分数据未观察。
- 完整数据下参数更新简单，但隐变量未知。

### 1.4 直觉与类比

像整理一叠来源标签脱落的化验单：先按当前各人群分布给每张单子分配软归属，再用这些分数重新估计各人群；新参数又会改变下一轮归属。

## 2. 核心思想与原理

### 2.1 困难来自 log-sum

观测似然为 $\log\sum_zp(X,z\mid\theta)$。对数外的求和使直接分离各隐状态很困难；如果 $Z$ 已知，完整数据似然通常容易最大化。

### 2.2 E 步构造代理目标

用当前参数下的 $p(Z\mid X,\theta^{(t)})$ 对完整数据对数似然取期望，得到 $Q$ 函数。它把未知标签替换为软权重。

### 2.3 M 步与单调性

M 步最大化 $Q$。精确 E/M 步通常保证观测对数似然不下降，但只保证到达局部驻点，不保证全局最优，也不保证潜在类别具有真实医学含义。

## 3. 数学形式

### 3.1 观测与完整数据似然

$$
\ell(\theta)=
\log p(X\mid\theta)
=\log\sum_Zp(X,Z\mid\theta)
$$

### 3.2 E 步

$$
Q(\theta\mid\theta^{(t)})
=
E_{Z\mid X,\theta^{(t)}}
[\log p(X,Z\mid\theta)]
$$

### 3.3 M 步

$$
\theta^{(t+1)}
=
\operatorname*{arg\,max}_{\theta}
Q(\theta\mid\theta^{(t)})
$$

在 GMM 中，E 步责任概率为：

$$
\gamma_{ik}=
\frac{\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)}
{\sum_j\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)}
$$

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 模型与隐变量结构可识别 | 多组参数给出同样分布 | 约束、模拟与理论审查 |
| E/M 步正确且数值稳定 | 似然异常或不收敛 | 监控每轮似然 |
| 多个初值得到稳定解 | 陷入不同局部最优 | 多初始化比较 |
| 缺失机制假设合理 | 缺失数据估计偏倚 | 明确 MCAR/MAR/MNAR |

## 4. 手把手算例

沿用 [[高斯混合模型（Gaussian Mixture Model, GMM）]] 的一维两成分例子：

$$
x=(0,1,4),\quad
\pi_1=\pi_2=0.5,\quad
\mu_1=0,\quad\mu_2=4,\quad
\sigma_1^2=\sigma_2^2=1
$$

**E 步：计算软标签。** 根据贝叶斯公式得到：

| $x$ | $\gamma_{i1}$ | $\gamma_{i2}$ |
| ---: | ---: | ---: |
| 0 | 0.9997 | 0.0003 |
| 1 | 0.9820 | 0.0180 |
| 4 | 0.0003 | 0.9997 |

例如 $x=1$：

$$
\gamma_{1,1}
=
\frac{\exp(-0.5)}
{\exp(-0.5)+\exp(-4.5)}
\approx0.982
$$

**M 步：把软标签当权重。**

$$
N_1=\sum_i\gamma_{i1}\approx1.982,\qquad
N_2\approx1.018
$$

更新混合比例：

$$
\pi_1^{new}=\frac{1.982}{3}\approx0.661,\qquad
\pi_2^{new}\approx0.339
$$

更新均值：

$$
\mu_1^{new}
=
\frac{0(0.9997)+1(0.9820)+4(0.0003)}
{1.982}
\approx0.496
$$

$$
\mu_2^{new}\approx3.946
$$

然后用新参数进入下一轮 E 步，直到对数似然变化足够小。

**结论：** E 步没有“填死”一个标签，而是传播不确定性；M 步用这些概率加权完成普通参数估计。

## 5. 数据形式与输入输出

### 5.1 数据要求

- EM 是算法而非特定数据模型，可用于连续、类别、计数、缺失或序列数据。
- 可用性取决于完整数据似然、E 步后验与 M 步更新是否可计算。
- 用于缺失数据时，不能绕过缺失机制假设。

### 5.2 输入与产出

输入为概率模型、隐变量定义、初值与收敛阈值。输出为参数估计、后验隐状态、对数似然轨迹和收敛信息；具体含义由模型决定。

## 6. 适用场景

- GMM、潜类别、隐马尔可夫、随机效应或某些缺失数据模型。
- 完整数据估计简单、观测似然直接优化困难。
- 不适合模型不可识别、E/M 步均无可行计算或多局部最优无法区分的场景。

## 7. 实现

### 7.1 Python

```python
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

X_s = StandardScaler().fit_transform(X)
model = GaussianMixture(
    n_components=3,
    covariance_type="full",
    n_init=30,
    max_iter=500,
    tol=1e-5,
    reg_covar=1e-6,
    random_state=42,
)
model.fit(X_s)

posterior = model.predict_proba(X_s)
cluster = model.predict(X_s)
print("converged:", model.converged_)
print("iterations:", model.n_iter_)
print("lower bound:", model.lower_bound_)
```

### 7.2 R

```r
library(mclust)

x <- scale(df[, c("CRP", "IL6", "BMI", "FastingGlucose")])
fit <- Mclust(
  x,
  G = 1:6,
  modelNames = NULL,
  control = emControl(tol = c(1e-5, 1e-5))
)

posterior <- fit$z
cluster <- fit$classification
fit$loglik
fit$bic
```

软件通常封装了特定模型的 EM；不存在一个脱离模型、可对任意数据直接调用的通用 EM 聚类器。

## 8. 结果如何解释

- `converged` 只说明停止标准满足，不说明达到全局最大值。
- 隐状态编号可交换，参数比较前需进行成分匹配。
- 后验责任度是给定模型与参数的条件概率，不是亚型真实性概率。
- EM 本身没有医学结论，结论来自所拟合模型、数据和外部验证。

## 9. 诊断与稳健性

1. 保存每轮对数似然，确认无异常下降并识别平台期。
2. 运行多个随机初值，比较最终似然和参数。
3. 检查极小成分、奇异协方差与参数边界。
4. 用模拟数据检验模型可识别性和恢复能力。
5. 比较成分数、模型约束及 bootstrap 稳定性。

## 10. 推荐可视化

- 对数似然或下界随迭代变化图。
- 不同初始化的最终似然分布。
- 后验责任概率热图。
- BIC 随模型和成分数变化图。

## 11. 优势、局限与常见坑

**优势：** 将困难的隐变量似然拆成可解释两步，常有闭式更新，精确迭代保证似然不降。

**局限：** 局部最优、初始化敏感、收敛可能慢，标准误与模型选择需额外处理。

**常见坑：** 把 EM 当聚类模型；把收敛当全局最优；只跑一次；忽略成分退化；认为 EM 自动解决 MNAR 缺失。

## 12. 与相近方法的区别

- [[高斯混合模型（Gaussian Mixture Model, GMM）]]：GMM 是模型，EM 是其常用估计算法。
- [[K-means聚类（K-means Clustering）]]：可视为特定极限下的硬分配迭代，但目标和概率解释不同。
- [[模糊C均值聚类（Fuzzy C-Means Clustering, FCM）]]：隶属度来自模糊目标，不是后验概率。
- 选择经验：先明确生成模型和隐变量，再判断 EM 是否是合适的估计器。

## 13. 医学研究中的典型应用

- 潜在疾病亚型和生物标志物混合分布。
- 隐马尔可夫疾病状态与纵向潜类别。
- 某些 MAR 缺失数据、测量误差和随机效应模型估计。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| latent variable | 模型中存在但未直接观察的变量 |
| complete-data likelihood | 假设隐变量已知时的似然 |
| E-step | 对隐变量后验取完整似然期望 |
| M-step | 最大化 E 步构造的期望目标 |
| local optimum | 只在邻域内最优而非全局最优的解 |

## 15. 相关方法

- [[高斯混合模型（Gaussian Mixture Model, GMM）]]
- [[K-means聚类（K-means Clustering）]]
- [[模糊C均值聚类（Fuzzy C-Means Clustering, FCM）]]
- [[贝叶斯回归（Bayesian Regression）]]

## 16. 参考资料

- Dempster AP, Laird NM, Rubin DB. Maximum likelihood from incomplete data via the EM algorithm. *J R Stat Soc Series B*. 1977;39(1):1-38.
- McLachlan GJ, Krishnan T. *The EM Algorithm and Extensions*. 2nd ed. Wiley; 2008.
- Bishop CM. *Pattern Recognition and Machine Learning*. Springer; 2006.
