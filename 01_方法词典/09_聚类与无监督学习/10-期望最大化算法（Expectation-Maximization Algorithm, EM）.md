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

### 1.1 定义

期望最大化算法是一种用于含隐变量或缺失数据模型的迭代最大似然估计算法。它交替执行 E 步和 M 步：先估计隐变量的条件期望，再在该期望下更新模型参数。

### 1.2 它主要解决什么问题

- 研究问题：当模型中有看不见的类别、成分或缺失信息时，如何估计参数。
- 适用任务：混合模型估计、软聚类、缺失数据模型、隐变量模型。
- 常见医学场景：疾病潜在亚型建模，患者群体混合分布估计，带隐含类别的生物标志物模式分析。

### 1.3 直觉理解

EM 像是在“猜标签”和“更新模型”之间反复来回。先根据当前参数猜每个样本属于各潜在成分的概率，再用这些软标签重新估计参数，直到似然稳定。

## 2. 数学形式

### 2.1 核心公式

设观测数据为 $X$，隐变量为 $Z$，参数为 $\theta$。目标是最大化观测数据似然：

$$
\ell(\theta)=\log p(X\mid\theta)=\log\sum_Z p(X,Z\mid\theta)
$$

E 步构造辅助函数：

$$
Q(\theta\mid\theta^{(t)})=
E_{Z\mid X,\theta^{(t)}}[\log p(X,Z\mid\theta)]
$$

M 步更新参数：

$$
\theta^{(t+1)}=\arg\max_\theta Q(\theta\mid\theta^{(t)})
$$

在 [[高斯混合模型（Gaussian Mixture Model, GMM）]] 中，责任概率为：

$$
\gamma_{ik}=
\frac{\pi_k\mathcal{N}(x_i\mid \mu_k,\Sigma_k)}
{\sum_{j=1}^{K}\pi_j\mathcal{N}(x_i\mid \mu_j,\Sigma_j)}
$$

### 2.2 参数或统计量含义

- $X$：观测数据。
- $Z$：隐变量，如潜在类别或混合成分标签。
- $\theta$：待估计参数。
- $Q(\theta\mid\theta^{(t)})$：完整数据对数似然的条件期望。
- 责任概率：样本属于各隐含成分的后验权重。
- log-likelihood：迭代过程中应非下降的目标函数。

### 2.3 关键假设

- 模型形式已指定，隐变量结构合理。
- E 步和 M 步可以计算或近似计算。
- 似然函数可能有局部最优，需要多次初始化。
- 收敛到的是驻点，不保证全局最优。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：取决于具体隐变量模型，可为连续、分类、计数或混合数据。
- 因变量类型：通常用于无监督或半监督场景。
- 数据结构：观测数据加隐含结构。
- 是否适合高维数据：可用，但参数估计和局部最优问题会更突出。
- 是否适合缺失较多数据：EM 可用于某些缺失数据模型，但需明确缺失机制和模型假设。
- 是否适合删失数据：可扩展到删失或截断模型，但普通聚类实现不直接处理。
- 是否适合重复测量数据：可用于隐马尔可夫模型、混合效应模型等扩展，但需专门建模。

### 3.2 示例表格

以潜在疾病亚型混合模型为例：

| CRP | IL6 | BMI | FastingGlucose | LatentSubtype |
| --- | --- | --- | --- | --- |
| 5.2 | 8.1 | 31.2 | 7.8 | 未观测 |
| 1.1 | 2.0 | 24.8 | 5.2 | 未观测 |
| 3.8 | 5.5 | 29.5 | 6.9 | 未观测 |
| 0.8 | 1.5 | 22.9 | 4.9 | 未观测 |

### 3.3 输入与产出

#### 输入

- 输入数据：观测变量矩阵。
- 关键变量：模型形式、隐变量数量、初始化、收敛阈值。
- 需要预处理的内容：缺失机制判断、标准化、异常值检查、模型可识别性评估。

#### 产出

- 模型对象/统计结果：参数估计、隐变量后验权重、对数似然、AIC/BIC。
- 参数估计：取决于具体模型，如混合比例、均值、协方差。
- 预测结果：隐类别软归属或硬标签。
- 不确定性指标：后验权重、标准误、似然变化、多初始化稳定性。

## 4. 适用场景

- 适合：模型中有潜在类别、混合成分或可建模缺失信息，且希望做最大似然估计的场景。
- 不适合：模型形式不清楚、隐变量不可识别、似然面高度复杂且无稳定初始化的场景。
- 使用前需要特别检查的点：初始化、局部最优、收敛标准、模型选择和隐变量解释。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("latent_subtype_features.csv")
X = df[["CRP", "IL6", "BMI", "FastingGlucose"]]
X_scaled = StandardScaler().fit_transform(X)

fit = GaussianMixture(
    n_components=3,
    covariance_type="full",
    n_init=20,
    random_state=42
)
fit.fit(X_scaled)

posterior = fit.predict_proba(X_scaled)
cluster = fit.predict(X_scaled)
print("BIC:", fit.bic(X_scaled))
print(posterior[:5].round(3))
```

### 5.2 R

常用包：

- `mclust`

```r
library(mclust)

x <- scale(df[, c("CRP", "IL6", "BMI", "FastingGlucose")])
fit <- Mclust(x, G = 1:6)

cluster <- fit$classification
posterior <- fit$z
fit$bic
head(posterior)
```

## 6. 结果如何解释

- 核心结果看什么：对数似然是否收敛、参数估计是否稳定、隐变量后验权重是否清晰。
- 每个主要参数如何解释：在 GMM 中，混合比例表示潜在成分占比，责任概率表示样本对各成分的软归属。
- 临床或医学意义如何表达：EM 只是估计算法，医学解释来自具体模型及其成分特征。
- 常见误读：EM 不是一个单独的聚类模型；它是估计 GMM 等隐变量模型的算法。

## 7. 推荐可视化

- 对数似然随迭代变化曲线。
- BIC/AIC 随成分数变化图。
- 后验责任概率热图。
- 混合成分在降维空间中的分布图。

## 8. 优势、局限与常见坑

### 优势

- 适合含隐变量模型的最大似然估计。
- 每次迭代通常保证观测似然不下降。
- 能自然产生软归属或后验权重。

### 局限

- 可能陷入局部最优。
- 对初始化敏感。
- 收敛速度可能较慢，且依赖模型可识别性。

### 常见坑

- 只运行一次初始化就解释成分。
- 把 EM 的收敛结果当作全局最优。
- 忽略模型本身假设，只讨论算法步骤。

## 9. 与相近方法的区别

- 和 [[高斯混合模型（Gaussian Mixture Model, GMM）]] 的区别：GMM 是模型，EM 是常用的参数估计算法。
- 和 [[K-means聚类（K-means Clustering）]] 的区别：K-means 可看作某些限制条件下的硬分配原型聚类，EM 在混合模型中进行软分配和参数更新。
- 和 [[模糊C均值聚类（Fuzzy C-Means Clustering, FCM）]] 的区别：FCM 的隶属度来自模糊目标函数，EM 的责任概率来自概率模型后验。

## 10. 医学研究中的典型应用

- 潜在疾病亚型或患者成分模型估计。
- 生物标志物混合分布分解。
- 隐变量模型、缺失数据模型或测量误差模型的参数估计。

## 11. 相关方法

- [[高斯混合模型（Gaussian Mixture Model, GMM）]]
- [[K-means聚类（K-means Clustering）]]
- [[模糊C均值聚类（Fuzzy C-Means Clustering, FCM）]]
- [[贝叶斯回归（Bayesian Regression）]]

## 12. 参考资料

- Dempster AP, Laird NM, Rubin DB. Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*. 1977;39(1):1-38.
- McLachlan GJ, Krishnan T. *The EM Algorithm and Extensions*. 2nd ed. Wiley; 2008.
- Bishop CM. *Pattern Recognition and Machine Learning*. Springer; 2006.
