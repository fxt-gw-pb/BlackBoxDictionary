---
title: 高斯混合模型
english_name: Gaussian Mixture Model, GMM
slug: gaussian-mixture-model-gmm
aliases: [GMM, gaussian mixture model, 高斯混合, "高斯混合模型（Gaussian Mixture Model, GMM）"]
category: 聚类与无监督学习
subcategory: 概率聚类
tags: [医学统计, 数据科学, 聚类, 无监督学习, 概率模型]
status: 已建
difficulty: intermediate
question_type: 软聚类与密度估计
data_type: [表格数据]
outcome_type: [无监督分群]
python_packages: [scikit-learn]
r_packages: [mclust]
---

# 高斯混合模型（Gaussian Mixture Model, GMM）

## 1. 方法概览

### 1.1 一句话本质

GMM 把总体密度表示为若干高斯成分的加权和，并为每名患者给出属于各成分的后验概率。

### 1.2 定义

高斯混合模型假设每个观测由一个未观察到的成分生成，每个成分服从自己的多元高斯分布。模型同时估计成分比例、均值、协方差和病例责任概率，可用于密度估计与软聚类。

### 1.3 它主要解决什么问题

- 潜在亚群彼此重叠，硬标签过于武断。
- 亚群可能呈不同方向和大小的椭圆分布。
- 既希望分群，也希望得到完整概率密度。

### 1.4 直觉与类比

一张混合人群的指标直方图可能由两个钟形分布叠加而成。GMM 尝试还原每个钟形分布，并回答“这个患者更像哪个成分，以及有多不确定”。

## 2. 核心思想与原理

### 2.1 从距离到生成概率

[[K-means聚类（K-means Clustering）]] 只比较到质心的距离；GMM 还考虑簇的大小、方向和先验占比。同样距离下，位于高方差成分中的点可能得到更高概率。

### 2.2 软归属

每个病例对所有成分都有责任概率，且总和为 1。接近重叠区的病例可呈 0.55/0.45，而不是被强行解释为确定亚型。

### 2.3 EM 估计

成分标签未知使直接最大化似然困难。[[期望最大化算法（Expectation-Maximization Algorithm, EM）]] 交替计算责任概率，并用这些软权重更新参数。

## 3. 数学形式

### 3.1 混合密度

$$
p(x)=\sum_{k=1}^{K}\pi_k
\mathcal N(x\mid\mu_k,\Sigma_k)
$$

其中 $\pi_k\ge0$ 且 $\sum_k\pi_k=1$。

### 3.2 责任概率

$$
\gamma_{ik}
=P(z_i=k\mid x_i)
=
\frac{\pi_k\mathcal N(x_i\mid\mu_k,\Sigma_k)}
{\sum_{j=1}^{K}\pi_j\mathcal N(x_i\mid\mu_j,\Sigma_j)}
$$

### 3.3 M 步更新

令 $N_k=\sum_i\gamma_{ik}$：

$$
\pi_k=\frac{N_k}{n},\qquad
\mu_k=\frac{1}{N_k}\sum_i\gamma_{ik}x_i
$$

$$
\Sigma_k=
\frac{1}{N_k}
\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top
$$

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 成分近似高斯 | 非椭圆结构拟合较差 | 密度图与后验检查 |
| 模型可识别 | 成分重叠或退化 | 多初始化、协方差正则 |
| 样本足以估协方差 | 高维协方差不稳定 | 降维或约束协方差 |
| 成分数合理 | 过分拆分或合并亚群 | BIC、稳定性与临床解释 |

## 4. 手把手算例

一维 GMM 有两个成分，初始参数为：

$$
\pi_1=\pi_2=0.5,\quad
\mu_1=0,\quad\mu_2=4,\quad
\sigma_1^2=\sigma_2^2=1
$$

观测为 $x=(0,1,4)$。

**Step 1：计算 $x=1$ 的责任概率。** 两个高斯密度共有的常数会抵消，只需比较指数部分：

$$
\exp[-(1-0)^2/2]\approx0.607
$$

$$
\exp[-(1-4)^2/2]\approx0.011
$$

因此：

$$
\gamma_{1,1}=
\frac{0.607}{0.607+0.011}
\approx0.982
$$

该点属于成分 1 的概率约为 98.2%。

**Step 2：三个观测的责任度。**

| $x$ | 成分 1 | 成分 2 |
| ---: | ---: | ---: |
| 0 | 0.9997 | 0.0003 |
| 1 | 0.9820 | 0.0180 |
| 4 | 0.0003 | 0.9997 |

**Step 3：更新均值与权重。**

$$
N_1\approx1.982,\qquad N_2\approx1.018
$$

$$
\mu_1^{new}\approx0.496,\qquad
\mu_2^{new}\approx3.946
$$

$$
\pi_1^{new}\approx0.661,\qquad
\pi_2^{new}\approx0.339
$$

**结论：** GMM 没有先把 $x=1$ 生硬贴标签，而是用 0.982/0.018 的软归属参与两组参数更新。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 以连续变量为主，通常需标准化。
- 缺失需妥善处理；高维时常先筛选或降维。
- 普通 GMM 不直接处理类别变量、删失或重复测量相关性。

### 5.2 输入与产出

输入包括成分数、协方差结构、初始化和正则项。输出包括 $\pi_k,\mu_k,\Sigma_k$、对数似然、BIC、硬标签和责任概率矩阵。

## 6. 适用场景

- 患者亚群可能重叠，希望保留归属不确定性。
- 簇近似椭圆形且可由高斯成分描述。
- 不适合任意形状、重尾异常点很多或主要为类别变量的数据。

## 7. 实现

### 7.1 Python

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
X_s = scaler.transform(X)

candidates = []
for k in range(1, 7):
    model = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=20,
        reg_covar=1e-6,
        random_state=42,
    ).fit(X_s)
    candidates.append((model.bic(X_s), model))

best_bic, model = min(candidates, key=lambda item: item[0])
cluster = model.predict(X_s)
prob = model.predict_proba(X_s)
uncertainty = 1 - prob.max(axis=1)
print(best_bic, model.converged_, model.weights_)
```

### 7.2 R

```r
library(mclust)

x <- scale(df[, c("BMI", "TG", "HDL", "FastingGlucose", "SBP")])
fit <- Mclust(x, G = 1:6)

cluster <- fit$classification
prob <- fit$z
uncertainty <- fit$uncertainty
summary(fit)
```

## 8. 结果如何解释

- 成分编号可交换，本身没有从低风险到高风险的固定顺序。
- 责任概率接近 0.5 表示边界病例，不应只报告硬标签。
- BIC 较低表示在拟合与复杂度之间更优，但不能单独证明生物亚型真实存在。
- 应在独立结局或外部队列中描述簇的临床可重复性。

## 9. 诊断与稳健性

1. 比较不同成分数和协方差结构的 BIC。
2. 用多个随机初始化检查局部最优。
3. 检查最小成分比例、协方差特征值和收敛状态。
4. bootstrap 重采样后评估标签和责任度稳定性。
5. 比较 K-means、层次聚类及不同标准化方案。

## 10. 推荐可视化

- BIC 随成分数和协方差模型变化图。
- 二维投影上的成分椭圆与责任度透明度。
- 成分均值热图和归属概率热图。

下图展示 GMM 在二维空间中的聚类结果：

![](../../04_示例图像/gmm_clustering_result.png)

## 11. 优势、局限与常见坑

**优势：** 概率软聚类，可拟合椭圆簇，同时提供密度模型。

**局限：** 对高斯假设、初始化和成分数敏感，高维协方差参数很多。

**常见坑：** 把统计成分直接称为疾病亚型；只运行一次；忽略小成分退化；只报告硬标签；在全数据降维后夸大稳定性。

## 12. 与相近方法的区别

- [[K-means聚类（K-means Clustering）]]：硬分配、球形等方差是其更强限制。
- [[模糊C均值聚类（Fuzzy C-Means Clustering, FCM）]]：隶属度来自模糊目标，而非生成概率后验。
- [[DBSCAN（Density-Based Spatial Clustering of Applications with Noise）]]：按密度连通找任意形状簇并标噪声。
- 选择经验：重视概率归属且簇近似椭圆时考虑 GMM。

## 13. 医学研究中的典型应用

- 代谢、炎症或生物标志物表型分群。
- 多组学降维后的潜在患者成分识别。
- 连续指标混合分布分解与异常概率探索。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| mixture component | 混合密度中的一个潜在分布 |
| mixing weight | 成分在总体中的比例 |
| responsibility | 病例属于某成分的后验概率 |
| covariance type | 对成分形状和方向的约束 |
| label switching | 成分编号可互换但模型密度不变 |

## 15. 相关方法

- [[期望最大化算法（Expectation-Maximization Algorithm, EM）]]
- [[K-means聚类（K-means Clustering）]]
- [[模糊C均值聚类（Fuzzy C-Means Clustering, FCM）]]
- [[主成分分析（Principal Component Analysis, PCA）]]

## 16. 参考资料

- McLachlan G, Peel D. *Finite Mixture Models*. Wiley; 2000.
- Bishop CM. *Pattern Recognition and Machine Learning*. Springer; 2006.
- scikit-learn Developers. `GaussianMixture` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) （访问日期：2026-07-09）
