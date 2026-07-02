---
title: 逆概率处理加权
english_name: Inverse Probability of Treatment Weighting, IPTW
slug: inverse-probability-of-treatment-weighting-iptw
aliases: [IPTW, IPW, inverse probability weighting, 逆概率加权, "逆概率处理加权（Inverse Probability of Treatment Weighting, IPTW）"]
category: 因果推断
subcategory: 倾向评分方法
tags: [医学统计, 流行病学, 因果推断, 混杂调整, 边际结构模型]
status: 已建
difficulty: intermediate
question_type: 观察性研究混杂调整与边际处理效应估计
data_type: [观察性研究数据]
outcome_type: [二分类, 连续型, 时间到事件]
python_packages: [scikit-learn]
r_packages: [WeightIt, survey]
---

# 逆概率处理加权（Inverse Probability of Treatment Weighting, IPTW）

> 本卡是 [[因果推断（Causal Inference）]] 家族下的具体方法，因果识别假设与潜在结果框架见主条目，这里聚焦 IPTW 的构造、稳定化与实践要点。

## 1. 方法概览

### 1.1 定义

IPTW 用倾向评分的倒数给每个个体加权，构造一个“伪人群”，在其中处理分配与已测混杂独立，从而估计边际（population-average）处理效应。

### 1.2 它主要解决什么问题

- 研究问题：观察性数据中如何在保留全部样本的前提下调整混杂，估计总体平均处理效应。
- 适用任务：ATE 估计、边际结构模型、时依处理与时依混杂建模。
- 常见医学场景：真实世界疗效比较、随时间变化的用药暴露、需要总体效应而非仅处理组效应的评估。

### 1.3 直觉理解

被处理概率越低却接受了处理的人越“稀有”，给他更大权重来代表那些本该像他一样却没被处理的人。加权后处理组和对照组在混杂上看起来分布相同，组间对比就更接近随机化。

## 2. 数学形式

### 2.1 核心公式

倾向评分 $e(X)=P(T=1\mid X)$。估计 ATE 的（未稳定化）权重为：

$$
w_i=\frac{T_i}{e(X_i)}+\frac{1-T_i}{1-e(X_i)}
$$

稳定化权重降低极端权重的方差：

$$
sw_i=\frac{T_i\,P(T=1)}{e(X_i)}+\frac{(1-T_i)\,P(T=0)}{1-e(X_i)}
$$

加权后用边际结构模型 $E[Y^t]=g^{-1}(\beta_0+\beta_1 t)$ 估计效应。

### 2.2 参数或统计量含义

- $e(X)$：倾向评分。
- $w_i$：ATE 权重；$sw_i$：稳定化权重（均值应接近 1）。
- 边际结构模型系数：加权回归给出的边际处理效应。
- 有效样本量：权重离散度的函数，反映加权带来的信息损失。

### 2.3 关键假设

- 可忽略性（无未测混杂）与一致性、SUTVA。
- 正值性尤为关键：任何协变量组合下处理概率不能接近 0 或 1，否则权重爆炸。
- 倾向评分模型近似正确（否则考虑双重稳健 AIPW）。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：处理变量、混杂协变量。
- 因变量类型：连续、二分类或时间到事件。
- 数据结构：队列/横断面观察性数据；纵向数据可做时依加权。
- 是否适合高维数据：可用机器学习估权，但正值性与极端权重更需警惕。
- 是否适合缺失较多数据：需先处理缺失，可与逆概率删失加权（IPCW）结合。
- 是否适合删失数据：可结合 IPCW 处理删失。
- 是否适合重复测量数据：适合，边际结构模型是其经典应用场景。

### 3.2 示例表格

| 个体 | 处理 T | 倾向评分 e(X) | 权重 w | 稳定化权重 sw | 结局 Y |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0.80 | 1.25 | 0.50 | 1 |
| 2 | 0 | 0.30 | 1.43 | 0.86 | 0 |
| 3 | 1 | 0.20 | 5.00 | 2.00 | 1 |
| 4 | 0 | 0.70 | 3.33 | 2.00 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：处理变量、混杂协变量、结局。
- 关键变量：处理定义、协变量集合、权重类型（稳定化/截断）。
- 需要预处理的内容：缺失处理、正值性/重叠检查、权重诊断。

#### 产出

- 模型对象/统计结果：倾向评分、权重、加权结局模型。
- 参数估计：ATE（风险差、风险比、率比或 HR）。
- 预测结果：加权后的组间结局对比。
- 不确定性指标：稳健（三明治）或 bootstrap 标准误、加权后平衡表。

## 4. 适用场景

- 适合：需要总体平均效应、重叠良好、有时依处理/混杂的纵向数据。
- 不适合：正值性严重违背、极端权重主导估计的场景。
- 使用前需要特别检查的点：权重分布与极值、加权后 SMD 平衡、稳定化与截断、正值性。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

X = df[["Age", "Charlson", "PriorAdmission"]]
T = df["Drug"].values
Y = df["Y"].values

e = LogisticRegression(max_iter=5000).fit(X, T).predict_proba(X)[:, 1]
pT = T.mean()
sw = np.where(T == 1, pT / e, (1 - pT) / (1 - e))   # 稳定化权重

wls = sm.WLS(Y, sm.add_constant(T), weights=sw).fit(cov_type="HC1")
print("加权风险差(ATE):", round(wls.params[1], 3))
```

### 5.2 R

常用包：

- `WeightIt`
- `survey`

```r
library(WeightIt)
library(survey)

w <- weightit(Drug ~ Age + Charlson + PriorAdmission,
              data = df, method = "ps",
              estimand = "ATE", stabilize = TRUE)

des <- svydesign(ids = ~1, weights = ~w$weights, data = df)
fit <- svyglm(Y ~ Drug, design = des, family = quasibinomial())
summary(fit)
```

## 6. 结果如何解释

- 核心结果看什么：ATE 估计与稳健区间、权重分布、加权后平衡。
- 每个主要参数如何解释：加权模型的处理系数是边际处理效应（对整个目标人群）。
- 临床或医学意义如何表达：强调“在识别假设与正值性成立时”的总体效应；报告风险差/比时说明尺度。
- 常见误读：忽略极端权重导致的高方差；用普通（未稳健）方差低估不确定性；把 ATE 当条件效应。

## 7. 推荐可视化

- 权重分布直方图（识别极端权重）。
- 加权前后协变量 SMD 的 Love plot。
- 倾向评分重叠图。

## 8. 优势、局限与常见坑

### 优势

- 保留全部样本，效率通常优于匹配。
- 直接估计边际（总体）效应，尺度可解释。
- 天然扩展到时依处理的边际结构模型。

### 局限

- 对正值性违背和极端权重敏感，方差可能很大。
- 只调整已测混杂。
- 依赖倾向评分模型正确设定。

### 常见坑

- 不做权重截断/稳定化，少数极端权重主导结果。
- 用普通标准误而非稳健/bootstrap。
- 正值性明显违背仍强行加权。

## 9. 与相近方法的区别

- 和 [[倾向评分匹配（Propensity Score Matching）]] 的区别：IPTW 加权保留样本、常估 ATE，匹配丢弃样本、常估 ATT。
- 和双重稳健（AIPW）的关系：AIPW 同时建倾向评分和结局模型，任一正确即一致，比单纯 IPTW 更稳健。
- 和 [[因果推断（Causal Inference）]] 主条目的关系：本卡是其倾向评分加权的具体实现。

## 10. 医学研究中的典型应用

- 真实世界数据中总体层面的疗效比较。
- 时依用药暴露与结局的边际结构模型分析。
- 结合 IPCW 处理随访删失的加权生存分析。

## 11. 相关方法

- [[因果推断（Causal Inference）]]
- [[倾向评分匹配（Propensity Score Matching）]]
- [[Logistic回归（Logistic Regression）]]
- [[广义估计方程（Generalized Estimating Equations, GEE）]]

## 12. 参考资料

- Robins JM, Hernán MA, Brumback B. Marginal structural models and causal inference in epidemiology. *Epidemiology*. 2000;11(5):550-560.
- Austin PC, Stuart EA. Moving towards best practice when using inverse probability of treatment weighting (IPTW). *Stat Med*. 2015;34(28):3661-3679.
- Hernán MA, Robins JM. *Causal Inference: What If*. Chapman and Hall/CRC; 2020.
