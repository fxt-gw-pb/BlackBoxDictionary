---
title: 倾向评分匹配
english_name: Propensity Score Matching
slug: propensity-score-matching
aliases: [PSM, propensity score matching, 倾向性评分匹配, "倾向评分匹配（Propensity Score Matching）"]
category: 因果推断
subcategory: 倾向评分方法
tags: [医学统计, 流行病学, 因果推断, 混杂调整, 观察性研究]
status: 已建
difficulty: intermediate
question_type: 观察性研究混杂调整与处理效应估计
data_type: [观察性研究数据]
outcome_type: [二分类, 连续型, 时间到事件]
python_packages: [scikit-learn]
r_packages: [MatchIt, cobalt]
---

# 倾向评分匹配（Propensity Score Matching）

> 本卡是 [[因果推断（Causal Inference）]] 家族下的具体方法，因果识别假设与潜在结果框架见主条目，这里聚焦倾向评分匹配的做法与实践要点。

## 1. 方法概览

### 1.1 定义

倾向评分匹配先估计每个个体接受处理的条件概率（倾向评分），再把处理组个体与倾向评分相近的对照组个体配对，从而在已测混杂上构造可比的两组，估计处理效应。

### 1.2 它主要解决什么问题

- 研究问题：在无法随机化的观察性研究中，如何减少已测混杂、更接近可比较的组间对比。
- 适用任务：平均处理效应（尤其处理组的 ATT）估计、混杂调整、真实世界治疗效果评估。
- 常见医学场景：两种治疗方案疗效比较、暴露与结局关联、器械或术式的真实世界评价。

### 1.3 直觉理解

Rosenbaum 与 Rubin 证明：只要按倾向评分平衡，就等价于按全部已测混杂平衡。于是我们不必逐个匹配所有变量，只需匹配一个“综合分数”，让处理组和对照组在这个分数上分布相近。

## 2. 数学形式

### 2.1 核心公式

倾向评分是给定协变量下接受处理的概率：

$$
e(X)=P(T=1\mid X)
$$

在可忽略性与正值性下，倾向评分是平衡评分：

$$
\{Y(1),Y(0)\}\perp T\mid X\ \Rightarrow\ \{Y(1),Y(0)\}\perp T\mid e(X)
$$

匹配后常估计处理组平均处理效应：

$$
\mathrm{ATT}=E\big[Y(1)-Y(0)\mid T=1\big]
$$

### 2.2 参数或统计量含义

- $e(X)$：倾向评分，通常用 Logistic 回归或机器学习估计。
- $T$：处理/暴露指示；$Y$：结局。
- 卡钳（caliper）：允许配对的倾向评分（或其 logit）最大距离，常取 $0.2\times$ logit 标准差。
- 标准化均差（SMD）：匹配后协变量平衡的核心诊断，通常要求 $<0.1$。

### 2.3 关键假设

- 可忽略性（无未测混杂）：条件于已测协变量后处理可交换。
- 正值性/重叠：处理与对照的倾向评分分布充分重叠。
- 倾向评分模型近似正确；只纳入混杂与结局预测变量，不纳入处理后变量。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：处理变量、混杂协变量（连续/分类）。
- 因变量类型：连续、二分类或时间到事件结局。
- 数据结构：横断面或队列观察性数据，每行一个个体。
- 是否适合高维数据：可用，但高维需正则化/机器学习估计倾向评分并谨慎评估重叠。
- 是否适合缺失较多数据：需先处理缺失（如多重插补）。
- 是否适合删失数据：结局为生存时可在匹配样本上做 Cox/KM。
- 是否适合重复测量数据：基础版针对个体；纵向处理需边际结构模型等扩展。

### 3.2 示例表格

| 个体 | Age | Charlson | 处理 T | 倾向评分 e(X) | 结局 Y |
| --- | --- | --- | --- | --- | --- |
| 1 | 73 | 4 | 1 | 0.78 | 1 |
| 2 | 70 | 4 | 0 | 0.75 | 0 |
| 3 | 58 | 2 | 1 | 0.41 | 0 |
| 4 | 60 | 2 | 0 | 0.43 | 1 |

### 3.3 输入与产出

#### 输入

- 输入数据：处理变量、混杂协变量、结局。
- 关键变量：处理定义、协变量集合、匹配设置（比例、卡钳、有无放回）。
- 需要预处理的内容：缺失处理、变量编码、重叠检查。

#### 产出

- 模型对象/统计结果：倾向评分、匹配样本、平衡表。
- 参数估计：匹配后 ATT 及其区间。
- 预测结果：匹配对/权重、未匹配样本。
- 不确定性指标：考虑匹配结构的标准误、敏感性分析（如 Rosenbaum bounds）。

## 4. 适用场景

- 适合：已测混杂较充分、处理与对照有良好重叠的观察性研究。
- 不适合：关键混杂未测量、重叠差、样本量小到匹配后损失过多。
- 使用前需要特别检查的点：倾向评分重叠、匹配后 SMD 平衡、匹配丢弃了多少样本、未测混杂敏感性。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# df: 含处理 T、混杂 X1..Xk、结局 Y
X = df[["Age", "Charlson", "PriorAdmission"]]
T = df["Drug"].values
ps = LogisticRegression(max_iter=5000).fit(X, T).predict_proba(X)[:, 1]
df["logit_ps"] = np.log(ps / (1 - ps))

treated = df[T == 1]
control = df[T == 0]
nn = NearestNeighbors(n_neighbors=1).fit(control[["logit_ps"]])
dist, idx = nn.kneighbors(treated[["logit_ps"]])
caliper = 0.2 * df["logit_ps"].std()
keep = dist.ravel() <= caliper

matched = pd.concat([treated[keep], control.iloc[idx.ravel()[keep]]])
att = matched.loc[matched.Drug == 1, "Y"].mean() - matched.loc[matched.Drug == 0, "Y"].mean()
print("匹配后 ATT(风险差):", round(att, 3))
```

### 5.2 R

常用包：

- `MatchIt`
- `cobalt`

```r
library(MatchIt)
library(cobalt)

m <- matchit(Drug ~ Age + Charlson + PriorAdmission,
             data = df, method = "nearest",
             caliper = 0.2, distance = "glm")
bal.tab(m, un = TRUE)                 # 匹配前后 SMD
md <- match.data(m)

fit <- glm(Y ~ Drug, data = md, weights = weights, family = binomial())
summary(fit)
```

## 6. 结果如何解释

- 核心结果看什么：匹配后协变量平衡（SMD）、ATT 估计与区间、样本损失情况。
- 每个主要参数如何解释：ATT 是“在处理组人群中”的平均处理效应；匹配改变了目标人群，解读要限定于此。
- 临床或医学意义如何表达：强调“在可忽略性成立的前提下”的效应估计，避免因果定论。
- 常见误读：用匹配后 p 值判断平衡（应看 SMD）；把 ATT 当成对全人群的 ATE；忽略未测混杂。

## 7. 推荐可视化

- 匹配前后倾向评分分布重叠图。
- 协变量标准化均差的 Love plot。
- 匹配后结局比较（如 KM 曲线或森林图）。

## 8. 优势、局限与常见坑

### 优势

- 概念直观、结果易于向临床读者解释。
- 把多维混杂压缩为单一评分，便于诊断平衡。
- 分析与结局模型分离，降低“看着结果调模型”的风险。

### 局限

- 只能平衡已测混杂，对未测混杂无能为力。
- 匹配常丢弃未匹配样本，降低效率并改变目标人群。
- 结果对匹配设置（卡钳、比例、是否放回）敏感。

### 常见坑

- 把处理后变量或中介纳入倾向评分模型。
- 只报显著性不报平衡与重叠。
- 忽略匹配结构对标准误的影响；未做未测混杂敏感性分析。

## 9. 与相近方法的区别

- 和 [[因果推断（Causal Inference）]] 主条目的关系：本卡是其倾向评分家族的具体实现。
- 和 [[逆概率处理加权（Inverse Probability of Treatment Weighting, IPTW）]] 的区别：IPTW 用权重保留全部样本、常估 ATE，匹配丢弃样本、常估 ATT；King & Nielsen 建议优先考虑加权或全匹配。
- 和直接协变量回归调整的区别：匹配对结局模型形式依赖更小，但会损失样本。

## 10. 医学研究中的典型应用

- 真实世界数据中两种药物/术式的疗效与安全性比较。
- 暴露（如吸烟、用药）与结局关联的混杂调整。
- 注册登记数据的观察性疗效评估。

## 11. 相关方法

- [[因果推断（Causal Inference）]]
- [[逆概率处理加权（Inverse Probability of Treatment Weighting, IPTW）]]
- [[Logistic回归（Logistic Regression）]]
- [[Cox比例风险模型（Cox Proportional Hazards Model）]]

## 12. 参考资料

- Rosenbaum PR, Rubin DB. The central role of the propensity score in observational studies for causal effects. *Biometrika*. 1983;70(1):41-55.
- Austin PC. An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behav Res*. 2011;46(3):399-424.
- King G, Nielsen R. Why propensity scores should not be used for matching. *Political Analysis*. 2019;27(4):435-454.
- Ho DE, Imai K, King G, Stuart EA. MatchIt: nonparametric preprocessing for parametric causal inference. *J Stat Softw*. 2011;42(8):1-28.
