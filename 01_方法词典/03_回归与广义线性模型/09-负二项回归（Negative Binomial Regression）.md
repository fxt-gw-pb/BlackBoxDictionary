---
title: 负二项回归
english_name: Negative Binomial Regression
slug: negative-binomial-regression
aliases: [negative binomial regression, NB回归, 负二项模型, "负二项回归（Negative Binomial Regression）"]
category: 回归与广义线性模型
subcategory: 计数结局建模
tags: [医学统计, 数据科学, 计数数据, 过度离散, GLM]
status: 已建
difficulty: intermediate
question_type: 过度离散计数结局建模
data_type: [表格数据]
outcome_type: [计数型]
python_packages: [statsmodels]
r_packages: [MASS]
---

# 负二项回归（Negative Binomial Regression）

## 1. 方法概览

### 1.1 定义

负二项回归是计数结局的回归模型，通过在 Poisson 基础上引入一个额外的离散参数，允许方差大于均值，从而处理过度离散的计数数据。

### 1.2 它主要解决什么问题

- 研究问题：某计数结局如何随协变量变化，且数据方差明显超过均值。
- 适用任务：过度离散计数建模、率比估计、含 offset 的发生率建模。
- 常见医学场景：住院次数、急诊就诊次数、癫痫发作次数、感染事件计数等常见的过度离散计数。

### 1.3 直觉理解

Poisson 回归假设“均值等于方差”，但真实计数常因个体异质性而更分散。负二项模型可看作“每个个体的 Poisson 率本身服从 Gamma 分布”的混合，因而天然容纳这种额外波动。

## 2. 数学形式

### 2.1 核心公式

均值用 log 链接建模，方差随离散参数 $\alpha$ 增大：

$$
\log(\mu_i)=X_i^\top\boldsymbol{\beta}\ (+\log t_i),\qquad
\mathrm{Var}(Y_i)=\mu_i+\alpha\,\mu_i^2
$$

其中 $\log t_i$ 为可选 offset（暴露时间/人时）。$\alpha\to 0$ 时退化为 Poisson。

### 2.2 参数或统计量含义

- $\boldsymbol{\beta}$：log 率尺度上的效应；$\exp(\beta_j)$ 为发生率比（IRR）。
- $\alpha$：离散参数，$\alpha>0$ 表示过度离散。
- offset $\log t_i$：把计数转为率（每单位人时）。
- 均值–方差关系：$\mu+\alpha\mu^2$（NB2 参数化）。

### 2.3 关键假设

- 结局为非负整数计数。
- log 均值模型正确、观测独立。
- 方差为 $\mu+\alpha\mu^2$ 形式；若仍不符可考虑零膨胀或其他分布。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、二分类、多分类。
- 因变量类型：计数型结局。
- 数据结构：每行一个观测，含计数与（可选）暴露量。
- 是否适合高维数据：可结合正则化，但需谨慎。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：计数删失需专门模型。
- 是否适合重复测量数据：聚类计数用 NB-GLMM 或 GEE。

### 3.2 示例表格

| 个体 | 就诊次数 count | 随访月数 t | 年龄 | 合并症 |
| --- | --- | --- | --- | --- |
| 1 | 3 | 12 | 67 | 1 |
| 2 | 0 | 12 | 54 | 0 |
| 3 | 9 | 12 | 72 | 1 |
| 4 | 1 | 6 | 60 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：计数结局、协变量、可选 offset。
- 关键变量：计数、暴露时间、协变量。
- 需要预处理的内容：确认过度离散、构造 offset、编码分类变量。

#### 产出

- 模型对象/统计结果：系数、IRR、离散参数 $\alpha$。
- 参数估计：$\boldsymbol{\beta}$、$\exp(\beta)$。
- 预测结果：期望计数或率。
- 不确定性指标：标准误、IRR 区间、与 Poisson 的似然比检验。

## 4. 适用场景

- 适合：过度离散的计数结局、需要率比解释、含暴露时间。
- 不适合：无过度离散（Poisson 足够）、零过多且结构性零明显（考虑零膨胀）。
- 使用前需要特别检查的点：过度离散检验、零的比例、offset 是否需要。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`

```python
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# df 含 count, t(随访), age, comorb
m = smf.glm("count ~ age + comorb",
            data=df,
            family=sm.families.NegativeBinomial(),
            offset=np.log(df["t"])).fit()
print(m.summary())
print("IRR:", np.exp(m.params))
```

### 5.2 R

常用包：

- `MASS`

```r
library(MASS)

fit <- glm.nb(count ~ age + comorb + offset(log(t)), data = df)
summary(fit)
exp(cbind(IRR = coef(fit), confint(fit)))
```

## 6. 结果如何解释

- 核心结果看什么：IRR 的方向、大小、区间，以及离散参数 $\alpha$。
- 每个主要参数如何解释：IRR=1.3 表示该因素使事件发生率约提高 30%（其他不变）。
- 临床或医学意义如何表达：结合 offset 说明是“每人时发生率”的比值。
- 常见误读：把 IRR 当风险差；忽略过度离散仍用 Poisson 导致标准误偏小、假阳性。

## 7. 推荐可视化

- IRR 森林图。
- 观测计数分布与模型预测分布对比（含零的比例）。
- 残差/离散诊断图。

## 8. 优势、局限与常见坑

### 优势

- 正确处理过度离散，标准误更可信。
- 保留 Poisson 的率比解释与 offset 机制。
- 有对 Poisson 的嵌套检验（$\alpha=0$）。

### 局限

- 假设特定的均值–方差形式。
- 结构性零多时仍不足（需零膨胀/障碍模型）。
- 极端离散或小样本估计不稳。

### 常见坑

- 未检验过度离散就默认 Poisson。
- 建模率却漏掉 offset。
- 零膨胀数据强套普通 NB。

## 9. 与相近方法的区别

- 和 [[Poisson回归（Poisson Regression）]] 的区别：Poisson 要求均值=方差，NB 放宽为方差随均值二次增长。
- 和 [[Quasi-Likelihood与过度离散（Quasi-Likelihood and Overdispersion）]] 的区别：quasi-Poisson 只调方差不设完整分布，NB 是完整似然可比较 AIC。
- 和零膨胀/障碍模型的区别：后者显式建模“额外的零”，适合结构性零多的数据。

## 10. 医学研究中的典型应用

- 医疗利用（住院/急诊次数）建模。
- 发作/事件计数（癫痫、跌倒、感染）与危险因素分析。
- 以人时为分母的发生率比较。

## 11. 相关方法

- [[Poisson回归（Poisson Regression）]]
- [[Quasi-Likelihood与过度离散（Quasi-Likelihood and Overdispersion）]]
- [[广义线性模型（Generalized Linear Model, GLM）]]

## 12. 参考资料

- Hilbe JM. *Negative Binomial Regression*. 2nd ed. Cambridge University Press; 2011.
- Cameron AC, Trivedi PK. *Regression Analysis of Count Data*. 2nd ed. Cambridge University Press; 2013.
- Venables WN, Ripley BD. *Modern Applied Statistics with S*. 4th ed. Springer; 2002. （`MASS::glm.nb`）
