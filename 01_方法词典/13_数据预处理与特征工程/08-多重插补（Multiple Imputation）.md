---
title: 多重插补
english_name: Multiple Imputation
slug: multiple-imputation
aliases: [MI, multiple imputation, MICE, 链式方程多重插补, "多重插补（Multiple Imputation）"]
category: 数据预处理与特征工程
subcategory: 缺失数据处理
tags: [医学统计, 数据科学, 缺失数据, 插补, 不确定性]
status: 已建
difficulty: intermediate
question_type: 缺失数据处理与不确定性传播
data_type: [表格数据]
outcome_type: [连续型, 二分类, 时间到事件]
python_packages: [statsmodels, scikit-learn]
r_packages: [mice]
---

# 多重插补（Multiple Imputation）

## 1. 方法概览

### 1.1 定义

多重插补通过为每个缺失值生成多套（$m$ 套）“合理猜测”得到多份完整数据集，分别分析后按 Rubin 规则合并结果，使插补带来的不确定性被正确传播到最终推断中。

### 1.2 它主要解决什么问题

- 研究问题：数据有缺失时，如何在不丢弃样本、又不低估不确定性的前提下做有效推断。
- 适用任务：缺失协变量/结局处理、含缺失数据的回归与生存分析、敏感性分析。
- 常见医学场景：随访失访、实验室指标漏测、量表条目缺失、多中心数据字段不齐。

### 1.3 直觉理解

单一插补（如用均值填补）假装“我们确切知道缺的是多少”，会人为缩小方差。多重插补做多套不同的合理填补，让“我们并不确定缺的是多少”这件事体现在结果的波动里，从而给出诚实的置信区间。

## 2. 数学形式

### 2.1 核心公式

对 $m$ 套插补数据分别得到估计 $\hat{\theta}_k$ 与方差 $U_k$，Rubin 合并规则为：

$$
\bar{\theta}=\frac{1}{m}\sum_{k=1}^{m}\hat{\theta}_k,\qquad
\bar{U}=\frac{1}{m}\sum_{k=1}^{m}U_k,\qquad
B=\frac{1}{m-1}\sum_{k=1}^{m}(\hat{\theta}_k-\bar{\theta})^2
$$

总方差与缺失信息比为：

$$
T=\bar{U}+\Big(1+\frac{1}{m}\Big)B,\qquad
\mathrm{FMI}\approx\frac{(1+1/m)B}{T}
$$

### 2.2 参数或统计量含义

- $\bar{\theta}$：合并后的点估计。
- $\bar{U}$（within）：插补内方差；$B$（between）：插补间方差，反映缺失带来的额外不确定性。
- $T$：合并后总方差，用于构造区间。
- FMI：缺失信息比，衡量缺失对该参数的影响程度。

### 2.3 关键假设

- 缺失机制通常假设为 MAR（随机缺失，可由观测变量解释）。
- 插补模型与分析模型“协调”（congenial），插补时应纳入结局与相关辅助变量。
- 插补模型近似正确；MNAR 时需专门的敏感性分析。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、二分类、多分类均可（分变量类型选插补模型）。
- 因变量类型：连续、二分类或时间到事件。
- 数据结构：含缺失的表格数据。
- 是否适合高维数据：可用，但变量很多时需筛选辅助变量、控制共线。
- 是否适合缺失较多数据：正是为此设计；但缺失比例极高时结论对模型更敏感。
- 是否适合删失数据：可结合生存插补（如纳入 Nelson-Aalen 累积风险作为预测量）。
- 是否适合重复测量数据：需用纵向/多层插补模型。

### 3.2 示例表格

| 个体 | Age | SBP | Chol | 是否缺失 |
| --- | --- | --- | --- | --- |
| 1 | 61 | 140 | 5.2 | 完整 |
| 2 | 58 | NA | 6.1 | SBP 缺失 |
| 3 | 72 | 155 | NA | Chol 缺失 |
| 4 | 49 | 128 | 4.8 | 完整 |

### 3.3 输入与产出

#### 输入

- 输入数据：含缺失的完整变量集合（含结局与辅助变量）。
- 关键变量：缺失变量、预测变量、插补次数 $m$、每变量的插补模型。
- 需要预处理的内容：判断缺失机制、选择插补方法、检查插补合理性。

#### 产出

- 模型对象/统计结果：$m$ 套完整数据与各自分析结果。
- 参数估计：Rubin 合并后的估计与区间。
- 预测结果：可在合并层面报告。
- 不确定性指标：合并方差、FMI、自由度调整后的检验。

## 4. 适用场景

- 适合：MAR 假设合理、缺失非极端、需要有效区间估计的分析。
- 不适合：缺失强烈依赖未观测值（MNAR）且无敏感性框架、缺失比例极高时。
- 使用前需要特别检查的点：缺失机制、是否纳入结局与辅助变量、插补诊断、$m$ 是否足够。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`
- `scikit-learn`

```python
import numpy as np
import statsmodels.api as sm
from statsmodels.imputation import mice

# df 含缺失; 用链式方程插补并按 Rubin 规则合并回归结果
imp = mice.MICEData(df)
fit = mice.MICE("Chol ~ Age + SBP", sm.OLS, imp).fit(n_burnin=10, n_imputations=20)
print(fit.summary())   # 已按 Rubin 规则合并，含 FMI
```

### 5.2 R

常用包：

- `mice`

```r
library(mice)

imp <- mice(df, m = 20, method = "pmm", seed = 1)   # 预测均值匹配
fit <- with(imp, lm(Chol ~ Age + SBP))
pooled <- pool(fit)
summary(pooled)        # 合并估计、区间与 FMI
```

## 6. 结果如何解释

- 核心结果看什么：合并后的点估计与区间、FMI、插补诊断是否合理。
- 每个主要参数如何解释：区间已包含插补不确定性；FMI 高说明该参数受缺失影响大。
- 临床或医学意义如何表达：报告应说明缺失比例、机制假设与插补策略，而非默默删除缺失样本。
- 常见误读：只取第一套插补分析；用单一插补冒充多重插补；忽略 MAR 假设的合理性。

## 7. 推荐可视化

- 缺失模式图（缺失共现结构）。
- 观测值与插补值的密度/分布对比（诊断插补合理性）。
- 迭代收敛轨迹图（链式方程收敛检查）。

## 8. 优势、局限与常见坑

### 优势

- 保留全部样本，避免完整个案分析的效率损失与潜在偏倚。
- 正确传播不确定性，给出诚实区间。
- 框架通用，可配合各类分析模型。

### 局限

- 依赖 MAR 与插补模型设定，MNAR 需额外假设。
- 实现与诊断比单一插补复杂。
- 极高缺失比例下结论对模型很敏感。

### 常见坑

- 插补时漏掉结局变量，导致协变量与结局关系被削弱。
- 先插补再单一分析（不合并），低估方差。
- 用均值/末次观测结转（LOCF）冒充多重插补。

## 9. 与相近方法的区别

- 和完整个案分析（listwise deletion）的区别：后者在 MAR 下可能有偏且低效，MI 更稳健。
- 和单一插补（均值/回归）的区别：单一插补低估方差，MI 通过多套插补修正。
- 和 [[逆概率处理加权（Inverse Probability of Treatment Weighting, IPTW）]] 的关系：IPW 用加权处理缺失/删失，MI 用生成数据，二者可视问题选择或结合。

## 10. 医学研究中的典型应用

- 队列/RCT 中缺失协变量或结局的主分析与敏感性分析。
- 多中心真实世界数据字段不齐时的统一分析。
- 生存分析中缺失预后因子的处理。

## 11. 相关方法

- [[线性回归（Linear Regression）]]
- [[Logistic回归（Logistic Regression）]]
- [[因果推断（Causal Inference）]]
- [[逆概率处理加权（Inverse Probability of Treatment Weighting, IPTW）]]

## 12. 参考资料

- Rubin DB. *Multiple Imputation for Nonresponse in Surveys*. Wiley; 1987.
- van Buuren S. *Flexible Imputation of Missing Data*. 2nd ed. Chapman and Hall/CRC; 2018.
- White IR, Royston P, Wood AM. Multiple imputation using chained equations: issues and guidance for practice. *Stat Med*. 2011;30(4):377-399.
- van Buuren S, Groothuis-Oudshoorn K. mice: multivariate imputation by chained equations in R. *J Stat Softw*. 2011;45(3):1-67.
