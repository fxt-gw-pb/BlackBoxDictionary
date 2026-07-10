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

> 本卡是 [[因果推断（Causal Inference）]] 家族下的具体方法，因果识别假设与潜在结果框架见主条目；这里聚焦如何用权重构造伪人群。

## 1. 方法概览

### 1.1 一句话本质

IPTW 给「在自身条件下不太可能接受实际处理」的人更大权重，让加权后的数据看起来像处理被随机分配过。

### 1.2 定义

逆概率处理加权使用倾向评分的倒数作为权重，构造一个伪人群。在这个伪人群中，处理组和对照组在已测处理前协变量上的分布应接近平衡，从而可以估计边际处理效应。

### 1.3 它主要解决什么问题

- 研究问题：如何在保留全部或大部分样本的情况下调整已测混杂，估计总体平均处理效应。
- 适用任务：ATE 估计、边际结构模型、时依处理和时依混杂调整、IPCW 删失调整。
- 常见医学场景：真实世界疗效比较、随时间变化的用药暴露、需要报告总体平均风险差或风险比的政策评估。

### 1.4 直觉与类比

IPTW 像是在补齐一个不均衡的抽样箱。某类患者本来很少接受治疗，但恰好有一位接受了治疗，那么他提供的信息很稀有，就让他代表更多类似患者；某类患者很常接受治疗，则每个这类治疗者代表的人数较少。加权后，两组的病情和基线风险分布被拉到同一个目标人群。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

观察性数据中，处理概率随病情、年龄、既往病史等改变。直接比较处理组和对照组时，某些协变量组合在一组中过多、另一组中过少。IPTW 解决的是**处理分配不均衡导致的混杂**：通过加权让每个协变量组合在处理和对照状态下都被充分代表。

### 2.2 关键洞察

关键洞察是「按实际处理概率的倒数重建伪随机化」。如果一个人接受了处理，而模型认为他接受处理的概率只有 0.25，那么他的权重是 4；直觉上，他代表了 4 个类似但在随机世界中可能有不同处理状态的人。加权后，处理不再由 $X$ 强烈决定，混杂变量达到平衡。

### 2.3 与朴素/相邻做法的对比

- 相对朴素均值差：IPTW 先改变每个观测的贡献大小，再比较加权均值。
- 相对 [[倾向评分匹配（Propensity Score Matching）]]：匹配靠丢弃和配对，IPTW 靠加权，通常更接近 ATE。
- 相对结局回归：IPTW 建模处理机制；若结合结局模型形成 AIPW，可获得双重稳健性。

## 3. 数学形式

### 3.1 核心公式

倾向评分为：

$$
e(X)=P(T=1\mid X)
$$

ATE 的未稳定化权重为：

$$
w_i=\frac{T_i}{e(X_i)}+\frac{1-T_i}{1-e(X_i)}
$$

稳定化权重为：

$$
sw_i=\frac{T_iP(T=1)}{e(X_i)}+\frac{(1-T_i)P(T=0)}{1-e(X_i)}
$$

归一化加权均值差（Hájek 形式）为：

$$
\widehat{\mathrm{ATE}}=
\frac{\sum_i T_iw_iY_i}{\sum_i T_iw_i}
-
\frac{\sum_i (1-T_i)w_iY_i}{\sum_i (1-T_i)w_i}
$$

这个式子在说：先用权重把处理组和对照组各自重构成目标人群，再比较两个加权平均结局。

### 3.2 推导脉络

1. 在可交换性和正值性下，给定 $X$ 后处理像随机分配。
2. 但不同 $X$ 层里处理概率不同，所以直接平均会让某些层过度或不足代表。
3. 用 $1/e(X)$ 或 $1/\{1-e(X)\}$ 重新加权，可把每个处理状态扩展到目标总体。
4. 稳定化权重把总体处理比例放到分子，通常不改变目标 estimand，却能减少方差和极端权重影响。

### 3.3 参数与统计量含义

- $e(X)$：给定协变量后接受处理的概率。
- $w_i$：未稳定化 IPTW 权重。
- $sw_i$：稳定化权重；均值通常应接近 1。
- ATE：加权伪人群中的总体平均处理效应。
- 有效样本量（ESS）：权重离散度造成的信息损失，权重越极端 ESS 越小。
- 截断（truncation）：把极端权重限制在某些分位数内，以换取偏倚和方差之间的折中。

### 3.4 关键假设(含违反后果)

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 无未测混杂 | 已测 $X$ 足以解释处理选择中的混杂 | 加权后仍偏倚 | DAG、负对照、敏感性分析 |
| 正值性 | $0\lt e(X)\lt 1$，且不能太接近边界 | 极端权重主导估计 | 权重分布、倾向评分重叠图 |
| 倾向评分模型合适 | 权重能平衡处理前协变量 | 加权后 SMD 仍大 | Love plot、平衡表 |
| 独立观测或正确聚类 | 标准误需符合设计 | 区间过窄 | 聚类稳健方差或 GEE |
| 缺失/删失处理正确 | 缺失或删失不能被忽略时需额外建权重 | 选择偏倚 | 缺失模型、IPCW、敏感性分析 |

## 4. 手把手算例

沿用 [[倾向评分匹配（Propensity Score Matching）]] 的 6 人小数据，$Y$ 为住院天数，越少越好。

| 患者 | T | 倾向评分 e | Y | IPTW 权重 |
| --- | --- | --- | --- | --- |
| A | 1 | 0.70 | 8 | $1/0.70=1.43$ |
| B | 1 | 0.55 | 6 | $1/0.55=1.82$ |
| C | 1 | 0.35 | 4 | $1/0.35=2.86$ |
| D | 0 | 0.68 | 10 | $1/(1-0.68)=3.13$ |
| E | 0 | 0.50 | 7 | $1/(1-0.50)=2.00$ |
| F | 0 | 0.32 | 5 | $1/(1-0.32)=1.47$ |

**Step 1：算加权处理组均值。**

$$
\bar Y_1^w=
\frac{1.43\times 8+1.82\times 6+2.86\times 4}{1.43+1.82+2.86}
=\frac{33.80}{6.11}=5.53
$$

**Step 2：算加权对照组均值。**

$$
\bar Y_0^w=
\frac{3.13\times 10+2.00\times 7+1.47\times 5}{3.13+2.00+1.47}
=\frac{52.65}{6.60}=7.98
$$

**Step 3：计算 ATE。**

$$
\widehat{\mathrm{ATE}}=5.53-7.98=-2.45
$$

**结论：** IPTW 估计药物在目标总体中平均减少约 2.45 天住院。和 PSM 的 ATT 不同，这里每个人都被保留，只是贡献大小不同；权重越不均，结果越需要做稳健性检查。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：处理前混杂变量，可连续、分类或高维。
- 因变量类型：连续、二分类、计数、时间到事件。
- 数据结构：个体级观察性队列；纵向资料可扩展为时依权重。
- 是否适合高维数据：可以，但需正则化或交叉拟合，并重点看平衡和极端权重。
- 是否适合缺失较多数据：需先处理缺失；缺失可能与处理或结局相关时需额外权重或插补。
- 是否适合删失数据：适合与 IPCW 结合处理随访删失。
- 是否适合重复测量数据：适合，边际结构模型是经典应用。

### 5.2 示例表格

| 个体 | Age | Charlson | Drug | ps | sw | Readmit30d |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 73 | 4 | 1 | 0.80 | 0.63 | 1 |
| 2 | 58 | 2 | 0 | 0.30 | 0.71 | 0 |
| 3 | 69 | 5 | 1 | 0.35 | 1.43 | 1 |
| 4 | 46 | 1 | 0 | 0.20 | 0.63 | 0 |

### 5.3 输入与产出

#### 输入

- 输入数据：处理变量、处理前混杂变量、结局变量。
- 关键变量：目标 estimand、权重类型、是否稳定化、是否截断。
- 需要预处理的内容：缺失处理、倾向评分估计、权重诊断、平衡诊断。

#### 产出

- 模型对象/统计结果：倾向评分、权重、加权结局模型。
- 参数估计：ATE、风险差、风险比、均值差、HR 或边际 OR。
- 预测结果：不是核心；主要产出是加权伪人群中的效应。
- 不确定性指标：稳健标准误、bootstrap 区间、权重分布、有效样本量。

## 6. 适用场景

- 适合：重叠较好、希望估计总体边际效应、样本量足以承受加权方差的研究。
- 不适合：处理概率接近 0 或 1、极端权重很多、关键混杂未测量、样本很小。
- 使用前需要特别检查的点：权重均值、最大权重、分位数、ESS、加权后 SMD、截断敏感性。

## 7. 实现

### 7.1 Python

常用包:

- `scikit-learn`
- `statsmodels`

```python
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

X = df[["Age", "Charlson", "PriorAdmission"]]
T = df["Drug"].to_numpy()

ps = LogisticRegression(max_iter=5000).fit(X, T).predict_proba(X)[:, 1]
p_treated = T.mean()
sw = np.where(T == 1, p_treated / ps, (1 - p_treated) / (1 - ps))

df = df.copy()
df["sw"] = sw

fit = smf.wls("Readmit30d ~ Drug", data=df, weights=df["sw"]).fit(
    cov_type="HC1"
)
print(fit.params["Drug"])       # 加权风险差
print(df["sw"].describe())      # 权重诊断
```

### 7.2 R

常用包:

- `WeightIt`
- `cobalt`
- `survey`

```r
library(WeightIt)
library(cobalt)
library(survey)

w <- weightit(
  Drug ~ Age + Charlson + PriorAdmission,
  data = df,
  method = "ps",
  estimand = "ATE",
  stabilize = TRUE
)

bal.tab(w, un = TRUE)
summary(w$weights)

des <- svydesign(ids = ~1, weights = ~w$weights, data = df)
fit <- svyglm(Readmit30d ~ Drug, design = des, family = quasibinomial())
summary(fit)
```

## 8. 结果如何解读

- 核心结果看什么：ATE 点估计、稳健区间、加权后平衡、权重分布和 ESS。
- 每个主要参数如何解读：加权结局模型中 `Drug` 的系数是目标总体的边际处理效应，解释时要说明效应尺度。
- 临床或医学意义如何表达：例如「在目标总体中，用药使 30 天再入院风险平均降低多少个百分点」。
- 常见误读：把加权后个体复制当成真实样本量增加；实际上极端权重会降低有效样本量。

## 9. 假设诊断与稳健性

- 权重诊断：报告均值、最大值、1% 和 99% 分位数、ESS；稳定化权重均值应接近 1。
- 平衡诊断：检查加权后 SMD，目标是绝对值小于 0.1；若不达标，重新设定倾向评分模型。
- 正值性诊断：画倾向评分重叠图，识别只存在处理或只存在对照的区域。
- 截断敏感性：比较不截断、1%/99% 截断、5%/95% 截断的效应是否稳定。
- 方差估计：使用稳健三明治方差或 bootstrap；聚类/重复测量数据用聚类稳健方法。

## 10. 推荐可视化

- 权重直方图或箱线图：识别极端权重。
- 倾向评分重叠图：判断共同支持。
- Love plot：展示加权前后协变量平衡。
- 截断敏感性图：横轴为截断规则，纵轴为效应估计和区间。

## 11. 优势、局限与常见坑

### 优势

- 保留样本信息，直接估计边际总体效应。
- 可自然扩展到时依处理和边际结构模型。
- 与结局模型分离，便于先做设计诊断。

### 局限

- 对正值性违背和极端权重非常敏感。
- 倾向评分模型错设会导致平衡失败和偏倚。
- 只调整已测混杂，不能自动处理未测混杂。

### 常见坑

- 不看权重分布就直接报告加权模型结果。
- 极端权重明显却不做截断或目标人群限制。
- 用普通标准误而不是稳健或 bootstrap 标准误。
- 加权后仍不平衡，却把结果解释成因果效应。

## 12. 与相近方法的区别

- 和 [[倾向评分匹配（Propensity Score Matching）]] 的区别：IPTW 保留样本并改变贡献大小，PSM 保留可匹配样本并做配对比较。
- 和结局回归的区别：IPTW 建模处理分配，结局回归建模 $E(Y\mid T,X)$；二者结合可形成双重稳健估计。
- 和 [[广义估计方程（Generalized Estimating Equations, GEE）]] 的关系：GEE 估边际均值，IPTW 可作为权重放入边际模型，尤其用于重复测量和删失。
- 如何选择：目标是 ATE 且重叠良好时优先考虑 IPTW；极端权重多时考虑限制人群、匹配或双重稳健方法。

## 13. 医学研究中的典型应用

- 真实世界数据库中估计两种治疗策略对死亡、再入院或不良事件的总体效应。
- 随时间变化的药物使用、依从性和疾病严重度相互影响时，构建边际结构模型。
- 生存分析中结合 IPCW 处理失访或治疗转换。

## 14. 关键术语

- **逆概率权重（Inverse Probability Weight）**：实际处理状态发生概率的倒数。
- **稳定化权重（Stabilized Weight）**：分子加入总体处理概率以降低方差的权重。
- **伪人群（Pseudo-population）**：加权后构造出的、处理与协变量近似独立的人群。
- **正值性（Positivity）**：每类协变量组合下处理概率既不是 0 也不是 1。
- **极端权重（Extreme Weight）**：由倾向评分接近边界产生的过大权重。
- **有效样本量（Effective Sample Size, ESS）**：考虑权重不均后真正可用的信息量。
- **边际结构模型（Marginal Structural Model, MSM）**：用权重调整时依混杂后估计边际因果效应的模型。

## 15. 相关方法

- [[因果推断（Causal Inference）]]
- [[倾向评分匹配（Propensity Score Matching）]]
- [[Logistic回归（Logistic Regression）]]
- [[广义估计方程（Generalized Estimating Equations, GEE）]]
- [[Cox比例风险模型（Cox Proportional Hazards Model）]]

## 16. 参考资料

- Robins JM, Hernán MA, Brumback B. Marginal structural models and causal inference in epidemiology. *Epidemiology*. 2000;11(5):550-560.
- Austin PC, Stuart EA. Moving towards best practice when using inverse probability of treatment weighting (IPTW). *Stat Med*. 2015;34(28):3661-3679.
- Hernán MA, Robins JM. *Causal Inference: What If*. Chapman and Hall/CRC; 2020.
- Cole SR, Hernán MA. Constructing inverse probability weights for marginal structural models. *Am J Epidemiol*. 2008;168(6):656-664.
