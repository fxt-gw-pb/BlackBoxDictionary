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

> 本卡是 [[因果推断（Causal Inference）]] 家族下的具体方法，因果识别假设与潜在结果框架见主条目；这里聚焦如何用倾向评分构造可比样本。

## 1. 方法概览

### 1.1 一句话本质

倾向评分匹配把一堆混杂变量压缩成「接受处理的概率」这一维分数，再在分数相近的人之间比较结局。

### 1.2 定义

倾向评分匹配先估计 $e(X)=P(T=1\mid X)$，再为每个处理组个体寻找倾向评分相近的对照组个体。匹配后的样本应在处理前协变量上接近平衡，因而更像一个可比的非随机试验。

### 1.3 它主要解决什么问题

- 研究问题：观察性研究中，如何把接受治疗的人和未接受治疗但「原本相似」的人放在一起比较。
- 适用任务：已测混杂调整、ATT 估计、匹配后结局分析。
- 常见医学场景：药物真实世界疗效比较、术式或器械评价、暴露与疾病结局的观察性队列分析。

### 1.4 直觉与类比

把倾向评分想成「进入治疗组的入场券概率」。两个人年龄、合并症、既往入院等不同变量不完全一样，但如果他们拿到治疗的概率都约为 0.55，就说明在医生眼里他们的治疗倾向相近。PSM 就是在这种「同样容易被治疗」的人之间做比较。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

观察性研究的困难在于混杂维度很多：年龄相近但合并症不同、合并症相近但病程不同，逐个变量精确匹配会迅速变得不可能。PSM 试图解决的是**多维可比性难以直接构造**的问题。

### 2.2 关键洞察

Rosenbaum 与 Rubin 的关键洞察是：如果给定全部处理前协变量 $X$ 后可交换，那么给定倾向评分 $e(X)$ 后也可交换。也就是说，$e(X)$ 是一个**平衡评分**：处理组和对照组只要在这一个分数上分布相近，理论上也会在建模所用的 $X$ 上平衡。

### 2.3 与朴素/相邻做法的对比

- 相对逐变量精确匹配：PSM 把多维匹配转成一维距离匹配，操作更可行。
- 相对直接回归调整：PSM 在看结局前先做设计和可比性诊断，减少「看着结果调模型」。
- 相对 [[逆概率处理加权（Inverse Probability of Treatment Weighting, IPTW）]]：PSM 常丢弃不可匹配样本，目标更接近 ATT；IPTW 多保留样本，常估 ATE。

## 3. 数学形式

### 3.1 核心公式

倾向评分定义为：

$$
e(X)=P(T=1\mid X)
$$

若满足条件可交换性和正值性：

$$
\{Y(1),Y(0)\}\perp T\mid X
\quad\Rightarrow\quad
\{Y(1),Y(0)\}\perp T\mid e(X)
$$

最近邻匹配常寻找：

$$
j(i)=\operatorname*{arg\,min}_{j:T_j=0}\left|\operatorname{logit}\{e(X_i)\}-\operatorname{logit}\{e(X_j)\}\right|
$$

匹配后处理组平均处理效应可估为：

$$
\widehat{\mathrm{ATT}}=\frac{1}{n_1}\sum_{i:T_i=1}\left(Y_i-Y_{j(i)}\right)
$$

这个式子在说：对每个处理组个体，找到一个最像他的对照者，再把一对一的结局差求平均。

### 3.2 推导脉络

1. 先用处理前协变量 $X$ 建模 $P(T=1\mid X)$，得到每人的倾向评分。
2. 在倾向评分或 logit 倾向评分上定义距离，常加卡钳限制，避免差太远的人硬配。
3. 匹配后检查协变量平衡；若仍不平衡，重新设定倾向评分模型或匹配规则。
4. 在匹配样本上估计结局差异，并用考虑匹配结构的标准误或 bootstrap 评估不确定性。

### 3.3 参数与统计量含义

- $e(X)$：给定处理前协变量后接受处理的概率。
- 卡钳（caliper）：允许匹配的最大距离，常在 logit 倾向评分尺度上设定。
- $j(i)$：处理组个体 $i$ 匹配到的对照组个体。
- ATT：处理组患者若未接受处理时，平均结局会改变多少。
- 标准化均差（SMD）：平衡诊断核心指标，匹配后通常希望绝对值小于 0.1。

### 3.4 关键假设(含违反后果)

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 无未测混杂 | 所有共同影响处理和结局的处理前变量都已测量 | 匹配后仍有隐藏偏倚 | DAG、负对照、Rosenbaum bounds |
| 正值性/重叠 | 每类患者都有相似的处理和对照 | 大量样本无法匹配，目标人群变窄 | 倾向评分密度图 |
| 倾向评分模型足够好 | 能把需要平衡的协变量结构捕捉到 | 匹配后 SMD 仍大 | Love plot、平衡表 |
| 不调整处理后变量 | 匹配变量必须在处理前 | 调整中介或碰撞变量引入偏倚 | 时间轴和 DAG |

## 4. 手把手算例

沿用一个 6 人小例子，$T=1$ 为用药，$Y$ 为住院天数，越少越好。假设倾向评分已经由年龄、病情和既往入院估计好。

| 患者 | T | 倾向评分 e | Y |
| --- | --- | --- | --- |
| A | 1 | 0.70 | 8 |
| B | 1 | 0.55 | 6 |
| C | 1 | 0.35 | 4 |
| D | 0 | 0.68 | 10 |
| E | 0 | 0.50 | 7 |
| F | 0 | 0.32 | 5 |

**Step 1：找最近邻。**

- A 的最近对照是 D，距离 $|0.70-0.68|=0.02$。
- B 的最近对照是 E，距离 $|0.55-0.50|=0.05$。
- C 的最近对照是 F，距离 $|0.35-0.32|=0.03$。

若卡钳设为 0.06，三对都保留。

**Step 2：计算每一对差值。**

$$
A-D=8-10=-2,\qquad B-E=6-7=-1,\qquad C-F=4-5=-1
$$

**Step 3：求 ATT。**

$$
\widehat{\mathrm{ATT}}=\frac{-2-1-1}{3}=-1.33
$$

**结论：** 在可匹配的处理组患者中，用药平均减少约 1.33 天住院。这里的目标不是全体患者，而是「实际接受用药且能找到相似对照的人」。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：处理前混杂变量，可为连续、二分类或多分类。
- 因变量类型：连续、二分类、计数或时间到事件。
- 数据结构：每行一个个体的观察性队列或注册登记数据。
- 是否适合高维数据：可用正则化或机器学习估计倾向评分，但匹配后仍必须看平衡。
- 是否适合缺失较多数据：不直接适合；常先做 [[多重插补（Multiple Imputation）]]。
- 是否适合删失数据：可在匹配样本上做生存分析，并处理删失。
- 是否适合重复测量数据：基础 PSM 针对基线处理；时变处理需边际结构模型等扩展。

### 5.2 示例表格

| 个体 | Age | Charlson | PriorAdmission | Drug | ps | Readmit30d |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 73 | 4 | 1 | 1 | 0.78 | 1 |
| 2 | 70 | 4 | 0 | 0 | 0.75 | 0 |
| 3 | 58 | 2 | 1 | 1 | 0.41 | 0 |
| 4 | 60 | 2 | 0 | 0 | 0.43 | 1 |

### 5.3 输入与产出

#### 输入

- 输入数据：处理变量、处理前混杂变量、结局。
- 关键变量：处理定义、匹配比例、是否放回、卡钳、距离尺度。
- 需要预处理的内容：缺失处理、分类变量编码、倾向评分估计、重叠检查。

#### 产出

- 模型对象/统计结果：倾向评分、匹配对、平衡表、匹配权重。
- 参数估计：匹配样本中的 ATT 或指定 estimand。
- 预测结果：通常不是核心产出；倾向评分只是设计工具。
- 不确定性指标：匹配后标准误、bootstrap 区间、未测混杂敏感性分析。

## 6. 适用场景

- 适合：已测混杂较充分、处理和对照有明显重叠、希望得到直观可解释的匹配队列。
- 不适合：关键混杂未测量、样本太小、处理组和对照组几乎不重叠、需要保留全部样本估计总体效应。
- 使用前需要特别检查的点：匹配前后 SMD、样本丢弃比例、倾向评分重叠、匹配设置敏感性。

## 7. 实现

### 7.1 Python

常用包:

- `scikit-learn`
- `pandas`

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

X = df[["Age", "Charlson", "PriorAdmission"]]
T = df["Drug"].to_numpy()

ps = LogisticRegression(max_iter=5000).fit(X, T).predict_proba(X)[:, 1]
df = df.copy()
df["logit_ps"] = np.log(ps / (1 - ps))

treated = df[df["Drug"] == 1].copy()
control = df[df["Drug"] == 0].copy()

nn = NearestNeighbors(n_neighbors=1).fit(control[["logit_ps"]])
dist, idx = nn.kneighbors(treated[["logit_ps"]])
caliper = 0.2 * df["logit_ps"].std()
keep = dist.ravel() <= caliper

pairs = treated.loc[keep, ["Readmit30d"]].reset_index(drop=True)
pairs["control_y"] = control.iloc[idx.ravel()[keep]]["Readmit30d"].to_numpy()
att = (pairs["Readmit30d"] - pairs["control_y"]).mean()
print(round(att, 3))
```

### 7.2 R

常用包:

- `MatchIt`
- `cobalt`

```r
library(MatchIt)
library(cobalt)

m <- matchit(
  Drug ~ Age + Charlson + PriorAdmission,
  data = df,
  method = "nearest",
  distance = "glm",
  caliper = 0.2,
  replace = FALSE
)

bal.tab(m, un = TRUE)
love.plot(m, threshold = 0.1)

md <- match.data(m)
fit <- glm(Readmit30d ~ Drug, data = md, weights = weights, family = binomial())
summary(fit)
```

## 8. 结果如何解读

- 核心结果看什么：匹配后协变量平衡、匹配成功率、ATT 点估计和区间。
- 每个主要参数如何解读：ATT 是「已接受处理且留在匹配样本中的患者」的平均效应，不自动代表全体患者。
- 临床或医学意义如何表达：例如「在基线风险相近的可匹配患者中，用药组 30 天再入院风险低多少」。
- 常见误读：把匹配前倾向评分模型的 AUC 当成好坏标准；真正要看的是匹配后平衡，不是处理预测多准。

## 9. 假设诊断与稳健性

- 平衡诊断：报告匹配前后 SMD、方差比和 Love plot，不用 p 值作为主要平衡标准。
- 重叠诊断：画处理组与对照组倾向评分密度图，说明被丢弃的人群特征。
- 匹配设置敏感性：比较不同卡钳、1:1 与 1:k、是否放回、最近邻与全匹配。
- 未测混杂：做 Rosenbaum bounds 或定量偏倚分析。
- 结局分析：匹配后标准误要考虑匹配和权重；生存结局可用匹配样本的 KM、Cox 或稳健方差。

## 10. 推荐可视化

- 倾向评分重叠图：看是否存在共同支持区域。
- Love plot：展示匹配前后每个协变量的 SMD。
- 匹配对连线图：显示每个处理者与对照者的倾向评分距离。
- 匹配后结局图：二分类结局用风险差森林图，生存结局用 KM 曲线。

## 11. 优势、局限与常见坑

### 优势

- 直观，容易向临床读者解释「我们比较了相似患者」。
- 把设计阶段和结局分析分开，降低模型驱动结果的风险。
- 能清楚暴露缺乏重叠的人群，从而避免不可信外推。

### 局限

- 只能平衡已测且正确纳入模型的混杂。
- 会丢弃不可匹配样本，降低效率并改变目标人群。
- 结果对匹配规则、卡钳和放回设置敏感。

### 常见坑

- 把处理后变量、中介变量或碰撞变量放进倾向评分模型。
- 匹配后只报结局 p 值，不报平衡和样本丢弃。
- 用匹配后的显著性检验判断协变量是否平衡。
- 把 ATT 解释成全体患者的 ATE。

## 12. 与相近方法的区别

- 和 [[因果推断（Causal Inference）]] 的关系：PSM 是因果推断中构造可比样本的一种设计工具。
- 和 [[逆概率处理加权（Inverse Probability of Treatment Weighting, IPTW）]] 的区别：PSM 借「丢弃和配对」求可比，IPTW 借「给稀有人群更大权重」求伪随机化。
- 和直接回归调整的区别：回归调整依赖结局模型形式，PSM 更强调先看基线可比性；但匹配后仍可做结局回归以提高精度。
- 如何选择：重叠好且想估总体效应用 IPTW；想给临床读者展示相似患者比较、且能接受目标人群变窄时用 PSM。

## 13. 医学研究中的典型应用

- 两种药物、术式、器械或诊疗路径在真实世界中的疗效与安全性比较。
- 罕见治疗或新技术应用场景中，为处理组寻找可比对照。
- 注册登记研究中，构造平衡队列后比较死亡、复发、再入院等结局。

## 14. 关键术语

- **倾向评分（Propensity Score）**：给定处理前协变量后，个体接受处理的概率。
- **匹配（Matching）**：为处理组个体寻找协变量或倾向评分相近的对照。
- **卡钳（Caliper）**：允许匹配的最大距离，防止差太远的人硬配。
- **共同支持（Common Support）**：处理组和对照组倾向评分都有覆盖的区域。
- **标准化均差（Standardized Mean Difference, SMD）**：衡量协变量平衡的尺度化差异。
- **ATT（Average Treatment Effect on the Treated）**：处理组人群中的平均处理效应。
- **放回匹配（Matching with Replacement）**：同一个对照可被多个处理者匹配，提高匹配质量但增加方差处理复杂度。

## 15. 相关方法

- [[因果推断（Causal Inference）]]
- [[逆概率处理加权（Inverse Probability of Treatment Weighting, IPTW）]]
- [[Logistic回归（Logistic Regression）]]
- [[Cox比例风险模型（Cox Proportional Hazards Model）]]
- [[Bootstrap重抽样（Bootstrap Resampling）]]

## 16. 参考资料

- Rosenbaum PR, Rubin DB. The central role of the propensity score in observational studies for causal effects. *Biometrika*. 1983;70(1):41-55.
- Austin PC. An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behav Res*. 2011;46(3):399-424.
- Ho DE, Imai K, King G, Stuart EA. MatchIt: nonparametric preprocessing for parametric causal inference. *J Stat Softw*. 2011;42(8):1-28.
- Stuart EA. Matching methods for causal inference: a review and a look forward. *Stat Sci*. 2010;25(1):1-21.
