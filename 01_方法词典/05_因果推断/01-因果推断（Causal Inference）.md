---
title: 因果推断
english_name: Causal Inference
slug: causal-inference
aliases: [causal inference, 因果推断, "因果推断（Causal Inference）"]
category: 因果推断
subcategory: 因果推断框架
tags: [医学统计, 流行病学, 因果推断, 观察性研究]
status: 已建
difficulty: intermediate
question_type: 干预效应估计与混杂调整
data_type: [观察性研究数据, 随机试验数据, 表格数据]
outcome_type: [连续型, 二分类, 计数型, 时间到事件]
python_packages: [econml, dowhy, causallib]
r_packages: [MatchIt, WeightIt, cobalt, AIPW]
---

# 因果推断（Causal Inference）

## 1. 方法概览

### 1.1 定义

因果推断是一组用于估计干预或暴露对结局影响的方法框架。它关注的是“如果我们改变处理状态，结局会怎样变化”，而不是单纯描述变量之间是否相关。

### 1.2 它主要解决什么问题

- 研究问题：治疗、暴露或政策是否导致结局变化，以及影响大小是多少。
- 适用任务：平均处理效应、条件平均处理效应、异质性效应、混杂调整。
- 常见医学场景：药物真实世界效果评估、生活方式干预、筛查策略、治疗方案比较。

### 1.3 直觉理解

因果推断试图构造一个合理的反事实比较：同一个患者不可能同时接受和不接受治疗，因此需要用随机化、匹配、加权或建模方法，让处理组和对照组尽可能像“可比较的两组人”。

## 2. 数学形式

### 2.1 核心公式

在潜在结果框架中，每个个体 $i$ 有两个潜在结果：

$$
Y_i(1),\quad Y_i(0)
$$

个体处理效应为：

$$
\tau_i=Y_i(1)-Y_i(0)
$$

平均处理效应为：

$$
\mathrm{ATE}=E[Y(1)-Y(0)]
$$

条件平均处理效应为：

$$
\mathrm{CATE}(x)=E[Y(1)-Y(0)\mid X=x]
$$

在条件可交换性假设下：

$$
\{Y(1),Y(0)\}\perp T\mid X
$$

倾向评分为：

$$
e(X)=P(T=1\mid X)
$$

逆概率加权估计 ATE 的形式为：

$$
E\left[\frac{TY}{e(X)}-\frac{(1-T)Y}{1-e(X)}\right]
$$

### 2.2 参数或统计量含义

- $T$：处理或暴露变量。
- $Y$：结局变量。
- $X$：需要调整的协变量或混杂变量。
- $Y(1),Y(0)$：处理和未处理状态下的潜在结果。
- ATE：总体平均处理效应。
- CATE：特定协变量水平下的条件平均处理效应。
- 倾向评分：个体接受处理的条件概率。

### 2.3 关键假设

- 一致性：观测到的结局等于实际接受处理状态下的潜在结果。
- 可交换性：给定已测协变量后，处理分配与潜在结果独立。
- 正值性：每个协变量组合下都存在接受和不接受处理的可能。
- SUTVA：个体之间没有干预相互干扰，处理版本定义清晰。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：处理变量、结局变量、混杂变量、效应修饰变量。
- 因变量类型：连续型、二分类、计数型或时间到事件。
- 数据结构：随机试验数据或观察性研究数据。
- 是否适合高维数据：可用机器学习辅助，但假设和验证更关键。
- 是否适合缺失较多数据：需明确缺失机制并进行合理处理。
- 是否适合删失数据：可结合生存分析和 IPCW 等方法。
- 是否适合重复测量数据：可扩展到纵向处理和边际结构模型。

### 3.2 示例表格

以评估某药物对 30 天再入院风险的影响为例：

| Age | Charlson | PriorAdmission | Drug | Readmit30d |
| --- | --- | --- | --- | --- |
| 73 | 4 | 1 | 1 | 0 |
| 58 | 2 | 0 | 0 | 0 |
| 69 | 5 | 1 | 1 | 1 |
| 46 | 1 | 0 | 0 | 0 |
| 62 | 3 | 1 | 0 | 1 |

### 3.3 输入与产出

#### 输入

- 输入数据：处理变量、结局变量、混杂变量和可选效应修饰变量。
- 关键变量：处理定义、时间零点、随访窗口、调整变量集合。
- 需要预处理的内容：缺失处理、极端倾向评分检查、协变量平衡检查、删失处理。

#### 产出

- 模型对象/统计结果：ATE、ATT、CATE、权重或匹配结果。
- 参数估计：处理效应估计及置信区间。
- 预测结果：潜在结果预测或个体化效应估计。
- 不确定性指标：标准误、置信区间、bootstrap 区间、敏感性分析结果。

## 4. 适用场景

- 适合：无法随机化但希望尽量估计干预效果的观察性研究。
- 不适合：关键混杂变量未测量、处理定义模糊、时间顺序不清、缺乏重叠的场景。
- 使用前需要特别检查的点：DAG 或调整变量依据、协变量平衡、正值性、未测混杂敏感性。

## 5. 实现

### 5.1 Python

常用包：

- `dowhy`
- `econml`
- `causallib`

```python
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("drug_readmission.csv")
X = df[["Age", "Charlson", "PriorAdmission"]]
T = df["Drug"]
Y = df["Readmit30d"]

ps_model = sm.Logit(T, sm.add_constant(X)).fit(disp=False)
df["ps"] = ps_model.predict(sm.add_constant(X))

treated = T == 1
df["ipw"] = T / df["ps"] + (1 - T) / (1 - df["ps"])

ate_ipw = (
    (df.loc[treated, "ipw"] * df.loc[treated, "Readmit30d"]).sum() / df.loc[treated, "ipw"].sum()
    - (df.loc[~treated, "ipw"] * df.loc[~treated, "Readmit30d"]).sum() / df.loc[~treated, "ipw"].sum()
)

print("IPW risk difference:", ate_ipw)
```

### 5.2 R

常用包：

- `WeightIt`
- `cobalt`

```r
library(WeightIt)
library(cobalt)

w <- weightit(
  Drug ~ Age + Charlson + PriorAdmission,
  data = df,
  method = "ps",
  estimand = "ATE"
)

bal.tab(w)

fit <- glm(
  Readmit30d ~ Drug,
  data = df,
  weights = w$weights,
  family = binomial()
)

summary(fit)
```

## 6. 结果如何解释

- 核心结果看什么：处理效应估计、置信区间、协变量平衡、重叠情况和敏感性分析。
- 每个主要参数如何解释：ATE 是总体平均效应；ATT 是处理组人群中的平均效应；CATE 反映异质性。
- 临床或医学意义如何表达：应明确“在给定假设成立时”的效应估计，避免写成无条件因果定论。
- 常见误读：调整了很多变量不等于消除了混杂；预测性能高不等于因果效应估计可靠。

## 7. 推荐可视化

- DAG 因果图。
- 倾向评分重叠图。
- 加权或匹配前后的协变量平衡 Love plot。
- CATE 分布图或亚组效应森林图。

## 8. 优势、局限与常见坑

### 优势

- 能明确区分相关性和干预效应。
- 可用于真实世界数据中的治疗效果评估。
- 支持总体效应和异质性效应分析。

### 局限

- 强依赖不可完全验证的识别假设。
- 未测混杂会导致偏倚。
- 复杂机器学习方法可能提升灵活性，但也增加解释和验证难度。

### 常见坑

- 根据统计显著性而非因果结构选择调整变量。
- 调整中介变量或碰撞变量，反而引入偏倚。
- 不检查倾向评分重叠和加权后平衡。
- 把 CATE 结果当成已证实的个体化治疗规则。

## 9. 与相近方法的区别

- 和 [[Logistic回归（Logistic Regression）]] 的区别：Logistic 回归可用于结局建模或倾向评分估计，但因果推断还需要识别假设和可比性检查。
- 和 [[条件Logistic回归（Conditional Logistic Regression）]] 的区别：条件 Logistic 回归处理匹配数据关联分析，因果推断关注处理效应识别与估计。
- 和 [[Cox比例风险模型（Cox Proportional Hazards Model）]] 的区别：Cox 模型用于时间到事件结局建模，因果问题还需处理混杂、删失和目标效应。

## 10. 医学研究中的典型应用

- 观察性队列中比较两种治疗方案的真实世界效果。
- 评估筛查、药物、手术或康复干预对结局的影响。
- 分析不同患者亚群中的异质性治疗效应。

## 11. 相关方法

- [[倾向评分匹配（Propensity Score Matching）]]
- [[Logistic回归（Logistic Regression）]]
- [[Cox比例风险模型（Cox Proportional Hazards Model）]]
- [[Bootstrap重抽样（Bootstrap Resampling）]]

## 12. 参考资料

- Rubin DB. Estimating causal effects of treatments in randomized and nonrandomized studies. *J Educ Psychol*. 1974;66(5):688-701.
- Rosenbaum PR, Rubin DB. The central role of the propensity score in observational studies for causal effects. *Biometrika*. 1983;70(1):41-55.
- Pearl J. *Causality: Models, Reasoning, and Inference*. 2nd ed. Cambridge University Press; 2009.
- Hernan MA, Robins JM. *Causal Inference: What If*. Chapman and Hall/CRC; 2020.
