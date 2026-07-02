---
title: 中介分析
english_name: Mediation Analysis
slug: mediation-analysis
aliases: [中介效应, mediation, 因果中介分析, "中介分析（Mediation Analysis）"]
category: 因果推断
subcategory: 中介效应
tags: [医学统计, 流行病学, 因果推断, 机制分析, 中介]
status: 已建
difficulty: intermediate
question_type: 暴露对结局的作用机制分解
data_type: [表格数据, 观察性研究数据]
outcome_type: [连续型, 二分类]
python_packages: [statsmodels]
r_packages: [mediation]
---

# 中介分析（Mediation Analysis）

## 1. 方法概览

### 1.1 定义

中介分析把暴露对结局的总效应分解为「经由某中介变量传递的间接效应」与「不经该中介的直接效应」，用于揭示作用机制而非仅仅是否有效。

### 1.2 它主要解决什么问题

- 研究问题：暴露/治疗是「通过什么」影响结局，中介占多少。
- 适用任务：机制分解、间接效应估计、干预靶点识别。
- 常见医学场景：药物→降血压→降低卒中；肥胖→炎症因子→糖尿病；社会经济地位→行为→结局。

### 1.3 直觉理解

知道治疗有效还不够，我们常想知道它「怎么起效」。中介分析把总效应拆成两条路径：一条经过中介变量（间接），一条绕过它（直接），从而定位机制与潜在干预点。

## 2. 数学形式

### 2.1 核心公式

以线性模型为例，暴露 $X$、中介 $M$、结局 $Y$：

$$
M=\alpha_0+aX+\epsilon_M,\qquad
Y=\beta_0+cX+bM+\epsilon_Y
$$

间接效应为 $a\cdot b$，直接效应为 $c$，总效应 $=ab+c$。反事实框架下更一般地定义自然直接效应（NDE）与自然间接效应（NIE）：

$$
\text{总效应}=\underbrace{\text{NDE}}_{\text{直接}}+\underbrace{\text{NIE}}_{\text{间接}}
$$

### 2.2 参数或统计量含义

- $a$：暴露对中介的效应；$b$：调整暴露后中介对结局的效应。
- $ab$：间接（中介）效应；$c$：直接效应。
- 中介比例 = 间接效应 / 总效应。
- NDE/NIE：反事实定义，可含暴露-中介交互，适用于非线性结局。

### 2.3 关键假设

- 无未测混杂：暴露-结局、中介-结局、暴露-中介三组关系均无未测混杂。
- 中介的时序在暴露之后、结局之前，且测量可靠。
- 无「暴露影响的中介-结局混杂」；必要时建模暴露-中介交互。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：暴露、中介、混杂协变量。
- 因变量类型：连续或二分类结局（中介与结局各有对应模型）。
- 数据结构：每行一个体，含 X、M、Y 与混杂。
- 是否适合高维数据：多中介需专门方法。
- 是否适合缺失较多数据：需谨慎处理缺失。
- 是否适合删失数据：可扩展到生存结局中介。
- 是否适合重复测量数据：需纵向中介模型。

### 3.2 示例表格

| 个体 | 治疗X | 血压M | 卒中Y | 年龄 |
| --- | --- | --- | --- | --- |
| 1 | 1 | 128 | 0 | 61 |
| 2 | 0 | 150 | 1 | 70 |
| 3 | 1 | 132 | 0 | 58 |
| 4 | 0 | 145 | 1 | 66 |

### 3.3 输入与产出

#### 输入

- 输入数据：暴露、中介、结局、混杂变量。
- 关键变量：X、M、Y 及其时序、需调整的混杂集合。
- 需要预处理的内容：确认时序、设定中介与结局模型、是否含交互。

#### 产出

- 模型对象/统计结果：直接效应、间接效应、总效应、中介比例。
- 参数估计：NDE、NIE 及其区间。
- 预测结果：无。
- 不确定性指标：bootstrap 或蒙特卡洛置信区间、敏感性分析。

## 4. 适用场景

- 适合：已知或假设存在机制路径、且中介测量可靠、时序清晰。
- 不适合：中介-结局存在未测混杂、时序不清、中介测量误差大。
- 使用前需要特别检查的点：三组无未测混杂假设、暴露-中介交互、对未测混杂的敏感性分析。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`

```python
import statsmodels.api as sm
from statsmodels.stats.mediation import Mediation
import statsmodels.formula.api as smf

y_model = smf.logit("Y ~ X + M + age", data=df)
m_model = smf.ols("M ~ X + age", data=df)
med = Mediation(y_model, m_model, "X", "M").fit(n_rep=1000)
print(med.summary())   # 直接/间接/总效应与中介比例
```

### 5.2 R

常用包：

- `mediation`

```r
library(mediation)
m_model <- lm(M ~ X + age, data = df)
y_model <- glm(Y ~ X + M + age, data = df, family = binomial)
med <- mediate(m_model, y_model, treat = "X", mediator = "M",
               boot = TRUE, sims = 1000)
summary(med)          # ACME(间接)、ADE(直接)、总效应、中介比例
```

## 6. 结果如何解释

- 核心结果看什么：间接效应（ACME/NIE）、直接效应（ADE/NDE）、中介比例及区间。
- 每个主要参数如何解释：间接效应显著说明存在经该中介的机制；中介比例反映其重要性。
- 临床或医学意义如何表达：如「治疗降卒中风险中约 60% 由血压下降介导」。
- 常见误读：把中介当成已证实的因果机制；忽视未测混杂使中介比例有偏。

## 7. 推荐可视化

- 路径图（X→M→Y 与 X→Y）。
- 直接/间接/总效应的森林图。
- 对未测混杂的敏感性分析曲线。

## 8. 优势、局限与常见坑

### 优势

- 揭示作用机制，超越「是否有效」。
- 反事实框架可处理交互与非线性结局。
- 可识别潜在干预靶点。

### 局限

- 强依赖不可完全验证的无未测混杂假设。
- 中介测量误差会严重偏倚估计。
- 多中介与时变中介需更复杂方法。

### 常见坑

- 用横断面数据做时序不清的中介分析。
- 忽略暴露-中介交互直接用 Baron-Kenny。
- 不做未测混杂敏感性分析就下机制结论。

## 9. 与相近方法的区别

- 和 [[因果推断（Causal Inference）]] 的关系：中介分析是因果推断在「机制分解」上的专门分支。
- 和 [[线性回归（Linear Regression）]] 的区别:中介分析用多个回归组合并在反事实框架下定义效应，而非单一回归系数。
- 和调节（moderation）分析的区别:中介问「怎么起效」，调节问「对谁效应更强」。

## 10. 医学研究中的典型应用

- 治疗通过生理指标（血压、血脂）介导对硬结局的效应分解。
- 暴露→炎症/代谢标志物→疾病的机制研究。
- 社会决定因素经行为路径影响健康结局。

## 11. 相关方法

- [[因果推断（Causal Inference）]]
- [[线性回归（Linear Regression）]]
- [[Logistic回归（Logistic Regression）]]

## 12. 参考资料

- Baron RM, Kenny DA. The moderator-mediator variable distinction in social psychological research. *J Pers Soc Psychol*. 1986;51(6):1173-1182.
- VanderWeele TJ. *Explanation in Causal Inference: Methods for Mediation and Interaction*. Oxford University Press; 2015.
- Imai K, Keele L, Tingley D. A general approach to causal mediation analysis. *Psychol Methods*. 2010;15(4):309-334.
