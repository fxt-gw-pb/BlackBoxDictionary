---
title: 竞争风险模型
english_name: Competing Risks and Fine-Gray Model
slug: competing-risks-and-fine-gray-model
aliases: [competing risks, Fine-Gray, 竞争风险, 亚分布风险模型, "竞争风险模型（Competing Risks and Fine-Gray Model）"]
category: 生存分析与纵向数据
subcategory: 生存回归
tags: [医学统计, 生存分析, 竞争风险, 累积发生率]
status: 已建
difficulty: advanced
question_type: 存在竞争事件的时间到事件建模
data_type: [生存数据]
outcome_type: [时间到事件]
python_packages: [lifelines]
r_packages: [cmprsk]
---

# 竞争风险模型（Competing Risks and Fine-Gray Model）

## 1. 方法概览

### 1.1 定义

竞争风险分析处理“存在多种互斥结局、某一事件的发生会阻止其他事件发生”的生存数据。它用累积发生率函数（CIF）描述各事件的发生，并用 cause-specific Cox 或 Fine-Gray 亚分布风险模型做回归。

### 1.2 它主要解决什么问题

- 研究问题：当患者可能死于多种原因（或先发生别的事件）时，如何正确估计某一特定事件的发生概率与其影响因素。
- 适用任务：竞争事件下的发生率估计、亚分布/病因别风险回归、预测特定事件累积概率。
- 常见医学场景：肿瘤特异性死亡 vs 其他原因死亡、移植后复发 vs 非复发死亡、心血管事件 vs 全因死亡。

### 1.3 直觉理解

若把竞争事件（如“其他原因死亡”）当作普通删失，用 1−KM 估某事件概率会系统性高估——因为已死于其他原因的人根本不可能再发生目标事件。竞争风险方法把这些人正确地留在风险集合的分母逻辑里，给出真实世界中的累积发生率。

## 2. 数学形式

### 2.1 核心公式

事件 $k$ 的病因别风险与累积发生率函数（CIF）：

$$
\lambda_k(t)=\lim_{\Delta t\to 0}\frac{P(t\le T<t+\Delta t,\,\varepsilon=k\mid T\ge t)}{\Delta t},\qquad
F_k(t)=\int_0^t S(u^-)\,\lambda_k(u)\,du
$$

Fine-Gray 模型对亚分布风险建比例模型：

$$
\lambda_k^{\mathrm{sd}}(t\mid X)=\lambda_{k0}^{\mathrm{sd}}(t)\exp(X^\top\boldsymbol{\beta})
$$

其系数 $\exp(\beta)$ 为亚分布风险比（sHR），直接关联到 CIF。

### 2.2 参数或统计量含义

- $\lambda_k(t)$：病因别风险（cause-specific hazard），关注病因机制。
- $F_k(t)$：CIF，事件 $k$ 到时刻 $t$ 的累积发生概率。
- 亚分布风险比 sHR：Fine-Gray 系数，反映协变量对 CIF 的作用。
- 病因别 HR：对某一事件用 cause-specific Cox 得到的风险比。

### 2.3 关键假设

- 竞争事件互斥且定义清晰；每个个体最多观察到一种事件或删失。
- 相应比例风险假设（cause-specific 或 subdistribution）成立。
- 删失（真正的删失，非竞争事件）独立、非信息性。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、二分类、多分类。
- 因变量类型：时间 + 多状态事件指示（0=删失，1=目标事件，2=竞争事件…）。
- 数据结构：每行一个个体，含 time 与 event 编码（可含竞争事件编码）。
- 是否适合高维数据：可扩展，需谨慎与正则化。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：专为“删失 + 竞争事件”设计。
- 是否适合重复测量数据：多事件/复发需多状态或频率模型扩展。

### 3.2 示例表格

| 个体 | time | status(0删失/1目标/2竞争) | 治疗 | 年龄 |
| --- | --- | --- | --- | --- |
| 1 | 540 | 1 | 1 | 62 |
| 2 | 720 | 0 | 0 | 55 |
| 3 | 300 | 2 | 1 | 78 |
| 4 | 900 | 1 | 0 | 49 |

### 3.3 输入与产出

#### 输入

- 输入数据：time、多类别 status、协变量。
- 关键变量：目标事件编码、竞争事件编码、删失编码。
- 需要预处理的内容：正确区分删失与竞争事件、选定目标事件。

#### 产出

- 模型对象/统计结果：CIF 估计、cause-specific HR 或 sHR。
- 参数估计：$\boldsymbol{\beta}$、sHR。
- 预测结果：给定协变量的特定事件累积发生概率。
- 不确定性指标：CIF 置信带、HR/sHR 区间、Gray 检验。

## 4. 适用场景

- 适合：存在不可忽视竞争事件的生存数据、需要真实累积发生概率或其影响因素。
- 不适合：无竞争事件（普通生存分析即可）、竞争事件极罕见可忽略时。
- 使用前需要特别检查的点：分析目标是病因机制还是预测、相应比例风险假设、事件编码是否正确。

## 5. 实现

### 5.1 Python

常用包：

- `lifelines`

```python
from lifelines import AalenJohansenFitter

# T: 时间; E: 事件类型(0删失,1目标,2竞争)
ajf = AalenJohansenFitter()
ajf.fit(T, E, event_of_interest=1)   # 目标事件的 CIF
ajf.plot()
print(ajf.cumulative_density_.tail())
```

### 5.2 R

常用包：

- `cmprsk`

```r
library(cmprsk)

# ftime: 时间; fstatus: 0删失/1目标/2竞争
ci <- cuminc(ftime = df$ftime, fstatus = df$fstatus)  # CIF 与 Gray 检验
plot(ci)

# Fine-Gray 亚分布风险回归(目标事件=1)
cov <- model.matrix(~ treat + age, df)[, -1]
fg <- crr(df$ftime, df$fstatus, cov1 = cov, failcode = 1, cencode = 0)
summary(fg)   # 亚分布风险比 sHR
```

## 6. 结果如何解释

- 核心结果看什么：各事件 CIF 曲线、目标事件的 sHR 或 cause-specific HR、Gray 检验。
- 每个主要参数如何解释：sHR>1 表示该因素提高目标事件的累积发生率；cause-specific HR 反映对瞬时病因风险的作用。
- 临床或医学意义如何表达：区分“机制问题用 cause-specific、预测/负担问题用 Fine-Gray”，并说明所报的是哪种。
- 常见误读：用 1−KM 报竞争风险下的发生率（高估）；把 sHR 当成 cause-specific HR；忽视竞争事件直接删失。

## 7. 推荐可视化

- 各事件的累积发生率函数（CIF）叠加图。
- 按组别的 CIF 对比（配 Gray 检验）。
- sHR 森林图。

## 8. 优势、局限与常见坑

### 优势

- 正确估计竞争事件下的真实累积发生概率。
- 提供机制导向（cause-specific）与预测导向（Fine-Gray）两条路径。
- 有配套的组间检验（Gray）与回归框架。

### 局限

- 两类模型解释不同，误用会误导结论。
- Fine-Gray 的“风险集合”定义不直观，机制解释受限。
- 比例（亚分布）风险假设可能不成立。

### 常见坑

- 把竞争事件当普通删失、用 KM/标准 Cox。
- 混淆 sHR 与 cause-specific HR。
- 只报一种模型而不说明分析目标。

## 9. 与相近方法的区别

- 和 [[Kaplan-Meier生存曲线（Kaplan-Meier Estimator）]] 的区别：KM 在竞争风险下会高估单事件累积发生率，应改用 CIF。
- 和 [[Cox比例风险模型（Cox Proportional Hazards Model）]] 的区别：标准 Cox 把竞争事件删失，Fine-Gray 直接对 CIF 建模；cause-specific Cox 用于病因机制。
- 和 [[Nelson-Aalen累积风险估计（Nelson-Aalen Estimator）]] 的区别：Nelson-Aalen 估累积风险，竞争风险聚焦累积发生率 CIF。

## 10. 医学研究中的典型应用

- 肿瘤研究中肿瘤特异性死亡与其他死因的区分分析。
- 移植/血液病中复发与非复发死亡的竞争风险评估。
- 老年或多病共存人群中特定事件累积发生率的估计。

## 11. 相关方法

- [[Kaplan-Meier生存曲线（Kaplan-Meier Estimator）]]
- [[Cox比例风险模型（Cox Proportional Hazards Model）]]
- [[Nelson-Aalen累积风险估计（Nelson-Aalen Estimator）]]
- [[加速失效时间模型（Accelerated Failure Time, AFT, Model）]]

## 12. 参考资料

- Fine JP, Gray RJ. A proportional hazards model for the subdistribution of a competing risk. *J Am Stat Assoc*. 1999;94(446):496-509.
- Austin PC, Lee DS, Fine JP. Introduction to the analysis of survival data in the presence of competing risks. *Circulation*. 2016;133(6):601-609.
- Putter H, Fiocco M, Geskus RB. Tutorial in biostatistics: competing risks and multi-state models. *Stat Med*. 2007;26(11):2389-2430.
