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

### 1.1 一句话本质

当「死于别的原因」会永久阻止目标事件发生时，把竞争事件当普通删失会系统性高估目标事件的发生概率；竞争风险方法改用**累积发生率函数（CIF）**如实计数，并用 cause-specific Cox（问机制）或 Fine-Gray（问负担）两条路做回归。

### 1.2 定义

竞争风险分析处理「存在多种互斥结局、一种事件发生后其他事件不再可能」的时间到事件数据。它用 CIF 描述每类事件的累积发生概率，用**病因别风险（cause-specific hazard）**回归探究机制、用 **Fine-Gray 亚分布风险**回归直接建模 CIF，产出亚分布风险比（sHR）。

### 1.3 它主要解决什么问题

- 研究问题：当患者可能先死于别的原因时，如何正确估计某一特定事件的真实发生概率及其影响因素？
- 适用任务：竞争事件下的累积发生率估计、亚分布/病因别风险回归、特定事件累积概率预测。
- 常见医学场景：肿瘤特异性死亡 vs 其他原因死亡、骨髓移植后复发 vs 非复发死亡、心血管事件 vs 全因死亡（尤其老年、多病共存人群竞争事件不可忽视）。

### 1.4 直觉与类比

想象一个房间有两扇出口门：「目标门」（目标事件）和「竞争门」（如其他原因死亡）。每个人最终只从一扇门离开。若想估「从目标门出去」的概率，却把「从竞争门出去」的人当成「暂时离开、以后还会回来」（普通删失），就会高估目标门——因为这些人其实**永远不会**再从目标门出去了。竞争风险方法承认「从竞争门走的人已经出局」，只让还在房间里的人有机会走目标门，给出真实世界的累积发生率。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

标准生存分析（KM、Cox）背后有个隐含假设：被删失的人「以后仍可能发生事件」，只是我们没观测到。竞争事件违反了这个假设——死于其他原因的人**根本不可能**再发生目标事件。若强行把竞争事件当删失：

- 用 $1-\text{KM}$ 估目标事件发生率会**高估**（把已出局者当成「还会出事」）；
- 各事件的 $1-\text{KM}$ 之和甚至可能**超过 100%**（逻辑上荒谬）。

根本困难是：**当一种事件会排除另一种事件时，如何给出加起来不超过 1、且反映真实世界的发生概率？**

### 2.2 关键洞察

引入**累积发生率函数** $F_k(t)=\int_0^t S(u^-)\lambda_k(u)\,du$：在时点 $u$ 增加的目标事件概率，必须先乘上「活到 $u$（还没被任何事件带走）」的总生存 $S(u^-)$。这个 $S(u^-)$ 因子把竞争事件的影响正确地扣了进来——它随任何一类事件发生而下降，因此已死于竞争事件的人自动不再贡献目标事件概率。于是各事件 CIF 之和恰好等于 $1-S(t)$（总发生概率），永不越界。回归有两条路：

- **cause-specific hazard**：只在「还在场」的人里问某类事件的瞬时风险——回答**机制/病因**问题（这个因素是否加快该病因的风险）。
- **Fine-Gray 亚分布风险**：巧妙地把「已发生竞争事件的人」保留在风险集里（当作「永不发生目标事件」），使模型系数直接连到 CIF——回答**预测/负担**问题（这个因素是否提高目标事件的累积发生率）。

### 2.3 与朴素/相邻做法的对比

- 相对 $1-$ [[Kaplan-Meier生存曲线（Kaplan-Meier Estimator）]]：KM 补法高估竞争风险下的发生率，CIF 是正确替代。
- 相对标准 [[Cox比例风险模型（Cox Proportional Hazards Model）]]：标准 Cox 把竞争事件删失，估的是 cause-specific hazard（机制），不能直接读成累积发生率；Fine-Gray 直接对 CIF 建模。
- 相对 [[Nelson-Aalen累积风险估计（Nelson-Aalen Estimator）]]：NA 估累积风险，竞争风险聚焦累积发生率（概率）。
- cause-specific 与 Fine-Gray 不是二选一的对错，而是**回答不同问题**：机制用前者，预测/负担用后者，常需同时报告。

## 3. 数学形式

### 3.1 核心公式

事件 $k$ 的病因别风险与累积发生率函数（CIF）：

$$
\lambda_k(t)=\lim_{\Delta t\to 0}\frac{P(t\le T\lt t+\Delta t,\ \varepsilon=k\mid T\ge t)}{\Delta t},\qquad
F_k(t)=\int_0^t S(u^-)\,\lambda_k(u)\,du
$$

这个式子在说：$\lambda_k$ 是「还在场的人里」发生第 $k$ 类事件的瞬时率；$F_k(t)$ 把每一刻新增的第 $k$ 类事件概率累加，但每一刻都要先乘「活到此刻」的总生存 $S(u^-)$，从而把竞争事件扣除。Fine-Gray 对**亚分布风险**建比例模型：

$$
\lambda_k^{\mathrm{sd}}(t\mid X)=\lambda_{k0}^{\mathrm{sd}}(t)\exp\!\bigl(X^\top\boldsymbol{\beta}\bigr)
$$

其系数 $\exp(\beta)$ 为**亚分布风险比（sHR）**，直接关联 CIF。

### 3.2 推导脉络

1. 总生存 $S(t)=\exp\!\bigl(-\sum_k \int_0^t\lambda_k\bigr)$ 随**任何**事件下降。
2. CIF 由 $S(u^-)\lambda_k(u)$ 累加，保证 $\sum_k F_k(t)=1-S(t)\le 1$。
3. 非参数估计：Aalen-Johansen 估计（KM 的多状态推广）给出 CIF。
4. cause-specific Cox：把其他事件当删失，对 $\lambda_k$ 直接套 Cox。
5. Fine-Gray：定义「亚分布风险集」——发生竞争事件的人**不移出**风险集（并赋随时间衰减的删失权重），使 $\beta$ 与 CIF 单调对应；对亚分布风险套比例模型。

### 3.3 参数与统计量含义

- $\lambda_k(t)$：病因别风险（cause-specific hazard），机制导向。
- $F_k(t)$：CIF，事件 $k$ 到时刻 $t$ 的累积发生概率（真实世界口径）。
- **亚分布风险比 sHR**：Fine-Gray 系数 $e^\beta$，反映协变量对 CIF 的作用（负担/预测）。
- **病因别 HR**：cause-specific Cox 的 $e^\beta$，反映对瞬时病因风险的作用（机制）。
- 二者可方向不一致：某因素可提高目标事件的病因风险，却因更强地提高竞争事件而使目标 CIF 反而下降。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 事件互斥、编码清晰 | 每人至多观察到一种事件或删失 | CIF/回归错乱 | 核对 status 编码（0/1/2…） |
| 相应比例风险 | cause-specific 或 subdistribution 的 PH 成立 | HR/sHR 变时间平均、误导 | Schoenfeld 类残差、分时段 |
| 真删失非信息性 | 真正的删失（非竞争事件）与预后无关 | 估计有偏 | 比较删失者特征 |
| 目标与分析问题匹配 | 机制问题用 cause-specific、负担用 Fine-Gray | 误用模型误导结论 | 明确研究问题再选模型 |

## 4. 手把手算例

5 名患者，时间（月）与结局类型（1=目标事件，2=竞争事件，0=删失）：
**(1, 竞争), (2, 目标), (3, 竞争), (4, 目标), (5, 删失)**。

**先算总生存 $S(t)$（任何事件都让它下降），再累加 CIF。**

| 时点 $t$ | 在场 $n$ | 事件类型 | $S(t^-)$ | 本步增量 $\Delta F=S(t^-)\cdot\frac1n$ | $F_1$（目标） | $F_2$（竞争） | 更新 $S(t)$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5 | 竞争 | 1.00 | 1.00×1/5 = 0.20 | 0.00 | 0.20 | 0.80 |
| 2 | 4 | 目标 | 0.80 | 0.80×1/4 = 0.20 | 0.20 | 0.20 | 0.60 |
| 3 | 3 | 竞争 | 0.60 | 0.60×1/3 = 0.20 | 0.20 | 0.40 | 0.40 |
| 4 | 2 | 目标 | 0.40 | 0.40×1/2 = 0.20 | 0.40 | 0.40 | 0.20 |
| 5 | 1 | 删失 | — | — | 0.40 | 0.40 | 0.20 |

**验算：** $F_1(4)+F_2(4)=0.40+0.40=0.80=1-S(4)=1-0.20$ ✓ ——两条 CIF 之和恰好等于总发生概率，不越界。**目标事件的正确累积发生率 = 0.40。**

**对比错误做法（把竞争事件当删失，用 $1-\text{KM}$）：**

- 目标事件 $1-\text{KM}$：事件时点 $t=2$（$n=4$）、$t=4$（$n=2$），$1-\tfrac{3}{4}\cdot\tfrac{1}{2}=1-0.375=\mathbf{0.625}$。
- 竞争事件 $1-\text{KM}$：$t=1$（$n=5$）、$t=3$（$n=3$），$1-\tfrac{4}{5}\cdot\tfrac{2}{3}=1-0.533=\mathbf{0.467}$。
- 两者相加 $0.625+0.467=\mathbf{1.09}\gt 1$ ——**逻辑上不可能**！

**结论：** 正确的目标事件累积发生率是 CIF 给的 **0.40**，而把竞争事件当删失的 $1-\text{KM}$ 报出 **0.625**，高估了逾一半，且两事件之和超过 100% 暴露了方法错误。竞争事件越常见，这种高估越严重——这正是竞争风险方法不可或缺的原因。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：连续、二分类、多分类。
- 因变量类型：时间 + 多状态事件指示（0=删失，1=目标事件，2=竞争事件…）。
- 数据结构：每行一个个体，含 time 与多类别 status。
- 是否适合高维数据：可扩展，需谨慎与正则化。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：专为「真删失 + 竞争事件」设计。
- 是否适合重复测量数据：多事件/复发需多状态或频率模型扩展。

### 5.2 示例表格

| 个体 | time | status(0删失/1目标/2竞争) | 治疗 | 年龄 |
| --- | --- | --- | --- | --- |
| 1 | 540 | 1 | 1 | 62 |
| 2 | 720 | 0 | 0 | 55 |
| 3 | 300 | 2 | 1 | 78 |
| 4 | 900 | 1 | 0 | 49 |

### 5.3 输入与产出

#### 输入

- 输入数据：time、多类别 status、协变量。
- 关键变量：目标事件编码、竞争事件编码、真删失编码。
- 需要预处理的内容：正确区分真删失与竞争事件、选定目标事件、明确分析问题（机制/负担）。

#### 产出

- 模型对象/统计结果：各事件 CIF、cause-specific HR 或 sHR、Gray 检验。
- 参数估计：$\boldsymbol\beta$、sHR / cause-specific HR。
- 预测结果：给定协变量的特定事件累积发生概率曲线。
- 不确定性指标：CIF 置信带、HR/sHR 区间、Gray 检验 p 值。

## 6. 适用场景

- 适合：存在不可忽视竞争事件的生存数据，需真实累积发生概率或其影响因素。
- 不适合：无竞争事件（普通生存分析即可）；竞争事件极罕见可忽略。
- 使用前需要特别检查的点：分析目标是机制还是预测、相应 PH 假设、事件编码是否正确区分删失与竞争。

## 7. 实现

### 7.1 Python

常用包：

- `lifelines`

```python
from lifelines import AalenJohansenFitter

# T: 时间; E: 事件类型(0删失,1目标,2竞争)
ajf = AalenJohansenFitter()
ajf.fit(T, E, event_of_interest=1)     # 目标事件的 CIF(正确, 非 1-KM)
ajf.plot()
print(ajf.cumulative_density_.tail())

# 病因别 Cox: 把竞争事件当删失, 对 λ_1 建模(机制)
from lifelines import CoxPHFitter
df_cs = df.assign(event1=(df["E"] == 1).astype(int))
CoxPHFitter().fit(df_cs, "T", "event1").print_summary()
```

### 7.2 R

常用包：

- `cmprsk`

```r
library(cmprsk)

# ftime: 时间; fstatus: 0删失/1目标/2竞争
ci <- cuminc(ftime = df$ftime, fstatus = df$fstatus)   # CIF + Gray 检验
plot(ci)

# Fine-Gray 亚分布风险回归(目标事件=1) -> sHR(负担/预测)
cov <- model.matrix(~ treat + age, df)[, -1]
fg  <- crr(df$ftime, df$fstatus, cov1 = cov, failcode = 1, cencode = 0)
summary(fg)               # exp(coef) = 亚分布风险比 sHR

# 机制路径: survival::coxph 把竞争事件当删失得 cause-specific HR
```

## 8. 结果如何解读

- 核心结果看什么：各事件 CIF 曲线、目标事件的 sHR（负担）或 cause-specific HR（机制）、Gray 检验。
- 每个主要参数如何解读：sHR\>1 表示该因素提高目标事件的累积发生率；cause-specific HR\>1 表示提高目标事件的瞬时病因风险。二者可方向相反。
- 临床或医学意义如何表达：明确区分「机制问题→cause-specific」「预测/负担问题→Fine-Gray」，并说明报的是哪种、CIF 是真实世界口径。
- 常见误读：用 $1-\text{KM}$ 报竞争风险下发生率（高估）；把 sHR 当 cause-specific HR；把竞争事件直接删失。

## 9. 假设诊断与稳健性

- 亚分布/病因别 PH：类 Schoenfeld 残差检验；违反时分时段或加时间交互。
- 模型选择与问题匹配：先明确是机制还是预测，再选 cause-specific 或 Fine-Gray——这是最常见的「用错模型」根源。
- 敏感性：同时报两类模型，看结论是否稳健、方向是否一致（不一致本身是重要发现）。
- 竞争事件占比：占比越高，忽视竞争风险的偏差越大，越需 CIF/Fine-Gray。

## 10. 推荐可视化

- 各事件 CIF 叠加图：一图看全目标与竞争事件的累积发生。
- 按组别的 CIF 对比（配 Gray 检验）：展示协变量对累积发生率的影响。
- sHR 森林图：Fine-Gray 各协变量的亚分布效应。

## 11. 优势、局限与常见坑

### 优势

- 正确估计竞争事件下的真实累积发生概率（各事件 CIF 之和 $=1-S$，不越界）。
- 提供机制导向（cause-specific）与预测导向（Fine-Gray）两条互补路径。
- 配套组间检验（Gray）与回归框架完整。

### 局限

- 两类模型解释不同，误用易误导。
- Fine-Gray 的「亚分布风险集」保留已发生竞争事件者，机制解释不直观。
- 相应比例（亚分布）风险假设可能不成立。

### 常见坑

- 把竞争事件当普通删失、用 KM/标准 Cox 报累积发生率。
- 混淆 sHR 与 cause-specific HR。
- 只报一种模型而不说明分析目标。

## 12. 与相近方法的区别

- 和 [[Kaplan-Meier生存曲线（Kaplan-Meier Estimator）]]：竞争风险下 KM 的 $1-\text{KM}$ 高估单事件累积发生率，应改用 CIF（Aalen-Johansen）。
- 和 [[Cox比例风险模型（Cox Proportional Hazards Model）]]：标准 Cox 把竞争事件删失得 cause-specific HR（机制）；Fine-Gray 直接对 CIF 建模得 sHR（负担）。
- 和 [[Nelson-Aalen累积风险估计（Nelson-Aalen Estimator）]]：NA 估累积风险，竞争风险聚焦累积发生率 CIF（概率）。
- 和 [[Log-rank检验（Log-rank Test）]]：组间 CIF 比较用 Gray 检验（log-rank 的竞争风险版）。
- 如何选择：**先判断有无不可忽视的竞争事件；有则机制问题用 cause-specific Cox、负担/预测问题用 Fine-Gray，并用 CIF 而非 1−KM 报发生率**。

## 13. 医学研究中的典型应用

- 肿瘤研究中肿瘤特异性死亡 vs 其他死因的区分分析。
- 移植/血液病中复发 vs 非复发死亡的竞争风险评估。
- 老年或多病共存人群中特定事件累积发生率的估计。

## 14. 关键术语

- **竞争事件（Competing event）**：其发生会永久阻止目标事件发生的事件（如其他原因死亡）。
- **累积发生率函数（Cumulative incidence function, CIF）**：$F_k(t)$，竞争风险下事件 $k$ 到 $t$ 的真实累积发生概率，各事件之和 $=1-S(t)$。
- **病因别风险（Cause-specific hazard）**：在场者中某类事件的瞬时率，机制导向。
- **亚分布风险（Subdistribution hazard）**：Fine-Gray 定义的、保留已发生竞争事件者于风险集的风险，直接连接 CIF。
- **亚分布风险比（sHR）**：Fine-Gray 系数指数，反映对累积发生率的作用。
- **Aalen-Johansen 估计**：CIF 的非参数估计，KM 的多状态推广。
- **Gray 检验（Gray's test）**：竞争风险下比较组间 CIF 的检验，log-rank 的对应物。

## 15. 相关方法

- [[Kaplan-Meier生存曲线（Kaplan-Meier Estimator）]]
- [[Cox比例风险模型（Cox Proportional Hazards Model）]]
- [[Nelson-Aalen累积风险估计（Nelson-Aalen Estimator）]]
- [[加速失效时间模型（Accelerated Failure Time, AFT, Model）]]
- [[Log-rank检验（Log-rank Test）]]

## 16. 参考资料

- Fine JP, Gray RJ. A proportional hazards model for the subdistribution of a competing risk. *J Am Stat Assoc*. 1999;94(446):496-509.
- Austin PC, Lee DS, Fine JP. Introduction to the analysis of survival data in the presence of competing risks. *Circulation*. 2016;133(6):601-609.
- Putter H, Fiocco M, Geskus RB. Tutorial in biostatistics: competing risks and multi-state models. *Stat Med*. 2007;26(11):2389-2430.
