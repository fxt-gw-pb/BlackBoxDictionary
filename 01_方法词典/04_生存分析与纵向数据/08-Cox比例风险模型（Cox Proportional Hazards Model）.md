---
title: Cox比例风险模型
english_name: Cox Proportional Hazards Model
slug: cox-proportional-hazards-model
aliases: [Cox PH model, Cox model, "Cox比例风险模型（Cox Proportional Hazards Model）"]
category: 生存分析与纵向数据
subcategory: 生存回归
tags: [医学统计, 数据科学, 生存分析, Cox模型]
status: 已建
difficulty: intermediate
question_type: 时间到事件回归建模
data_type: [生存数据]
outcome_type: [时间到事件]
python_packages: [lifelines]
r_packages: [survival]
---

# Cox比例风险模型（Cox Proportional Hazards Model）

## 1. 方法概览

### 1.1 一句话本质

Cox 模型把危险率写成「大家共享的基线风险 × 由协变量决定的倍数」$h(t\mid X)=h_0(t)\,e^{X^\top\beta}$，然后用**偏似然**巧妙地把未知的基线 $h_0(t)$ 消掉——只凭「每次出事的是谁」就能估出风险比，不必知道风险随时间的具体形状。

### 1.2 定义

Cox 比例风险模型是最常用的生存回归方法。它是半参数模型：协变量效应部分是参数化的（$e^{X^\top\beta}$），基线风险 $h_0(t)$ 完全不设形式。核心产出是**风险比（hazard ratio, HR）**，通过 Cox 提出的偏似然（partial likelihood）估计。

### 1.3 它主要解决什么问题

- 研究问题：在调整多个协变量后，哪些因素提高或降低事件发生风险，各提高/降低多少（相对尺度）？
- 适用任务：时间到事件回归、风险比估计、混杂调整、风险预测评分构建。
- 常见医学场景：调整年龄性别后某生物标志物对死亡风险的影响、治疗对复发风险的效应、构建临床预后评分。

### 1.4 直觉与类比

想象每个人身上有一个「风险刻度盘」，指针位置随时间变化，但这个随时间变化的**形状对所有人是一样的**（基线 $h_0(t)$）——男性、老年、高 BMI 只是把各自的指针整体**按固定倍数放大**。Cox 的聪明之处在于：要比较「谁的指针更高」，其实不需要知道指针的绝对刻度。就像排队时想知道「A 比 B 高多少倍」，只要每次「有人被叫号（出事）」时看看排队的都有谁、他们的相对倍数如何，就能推出倍数关系——绝对刻度（基线）在比值里被约掉了。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

参数生存模型（如指数、Weibull）要求你**事先假设风险率随时间的形状**——但真实的基线风险常是先高后低（如术后）、或波浪形，猜错形状就会系统性偏差。而如果放弃参数、又想做「调整多个协变量的回归」，似乎无从下手：似然里含有未知函数 $h_0(t)$，一个无穷维的东西没法直接估。根本困难是：**如何在完全不假设基线风险形状的前提下，仍然估计协变量的效应？**

### 2.2 关键洞察

Cox 的洞察是**偏似然**：只利用「每个事件发生时，出事的偏偏是风险集里的这个人」这一信息。在事件时点 $t_i$，出事者是个体 $j$ 的条件概率为

$$
\frac{h_0(t_i)e^{X_j^\top\beta}}{\sum_{k\in R(t_i)}h_0(t_i)e^{X_k^\top\beta}}=\frac{e^{X_j^\top\beta}}{\sum_{k\in R(t_i)}e^{X_k^\top\beta}}
$$

**基线 $h_0(t_i)$ 在分子分母同时出现、被约掉了！** 于是不必知道基线形状，就能从「出事顺序」估出 $\beta$。代价是丢弃了「事件发生在什么绝对时刻」的信息（只用了排序），但换来了对基线形状的完全免疫。

### 2.3 与朴素/相邻做法的对比

- 相对**参数模型（[[加速失效时间模型（Accelerated Failure Time, AFT, Model）]]、指数/Weibull PH）**：Cox 不设基线形状，更稳健；参数模型在分布正确时更有效率、且能外推预测绝对生存时间。
- 相对 [[Log-rank检验（Log-rank Test）]]：log-rank 只给 p 值，Cox 给 HR 及区间、能同时调整多个协变量；单二分类协变量时 Cox 的 score 检验 = log-rank。
- 相对 [[Kaplan-Meier生存曲线（Kaplan-Meier Estimator）]]：KM 纯描述、无协变量；Cox 是它的回归推广。
- 相对**竞争风险模型**：标准 Cox 把竞争事件当删失（估病因别风险），关注累积发生率负担时改用 [[竞争风险模型（Competing Risks and Fine-Gray Model）]]。

## 3. 数学形式

### 3.1 核心公式

$$
h(t\mid X_i)=h_0(t)\,\exp\!\bigl(X_i^\top\boldsymbol{\beta}\bigr)
$$

这个式子在说：任何人的危险率 = 共享的时间形状 $h_0(t)$ 乘上一个只由其协变量决定、**不随时间变化**的放大倍数。两个人的风险比

$$
HR=\frac{h(t\mid X_1)}{h(t\mid X_2)}=\exp\!\bigl[(X_1-X_2)^\top\boldsymbol{\beta}\bigr]
$$

与 $t$ 无关——这就是「比例风险」之名的由来。

### 3.2 推导脉络

1. 设定 $h(t\mid X)=h_0(t)e^{X^\top\beta}$，比例风险是核心结构假设。
2. 构造偏似然 $L(\beta)=\prod_{i:\,\delta_i=1}\dfrac{e^{X_i^\top\beta}}{\sum_{k\in R(t_i)}e^{X_k^\top\beta}}$，每个事件贡献一项「出事的是他而非风险集里其他人」的条件概率，基线被约掉。
3. 对 $\log L$ 求导得 score $U(\beta)=\sum_i\bigl(X_i-\bar X(t_i)\bigr)$，其中 $\bar X(t_i)$ 是风险集按 $e^{X\beta}$ 加权的协变量均值；令 $U=0$，Newton-Raphson 迭代求解。
4. ties 处理：Efron（默认，推荐）、Breslow（快、近似）、exact（精确、慢）。
5. 基线累积风险由 Breslow 估计补上（即协变量加权版的 [[Nelson-Aalen累积风险估计（Nelson-Aalen Estimator）]]），从而可预测个体生存曲线。

### 3.3 参数与统计量含义

- $h_0(t)$：基线危险率（所有协变量取 0 时的风险随时间形状），不被估计，但可事后由 Breslow 估累积版。
- $\beta_j$：协变量在 log-hazard 尺度的效应。
- $\exp(\beta_j)=HR$：$X_j$ 每增 1 单位、其他不变时风险乘以的倍数。$HR\gt 1$ 有害、$\lt 1$ 有益、$=1$ 无关。
- 偏似然（partial likelihood）：估计 $\beta$ 的目标函数。
- 三大检验：Wald、似然比（LR）、score（=log-rank）。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 比例风险（PH） | HR 不随时间变化 | HR 变成「时间平均」、误导 | Schoenfeld 残差检验、$\log(-\log S)$ 曲线是否平行 |
| log-线性 | log-hazard 与连续协变量线性 | 效应形状被扭曲 | Martingale 残差、加样条 [[限制性立方样条（Restricted Cubic Splines, RCS）]] |
| 非信息性删失 | 删失与预后无关 | 系数有偏 | 比较删失者与在随访者特征 |
| 无重要遗漏/无过强共线 | 模型设定合理 | 混杂残留或系数不稳 | VIF、领域知识、影响点诊断 |

## 4. 手把手算例

用一个只含**一个二分类协变量**（$x=1$ 治疗组，$x=0$ 对照组）的最小例子，亲手把偏似然推到收敛。沿用生存分析卡的同一数据：

- 对照组 $x=0$：2, 4, 6+, 8, 10+
- 治疗组 $x=1$：5, 7+, 9, 12+, 14+（`+` 为删失）

共 5 个事件，时点为 2, 4, 5, 8, 9。每个事件的偏似然项 = $\dfrac{e^{\beta x_{死者}}}{\sum_{风险集}e^{\beta x}}$。

**Step 1：写出对数偏似然的 score 与信息量。** 令 $r=e^\beta$。在时点 $t_i$，风险集有 $n_{Ci}$ 个对照（$e^{0}=1$）和 $n_{Ti}$ 个治疗（$e^\beta=r$），加权总和 $S_0=n_{Ci}+n_{Ti}\,r$，治疗方的加权和 $S_1=n_{Ti}\,r$。死者的 $x$ 记为 $x_i$。

$$
U(\beta)=\sum_i\Bigl(x_i-\frac{S_1}{S_0}\Bigr),\qquad
I(\beta)=\sum_i\Bigl(\frac{S_2}{S_0}-\bigl(\tfrac{S_1}{S_0}\bigr)^2\Bigr),\quad S_2=n_{Ti}\,r\ (\text{因 }x^2=x)
$$

**Step 2：从 $\beta=0$ 起 Newton 迭代**（$\beta\leftarrow\beta+U/I$）：

| 迭代 | $\beta$ | $U(\beta)$ | $I(\beta)$ | 对数偏似然 |
| --- | --- | --- | --- | --- |
| 0 | 0.0000 | −1.031 | 1.159 | −9.575 |
| 1 | −0.889 | +0.030 | 1.155 | −9.130 |
| 2 | −0.864 | −0.000 | 1.161 | −9.130 |
| 3 | −0.864 | 收敛 | 1.161 | −9.130 |

（$\beta=0$ 处的 $U=-1.031$、$I=1.159$ 与 [[Log-rank检验（Log-rank Test）]] 卡手算的 $U_C=1.031$、$V=1.159$ **完全一致**——因为对照组视角的 $U_C$ 就是治疗组视角 score 的相反数，这正是「Cox score 检验 = log-rank」的算术根源。）

**Step 3：读结果。**

- $\hat\beta=-0.864$，风险比 $HR=e^{-0.864}=\mathbf{0.42}$。
- 标准误 $SE=1/\sqrt{I}=1/\sqrt{1.161}=0.928$；95% CI（HR）$=e^{-0.864\pm1.96\times0.928}=[0.068,\ 2.60]$。
- score 检验卡方 $=U(0)^2/I(0)=1.031^2/1.159=0.917$，$p\approx0.34$——与 log-rank 分毫不差。

**结论：** 治疗组的死亡风险约为对照组的 0.42 倍（下降约 58%），方向上有利，但 5 个事件的信息量太少，95% CI 从 0.07 跨到 2.6、包含 1，$p=0.34$ 不显著。这个例子把 Cox 的核心机制暴露无遗：**基线 $h_0(t)$ 从头到尾没出现，只靠「每次出事的是谁」就估出了 HR。**

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：连续、二分类、多分类均可（分类做哑变量）。
- 因变量类型：时间到事件（time + event）。
- 数据结构：每行一个个体；时间依赖协变量用 (start, stop, event] 长表。
- 是否适合高维数据：可扩展，但需正则化（LASSO-Cox 等）。
- 是否适合缺失较多数据：需明确缺失机制、多重插补。
- 是否适合删失数据：非常适合（右删失；左截断用延迟进入）。
- 是否适合重复测量数据：时间依赖协变量/复发事件用扩展（counting process、frailty）。

### 5.2 示例表格

Framingham 风格的生存回归数据：

| RANDID | TIMEDTH | DEATH | SEX | BMI | AGE_group |
| --- | --- | --- | --- | --- | --- |
| 2448 | 8766 | 0 | 0 | 26.97 | 1 |
| 10552 | 2956 | 1 | 1 | 28.58 | 2 |
| 11252 | 8766 | 0 | 1 | 23.10 | 1 |

一个多变量 Cox 模型的 HR 结果可整理为：

| 变量 | HR | 95% CI |
| --- | --- | --- |
| BMI（每 +1） | 1.022 | 1.009 – 1.036 |
| Female vs Male | 0.586 | 0.527 – 0.651 |
| AGE_group 2 vs 1 | 3.282 | 2.938 – 3.666 |
| AGE_group 3 vs 1 | 8.261 | 6.653 – 10.258 |

### 5.3 输入与产出

#### 输入

- 输入数据：时间、事件指示、协变量矩阵。
- 关键变量：`time`、`event`、协变量、ties 处理方法（Efron 默认）。
- 需要预处理的内容：删失编码、分类变量哑编码、连续变量的非线性检查、必要时构造 (start, stop] 长表。

#### 产出

- 模型对象/统计结果：系数、HR、SE、Wald/LR/score 检验、C-index。
- 参数估计：$\boldsymbol\beta$ 与 HR 及区间。
- 预测结果：相对风险、个体预测生存曲线、Breslow 基线累积风险。
- 不确定性指标：HR 置信区间、PH 诊断（Schoenfeld）。

## 6. 适用场景

- 适合：需要调整多个协变量的时间到事件分析——医学生存研究的默认回归工具。
- 不适合：PH 假设明显违反又不做扩展；需要预测绝对生存时间且已知分布（用 AFT）；存在重要竞争风险（用 Fine-Gray）。
- 使用前需要特别检查的点：PH 假设、连续变量线性、ties 比例、影响点、删失机制。

## 7. 实现

### 7.1 Python

常用包：

- `lifelines`

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df, duration_col="TIMEDTH", event_col="DEATH",
        formula="BMI + C(SEX) + C(AGE_group)")
cph.print_summary()                 # HR = exp(coef), 含 95% CI
cph.check_assumptions(df, show_plots=False)   # Schoenfeld 残差 PH 检验
print(cph.concordance_index_)       # C-index
cph.plot_partial_effects_on_outcome("BMI", values=[20, 30])  # 预测生存曲线
```

### 7.2 R

常用包：

- `survival`

```r
library(survival)

fit <- coxph(Surv(TIMEDTH, DEATH) ~ BMI + sex_label + factor(AGE_group),
             data = df, ties = "efron")
summary(fit)                    # coef / exp(coef)=HR / 95% CI / 三种检验
cox.zph(fit)                    # Schoenfeld 残差 PH 检验(p 小=违反 PH)
plot(survfit(fit))              # 预测生存曲线
```

## 8. 结果如何解读

- 核心结果看什么：各 HR 的方向、大小、95% CI 是否含 1、PH 诊断、整体 LR 检验与 C-index。
- 每个主要参数如何解读：$HR=1.022$（BMI）读作「BMI 每增 1，死亡风险约升 2.2%」；$HR=0.586$（女 vs 男）读作「女性风险约为男性的 0.59 倍」。
- 临床或医学意义如何表达：HR 是**相对风险率**尺度，不是绝对风险差，也不是某时点生存率差；需要绝对风险时用预测生存曲线补充。
- 常见误读：把 HR 当 odds ratio 或绝对风险差；PH 违反时仍把单个 HR 当「恒定效应」；只报 p 值不报 HR 区间。

## 9. 假设诊断与稳健性

- 比例风险：`cox.zph`/Schoenfeld 残差随时间无趋势则 PH 成立；违反时可加**时间依赖协变量**（$\beta(t)$）、分层（`strata()` 让不同层各有基线）、或分时段建模。
- 连续变量线性：Martingale 残差图；非线性用 [[限制性立方样条（Restricted Cubic Splines, RCS）]]。
- 影响点：dfbeta/score 残差找单个受试者对某 HR 的过大影响。
- 稳健性：聚集数据用 `cluster()` 稳健方差或 frailty 随机效应；小样本/稀疏事件（EPV\<10）时 HR 不稳，考虑 Firth 惩罚。
- ties 多：优先 Efron；Breslow 在大量 ties 时偏差较大。

## 10. 推荐可视化

- HR 森林图：一图看全所有协变量的效应与区间。
- 分组预测生存曲线：把 HR 翻译成直观的绝对生存差。
- Schoenfeld 残差图：诊断 PH 假设。

### 10.1 图像示例

下图用区间图展示 Cox 模型的 hazard ratio 及其 95% 置信区间。

![](../../04_示例图像/cox_hr_forest.png)

## 11. 优势、局限与常见坑

### 优势

- 半参数：不需假设基线风险形状，稳健且应用极广。
- 直接产出可解释的 HR，天然处理右删失。
- 生态成熟：诊断、扩展（时间依赖、分层、frailty）齐全。

### 局限

- 依赖比例风险假设。
- 只给相对尺度，不直接给绝对生存概率（需 Breslow 基线补上）。
- 高维/稀疏事件时需正则化或惩罚。

### 常见坑

- 不做 PH 诊断，PH 违反时误读「恒定 HR」。
- 连续变量硬套线性，漏掉非线性效应。
- 把竞争事件当普通删失却按「累积发生率」解读结果。
- 事件数过少（EPV\<10）硬塞多个协变量，HR 严重不稳。

## 12. 与相近方法的区别

- 和 [[Kaplan-Meier生存曲线（Kaplan-Meier Estimator）]] / [[Log-rank检验（Log-rank Test）]]：KM/log-rank 无协变量调整；Cox 是能调整混杂、给 HR 的回归推广。
- 和 [[加速失效时间模型（Accelerated Failure Time, AFT, Model）]]：Cox 在 hazard 尺度、半参数、给 HR；AFT 在时间尺度、参数化、给 time ratio。想解释「延长多久」用 AFT，想解释「风险几倍」用 Cox。
- 和 [[条件Logistic回归（Conditional Logistic Regression）]]：匹配病例对照下的条件 Logistic 偏似然与 ties 精确的 Cox 结构同形。
- 和 [[竞争风险模型（Competing Risks and Fine-Gray Model）]]：有竞争事件时，机制问题用 cause-specific Cox，累积发生率负担用 Fine-Gray。
- 如何选择：**默认 Cox；PH 严重违反或要绝对时间预测转 AFT；有竞争风险转 Fine-Gray/cause-specific**。

## 13. 医学研究中的典型应用

- 调整年龄、性别等混杂后评估生物标志物或治疗对死亡/复发风险的效应。
- 构建临床预后评分（如基于多个 HR 的风险分层）。
- 队列研究中多因素时间到事件分析的标准工具。

## 14. 关键术语

- **危险率/风险率（Hazard rate）**：$h(t)$，活到 $t$ 条件下下一瞬发生事件的强度。
- **风险比（Hazard ratio, HR）**：$e^{\beta}$，协变量每增 1 单位风险乘的倍数，Cox 的核心产出。
- **基线风险（Baseline hazard）**：$h_0(t)$，所有协变量为 0 时的风险随时间形状；Cox 不估其形式。
- **偏似然（Partial likelihood）**：只用「事件发生时出事的是谁」构造的似然，把基线约掉，Cox 的估计核心。
- **比例风险假设（Proportional hazards）**：组间 HR 不随时间变化，Cox 的关键前提。
- **Schoenfeld 残差**：逐事件的协变量偏差，用于检验 PH 是否随时间漂移。
- **ties 处理（Efron/Breslow/exact）**：多个事件同时点时偏似然的近似方式。
- **C-index（Concordance index）**：模型对风险排序的判别能力，生存版的 AUC。

## 15. 相关方法

- [[Kaplan-Meier生存曲线（Kaplan-Meier Estimator）]]
- [[Log-rank检验（Log-rank Test）]]
- [[加速失效时间模型（Accelerated Failure Time, AFT, Model）]]
- [[竞争风险模型（Competing Risks and Fine-Gray Model）]]
- [[限制性立方样条（Restricted Cubic Splines, RCS）]]

## 16. 参考资料

- Cox DR. Regression models and life-tables. *J R Stat Soc Ser B*. 1972;34(2):187-220.
- Therneau TM, Grambsch PM. *Modeling Survival Data: Extending the Cox Model*. Springer; 2000.
- Harrell FE. *Regression Modeling Strategies*. 2nd ed. Springer; 2015.
- R Core Team / survival package. `coxph`. R Manual. [https://stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html) （访问日期：2026-07-02）
