---
title: Kaplan-Meier生存曲线
english_name: Kaplan-Meier Estimator
slug: kaplan-meier-estimator
aliases: [Kaplan-Meier, KM curve, product-limit estimator, "Kaplan-Meier生存曲线（Kaplan-Meier Estimator）"]
category: 生存分析与纵向数据
subcategory: 生存函数非参数估计
tags: [医学统计, 数据科学, 生存分析, 非参数估计]
status: 已建
difficulty: basic
question_type: 生存函数估计
data_type: [生存数据]
outcome_type: [时间到事件]
python_packages: [lifelines]
r_packages: [survival]
---

# Kaplan-Meier生存曲线（Kaplan-Meier Estimator）

## 1. 方法概览

### 1.1 一句话本质

Kaplan-Meier 把「活过时刻 $t$」拆成一串「活过每个事件时点」的条件概率**连乘**——删失者在退出前为每个分母出力、退出后安静离场，删失携带的信息被用尽而不被误用。

### 1.2 定义

Kaplan-Meier 估计（又称乘积极限估计，product-limit estimator）是在存在右删失时估计生存函数 $S(t)=P(T\gt t)$ 的经典非参数方法。它不假设生存时间服从任何分布，只在每个观察到事件的时点上更新估计，输出一条阶梯状下降的生存曲线。

### 1.3 它主要解决什么问题

- 研究问题：随访到某个时间点，个体「仍未发生事件」的概率是多少？中位生存时间是多久？
- 适用任务：时间到事件结局的描述性估计——生存曲线、任意时点生存率、中位生存时间；配合 [[Log-rank检验（Log-rank Test）]] 做组间比较。
- 常见医学场景：肿瘤患者的总生存（OS）与无进展生存（PFS）曲线、术后复发前时间、植入装置失效时间。结局须为「时间 + 事件指示」形式的时间到事件数据，且几乎必然带右删失。

### 1.4 直觉与类比

想象一场跨年马拉松：想知道「跑满 8 小时仍在跑」的比例，但有些人中途被家人接走了（删失）——他们退出时还在跑，只是之后你看不到了。合理的算法是：每当有人**倒下**（事件），就问「此刻还在场的人里，倒下的占几分之几」，把「挺过这一关」的比例记下来；被接走的人在被接走之前算「在场」，之后不再进入任何分母。最后把每一关的「挺过率」连乘，就是「跑满 8 小时」的概率。KM 曲线因此长成阶梯形：每次事件掉一格，删失只做个小标记、不掉格。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

没有删失时，估计 $S(t)$ 很简单：数一数活过 $t$ 的人占几成。**删失把这个简单比例毁掉了**——对被删失者，我们只知道「至少活过删失时刻」，不知道之后如何。朴素补救全都有偏：

- **丢掉删失者**：留下的多是发生了事件的人，系统性**低估**生存；
- **把删失当事件**：让「还活着的人」提前死亡，也**低估**生存；
- **把删失当活到天荒地老**：又**高估**生存。

根本困难是：如何让「他至少活过了 $c$」这条**不完整但真实**的信息，不多不少恰好贡献它该贡献的份额？

### 2.2 关键洞察

把无条件概率拆成条件概率的连乘：

$$
P(T\gt t_2)=P(T\gt t_1)\times P(T\gt t_2\mid T\gt t_1)
$$

每个条件概率 $P(T\gt t_i\mid T\gt t_{i-1})$ 只依赖「该时点还在风险集里的人中有几个出事」——这恰好是删失数据也能提供的信息：删失者在删失前对每个风险集分母都有效，删失后自然退出，不需要对他的未来做任何假设。**「拆成条件概率」让删失从麻烦变成了可用信息**。

### 2.3 与朴素/相邻做法的对比

- 相对**简单比例（经验生存函数 / ECDF 的补）**：无删失时 KM 与 $1-\text{ECDF}$ 完全相同；KM 是 ECDF 在删失数据上的推广。
- 相对**寿命表法（actuarial method）**：寿命表把时间切成固定区间（如每年），KM 在每个精确事件时点更新，更充分利用连续随访数据。
- 相对 [[Cox比例风险模型（Cox Proportional Hazards Model）]]：KM 是无协变量的描述工具，只能按组画曲线；要调整多个协变量需上回归模型。

## 3. 数学形式

### 3.1 核心公式

$$
\hat S(t)=\prod_{t_i \le t}\left(1-\frac{d_i}{n_i}\right)
$$

这个式子在说：把 $t$ 之前每个事件时点 $t_i$ 的「即刻死亡率」$d_i/n_i$ 换成「即刻存活率」$1-d_i/n_i$，然后全部连乘——「活到 $t$」就是「每一关都挺过去」。

### 3.2 推导脉络

1. 只在观察到事件的时点 $t_1\lt t_2\lt\cdots$ 上考虑离散风险 $h_i=P(T=t_i\mid T\ge t_i)$。
2. 每个 $h_i$ 的自然估计是 $\hat h_i=d_i/n_i$（风险集中出事的比例）——这一步删失者已按「删失前在分母、删失后退出」正确计入。
3. 由条件概率连乘 $S(t)=\prod_{t_i\le t}(1-h_i)$ 代入 $\hat h_i$ 即得 KM 公式。可以证明它同时是删失数据下生存函数的非参数最大似然估计。
4. 方差用 **Greenwood 公式**：

$$
\widehat{\mathrm{Var}}[\hat S(t)]=\hat S(t)^2\sum_{t_i\le t}\frac{d_i}{n_i(n_i-d_i)}
$$

置信区间实践中常在 $\log(-\log S)$ 尺度上构造（R 的 `conf.type = "log-log"`），保证区间落在 $[0,1]$ 内。

### 3.3 参数与统计量含义

- $t_i$：第 $i$ 个**观察到事件**的时点（删失时点不引起更新）。
- $n_i$：$t_i$ 时刻前一瞬还在风险集（at risk）的人数——尚未发生事件、也尚未被删失。
- $d_i$：$t_i$ 时刻发生事件的人数。
- $\hat S(t)$：生存函数估计；**中位生存时间**是 $\hat S(t)$ 首次降到 0.5 及以下的时刻。
- 无回归系数——KM 是纯描述性估计。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 非信息性删失 | 删失与预后无关（不是「病重才失访」） | 病重者先失访→高估生存；反之低估 | 比较删失者与在随访者的基线特征 |
| 个体独立 | 各人的事件时间互不影响 | 方差估计失真 | 看设计（家系/中心聚集需谨慎） |
| 事件与时间定义清晰 | 起点、终点、事件编码一致 | 曲线整体错位 | 核对 time 单位与 event 编码 |
| 无竞争风险混入 | 「其他原因死亡」未被当普通删失 | 高估目标事件发生率 | 若存在竞争事件改用 CIF（见 [[竞争风险模型（Competing Risks and Fine-Gray Model）]]） |

## 4. 手把手算例

5 名对照组患者随访（单位：月），`+` 表示右删失：**2, 4, 6+, 8, 10+**。

（本卡与 [[Nelson-Aalen累积风险估计（Nelson-Aalen Estimator）]]、[[Log-rank检验（Log-rank Test）]]、[[Cox比例风险模型（Cox Proportional Hazards Model）]] 共用这套数据，可对照阅读。）

**一步步计算：**

| 时点 $t_i$ | 在场 $n_i$ | 事件 $d_i$ | 本关存活率 $1-d_i/n_i$ | $\hat S(t_i)$ |
| --- | --- | --- | --- | --- |
| 2 | 5 | 1 | 4/5 = 0.80 | 0.80 |
| 4 | 4 | 1 | 3/4 = 0.75 | 0.80×0.75 = **0.60** |
| 6+ | （删失：$n$ 由 3 减为 2，曲线不掉） | — | — | 0.60 |
| 8 | 2 | 1 | 1/2 = 0.50 | 0.60×0.50 = **0.30** |
| 10+ | （删失，随访结束） | — | — | 0.30 |

注意第 6 个月删失的那位：他为 $t=2$ 和 $t=4$ 的分母各出了一次力（我们确知他活过了这两关），但 $t=8$ 那关他不在分母里——我们不知道他第 8 个月的死活，KM 也不假装知道。

**中位生存时间**：$\hat S(t)$ 首次不超过 0.5 发生在 $t=8$（0.30），故中位生存时间 = 8 个月。

**对比朴素做法**：丢掉 2 名删失者只剩 3 人全部死亡，会得出「10 个月时生存率 0%」；把删失当事件同样在 10 个月归零。KM 给出的 30% 才正确使用了「删失者至少活到了 6 和 10 个月」的信息。

**结论：** 该组 4 个月生存率约 60%，8 个月生存率约 30%，中位生存 8 个月。曲线尾部只剩 1–2 人，末端的 0.30 不宜过度解读。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：可无分组；或加一个分组变量画分层曲线。
- 因变量类型：时间到事件（time + event 两列）。
- 数据结构：每行一个个体。
- 是否适合高维数据：不涉及协变量建模，谈不上高维。
- 是否适合缺失较多数据：time/event 缺失需先处理。
- 是否适合删失数据：为右删失而生；左截断需扩展（延迟进入）。
- 是否适合重复测量数据：不适合，时间依赖协变量需其他方法。

### 5.2 示例表格

Framingham 队列风格的生存数据（TIMEDTH=随访天数，DEATH=1 死亡/0 删失）：

| RANDID | TIMEDTH | DEATH | SEX | BMI | AGE_group |
| --- | --- | --- | --- | --- | --- |
| 2448 | 8766 | 0 | 0 | 26.97 | 1 |
| 6238 | 8766 | 0 | 1 | 28.73 | 1 |
| 10552 | 2956 | 1 | 1 | 28.58 | 2 |
| 11252 | 8766 | 0 | 1 | 23.10 | 1 |

### 5.3 输入与产出

#### 输入

- 输入数据：事件时间、事件指示（1 事件 / 0 删失）、可选分组变量。
- 关键变量：`time`、`event`、`group`。
- 需要预处理的内容：确认删失编码方向、时间单位、随访起点定义一致。

#### 产出

- 模型对象/统计结果：阶梯状生存曲线、风险集人数表。
- 参数估计：非参数，无回归系数。
- 预测结果：任意时点生存率、中位生存时间及其区间。
- 不确定性指标：Greenwood 标准误、log-log 置信区间/置信带。

## 6. 适用场景

- 适合：描述性生存分析、试验报告的主图、组间生存的初步比较（配 log-rank）。
- 不适合：需要调整多个协变量（用 Cox）；存在竞争风险（用 CIF）；想要平滑的风险率曲线。
- 使用前需要特别检查的点：删失比例与模式、事件编码、时间零点（入组？确诊？手术？）是否统一。

## 7. 实现

### 7.1 Python

常用包：

- `lifelines`

```python
from lifelines import KaplanMeierFitter

km = KaplanMeierFitter()
km.fit(durations=df["TIMEDTH"], event_observed=df["DEATH"])
km.plot_survival_function()            # 阶梯曲线 + 置信带
print(km.median_survival_time_)        # 中位生存时间
print(km.survival_function_at_times([1825, 3650]))  # 5 年 / 10 年生存率

# 分组曲线:按性别分别 fit 后画在同一坐标轴
for sex, sub in df.groupby("SEX"):
    KaplanMeierFitter().fit(sub["TIMEDTH"], sub["DEATH"],
                            label=f"SEX={sex}").plot_survival_function()
```

### 7.2 R

常用包：

- `survival`

```r
library(survival)

fit <- survfit(Surv(TIMEDTH, DEATH) ~ sex_label, data = df,
               conf.type = "log-log")   # log-log 区间保证落在 [0,1]
summary(fit, times = c(1825, 3650))     # 5 年 / 10 年生存率
print(fit)                              # 各组中位生存时间
plot(fit, mark.time = TRUE)             # mark.time 标出删失点
# 更美观:survminer::ggsurvplot(fit, risk.table = TRUE)
```

## 8. 结果如何解读

- 核心结果看什么：曲线整体形状与下降速度、特定时点生存率、中位生存时间、组间曲线是否分离及何时分离。
- 每个主要参数如何解读：$\hat S(1825)=0.80$ 读作「估计 80% 的个体在 5 年时尚未发生事件」；中位生存 8 年读作「估计一半个体在 8 年内发生事件」。
- 临床或医学意义如何表达：报告「x 年生存率 xx%（95% CI …）」与中位生存时间；曲线须配各时点风险集人数表（number at risk）。
- 常见误读：删失记号不是事件；曲线是**生存概率**不是危险率；尾部只剩几个人时的骤降常是假象；两组曲线分离不等于因果差异（未调整混杂）。

## 9. 假设诊断与稳健性

- 非信息性删失无法用数据直接检验：比较删失者与继续随访者的基线特征、按删失原因做敏感性分析（极端情形：删失者全当事件/全当长期存活，看结论翻不翻转）。
- 尾部不稳定：报告曲线时标注风险集人数，风险集低于 10–15 人的区段慎读；必要时截断展示时间轴。
- 竞争风险检查：若「其他原因死亡」占比不小，用 1−KM 会高估目标事件累积发生率，应改用 Aalen-Johansen/CIF。
- 组间比较不要只靠肉眼：配 [[Log-rank检验（Log-rank Test）]] 或 Cox 模型。

## 10. 推荐可视化

- 分组 KM 曲线 + 删失记号 + 风险集人数表：一图看全生存差异与数据支撑。
- 单组 KM 曲线 + 置信带：报告某人群的生存概况。
- KM 与拟合的参数模型曲线叠加：检查参数分布（如 Weibull）拟合是否贴合。

### 10.1 图像示例

下图展示按性别分层后的 Kaplan-Meier 生存曲线，可直接用于比较不同组的整体生存模式。

![](../../04_示例图像/km_survival_by_sex.png)

## 11. 优势、局限与常见坑

### 优势

- 非参数：不需要假设生存时间的分布形状。
- 正确、充分地利用右删失信息。
- 直观、标准，是几乎所有生存分析报告的入口图。

### 局限

- 不能调整协变量，只能按组分层，组一多每组样本就薄。
- 只给概率曲线，不直接给效应量（组间差异要靠 log-rank/Cox 补充）。
- 尾部风险集小时估计极不稳定。

### 常见坑

- 删失编码写反（0/1 颠倒），曲线含义完全错误。
- 存在竞争风险仍用 1−KM 报「发生率」，系统性高估。
- 对着尾部只剩 2 人的曲线末端解读「长期生存率」。
- 不同组随访起点不一致（永生时间偏倚，immortal time bias）。

## 12. 与相近方法的区别

- 和 [[Nelson-Aalen累积风险估计（Nelson-Aalen Estimator）]]：KM 连乘估计生存概率 $S(t)$，Nelson-Aalen 累加估计累积风险 $H(t)$；二者经 $S=\exp(-H)$ 渐近互通。
- 和 [[Log-rank检验（Log-rank Test）]]：KM 负责「画出来」，log-rank 负责「组间差异显不显著」。
- 和 [[Cox比例风险模型（Cox Proportional Hazards Model）]]：Cox 调整协变量、给出风险比；KM 无协变量、给出概率曲线。
- 和寿命表法：寿命表按固定区间汇总（适合大规模登记数据），KM 按精确事件时点更新。
- 如何选择：**先画 KM 看形状，再用 log-rank 检验组间差异，需要调整混杂时上 Cox**——这是生存分析的标准三步曲。

## 13. 医学研究中的典型应用

- 肿瘤临床试验主分析图：两臂 OS/PFS 的 KM 曲线 + 风险集表 + log-rank p 值。
- 队列研究描述不同暴露组的长期死亡/发病概率。
- 器械研究中植入物的无失效生存曲线。

## 14. 关键术语

- **生存函数（Survival function）**：$S(t)=P(T\gt t)$，事件时间超过 $t$ 的概率。
- **右删失（Right censoring）**：只知道事件时间大于某时刻（失访、研究结束仍未发生事件）。
- **风险集（Risk set）**：某时点尚未发生事件也未被删失、仍「有资格出事」的个体集合。
- **乘积极限估计（Product-limit estimator）**：KM 的学名，由各事件时点条件存活率连乘而得。
- **中位生存时间（Median survival time）**：生存曲线首次降到 0.5 及以下的时刻。
- **Greenwood 公式（Greenwood's formula）**：KM 估计方差的经典公式。
- **非信息性删失（Non-informative censoring）**：删失机制与预后无关，是 KM 无偏的关键前提。
- **风险集人数表（Number at risk table）**：曲线下方标注各时点在场人数的表，用来判断曲线哪段可信。

## 15. 相关方法

- [[Nelson-Aalen累积风险估计（Nelson-Aalen Estimator）]]
- [[Log-rank检验（Log-rank Test）]]
- [[Cox比例风险模型（Cox Proportional Hazards Model）]]
- [[竞争风险模型（Competing Risks and Fine-Gray Model）]]
- [[经验分布函数（Empirical Cumulative Distribution Function, ECDF）]]

## 16. 参考资料

- Kaplan EL, Meier P. Nonparametric estimation from incomplete observations. *J Am Stat Assoc*. 1958;53(282):457-481.
- Klein JP, Moeschberger ML. *Survival Analysis: Techniques for Censored and Truncated Data*. 2nd ed. Springer; 2003.
- Bland JM, Altman DG. Survival probabilities (the Kaplan-Meier method). *BMJ*. 1998;317(7172):1572.
- R Core Team / survival package. `survfit`. R Manual. [https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survfit.html](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survfit.html) （访问日期：2026-07-02）
