---
title: 加速失效时间模型
english_name: Accelerated Failure Time, AFT, Model
slug: accelerated-failure-time-aft-model
aliases: [AFT, survreg model, "加速失效时间模型（Accelerated Failure Time, AFT, Model）"]
category: 生存分析与纵向数据
subcategory: 参数生存回归
tags: [医学统计, 数据科学, 生存分析, AFT]
status: 已建
difficulty: intermediate
question_type: 生存时间回归建模
data_type: [生存数据]
outcome_type: [时间到事件]
python_packages: [lifelines]
r_packages: [survival]
---

# 加速失效时间模型（Accelerated Failure Time, AFT, Model）

## 1. 方法概览

### 1.1 一句话本质

AFT 不动风险率、直接对**生存时间本身**建模：$\log T=X^\top\beta+\epsilon$，协变量的作用是把每个人的「生命时钟」整体拨快或拨慢——系数指数化就是**时间比（time ratio）**，回答「事件时间被拉长/压缩了几倍」。

### 1.2 定义

加速失效时间模型是参数生存回归：假设 $\log(\text{生存时间})$ 等于协变量线性组合加一个服从特定分布的误差项。常用误差分布对应 Weibull、log-normal、log-logistic 等生存时间分布。它与 [[Cox比例风险模型（Cox Proportional Hazards Model）]] 是生存回归的两大范式，只是解释尺度不同（时间 vs 风险）。

### 1.3 它主要解决什么问题

- 研究问题：某因素让事件**更早**还是**更晚**发生？把生存时间延长/缩短了多少比例？
- 适用任务：参数生存回归、时间比解释、需要外推绝对生存时间/分位数时的建模。
- 常见医学场景：治疗是否延长中位生存时间、危险因素是否加速无事件时间的到来、工程/器械的寿命加速试验。

### 1.4 直觉与类比

想象每个人有一块「衰老时钟」，事件在时钟走到某刻度时发生。AFT 说：协变量的作用是**调节时钟的走速**。时间比 TR=1.5 意味着治疗组的时钟走得慢 1.5 倍——同样的衰老进程，他们要多花 50% 的现实时间才走到，于是各个分位数的生存时间都乘以 1.5。这和汽车「里程加速试验」同理：想知道正常使用能撑多久，就让它高速运转（加速时钟），再把时间换算回去。TR\<1 则相反：时钟走得快，事件提前到来。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

Cox 模型极其成功，但它的产出 HR 是「风险率的倍数」，对很多人（尤其临床沟通、卫生经济）不够直观——病人想问的是「这个治疗能让我多活多久」，而不是「我的瞬时风险是别人的几倍」。而且 Cox 只用事件排序、不预测绝对生存时间。根本困难是：**如何得到一个直接讲『时间延长多少』、并能外推绝对生存时间的生存回归？**

### 2.2 关键洞察

把回归直接架在 $\log T$ 上，就像普通线性回归架在结果变量上一样——只不过误差项换成能配合删失的生存分布，用（含删失的）最大似然估计。这样系数天然活在「时间尺度」：$e^{\beta_j}$ 直接是「$X_j$ 增 1 单位使生存时间乘的倍数」。关键洞察是**换一个建模尺度**（时间而非风险），把生存问题转化为一个「带删失的对数线性回归」，从而得到时间比这种更贴近生活的效应量，并顺带获得完整的参数生存分布用于预测。

### 2.3 与朴素/相邻做法的对比

- 相对 [[Cox比例风险模型（Cox Proportional Hazards Model）]]：AFT 参数化、在时间尺度解释、能外推绝对生存；Cox 半参数、更稳健、无需选分布。分布选对时 AFT 更有效率，选错则有偏。
- 相对**直接对 $\log T$ 做 OLS**：删失数据里 $T$ 常观测不全，OLS 会有偏；AFT 用删失似然正确处理。
- 相对 [[Kaplan-Meier生存曲线（Kaplan-Meier Estimator）]]：KM 非参数、无协变量；AFT 是参数回归。
- 特例关系：**Weibull（含指数）分布同时满足 AFT 和 PH**，是唯一常用的「两栖」分布；log-normal、log-logistic 只是 AFT、不是 PH。

## 3. 数学形式

### 3.1 核心公式

$$
\log T = X^\top\beta + \sigma\,\epsilon
\quad\Longleftrightarrow\quad
T = e^{X^\top\beta}\cdot e^{\sigma\epsilon}
$$

这个式子在说：对数生存时间是协变量的线性函数加随机扰动；等价地，生存时间等于一个「由协变量决定的时间缩放因子 $e^{X^\top\beta}$」乘上一个随机基线时间。协变量 $X_j$ 增 1 单位，把整条时间轴乘以 $e^{\beta_j}$（时间比 TR）。

### 3.2 推导脉络

1. 设 $\log T=X^\top\beta+\sigma\epsilon$，$\epsilon$ 服从标准分布（极值分布→Weibull、正态→log-normal、logistic→log-logistic）。
2. 加速性质：生存函数满足 $S(t\mid X)=S_0\!\bigl(t\cdot e^{-X^\top\beta}\bigr)$——协变量把基线生存函数在**时间轴上伸缩**，故所有分位数生存时间同比缩放。
3. 参数用删失最大似然估计：事件贡献密度 $f(t)$，删失贡献生存 $S(t)$，最大化 $\prod f(t_i)^{\delta_i}S(t_i)^{1-\delta_i}$。
4. Weibull 的两栖性：若 $\epsilon$ 为极值分布、shape $=1/\sigma$，则同一模型既可写成 AFT（时间比 $e^{\beta}$），又可写成 PH（风险比 $HR=e^{-\beta/\sigma}$），二者换算 $\log HR=-\log TR/\sigma$。

### 3.3 参数与统计量含义

- $\beta_j$：$X_j$ 在 log 生存时间尺度的效应。
- $\exp(\beta_j)=\text{TR}$：时间比。TR\>1 延长事件时间（保护），TR\<1 缩短（有害），TR=1 无关。
- $\sigma$（scale）：误差尺度；决定分布形状（Weibull 的 shape $=1/\sigma$：$\sigma\lt1$ 风险递增，$\sigma\gt1$ 递减，$\sigma=1$ 恒定即指数）。
- 分布选择：Weibull / log-normal / log-logistic，用 AIC 与残差诊断挑选。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 分布设定正确 | 生存时间近似所选分布 | 系数与预测有偏 | AIC 比较、QQ 图、KM 与拟合曲线叠合 |
| 加速性（时间伸缩） | 协变量效应是时间尺度的常数缩放 | TR 解释失真 | 分组 $\log$ 时间的 QQ 图是否平行 |
| 非信息性删失 | 删失与预后无关 | 估计有偏 | 比较删失者与在随访者特征 |
| log-线性 | log T 与连续协变量线性 | 效应形状扭曲 | 残差图、加样条 |

## 4. 手把手算例

难以像 Cox 那样只用几行数据手推最大似然，这里用一个**能手算的换算与解释例子**（Weibull/指数 AFT），把「时间比」的含义算清楚。

**设定：** 单个二分类协变量 $x$（1=治疗，0=对照）。数据近似**指数分布**（Weibull 的 $\sigma=1$，shape $p=1$，风险恒定）。观察到：对照组中位生存 $m_0=10$ 个月，治疗组中位生存 $m_1=15$ 个月。

**Step 1：算时间比。** 指数分布下中位数 $=\text{scale}\times\ln 2$，而 scale $=e^{\beta_0+\beta_1 x}$。故

$$
\text{TR}=\frac{m_1}{m_0}=\frac{15}{10}=1.5,\qquad \beta_1=\log(1.5)=0.405
$$

治疗组的**所有**生存分位数都乘 1.5：25% 分位、中位、90% 分位一律延长 50%——这就是「加速性」的含义（整条时间轴伸缩，而非某一点）。

**Step 2：换算成 HR，与 Cox 对照。** 指数/Weibull 两栖，$\log HR=-\beta_1/\sigma$：

$$
HR=\text{TR}^{-1/\sigma}=1.5^{-1/1}=\frac{1}{1.5}=0.667
$$

读作：治疗把生存时间延长到 1.5 倍，**等价于**把死亡风险率降到 0.667 倍。若换一个 shape（$\sigma=0.5$，风险递增），同样 TR=1.5 会对应 $HR=1.5^{-2}=0.444$——**同一个时间比，风险比取决于形状参数**，这正是 TR 与 HR 不可混为一谈的原因。

**Step 3：读结论。** TR=1.5（95% CI 若不含 1 则显著）读作「治疗预计把生存时间延长约 50%」；若面向临床沟通，这比「HR=0.67」更贴近病人关心的「能多活多久」。

**结论：** AFT 把治疗效应说成「时间轴拉长 1.5 倍」，指数/Weibull 情形下可无缝换算成 HR=0.67；换算依赖形状参数 $\sigma$，非 Weibull 分布（log-normal/log-logistic）则根本没有恒定 HR，只能用 TR 解释。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：连续、二分类、多分类均可。
- 因变量类型：时间到事件（time + event）。
- 数据结构：每行一个个体。
- 是否适合高维数据：非默认首选；高维需正则化。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：适合右删失，并天然支持**区间删失/左删失**（AFT 相对 Cox 的一个优势）。
- 是否适合重复测量数据：时间依赖协变量不易纳入（不如 Cox 灵活）。

### 5.2 示例表格

Framingham 风格生存数据：

| RANDID | TIMEDTH | DEATH | SEX | BMI | AGE_group |
| --- | --- | --- | --- | --- | --- |
| 2448 | 8766 | 0 | 0 | 26.97 | 1 |
| 10552 | 2956 | 1 | 1 | 28.58 | 2 |
| 11252 | 8766 | 0 | 1 | 23.10 | 1 |

一个 Weibull AFT 模型的时间比可整理为：

| 变量 | Time Ratio | 95% CI |
| --- | --- | --- |
| BMI（每 +1） | 0.988 | 0.981 – 0.995 |
| Female vs Male | 1.350 | 1.269 – 1.435 |
| AGE_group 2 vs 1 | 0.513 | 0.479 – 0.550 |
| AGE_group 3 vs 1 | 0.308 | 0.271 – 0.351 |

（TR\<1 的 BMI/高龄组缩短生存时间，TR\>1 的女性延长生存时间——与 Cox 卡里 HR 的方向恰好相反：延长时间⇔降低风险。）

### 5.3 输入与产出

#### 输入

- 输入数据：时间、事件指示、协变量。
- 关键变量：`time`、`event`、分布假设、scale 参数。
- 需要预处理的内容：删失编码、时间单位、分布选择（AIC/残差）。

#### 产出

- 模型对象/统计结果：系数、时间比、分布参数（shape/scale）、AIC。
- 参数估计：$\beta$ 与 TR。
- 预测结果：条件生存曲线、任意分位数生存时间、中位生存时间预测。
- 不确定性指标：标准误、TR 置信区间。

## 6. 适用场景

- 适合：希望以「延长/缩短多少时间」解释效应；需要外推绝对生存时间或分位数；存在区间删失。
- 不适合：不愿承担分布假设；有大量时间依赖协变量；只需相对风险（Cox 更省事）。
- 使用前需要特别检查的点：分布拟合优度、加速性、删失机制、与 PH 解释的差异。

## 7. 实现

### 7.1 Python

常用包：

- `lifelines`

```python
from lifelines import WeibullAFTFitter, LogNormalAFTFitter

aft = WeibullAFTFitter()
aft.fit(df, duration_col="TIMEDTH", event_col="DEATH",
        formula="BMI + C(SEX) + C(AGE_group)")
aft.print_summary()                       # 时间比 = exp(coef)
print(aft.median_survival_times(df.head()))   # 预测个体中位生存时间

# 选分布:比 AIC
ln = LogNormalAFTFitter().fit(df, "TIMEDTH", "DEATH")
print(aft.AIC_, ln.AIC_)
```

### 7.2 R

常用包：

- `survival`

```r
library(survival)

fit <- survreg(Surv(TIMEDTH, DEATH) ~ BMI + sex_label + factor(AGE_group),
               data = df, dist = "weibull")
summary(fit)
exp(coef(fit))            # 时间比 time ratios
1 / fit$scale             # Weibull shape = 1/sigma; 换算 HR = TR^(-shape)
predict(fit, type = "quantile", p = 0.5)   # 预测中位生存时间
```

## 8. 结果如何解读

- 核心结果看什么：TR 的方向与大小、95% CI 是否含 1、分布拟合（AIC/QQ）。
- 每个主要参数如何解读：TR=1.35（女 vs 男）读作「女性事件时间约延长 35%」；TR=0.31（高龄）读作「高龄组事件时间压缩到约 31%」。
- 临床或医学意义如何表达：TR 直接回答「延长/缩短多久」，可再由模型算出各组中位生存时间做绝对对比。
- 常见误读：把 TR 当 HR（方向相反、数值不等）；不检查分布就信预测；只看系数显著忽略拟合质量。

## 9. 假设诊断与稳健性

- 分布选择：多分布拟合比 AIC；画 KM 曲线与参数拟合曲线叠合看贴合度；残差（Cox-Snell）应近似标准指数。
- 加速性检查：分组画 $\log t$ 的分位数-分位数图，平行支持时间尺度的常数缩放。
- Weibull 双重解释：若既想要 TR 又想要 HR，只有 Weibull/指数能同时给（$HR=TR^{-1/\sigma}$）；换 log-normal/log-logistic 则无恒定 HR。
- 稳健性：小样本或删失重时分布参数不稳；比较不同分布下结论是否一致。

## 10. 推荐可视化

- 时间比森林图：各协变量对生存时间的伸缩效应。
- 参数拟合曲线 vs KM 曲线叠合图：直观检验分布假设。
- 分组预测生存曲线：把 TR 翻译成绝对生存差。

### 10.1 图像示例

下图用区间图展示 Weibull AFT 模型中各协变量对应的时间比及其 95% 区间。

![](../../04_示例图像/aft_time_ratio_forest.png)

## 11. 优势、局限与常见坑

### 优势

- 时间尺度解释直观，直接回答「延长/缩短多久」。
- 参数化，能外推绝对生存时间与任意分位数。
- 天然支持区间删失/左删失。

### 局限

- 需指定并依赖分布假设，选错则有偏。
- 时间依赖协变量不易纳入。
- 实践使用频率低于 Cox，读者熟悉度较低。

### 常见坑

- 把 time ratio 当 hazard ratio 解读（方向相反）。
- 不做分布拟合优度检查就报预测。
- 只因系数显著而忽略模型对数据的贴合程度。
- 忘记 Weibull 才两栖，对 log-normal 硬求 HR。

## 12. 与相近方法的区别

- 和 [[Cox比例风险模型（Cox Proportional Hazards Model）]]：AFT 时间尺度、参数化、给 TR、能外推绝对时间；Cox 风险尺度、半参数、给 HR、更稳健。想讲「多活多久」用 AFT，想讲「风险几倍」且不愿设分布用 Cox。
- 和 [[Kaplan-Meier生存曲线（Kaplan-Meier Estimator）]]：KM 非参数描述、无协变量；AFT 参数回归。
- 和 Weibull PH 参数模型：同一 Weibull 拟合既可作 AFT 也可作 PH，报告哪种取决于想要 TR 还是 HR。
- 如何选择：**分布能拟合好、需绝对时间预测或区间删失→AFT；不愿设分布、要相对风险、有时间依赖协变量→Cox**。

## 13. 医学研究中的典型应用

- 评估治疗是否延长中位生存/无进展生存时间（以 TR 报告）。
- 加速寿命/退化试验中危险因素对失效时间的加速效应。
- 作为 Cox 分析的参数化敏感性分析与绝对生存时间预测补充。

## 14. 关键术语

- **时间比（Time ratio, TR）**：$e^{\beta}$，协变量每增 1 单位生存时间乘的倍数，AFT 的核心产出。
- **加速失效时间（Accelerated failure time）**：协变量把基线生存函数在时间轴上按常数比例伸缩的模型结构。
- **shape / scale 参数**：Weibull 分布的形状与尺度；shape $=1/\sigma$ 决定风险随时间递增/递减。
- **两栖分布（Weibull/指数）**：唯一常用的同时满足 AFT 与 PH 的分布族，$HR=TR^{-1/\sigma}$。
- **区间删失（Interval censoring）**：只知事件发生在某时间段内，AFT 相对 Cox 更易处理。
- **Cox-Snell 残差**：参数生存模型的拟合诊断残差，正确时近似标准指数分布。

## 15. 相关方法

- [[Cox比例风险模型（Cox Proportional Hazards Model）]]
- [[Kaplan-Meier生存曲线（Kaplan-Meier Estimator）]]
- [[竞争风险模型（Competing Risks and Fine-Gray Model）]]
- [[线性回归（Linear Regression）]]

## 16. 参考资料

- Kalbfleisch JD, Prentice RL. *The Statistical Analysis of Failure Time Data*. 2nd ed. Wiley; 2002.
- Klein JP, Moeschberger ML. *Survival Analysis: Techniques for Censored and Truncated Data*. 2nd ed. Springer; 2003.
- Collett D. *Modelling Survival Data in Medical Research*. 3rd ed. Chapman & Hall/CRC; 2015.
- R Core Team / survival package. `survreg`. R Manual. [https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html) （访问日期：2026-07-02）
