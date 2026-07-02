---
title: 置信区间
english_name: Confidence Interval
slug: confidence-interval
aliases: [confidence interval, CI, 可信区间, "置信区间（Confidence Interval）"]
category: 描述统计与统计推断
subcategory: 区间估计
tags: [医学统计, 数据科学, 统计推断, 区间估计, 不确定性]
status: 已建
difficulty: basic
question_type: 参数区间估计与不确定性表达
data_type: [表格数据]
outcome_type: [连续型, 二分类]
python_packages: [scipy, statsmodels]
r_packages: [stats]
---

# 置信区间（Confidence Interval）

## 1. 方法概览

### 1.1 定义

置信区间是围绕点估计构造的一段取值范围，用来表达对未知参数的估计不确定性；在反复抽样意义下，一定比例（如 95%）的区间会覆盖真值。

### 1.2 它主要解决什么问题

- 研究问题：一个估计值有多可信，真值大概落在什么范围。
- 适用任务：均值/比例/风险比等参数的区间估计、效应量不确定性表达、等效/非劣判断。
- 常见医学场景：疗效差异的区间、患病率区间、OR/HR/RR 的 95% CI 报告。

### 1.3 直觉理解

点估计只给一个数字，置信区间告诉你这个数字有多“晃”。区间越窄越精确；关键在于理解它是关于“方法长期覆盖率”的陈述，而不是“真值有 95% 概率在此区间”的概率陈述。

## 2. 数学形式

### 2.1 核心公式

大样本下参数 $\theta$ 的 Wald 型区间：

$$
\hat{\theta}\pm z_{1-\alpha/2}\cdot \mathrm{SE}(\hat{\theta})
$$

正态均值（未知方差、小样本）用 $t$ 分布：

$$
\bar{x}\pm t_{1-\alpha/2,\,n-1}\cdot \frac{s}{\sqrt{n}}
$$

比例可用更稳健的 Wilson 区间替代 Wald。

### 2.2 参数或统计量含义

- $\hat{\theta}$：点估计；$\mathrm{SE}$：标准误。
- $z_{1-\alpha/2}$、$t_{1-\alpha/2,n-1}$：置信水平对应的临界值。
- $1-\alpha$：置信水平（如 0.95）。
- 区间宽度：随样本量增大而变窄。

### 2.3 关键假设

- 估计量的抽样分布近似（正态/ $t$），或用精确/自助法替代。
- 观测独立、模型设定合理。
- 对于比例/率等，边界附近 Wald 近似可能失效，需专用区间。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：视所估参数而定。
- 因变量类型：连续、二分类、计数、时间到事件均可（各有对应区间）。
- 数据结构：任意可估计参数的数据。
- 是否适合高维数据：单参数区间；多参数需联合置信域或多重校正。
- 是否适合缺失较多数据：需先处理缺失并传播不确定性。
- 是否适合删失数据：生存参数有专门区间（如 KM 的对数变换区间）。
- 是否适合重复测量数据：需用稳健/聚类标准误。

### 3.2 示例表格

| 参数 | 点估计 | 95% CI |
| --- | --- | --- |
| 均值差 | −8.0 mmHg | −11.2 ~ −4.8 |
| 比例 | 0.34 | 0.28 ~ 0.41 |
| 风险比 HR | 0.72 | 0.58 ~ 0.90 |

### 3.3 输入与产出

#### 输入

- 输入数据：样本数据或点估计 + 标准误。
- 关键变量：估计量、标准误、置信水平。
- 需要预处理的内容：选择合适的区间方法（Wald / t / Wilson / bootstrap）。

#### 产出

- 模型对象/统计结果：区间上下限。
- 参数估计：点估计。
- 预测结果：无（区间是推断，不是预测；预测区间是另一概念）。
- 不确定性指标：区间宽度本身即不确定性度量。

## 4. 适用场景

- 适合：几乎所有效应量的报告，替代或补充 p 值。
- 不适合：把区间当作真值的概率分布来解读、边界近似失效却仍用 Wald。
- 使用前需要特别检查的点：抽样分布近似是否成立、比例/率是否需要专用区间、是否需要多重比较调整。

## 5. 实现

### 5.1 Python

常用包：

- `scipy`
- `statsmodels`

```python
import numpy as np
from scipy import stats

x = np.random.normal(130, 12, 60)
m, se = x.mean(), stats.sem(x)
ci = stats.t.interval(0.95, len(x) - 1, loc=m, scale=se)
print(f"均值={m:.1f}, 95% CI={ci[0]:.1f} ~ {ci[1]:.1f}")

# 比例的 Wilson 区间
from statsmodels.stats.proportion import proportion_confint
print(proportion_confint(count=34, nobs=100, method="wilson"))
```

### 5.2 R

常用包：

- `stats`

```r
x <- rnorm(60, 130, 12)
t.test(x)$conf.int            # 均值 95% CI

prop.test(34, 100)$conf.int   # 比例区间(含连续性校正)
```

## 6. 结果如何解释

- 核心结果看什么：区间是否包含无效值（均差 0、比值 1）、区间宽度。
- 每个主要参数如何解释：不含无效值≈在该水平上显著；宽区间提示证据不足。
- 临床或医学意义如何表达：报告区间端点是否都具临床意义（如下限是否仍有效）。
- 常见误读：说“真值有 95% 概率落在此区间”（频率派下不成立）；把不显著等同于无效应。

## 7. 推荐可视化

- 森林图（多个效应量的点估计与区间）。
- 误差棒图。
- 区间随样本量变化的示意。

## 8. 优势、局限与常见坑

### 优势

- 同时传达效应方向、大小与精度，信息量高于 p 值。
- 支持等效/非劣性判断。
- 各类参数都有对应构造方法。

### 局限

- 依赖抽样分布近似或重抽样。
- 频率派解释常被误读为概率陈述。
- 边界参数（如比例接近 0/1）需专用方法。

### 常见坑

- 比例接近 0 或 1 仍用 Wald 区间。
- 把 95% CI 解读为真值概率。
- 多重比较下不调整区间宽度。

## 9. 与相近方法的区别

- 和 p 值/假设检验的区别：CI 提供效应大小与精度，检验只给是否显著；二者互补。
- 和 [[Bootstrap重抽样（Bootstrap Resampling）]] 的关系：分布近似难成立时，可用 bootstrap 构造区间。
- 和贝叶斯可信区间的区别：贝叶斯区间可直接作概率陈述，解释不同。

## 10. 医学研究中的典型应用

- 疗效差异、OR/RR/HR 的 95% CI 报告（期刊规范要求）。
- 患病率、敏感度/特异度的区间估计。
- 等效性与非劣效性试验的区间判定。

## 11. 相关方法

- [[单样本t检验（One-Sample t-Test）]]
- [[Bootstrap重抽样（Bootstrap Resampling）]]
- [[两独立样本t检验（Two-Sample t-Test）]]

## 12. 参考资料

- Altman DG, Machin D, Bryant TN, Gardner MJ. *Statistics with Confidence*. 2nd ed. BMJ Books; 2000.
- Brown LD, Cai TT, DasGupta A. Interval estimation for a binomial proportion. *Stat Sci*. 2001;16(2):101-133.
- Bland M. *An Introduction to Medical Statistics*. 4th ed. Oxford University Press; 2015.
