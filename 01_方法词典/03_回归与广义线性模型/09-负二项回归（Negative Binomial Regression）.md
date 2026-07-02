---
title: 负二项回归
english_name: Negative Binomial Regression
slug: negative-binomial-regression
aliases: [negative binomial regression, NB回归, 负二项模型, "负二项回归（Negative Binomial Regression）"]
category: 回归与广义线性模型
subcategory: 计数结局建模
tags: [医学统计, 数据科学, 计数数据, 过度离散, GLM]
status: 已建
difficulty: intermediate
question_type: 过度离散计数结局建模
data_type: [表格数据]
outcome_type: [计数型]
python_packages: [statsmodels]
r_packages: [MASS]
---

# 负二项回归（Negative Binomial Regression）

## 1. 方法概览

### 1.1 一句话本质

负二项回归是 Poisson 回归的「加宽版」：它多带一个离散参数，允许方差**大于**均值，从而正确处理真实计数里普遍存在的「过度离散」。

### 1.2 定义

负二项回归用于过度离散的计数结局。它可看作「泊松率本身在个体间随机波动（服从 Gamma 分布）」后混合出的模型，方差为 $\mu+\alpha\mu^2$，比泊松多出 $\alpha\mu^2$ 来吸收超额波动。

### 1.3 它主要解决什么问题

- 研究问题：计数结局的方差明显超过均值时，如何得到可靠的率比与标准误？
- 适用任务：过度离散计数/率的关联分析。
- 常见医学场景：就诊/住院次数（少数人反复就诊）、癫痫发作次数、聚集性事件计数。

### 1.4 直觉与类比

Poisson 假设「人人事件率相同」，但现实是**有人天生高发、有人几乎不发**（未观测的个体差异）。这种隐藏的异质性让计数比泊松预期更「散」——大部分人 0–1 次，少数人十几次。负二项相当于承认「每个人的基础率不一样」，先让率在人群里按 Gamma 分布抖动，再叠加泊松计数，于是能容纳这种额外的分散。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

真实计数几乎总是过度离散（方差 ≫ 均值），源于未观测异质性或事件聚集。Poisson 强行假设方差=均值，会**严重低估标准误**、把不显著当显著。根本困难是：**如何在保留「计数、率比」框架的同时，让模型容纳超额的方差？**

### 2.2 关键洞察

把泊松的固定率 $\mu$ 换成一个**随机**的率：设个体率 $\lambda\sim\text{Gamma}$，再 $Y\mid\lambda\sim\text{Poisson}(\lambda)$。对 $\lambda$ 积分（混合）后，$Y$ 的边际分布正是**负二项**。这个「Gamma-Poisson 混合」使方差变成 $\mu+\alpha\mu^2$：多出的 $\alpha\mu^2$ 恰好度量并吸收了个体间率的差异。离散参数 $\alpha\to0$ 时退回泊松——所以负二项把泊松作为特例包含在内。

### 2.3 与朴素/相邻做法的对比

- 相对 **Poisson**：负二项显式建模过度离散，SE 更诚实；泊松在过度离散下过度自信。
- 相对 **quasi-Poisson**：quasi 只把方差设为 $\phi\mu$（线性）、不给完整分布；负二项方差是 $\mu+\alpha\mu^2$（二次）、是完整似然模型，可做 AIC 比较。
- 相对**零膨胀模型**：过多零的根源是「结构零」时用零膨胀；负二项处理的是整体过度分散。

## 3. 数学形式

### 3.1 核心公式

$$
\log(\mu_i)=X_i^\top\boldsymbol{\beta},\qquad
\mathrm{Var}(Y_i)=\mu_i+\alpha\,\mu_i^2
$$

均值结构与 Poisson 相同（log 连接、系数指数化为率比），差别在方差多了 $\alpha\mu^2$ 项。这个式子在说：平均计数照旧用 log 线性建模，但允许方差按 $\mu$ 的平方额外膨胀。

### 3.2 推导脉络

- 分层表述：$Y\mid\lambda\sim\text{Poisson}(\lambda)$，$\lambda\sim\text{Gamma}$（均值 $\mu$、形状与 $\alpha$ 相关）。
- 对 $\lambda$ 积分得边际负二项分布，方差 $\mu+\alpha\mu^2$。
- 用最大似然同时估计 $\boldsymbol{\beta}$ 与离散参数 $\alpha$（$\alpha$ 通常需迭代/交替估计）。
- 与泊松嵌套：$H_0:\alpha=0$ 可用似然比检验判断是否需要负二项。

### 3.3 参数与统计量含义

- $\mu_i$：期望计数；$\beta_j$：log 尺度效应，$\exp(\beta_j)$ = 率比。
- $\alpha$（离散参数）：越大过度离散越强，$\alpha=0$ 即泊松。
- 方差 $\mu+\alpha\mu^2$：均值小时接近泊松，均值大时额外膨胀明显。
- AIC/LRT：与泊松比较。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 均值 log 线性 | log 率与 X 线性 | 效应有偏 | 残差图 |
| 方差二次形式 | Var=μ+αμ² | 方差结构错则 SE 偏 | 均值-方差图 |
| 独立 | 观测独立 | SE 失真 | 看设计；聚集用 GLMM |
| 过度离散源自异质 | 非结构零 | 若是零膨胀应换模型 | 零的比例 |

## 4. 手把手算例

某计数结局观测均值 $\mu=5$，但样本方差高达 $20$（远大于 5，提示过度离散）。

**一步步计算（用负二项方差式解出 $\alpha$）：**

- 若用 Poisson，隐含方差 = 均值 = 5，却与观测方差 20 严重不符——SE 会被低估约 $\sqrt{20/5}=2$ 倍。
- 用负二项：令 $\mu+\alpha\mu^2=20$，即 $5+\alpha\times5^2=20\Rightarrow25\alpha=15\Rightarrow\alpha=0.6$。
- 于是方差 $=5+0.6\times25=20$，与观测吻合；$\alpha=0.6$ 定量刻画了过度离散的强度。

**结论：** 离散参数 $\alpha=0.6$ 把「多出来的方差」显式建模进来，标准误因此被正确放大约 2 倍——原本在 Poisson 下「显著」的效应，改用负二项后 p 值会更大、结论更稳健。率比的**点估计**与 Poisson 接近（均值结构相同），但**推断**（SE、CI、p）才是关键差别。实务中先看 Poisson 的过度离散诊断，再决定是否升级到负二项。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：连续、分类均可。
- 因变量类型：过度离散计数。
- 数据结构：每行一观测 + 计数（+ 暴露）。
- 是否适合高维数据：可结合正则化。
- 是否适合缺失较多数据：需先处理。
- 是否适合删失数据：率模型经 offset 部分处理。
- 是否适合重复测量数据：不适合；用负二项 GLMM/GEE。

### 5.2 示例表格

| 统计量 | 值 | 含义 |
| --- | --- | --- |
| 均值 μ | 5 | 平均计数 |
| 方差 | 20 | 远大于均值 |
| α | 0.6 | 过度离散强度 |

### 5.3 输入与产出

#### 输入

- 输入数据：计数结局 + 协变量（+ offset）。
- 关键变量：计数、协变量、暴露。
- 需要预处理的内容：先诊断过度离散、构造 offset。

#### 产出

- 模型对象/统计结果：系数、SE、$\alpha$、AIC。
- 参数估计：率比 IRR 与离散参数。
- 预测结果：期望计数/率。
- 不确定性指标：IRR 的（更诚实的）置信区间。

## 6. 适用场景

- 适合：过度离散的计数/率结局。
- 不适合：等离散（用 Poisson 更简）、零膨胀（用 ZINB）、连续/二分类。
- 使用前需要特别检查的点：过度离散是否存在、是否零膨胀、offset。

## 7. 实现

### 7.1 Python

常用包：

- `statsmodels`

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

fit = smf.glm("count ~ x", data=df,
              family=sm.families.NegativeBinomial(alpha=0.6)).fit()
# 或用 discrete.NegativeBinomial 同时估计 alpha
print(np.exp(fit.params))       # 率比 IRR
```

### 7.2 R

常用包：

- `MASS`

```r
library(MASS)
fit <- glm.nb(count ~ x, data = df)    # 自动估计 theta(=1/alpha)
exp(cbind(IRR = coef(fit), confint(fit)))
```

## 8. 结果如何解读

- 核心结果看什么：率比 IRR 及区间、离散参数 $\alpha$、与 Poisson 的 AIC 对比。
- 每个主要参数如何解读：$\alpha$ 显著大于 0 说明确实过度离散、该用负二项。
- 临床或医学意义如何表达：报告 IRR 与区间，说明已校正过度离散。
- 常见误读：只看点估计忽略 SE 的改善；把零膨胀问题当一般过度离散。

## 9. 假设诊断与稳健性

- 是否需要：先看 Poisson 的偏差/df；再用 LRT（$\alpha=0$）确认负二项的必要性。
- 零膨胀 vs 过度离散：零特别多且有「结构零」机制时改用零膨胀负二项。
- 均值-方差检查：残差图看方差是否呈二次膨胀。
- 相关数据：用负二项 GLMM/GEE。

## 10. 推荐可视化

- 观测 vs 预测计数分布（对比 Poisson/NB 拟合）。
- 均值-方差关系图。
- IRR 森林图。

## 11. 优势、局限与常见坑

### 优势

- 正确处理过度离散，SE 诚实。
- 完整似然，可 AIC 比较、含泊松为特例。
- 系数仍解释为率比。

### 局限

- 假设特定的二次方差形式。
- 不解决零膨胀。
- 不处理相关数据。

### 常见坑

- 把零膨胀误当一般过度离散。
- 不与 Poisson 比较就默认用 NB。
- 忽略 offset。

## 12. 与相近方法的区别

- 和 **Poisson**：NB 加离散参数、容纳过度离散；泊松是 $\alpha=0$ 的特例。
- 和 **quasi-Poisson**：quasi 方差 $\phi\mu$（线性、非似然）；NB 方差 $\mu+\alpha\mu^2$（二次、完整似然）。
- 和**零膨胀（ZINB）**：结构零过多时用 ZINB。
- 如何选择：过度离散且无结构零 → NB；只想放大 SE → quasi；零特别多 → 零膨胀。

## 13. 医学研究中的典型应用

- 就诊/住院次数（少数高频者导致过度离散）。
- 癫痫发作次数等聚集性事件。
- 每单位随访时间的过度离散事件率。

## 14. 关键术语

- **过度离散（Overdispersion）**：方差大于均值。
- **离散参数（$\alpha$）**：度量过度离散强度，0 即泊松。
- **Gamma-Poisson 混合**：负二项的生成机制。
- **未观测异质性**：个体基础率差异导致的额外分散。
- **率比（IRR）**：$\exp(\beta)$。

## 15. 相关方法

- [[Poisson回归（Poisson Regression）]]
- [[Quasi-Likelihood与过度离散（Quasi-Likelihood and Overdispersion）]]
- [[广义线性模型（Generalized Linear Model, GLM）]]

## 16. 参考资料

- Cameron AC, Trivedi PK. *Regression Analysis of Count Data*. 2nd ed. Cambridge University Press; 2013.
- Hilbe JM. *Negative Binomial Regression*. 2nd ed. Cambridge University Press; 2011.
- Ver Hoef JM, Boveng PL. Quasi-Poisson vs. negative binomial regression. *Ecology*. 2007;88(11):2766-2772.
