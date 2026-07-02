---
title: Poisson回归
english_name: Poisson Regression
slug: poisson-regression
aliases: [poisson regression, loglinear model, "Poisson回归（Poisson Regression）"]
category: 回归与广义线性模型
subcategory: 计数结局建模
tags: [医学统计, 数据科学, 计数数据, GLM, 率比]
status: 已建
difficulty: intermediate
question_type: 计数/率结局建模
data_type: [表格数据]
outcome_type: [计数型]
python_packages: [statsmodels]
r_packages: [stats]
---

# Poisson回归（Poisson Regression）

## 1. 方法概览

### 1.1 一句话本质

Poisson 回归对「事件发生的次数」建模：用 log 连接把「平均计数」连到协变量的线性组合，系数取指数后就是**率比**（某因素让事件发生率变为几倍）。

### 1.2 定义

Poisson 回归是结局为计数（非负整数）的 GLM，假设结局服从泊松分布、用对数连接。配合 offset 还能建模「率」（单位人时的事件数）。

### 1.3 它主要解决什么问题

- 研究问题：哪些因素影响某事件的发生次数或发生率？
- 适用任务：计数/率结局的关联分析与预测。
- 常见医学场景：一年内住院/发作次数、每千人年的发病率、单位随访时间内的事件数。

### 1.4 直觉与类比

计数数据有几个「怪脾气」：只能是 0、1、2…（非负整数）、通常右偏、且「事件多的时候波动也大」。直接用线性回归会预测出负的次数、还假设波动恒定，全错。Poisson 回归顺着计数的天性来：在 log 尺度上建线性模型，保证预测的平均次数永远为正，并自带「均值越大、方差越大」的特性。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

计数结局非负、离散、右偏，且方差随均值增大。线性回归的「可取任意实数、方差恒定、正态」假设全部不成立。根本困难是：**如何建模一个只能取非负整数、且波动随水平变化的结局？**

### 2.2 关键洞察

选泊松分布作随机成分（它天生描述「固定时间/空间里稀有事件的计数」），并用 **log 连接** $\log\mu=X\boldsymbol{\beta}$。log 连接一举两得：反连接 $\mu=e^{\eta}$ 保证均值恒正；且系数进入的是「乘法尺度」——$x$ 增 1 使率**乘以** $e^{\beta}$，正好符合「率比」这一流行病学核心量。泊松的「均值=方差」特性还内建了「计数越多越波动」。用 **offset**（把 $\log(\text{暴露时间})$ 作为系数固定为 1 的项）即可从「计数」切换到「率」。

### 2.3 与朴素/相邻做法的对比

- 相对**线性回归**：Poisson 不会预测负计数、方差结构正确。
- 相对**对计数取对数再线性回归**：Poisson 在均值的 log 尺度建模、能处理零、解释为率比。
- 相对**负二项/quasi**：当出现过度离散（方差远大于均值）时，Poisson 会低估 SE，需换负二项或 quasi-Poisson。

## 3. 数学形式

### 3.1 核心公式

$$
Y_i\sim\mathrm{Poisson}(\mu_i),\qquad
\log(\mu_i)=X_i^\top\boldsymbol{\beta}\ (+\ \log t_i)
$$

其中 $t_i$ 为暴露时间/人口（offset）。这个式子在说：结局是均值为 $\mu$ 的泊松计数；$\mu$ 的对数是协变量的线性组合；加 $\log t_i$ 就把「计数」变成「率」。

### 3.2 推导脉络

- 泊松分布 $P(Y=k)=\dfrac{\mu^k e^{-\mu}}{k!}$，均值与方差都等于 $\mu$（等离散）。
- 作为 GLM：随机成分=泊松、连接=log、$\mathrm{Var}(Y)=\mu$。
- 参数用最大似然（IRLS）估计。
- offset：率模型 $\log(\mu_i/t_i)=X\beta\Rightarrow\log\mu_i=X\beta+\log t_i$，$\log t_i$ 系数固定为 1。

### 3.3 参数与统计量含义

- $\mu_i$：第 $i$ 个观测的期望计数。
- $\beta_j$：log 尺度效应；$\exp(\beta_j)$ = **率比（IRR）**，$x_j$ 增 1 使率变为几倍。
- offset $\log t_i$：把计数标准化为率。
- 偏差/自由度：检查过度离散。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 等离散 | 方差=均值 | 过度离散→SE 过小、假显著 | 偏差/自由度、Pearson χ²/df |
| log 线性 | log 率与 X 线性 | 效应有偏 | 残差图 |
| 独立 | 事件/观测独立 | SE 失真 | 看设计；聚集用负二项/GEE |
| 无过多零 | 零不异常多 | 零膨胀→偏差 | 零的比例 |

## 4. 手把手算例

比较两组一年内的发作次数（率）：A 组 100 人年发生 30 次，B 组 100 人年发生 15 次。

**一步步计算：**

- 两组率：$r_A=30/100=0.30$/人年，$r_B=15/100=0.15$/人年。
- 率比：$\mathrm{IRR}=r_A/r_B=0.30/0.15=2.0$。
- 在 Poisson 模型 $\log(\mu)=\beta_0+\beta_1\cdot\text{组}$（B 为参照、含 offset $\log(\text{人年})$）中：
  - $\beta_0=\log(r_B)=\log(0.15)=-1.90$（参照组的对数率）。
  - $\beta_1=\log(\mathrm{IRR})=\log(2.0)=0.69$，故 $\exp(\beta_1)=2.0$，即 A 组事件率是 B 组的 2 倍。

**结论：** 系数指数化恰好还原出率比 2.0——**Poisson 回归把列率表写成了回归形式**，多变量时再推广为「调整其他因素后的率比」。注意这里用了 offset 把「计数」变「率」：若两组人年不同（如 A 组 50 人年 30 次），率会是 0.60、率比变 4.0，直接比「计数 30 vs 15」就会误导。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：连续、分类均可。
- 因变量类型：计数（非负整数）。
- 数据结构：每行一个观测 + 计数（+ 暴露量）。
- 是否适合高维数据：可结合正则化。
- 是否适合缺失较多数据：需先处理。
- 是否适合删失数据：率模型经 offset 部分处理随访差异。
- 是否适合重复测量数据：不适合；用 Poisson GLMM/GEE。

### 5.2 示例表格

| 组 | 事件数 | 人年 | 率 |
| --- | --- | --- | --- |
| A | 30 | 100 | 0.30 |
| B | 15 | 100 | 0.15 |

### 5.3 输入与产出

#### 输入

- 输入数据：计数结局 + 协变量（+ 暴露量）。
- 关键变量：计数、offset（暴露）、协变量。
- 需要预处理的内容：构造 offset、检查过度离散与零膨胀。

#### 产出

- 模型对象/统计结果：系数、SE、偏差、AIC。
- 参数估计：$\beta$ 与率比 IRR。
- 预测结果：期望计数/率。
- 不确定性指标：IRR 的置信区间。

## 6. 适用场景

- 适合：计数/率结局、等离散、事件相对稀有。
- 不适合：过度离散（用负二项/quasi）、零过多（用零膨胀）、连续/二分类结局。
- 使用前需要特别检查的点：过度离散、零膨胀、offset 是否需要。

## 7. 实现

### 7.1 Python

常用包：

- `statsmodels`

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

# events ~ group, 以 log(人年) 为 offset 建模率
fit = smf.glm("events ~ group", data=df,
              family=sm.families.Poisson(),
              offset=np.log(df["person_years"])).fit()
print(np.exp(fit.params))       # 率比 IRR
```

### 7.2 R

常用包：

- `stats`

```r
fit <- glm(events ~ group + offset(log(person_years)),
           family = poisson(link = "log"), data = df)
exp(cbind(IRR = coef(fit), confint(fit)))
```

## 8. 结果如何解读

- 核心结果看什么：率比 IRR 及区间、偏差/自由度（过度离散）。
- 每个主要参数如何解读：$\exp(\beta)=2$ 即该因素使事件率翻倍。
- 临床或医学意义如何表达：报告 IRR 与区间，说明是「率」还是「计数」。
- 常见误读：忽视过度离散把窄 SE 当真；不加 offset 直接比不同暴露量下的计数。

## 9. 假设诊断与稳健性

- 过度离散：Pearson $\chi^2/\text{df}$ 或偏差/df 明显大于 1 → 用 quasi-Poisson（放大 SE）或负二项（建模离散）。
- 零膨胀：零远多于泊松预期 → 零膨胀/障碍模型。
- 线性：log 尺度部分残差图。
- 相关/聚集：用 Poisson GLMM 或带稳健 SE 的 GEE。

## 10. 推荐可视化

- 观测 vs 预测计数。
- 率随协变量变化曲线。
- IRR 森林图。

### 10.1 图像示例

下图展示按类别的计数分布与泊松拟合。

![](../../04_示例图像/poisson_wordcount_label.png)

## 11. 优势、局限与常见坑

### 优势

- 天然适配计数/率，系数即率比。
- 经 offset 灵活处理暴露差异。
- 是计数建模的标准起点。

### 局限

- 等离散假设常被违反（过度离散）。
- 零过多时失效。
- 不处理相关数据。

### 常见坑

- 不检查过度离散，SE 过小、结论过度自信。
- 忘记 offset，比较不可比的计数。
- 零膨胀数据硬套普通泊松。

## 12. 与相近方法的区别

- 和**负二项回归**：过度离散时用负二项（额外离散参数）。
- 和 **quasi-Poisson**：quasi 只放大 SE、不指定完整分布。
- 和**零膨胀模型**：零异常多时用 ZIP/ZINB。
- 和 **Logistic 回归**：二分类用 Logistic、计数用 Poisson。

## 13. 医学研究中的典型应用

- 每千人年发病率的危险因素分析。
- 随访期内住院/发作次数建模。
- 生态学/传染病计数数据分析。

## 14. 关键术语

- **泊松分布**：稀有事件计数的分布，均值=方差。
- **率比（IRR）**：$\exp(\beta)$，事件率的倍数。
- **offset**：把计数转为率的暴露项（系数固定为 1）。
- **等离散（Equidispersion）**：方差等于均值。
- **过度离散**：方差大于均值。
- **log 连接**：Poisson 的典范连接。

## 15. 相关方法

- [[广义线性模型（Generalized Linear Model, GLM）]]
- [[负二项回归（Negative Binomial Regression）]]
- [[Quasi-Likelihood与过度离散（Quasi-Likelihood and Overdispersion）]]
- [[Logistic回归（Logistic Regression）]]

## 16. 参考资料

- Cameron AC, Trivedi PK. *Regression Analysis of Count Data*. 2nd ed. Cambridge University Press; 2013.
- McCullagh P, Nelder JA. *Generalized Linear Models*. 2nd ed. Chapman & Hall; 1989.
- Hilbe JM. *Modeling Count Data*. Cambridge University Press; 2014.
