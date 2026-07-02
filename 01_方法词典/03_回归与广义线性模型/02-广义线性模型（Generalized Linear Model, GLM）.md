---
title: 广义线性模型
english_name: Generalized Linear Model, GLM
aliases: [GLM, "广义线性模型（Generalized Linear Model, GLM）"]
category: 回归与广义线性模型
subcategory: 广义线性模型框架
tags: [医学统计, 数据科学, 回归分析, 广义线性模型]
status: 已建
difficulty: intermediate
question_type: 非高斯结局建模
data_type: [表格数据]
outcome_type: [连续型, 二分类, 计数型]
python_packages: [statsmodels]
r_packages: [stats]
related_methods: [线性回归, Logistic回归, Poisson回归, Quasi-Likelihood与过度离散]
---

# 广义线性模型（Generalized Linear Model, GLM）

## 1. 方法概览

### 1.1 定义

广义线性模型是把线性模型推广到非高斯结局的一套统一框架，通过“分布族 + 线性预测子 + 连接函数”来建模连续、二分类和计数等不同类型的结局。

### 1.2 它主要解决什么问题

- 研究问题：当结局不再适合普通线性回归时，如何仍然保留“协变量线性组合”的建模思路。
- 适用任务：二分类、计数、比例、率等结局建模。
- 常见医学场景：患病风险建模、事件发生率建模、分组二元结局分析。

### 1.3 直觉理解

GLM 的核心思想是：不要直接对原始结局做线性建模，而是先找到一个合适的均值函数和连接函数，把“难处理的结局”变换到可以线性表达的尺度上。

## 2. 数学形式

### 2.1 核心公式

$$
\begin{aligned}
Y_i &\sim \text{Exponential Family} \\
\mu_i &= E(Y_i) \\
g(\mu_i) &= \eta_i = \mathbf{X}_i^\top \boldsymbol{\beta}
\end{aligned}
$$

### 2.2 参数或统计量含义

- 随机成分：$Y_i$ 的分布，常见为正态、Bernoulli、Binomial、Poisson。
- 系统成分：线性预测子 $\eta_i=\mathbf X_i^\top\boldsymbol\beta$。
- 连接函数 $g(\cdot)$：把均值 $\mu_i$ 映射到线性预测子空间。
- canonical link：满足 $g_c(\mu_i)=\theta_i$ 的连接函数。
- deviance：GLM 里的“拟合优度差异”量，常用于模型比较与缺失拟合判断。

### 2.3 关键假设

- 观测之间独立。
- 均值模型正确指定。
- 所选分布族和连接函数与结局类型基本匹配。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、二分类、多分类都可。
- 因变量类型：二分类、计数、比例、率，以及某些连续结局。
- 数据结构：独立样本宽表最典型。
- 是否适合高维数据：可扩展，但高维时通常需要正则化。
- 是否适合缺失较多数据：可以，但需先明确缺失处理策略。
- 是否适合删失数据：不适合；删失应转向生存模型。
- 是否适合重复测量数据：普通 GLM 不适合，应转向 GLMM 或 GEE。

### 3.2 示例表格

GLM 是一个框架，因此它对应的数据形式并不唯一。最常见的是下面几类：

| 数据形式 | 示例字段 | 常见分布 | 常见连接函数 |
| --- | --- | --- | --- |
| 二元结局个体数据 | `BMI`, `SEX`, `PREVHYP` | Bernoulli | `logit` |
| 分组二元结局 | `success`, `total`, `group` | Binomial | `logit` |
| 计数结局 | `word_count`, `label` | Poisson | `log` |
| 率数据 | `count`, `exposure_time`, `group` | Poisson | `log` + offset |

### 3.3 输入与产出

#### 输入

- 输入数据：结局变量和协变量。
- 关键变量：结局类型、连接函数、是否有分组或 offset。
- 需要预处理的内容：缺失处理、分类变量编码、必要时构造 grouped binomial 或 exposure。

#### 产出

- 模型对象/统计结果：系数估计、标准误、Wald 检验、deviance、AIC。
- 参数估计：$\boldsymbol{\beta}$ 及其区间。
- 预测结果：均值、概率或期望计数。
- 不确定性指标：标准误、置信区间、稳健方差。

## 4. 适用场景

- 适合：结局不是正态连续型时的一般回归建模。
- 不适合：强依赖结构、删失结局、复杂非线性未经处理的情形。
- 使用前需要特别检查的点：结局类型、分布假设、连接函数、deviance 和残差诊断。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`

```python
import statsmodels.api as sm

# family 可替换成 Binomial(), Poisson(), Gaussian() 等
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()
print(result.summary())
print(result.deviance, result.aic)
```

### 5.2 R

常用包：

- `stats`

```r
# family 可替换成 binomial(), poisson(), gaussian() 等
fit <- glm(y ~ x1 + x2, family = binomial(), data = df)
summary(fit)
deviance(fit)
AIC(fit)
```

## 6. 结果如何解释

- 核心结果看什么：系数所在尺度、连接函数、deviance 和模型拟合。
- 每个主要参数如何解释：必须先回到连接函数所在的尺度，再决定是解释为均值差、log-odds 还是 log-rate。
- 临床或医学意义如何表达：报告时应把模型系数翻译回临床更容易理解的量，如 OR、率比或预测概率。
- 常见误读：不同 GLM 的系数不能直接互相比较；deviance 也不等于线性回归的 $R^2$。

## 7. 推荐可视化

- 拟合值 vs 观测值图。
- 残差 / deviance 残差图。
- 预测概率或预测均值曲线。

## 8. 优势、局限与常见坑

### 优势

- 统一处理多种结局类型。
- 与线性模型共享大量解释框架。
- 自然支持 grouped binary、count 和 offset 等场景。

### 局限

- 对分布与连接函数选择敏感。
- 普通 GLM 不处理相关数据。
- 系数解释必须依赖所选连接函数。

### 常见坑

- 用线性回归处理明显不合适的二元或计数结局。
- 把 logit、log link 下的系数按“均值差”解释。
- 只看 p 值，不看 deviance、残差和预测尺度。

## 9. 与相近方法的区别

- 和线性回归的区别：线性回归是 GLM 在正态分布 + identity link 下的特例。
- 和 GLMM / GEE 的区别：普通 GLM 假设观测独立，不处理重复测量。
- 应该如何选择：先判断结局类型，再决定是否需要相关结构建模。

## 10. 医学研究中的典型应用

- 基线危险因素与二元疾病状态的关系。
- 不良事件发生次数的建模。
- 基于暴露时间的人群事件率建模。

## 11. 相关方法

- [[线性回归（Linear Regression）]]
- [[Logistic回归（Logistic Regression）]]
- [[Poisson回归（Poisson Regression）]]
- [[Quasi-Likelihood与过度离散（Quasi-Likelihood and Overdispersion）]]

## 12. 参考资料

- McCullagh P, Nelder JA. *Generalized Linear Models*. 2nd ed. Chapman & Hall; 1989.
- statsmodels Developers. `statsmodels.genmod.generalized_linear_model.GLM`. statsmodels API Reference. [https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html) （访问日期：2026-07-02）
- R Core Team. `glm`. R Manual. [https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html) （访问日期：2026-07-02）
