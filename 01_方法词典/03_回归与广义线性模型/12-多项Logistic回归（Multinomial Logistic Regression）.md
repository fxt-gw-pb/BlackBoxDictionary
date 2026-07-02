---
title: 多项Logistic回归
english_name: Multinomial Logistic Regression
slug: multinomial-logistic-regression
aliases: [多分类Logistic, softmax回归, baseline-category logit, "多项Logistic回归（Multinomial Logistic Regression）"]
category: 回归与广义线性模型
subcategory: 无序多分类结局建模
tags: [医学统计, 数据科学, 多分类, GLM, softmax]
status: 已建
difficulty: intermediate
question_type: 无序多分类结局建模
data_type: [表格数据]
outcome_type: [多分类]
python_packages: [statsmodels, scikit-learn]
r_packages: [nnet, VGAM]
---

# 多项Logistic回归（Multinomial Logistic Regression）

## 1. 方法概览

### 1.1 定义

多项 Logistic 回归把二分类 Logistic 推广到三类及以上的无序结局，通过对每个类别相对参考类别建 logit，估计协变量对各类别相对概率的影响。

### 1.2 它主要解决什么问题

- 研究问题：协变量如何影响个体落入若干个无自然顺序类别中的哪一个。
- 适用任务：无序多分类结局建模、相对比值比估计、多类别概率预测。
- 常见医学场景：疾病亚型（无序）、多种治疗方案选择、就诊科室/结局去向分类。

### 1.3 直觉理解

若结局是几个彼此平行、没有大小顺序的类别，就以其中一类为「参照」，逐一比较「进入其他每一类相对于参照类」的倾向。每个协变量因此对每个非参照类别各有一组系数。

## 2. 数学形式

### 2.1 核心公式

以类别 $J$ 为参考，对 $j=1,\dots,J-1$：

$$
\log\frac{P(Y=j\mid x)}{P(Y=J\mid x)}=\beta_j^\top x
$$

概率由 softmax 给出：

$$
P(Y=j\mid x)=\frac{\exp(\beta_j^\top x)}{1+\sum_{k=1}^{J-1}\exp(\beta_k^\top x)}
$$

### 2.2 参数或统计量含义

- $\beta_j$：类别 $j$ 相对参考类别的系数向量。
- $\exp(\beta_{j})$:相对参考类别的相对风险比（relative risk ratio）。
- 参考类别的选择只影响系数呈现，不影响拟合概率。
- 类别数 $J$：结局的无序类别个数。

### 2.3 关键假设

- 结局类别互斥穷尽、无自然顺序（有序则用有序 Logistic 更省参数）。
- 观测独立。
- 无关方案独立性（IIA）：加入/移除某类别不改变其余类别的相对比值。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、二分类、多分类均可。
- 因变量类型：无序多分类（≥3 类）。
- 数据结构：每行一个体，含多分类结局与协变量。
- 是否适合高维数据：需正则化。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不适用。
- 是否适合重复测量数据：需多分类混合模型。

### 3.2 示例表格

| 病例 | 年龄 | 分子标志物 | 亚型 |
| --- | --- | --- | --- |
| 1 | 62 | 1 | LuminalA |
| 2 | 55 | 0 | HER2 |
| 3 | 71 | 1 | Basal |
| 4 | 48 | 0 | LuminalA |

### 3.3 输入与产出

#### 输入

- 输入数据：无序多分类结局 + 协变量。
- 关键变量：结局类别集合、参考类别、协变量矩阵。
- 需要预处理的内容：指定参考类别、分类变量编码、检查各类别样本量。

#### 产出

- 模型对象/统计结果：各非参考类别的系数与相对风险比。
- 参数估计：$\beta_j$。
- 预测结果：各类别的预测概率、最可能类别。
- 不确定性指标：系数标准误、RRR 区间。

## 4. 适用场景

- 适合：无序、平行的多分类结局。
- 不适合：有序结局（用 [[有序Logistic回归（Ordinal Logistic Regression）]] 更省参数）、二分类（用普通 Logistic）。
- 使用前需要特别检查的点：结局是否真无序、各类别样本是否充足、IIA 是否合理。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`

```python
import statsmodels.api as sm
import pandas as pd

X = sm.add_constant(df[["age", "marker"]])
mod = sm.MNLogit(df["subtype_code"], X).fit(disp=False)  # 结局编码为 0..J-1
print(mod.summary())          # 各非参考类别系数
```

### 5.2 R

常用包：

- `nnet`

```r
library(nnet)
df$subtype <- relevel(factor(df$subtype), ref = "LuminalA")
fit <- multinom(subtype ~ age + marker, data = df)
exp(coef(fit))                # 相对风险比 RRR
```

## 6. 结果如何解释

- 核心结果看什么：每个非参考类别相对参考类别的 RRR、方向与区间。
- 每个主要参数如何解释：RRR>1 表示协变量升高使「进入该类别相对参考类别」的相对概率增大。
- 临床或医学意义如何表达：如「该标志物阳性者相对 LuminalA，更可能为 Basal 亚型」。
- 常见误读：把 RRR 当成绝对概率；忽视参考类别改变会改变系数呈现。

## 7. 推荐可视化

- 各类别 RRR 的分面森林图。
- 协变量变化下各类别预测概率曲线。
- 预测概率的堆叠面积图。

## 8. 优势、局限与常见坑

### 优势

- 直接建模无序多分类，输出各类别概率。
- 系数可解释为相对风险比。
- 软件成熟、易扩展。

### 局限

- 参数随类别数线性增多，需较大样本。
- 依赖 IIA 假设。
- 忽略结局顺序信息（若有序则低效）。

### 常见坑

- 对有序结局误用多项模型，损失效率。
- 某些类别样本太少导致估计不稳。
- 不声明参考类别，结果难复现。

## 9. 与相近方法的区别

- 和 [[Logistic回归（Logistic Regression）]] 的区别：后者二分类，本方法推广到多分类。
- 和 [[有序Logistic回归（Ordinal Logistic Regression）]] 的区别：有序模型利用顺序、更省参数；多项模型不假设顺序、更灵活。
- 和多层感知机分类器的区别：多项 Logistic 是线性、可解释；神经网络非线性、可解释性弱。

## 10. 医学研究中的典型应用

- 肿瘤分子亚型（无序）的危险因素分析。
- 多种治疗方案选择的影响因素。
- 多类别结局去向（治愈/转院/失访等）的建模。

## 11. 相关方法

- [[Logistic回归（Logistic Regression）]]
- [[有序Logistic回归（Ordinal Logistic Regression）]]
- [[广义线性模型（Generalized Linear Model, GLM）]]

## 12. 参考资料

- Hosmer DW, Lemeshow S, Sturdivant RX. *Applied Logistic Regression*. 3rd ed. Wiley; 2013.
- Agresti A. *Categorical Data Analysis*. 3rd ed. Wiley; 2013.
- Kwak C, Clayton-Matthews A. Multinomial logistic regression. *Nurs Res*. 2002;51(6):404-410.
