---
title: 有序Logistic回归
english_name: Ordinal Logistic Regression
slug: ordinal-logistic-regression
aliases: [比例优势模型, proportional odds model, 有序回归, "有序Logistic回归（Ordinal Logistic Regression）"]
category: 回归与广义线性模型
subcategory: 有序结局建模
tags: [医学统计, 数据科学, 有序结局, GLM, 比例优势]
status: 已建
difficulty: intermediate
question_type: 有序分类结局建模
data_type: [表格数据]
outcome_type: [多分类]
python_packages: [statsmodels]
r_packages: [MASS, ordinal]
---

# 有序Logistic回归（Ordinal Logistic Regression）

## 1. 方法概览

### 1.1 定义

有序 Logistic 回归（比例优势模型）用于结局为有序类别的建模，通过对「累积概率」建 logit，估计协变量如何把个体推向更高等级。

### 1.2 它主要解决什么问题

- 研究问题：协变量如何影响一个有自然顺序的分级结局。
- 适用任务：有序结局的回归、比值比估计、有序风险预测。
- 常见医学场景：疾病严重度（轻/中/重）、肿瘤分级、疼痛/功能量表、Likert 评分。

### 1.3 直觉理解

把有序结局在每个「切点」处二分（如「≤中度」vs「重度」），得到多条累积 logit。比例优势假设让这些切点共享同一组斜率，因此一个协变量的效应可以用单一比值比概括「向更高等级移动的倾向」。

## 2. 数学形式

### 2.1 核心公式

对 $J$ 个有序类别，累积 logit 模型为：

$$
\operatorname{logit}\!\big(P(Y\le j\mid x)\big)=\alpha_j-\beta^\top x,\qquad j=1,\dots,J-1
$$

各切点截距 $\alpha_j$ 不同，斜率 $\beta$ 共享（比例优势）。

### 2.2 参数或统计量含义

- $\alpha_j$：第 $j$ 个切点的截距，随 $j$ 单调递增。
- $\beta$：协变量对累积 logit 的效应；$\exp(\beta)$ 为累积比值比（OR）。
- 负号约定使 $\beta>0$ 表示协变量增大更高等级的概率。

### 2.3 关键假设

- 比例优势（平行线）假设：各切点斜率相同。
- 结局有序、观测独立。
- 若比例优势不成立，改用偏比例优势或多项 Logistic。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、二分类、多分类均可。
- 因变量类型：有序分类（≥3 个有序等级）。
- 数据结构：每行一个体，含有序结局与协变量。
- 是否适合高维数据：需配合正则化。
- 是否适合缺失较多数据：需先处理缺失。
- 是否适合删失数据：不适用。
- 是否适合重复测量数据：需有序混合模型扩展。

### 3.2 示例表格

| 病例 | 年龄 | 分期治疗 | 疗效等级 |
| --- | --- | --- | --- |
| 1 | 62 | 1 | 好转 |
| 2 | 71 | 0 | 稳定 |
| 3 | 55 | 1 | 治愈 |
| 4 | 68 | 0 | 恶化 |

### 3.3 输入与产出

#### 输入

- 输入数据：有序结局（编码为有序因子）+ 协变量。
- 关键变量：结局的等级顺序、协变量矩阵。
- 需要预处理的内容：设定有序因子、分类变量编码、比例优势检验。

#### 产出

- 模型对象/统计结果：切点截距、系数、累积 OR。
- 参数估计：$\alpha_j$、$\beta$。
- 预测结果：各等级概率、累积概率。
- 不确定性指标：系数标准误、OR 区间、比例优势检验。

## 4. 适用场景

- 适合：有序等级结局、且各切点效应近似一致。
- 不适合：无序多分类（用多项 Logistic）、比例优势明显不成立。
- 使用前需要特别检查的点：比例优势假设、等级是否真有序、样本在各等级是否充足。

## 5. 实现

### 5.1 Python

常用包：

- `statsmodels`

```python
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

df = pd.read_csv("severity.csv")
y = pd.Categorical(df["grade"],
                   categories=["恶化","稳定","好转","治愈"], ordered=True)
mod = OrderedModel(y.codes, df[["age","treat"]], distr="logit")
res = mod.fit(method="bfgs", disp=False)
print(res.summary())
```

### 5.2 R

常用包：

- `MASS`

```r
library(MASS)
df$grade <- factor(df$grade,
                   levels = c("恶化","稳定","好转","治愈"), ordered = TRUE)
fit <- polr(grade ~ age + treat, data = df, Hess = TRUE)
exp(cbind(OR = coef(fit), confint(fit)))   # 累积 OR
```

## 6. 结果如何解释

- 核心结果看什么：各协变量的累积 OR、方向与区间；比例优势检验。
- 每个主要参数如何解释：OR>1 表示协变量升高使结局落到更高等级的比值增大（对所有切点相同）。
- 临床或医学意义如何表达：如「治疗使达到更好疗效等级的比值提高约 80%」。
- 常见误读：把有序 OR 当成某一具体等级的风险；忽视比例优势不成立。

## 7. 推荐可视化

- 各协变量的 OR 森林图。
- 不同协变量水平下的等级概率堆叠条形图。
- 比例优势诊断（各切点单独拟合的斜率对比）。

## 8. 优势、局限与常见坑

### 优势

- 充分利用结局的顺序信息，比多项 Logistic 更省参数。
- 单一 OR 概括效应，易于沟通。
- 软件成熟。

### 局限

- 依赖比例优势假设。
- 假设不成立时结论有偏。
- 等级极不平衡时估计不稳。

### 常见坑

- 不检验比例优势就直接报单一 OR。
- 把有序结局当连续变量做线性回归。
- 等级顺序编码错误导致方向反转。

## 9. 与相近方法的区别

- 和 [[Logistic回归（Logistic Regression）]] 的区别：后者用于二分类，前者用于有序多分类。
- 和多项 Logistic 回归的区别：多项 Logistic 不假设顺序、参数更多；有序模型利用顺序更简洁。
- 和 [[广义线性模型（Generalized Linear Model, GLM）]] 的关系：属累积 link 的 GLM 扩展。

## 10. 医学研究中的典型应用

- 疾病严重度分级的危险因素分析。
- 治疗对功能/疼痛量表等级的效应。
- 肿瘤分级与分子标志物关联建模。

## 11. 相关方法

- [[Logistic回归（Logistic Regression）]]
- [[广义线性模型（Generalized Linear Model, GLM）]]
- [[多项Logistic回归（Multinomial Logistic Regression）]]

## 12. 参考资料

- McCullagh P. Regression models for ordinal data. *J R Stat Soc B*. 1980;42(2):109-142.
- Agresti A. *Analysis of Ordinal Categorical Data*. 2nd ed. Wiley; 2010.
- Harrell FE. *Regression Modeling Strategies*. 2nd ed. Springer; 2015.
