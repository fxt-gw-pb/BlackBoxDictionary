---
title: 限制性立方样条
english_name: Restricted Cubic Splines, RCS
slug: restricted-cubic-splines-rcs
aliases: [RCS, 限制性三次样条, restricted cubic spline, natural spline, "限制性立方样条（Restricted Cubic Splines, RCS）"]
category: 回归与广义线性模型
subcategory: 非线性建模
tags: [医学统计, 数据科学, 非线性, 剂量反应, 样条]
status: 已建
difficulty: intermediate
question_type: 连续协变量的非线性效应建模
data_type: [表格数据]
outcome_type: [连续型, 二分类, 时间到事件]
python_packages: [patsy, statsmodels]
r_packages: [rms, splines]
---

# 限制性立方样条（Restricted Cubic Splines, RCS）

## 1. 方法概览

### 1.1 定义

限制性立方样条是一种把连续协变量展开为若干基函数的建模技术，在结点之间用三次多项式平滑拟合，并强制两端呈线性，从而灵活刻画非线性关系而不过度摆动。

### 1.2 它主要解决什么问题

- 研究问题：某个连续暴露与结局是否呈非线性（如 U 形、阈值）关系，形状如何。
- 适用任务：非线性剂量反应建模、连续协变量调整、非线性检验。
- 常见医学场景：BMI 与死亡风险的 U 形关系、年龄效应、生物标志物阈值。

### 1.3 直觉理解

直接放入连续变量默认它是直线，容易漏掉真实的弯曲关系；而高次多项式两端会剧烈摆动。RCS 在数据密集处用分段三次曲线灵活拟合，在两端「拉直」以避免尾部失控，兼顾灵活与稳健。

## 2. 数学形式

### 2.1 核心公式

给定结点 $t_1<\dots<t_k$，把 $x$ 展开为 $k-1$ 个基函数，模型对结局的线性预测子写作：

$$
g\big(E[Y\mid x]\big)=\beta_0+\beta_1 x+\sum_{j=1}^{k-2}\gamma_j\, s_j(x)
$$

其中 $s_j(x)$ 为由结点构造、两端线性的三次样条基。

### 2.2 参数或统计量含义

- $t_1,\dots,t_k$：结点，常取分位数（如 3 结点取 10/50/90 百分位）。
- $\beta_1,\gamma_j$:样条基系数，单个系数无直接意义，需整体解读。
- 非线性检验：联合检验所有 $\gamma_j=0$。
- $g(\cdot)$：随结局类型取恒等/logit/log 等 link，可嵌入线性、Logistic、Cox 模型。

### 2.3 关键假设

- 结点数与位置合理（3–5 个通常足够）。
- 展开后仍沿用所在回归模型的其余假设。
- 关注的是效应形状而非单个基系数。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：至少一个连续协变量（做样条展开）。
- 因变量类型：连续、二分类、计数或时间到事件（依所嵌回归而定）。
- 数据结构：标准回归表格。
- 是否适合高维数据：对少数关键连续变量使用，不宜对所有变量展开。
- 是否适合缺失较多数据：先处理缺失。
- 是否适合删失数据：嵌入 Cox 时适用。
- 是否适合重复测量数据：可嵌入混合模型。

### 3.2 示例表格

| 病例 | BMI | 年龄 | 事件 |
| --- | --- | --- | --- |
| 1 | 22.5 | 60 | 0 |
| 2 | 31.0 | 72 | 1 |
| 3 | 18.5 | 55 | 1 |
| 4 | 26.4 | 64 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：连续协变量 + 结局 + 其他调整变量。
- 关键变量：待展开变量、结点数与位置。
- 需要预处理的内容：确定结点（分位数）、参考值设定。

#### 产出

- 模型对象/统计结果：样条系数、非线性检验 p 值。
- 参数估计:样条基系数（整体解读）。
- 预测结果：效应形状曲线（如相对某参考值的 OR/HR）。
- 不确定性指标：预测曲线的置信带。

## 4. 适用场景

- 适合：怀疑连续暴露与结局非线性、需灵活又稳健地刻画形状。
- 不适合：关系确为线性、样本量太小支撑不了多结点、需要单一斜率解释时。
- 使用前需要特别检查的点：结点数/位置、是否过拟合、如何设定参考值展示效应。

## 5. 实现

### 5.1 Python

常用包：

- `patsy`
- `statsmodels`

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# cr() 为自然(限制性)三次样条基, df 控制结点数
fit = smf.logit("event ~ cr(BMI, df=3) + age", data=df).fit()
print(fit.summary())
```

### 5.2 R

常用包：

- `rms`

```r
library(rms)
dd <- datadist(df); options(datadist = "dd")
fit <- lrm(event ~ rcs(BMI, 4) + age, data = df)  # 4 个结点
anova(fit)                     # 含非线性检验
plot(Predict(fit, BMI))        # 效应形状曲线
```

## 6. 结果如何解释

- 核心结果看什么：效应形状曲线、非线性检验 p 值、是否存在阈值/拐点。
- 每个主要参数如何解释：单个样条系数无意义，应看整条曲线相对参考值的 OR/HR。
- 临床或医学意义如何表达：如「BMI 在 25 附近风险最低，两侧升高，呈 U 形」。
- 常见误读：解读单个基系数；把非线性显著等同于临床重要。

## 7. 推荐可视化

- 效应形状曲线（含 95% 置信带）。
- 结点位置标注。
- 与线性拟合的对比叠加。

## 8. 优势、局限与常见坑

### 优势

- 灵活刻画非线性且两端稳健，避免多项式尾部摆动。
- 可嵌入线性/Logistic/Cox 等多种模型。
- 提供正式的非线性检验。

### 局限

- 结点选择带主观性。
- 结果需图形化解读，不如单一斜率直观。
- 结点过多易过拟合。

### 常见坑

- 结点数过多导致过拟合与不稳。
- 外推到数据稀疏的尾部。
- 不报告参考值，使 OR/HR 曲线无法解读。

## 9. 与相近方法的区别

- 和 [[多项式回归（Polynomial Regression）]] 的区别：多项式全局拟合、尾部易摆动；RCS 分段且两端线性更稳。
- 和 [[局部加权回归（Locally Weighted Regression）]] 的区别：LOWESS 偏探索性，RCS 给出可推断的参数模型。
- 和分箱/分类的区别:样条保留连续信息，避免任意切点造成的信息损失。

## 10. 医学研究中的典型应用

- BMI、血压、生物标志物与结局的非线性剂量反应。
- Cox 模型中连续暴露的非线性风险刻画。
- 年龄等连续混杂因素的灵活调整。

## 11. 相关方法

- [[多项式回归（Polynomial Regression）]]
- [[局部加权回归（Locally Weighted Regression）]]
- [[Cox比例风险模型（Cox Proportional Hazards Model）]]

## 12. 参考资料

- Durrleman S, Simon R. Flexible regression models with cubic splines. *Stat Med*. 1989;8(5):551-561.
- Harrell FE. *Regression Modeling Strategies*. 2nd ed. Springer; 2015.
- Gauthier J, et al. Cubic splines to model relationships between continuous variables and outcomes. *Bone Marrow Transplant*. 2020;55:675-680.
