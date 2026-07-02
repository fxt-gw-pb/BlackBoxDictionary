---
title: 校准曲线
english_name: Calibration Curve
slug: calibration-curve
aliases: [calibration curve, calibration plot, reliability diagram, "校准曲线（Calibration Curve）"]
category: 模型评估与解释
subcategory: 校准评估
tags: [医学统计, 数据科学, 模型评估, 校准, 风险预测]
status: 已建
difficulty: intermediate
question_type: 预测概率校准评估
data_type: [表格数据]
outcome_type: [二分类]
python_packages: [scikit-learn]
r_packages: [rms]
---

# 校准曲线（Calibration Curve）

## 1. 方法概览

### 1.1 定义

校准曲线（可靠性图）比较模型的预测概率与实际观察到的事件发生频率，用来判断“模型说的 20% 风险，是不是真的大约 20% 的人发生了事件”。

### 1.2 它主要解决什么问题

- 研究问题：预测概率在数值上可信吗，是系统性偏高还是偏低。
- 适用任务：风险预测模型的校准评估、模型更新与重校准、不同模型的可靠性比较。
- 常见医学场景：临床风险评分（如心血管 10 年风险）、术后并发症概率、疾病复发概率的可信度评估。

### 1.3 直觉理解

把样本按预测概率分组（或用平滑曲线），横轴是预测概率、纵轴是该组真实发生率。理想模型的点应落在 45° 对角线上；偏离对角线说明预测概率系统性偏高或偏低。区分度好的模型也可能校准很差。

## 2. 数学形式

### 2.1 核心公式

理想校准要求：

$$
P(Y=1\mid \hat{p}=p)=p,\quad \forall p\in[0,1]
$$

常用弱校准回归（在 logit 尺度上）刻画：

$$
\operatorname{logit}\big(P(Y=1)\big)=\alpha+\beta\cdot\operatorname{logit}(\hat{p})
$$

理想时 $\alpha=0$（截距，calibration-in-the-large）、$\beta=1$（校准斜率）。整体准确度可用 Brier 分数：

$$
\mathrm{Brier}=\frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i-y_i)^2
$$

### 2.2 参数或统计量含义

- 校准截距 $\alpha$：预测概率整体是否偏高（$\alpha<0$）或偏低（$\alpha>0$）。
- 校准斜率 $\beta$：$\beta<1$ 表示预测过度极端（过拟合的典型表现）。
- Brier 分数：越小越好，兼含区分度与校准（可做校准/区分分解）。
- Hosmer-Lemeshow 统计量：分组拟合优度检验，但对分组和样本量敏感，不宜单独依赖。

### 2.3 关键假设

- 需要预测概率（而非仅类别标签）与二分类结局。
- 校准是针对特定人群的性质，换人群需重新评估（外部校准）。
- 分组法依赖分箱数量，平滑法（loess/样条）更稳健。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：模型预测概率。
- 因变量类型：二分类结局。
- 数据结构：每行含预测概率与真实标签。
- 是否适合高维数据：只作用于最终概率，与特征维度无关。
- 是否适合缺失较多数据：需先得到完整预测概率。
- 是否适合删失数据：不直接适用；生存模型用特定时间点的校准（如 1 年预测生存 vs KM 观测）。
- 是否适合重复测量数据：聚类结构会影响区间估计。

### 3.2 示例表格

| 预测概率分组 | 平均预测概率 | 组内实际发生率 | 例数 |
| --- | --- | --- | --- |
| 0.0–0.2 | 0.09 | 0.07 | 210 |
| 0.2–0.4 | 0.31 | 0.28 | 160 |
| 0.4–0.6 | 0.52 | 0.55 | 120 |
| 0.6–0.8 | 0.69 | 0.74 | 90 |
| 0.8–1.0 | 0.90 | 0.88 | 70 |

### 3.3 输入与产出

#### 输入

- 输入数据：预测概率向量、真实标签向量。
- 关键变量：预测概率、二分类结局。
- 需要预处理的内容：在独立/验证数据上评估；选择分箱或平滑方式。

#### 产出

- 模型对象/统计结果：校准曲线、截距与斜率、Brier 分数。
- 参数估计：calibration-in-the-large、校准斜率。
- 预测结果：可用于重校准（Platt scaling、isotonic）。
- 不确定性指标：曲线置信带、截距/斜率区间。

## 4. 适用场景

- 适合：风险预测模型报告（配合区分度）、模型外部验证、比较重校准前后。
- 不适合：只需类别判别、无法输出概率的模型（可先做概率化）。
- 使用前需要特别检查的点：评估数据是否独立、样本量是否足以稳定估计、是否同时报告了区分度。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

clf = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:, 1]

frac_pos, mean_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
print("Brier:", round(brier_score_loss(yte, p), 4))
```

### 5.2 R

常用包：

- `rms`

```r
library(rms)

# p: 预测概率; y: 0/1 结局
val <- val.prob(p = p, y = y)   # 输出截距、斜率、Brier、C 指数并绘制校准曲线
print(val)
```

## 6. 结果如何解释

- 核心结果看什么：曲线是否贴近对角线、校准截距与斜率、Brier 分数。
- 每个主要参数如何解释：斜率 <1 说明预测过于极端（高的偏高、低的偏低）；截距偏离 0 说明整体平移。
- 临床或医学意义如何表达：校准差的模型即便区分度好，用于个体风险沟通也会误导决策。
- 常见误读：只看 Hosmer-Lemeshow 的 p 值（不显著≠校准好，尤其小样本）；把校准好等同于模型有临床价值。

## 7. 推荐可视化

- 校准曲线（loess 或分箱），叠加 45° 参考线与置信带。
- 预测概率的直方图/密度（观察风险分布与覆盖范围）。
- 重校准前后对比曲线。

## 8. 优势、局限与常见坑

### 优势

- 直接回答“预测概率可信吗”，是风险模型不可或缺的评估维度。
- 可量化为截距/斜率，便于报告与外部验证比较。
- 可指导重校准（Platt / isotonic）。

### 局限

- 是特定人群性质，人群漂移需重新评估。
- 分箱法受箱数影响，结果可能不稳。
- 只评校准，不含临床成本权衡。

### 常见坑

- 在训练集上评估校准，过度乐观。
- 依赖 Hosmer-Lemeshow 检验下结论。
- 校准好即宣称模型可用，忽略 [[决策曲线分析（Decision Curve Analysis, DCA）]] 的净获益。

## 9. 与相近方法的区别

- 和 [[ROC曲线与AUC（Receiver Operating Characteristic and AUC）]] 的区别：AUC 评排序/区分，校准评概率数值准确性，二者正交。
- 和 [[决策曲线分析（Decision Curve Analysis, DCA）]] 的区别：DCA 评临床净获益，校准评概率可靠性。
- 和重校准方法（Platt scaling、isotonic）的关系：校准曲线诊断问题，重校准解决问题。

## 10. 医学研究中的典型应用

- 临床预测模型 TRIPOD 报告中的校准评估。
- 将外部模型移植到本地人群时的校准检查与重校准。
- 机器学习模型（树/提升模型常概率偏差）的概率可靠性评估。

## 11. 相关方法

- [[ROC曲线与AUC（Receiver Operating Characteristic and AUC）]]
- [[决策曲线分析（Decision Curve Analysis, DCA）]]
- [[Logistic回归（Logistic Regression）]]
- [[交叉验证（Cross-Validation）]]

## 12. 参考资料

- Van Calster B, McLernon DJ, van Smeden M, et al. Calibration: the Achilles heel of predictive analytics. *BMC Med*. 2019;17(1):230.
- Steyerberg EW, Vickers AJ, Cook NR, et al. Assessing the performance of prediction models: a framework for traditional and novel measures. *Epidemiology*. 2010;21(1):128-138.
- Harrell FE. *Regression Modeling Strategies*. 2nd ed. Springer; 2015.
- scikit-learn. Probability calibration. [https://scikit-learn.org/stable/modules/calibration.html](https://scikit-learn.org/stable/modules/calibration.html) （访问日期：2026-07-02）
