---
title: 交叉验证
english_name: Cross-Validation
slug: cross-validation
aliases: [cross-validation, k-fold, CV, "交叉验证（Cross-Validation）"]
category: 模型评估与解释
subcategory: 重采样验证
tags: [医学统计, 数据科学, 模型评估, 重采样, 泛化误差]
status: 已建
difficulty: basic
question_type: 泛化性能估计与超参数选择
data_type: [表格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [rsample, caret]
---

# 交叉验证（Cross-Validation）

## 1. 方法概览

### 1.1 定义

交叉验证是一类重采样方法，通过反复把数据切成训练与验证子集来估计模型在新数据上的表现，也常用于超参数选择。

### 1.2 它主要解决什么问题

- 研究问题：模型在没见过的数据上大概表现如何，如何选超参数而不被训练误差误导。
- 适用任务：泛化性能估计、超参数调优、模型选择、有限样本下的稳健评估。
- 常见医学场景：样本量有限的临床预测模型开发、组学特征筛选后的性能估计、影像模型调参。

### 1.3 直觉理解

只在训练数据上评估会过于乐观。交叉验证把数据轮流当“考试卷”，每份数据都当过一次没被训练的测试样本，最后把多次成绩平均，得到更接近真实泛化能力的估计。

## 2. 数学形式

### 2.1 核心公式

$K$ 折交叉验证把数据分成 $K$ 份，第 $k$ 折用其余数据训练、在第 $k$ 份评估，泛化误差估计为：

$$
\widehat{\mathrm{CV}}=\frac{1}{K}\sum_{k=1}^{K}\frac{1}{|D_k|}\sum_{i\in D_k} L\big(y_i,\hat{f}^{(-k)}(x_i)\big)
$$

其中 $\hat{f}^{(-k)}$ 是去掉第 $k$ 折训练得到的模型，$L$ 为损失。留一法（LOOCV）是 $K=n$ 的特例。

### 2.2 参数或统计量含义

- $K$：折数，常用 5 或 10；偏差–方差权衡随 $K$ 变化。
- $\hat{f}^{(-k)}$：在训练折上拟合的模型。
- CV 估计：多折验证误差的平均，及其跨折标准差（反映稳定性）。
- 嵌套 CV：外层估性能、内层选超参，避免选择偏倚。

### 2.3 关键假设

- 样本近似独立同分布；有聚类（同一患者多次测量）时须按组划分。
- 时间序列须用时序切分，不能随机打乱（会造成未来信息泄漏）。
- 所有依赖数据的步骤（标准化、特征选择、插补）必须放进 CV 循环内。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：任意可建模特征。
- 因变量类型：二分类、多分类或连续型（分类问题用分层折）。
- 数据结构：独立样本的表格数据；聚类/时序需特殊切分。
- 是否适合高维数据：适合，且高维小样本尤其需要，但要防止特征选择泄漏。
- 是否适合缺失较多数据：插补须在每折训练子集内拟合。
- 是否适合删失数据：可用生存专用评分（如 C-index）作为 CV 指标。
- 是否适合重复测量数据：需用 GroupKFold 等按个体划分。

### 3.2 示例表格

| 折 | 训练样本数 | 验证样本数 | 验证 AUC |
| --- | --- | --- | --- |
| 1 | 455 | 114 | 0.982 |
| 2 | 455 | 114 | 0.971 |
| 3 | 455 | 114 | 0.990 |
| 4 | 456 | 113 | 0.977 |
| 5 | 456 | 113 | 0.985 |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵、标签、可选分组变量。
- 关键变量：折数、切分策略（分层/分组/时序）、评估指标。
- 需要预处理的内容：把预处理封装进管线，随折重新拟合。

#### 产出

- 模型对象/统计结果：各折得分及其均值、标准差。
- 参数估计：调优选出的超参数。
- 预测结果：out-of-fold 预测（可再评估校准/区分度）。
- 不确定性指标：跨折得分离散度、重复 CV 的区间。

## 4. 适用场景

- 适合：样本有限、需要调参、需要相对稳健的性能估计。
- 不适合：有独立大型外部验证集时（外部验证更可信）、忽视聚类/时序结构时。
- 使用前需要特别检查的点：切分是否尊重分组与时间、预处理是否泄漏、调参是否需要嵌套 CV。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

X, y = load_breast_cancer(return_X_y=True)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print("AUC 均值:", round(scores.mean(), 3), "标准差:", round(scores.std(), 3))
```

### 5.2 R

常用包：

- `rsample`
- `caret`

```r
library(caret)

ctrl <- trainControl(method = "cv", number = 5,
                     classProbs = TRUE, summaryFunction = twoClassSummary)
fit <- train(y ~ ., data = df, method = "glm", family = "binomial",
             metric = "ROC", trControl = ctrl, preProcess = c("center", "scale"))
print(fit)
```

## 6. 结果如何解释

- 核心结果看什么：CV 得分均值及跨折波动；波动大提示模型/数据不稳定。
- 每个主要参数如何解释：均值近似泛化性能；标准差反映稳定性；嵌套 CV 的外层得分才是无偏性能估计。
- 临床或医学意义如何表达：CV 是内部验证，仍需外部验证支撑临床推广。
- 常见误读：把“先在全数据调参再 CV”的得分当作无偏估计；用 CV 结果替代外部验证。

## 7. 推荐可视化

- 各折/重复得分的箱线图或散点。
- 学习曲线（性能随训练样本量变化）。
- 超参数网格上的验证得分热图。

## 8. 优势、局限与常见坑

### 优势

- 充分利用有限样本，得到较稳健的性能估计。
- 天然支持超参数选择与模型比较。
- 可产生 out-of-fold 预测用于进一步评估。

### 局限

- 计算成本随折数和调参空间上升。
- iid 假设破坏（聚类/时序）时会高估性能。
- 内部验证不能替代外部验证。

### 常见坑

- 在 CV 之外做标准化、特征选择或插补，导致数据泄漏。
- 用同一层 CV 既调参又报告性能（应嵌套 CV）。
- 同一患者的多条记录被拆到训练与验证两边。

## 9. 与相近方法的区别

- 和 [[Bootstrap重抽样（Bootstrap Resampling）]] 的区别：CV 靠不重叠切分，Bootstrap 靠有放回重抽样，二者都可估泛化误差与不确定性。
- 和单次留出法（hold-out）的区别：CV 用满所有数据、方差更低，小样本更可靠。
- 和外部验证的区别：CV 是内部验证，外部验证在独立人群评估可推广性。

## 10. 医学研究中的典型应用

- 有限样本临床预测模型的内部验证（配合 optimism 校正）。
- 组学高维数据中特征筛选＋建模的嵌套 CV 评估。
- 机器学习/影像模型的超参数选择。

## 11. 相关方法

- [[Bootstrap重抽样（Bootstrap Resampling）]]
- [[ROC曲线与AUC（Receiver Operating Characteristic and AUC）]]
- [[校准曲线（Calibration Curve）]]

## 12. 参考资料

- Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. 2nd ed. Springer; 2009. （第 7 章）
- Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7:91.
- scikit-learn. Cross-validation. [https://scikit-learn.org/stable/modules/cross_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html) （访问日期：2026-07-02）
