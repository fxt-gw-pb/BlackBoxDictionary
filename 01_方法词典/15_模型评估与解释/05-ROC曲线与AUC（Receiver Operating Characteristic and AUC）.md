---
title: ROC曲线与AUC
english_name: Receiver Operating Characteristic and AUC
slug: receiver-operating-characteristic-and-auc
aliases: [ROC, AUC, ROC-AUC, "ROC曲线与AUC（Receiver Operating Characteristic and AUC）"]
category: 模型评估与解释
subcategory: 区分度评估
tags: [医学统计, 数据科学, 模型评估, 区分度, 诊断试验]
status: 已建
difficulty: basic
question_type: 二分类模型区分度评估
data_type: [表格数据]
outcome_type: [二分类]
python_packages: [scikit-learn]
r_packages: [pROC]
---

# ROC曲线与AUC（Receiver Operating Characteristic and AUC）

## 1. 方法概览

### 1.1 一句话本质

ROC 扫描所有阈值观察灵敏度—假阳性权衡，AUC 则等于随机一名阳性得分高于随机一名阴性的概率。

### 1.2 定义

ROC 曲线（受试者工作特征曲线）把一个连续评分或预测概率在所有可能阈值下的敏感度与（1−特异度）连成一条曲线；AUC 是曲线下面积，用一个 0–1 的数概括模型的整体区分能力。

### 1.3 它主要解决什么问题

- 研究问题：一个诊断指标或风险模型，把病例和非病例区分开的能力有多强。
- 适用任务：二分类模型区分度评估、诊断试验评价、不同模型/生物标志物的区分度比较。
- 常见医学场景：肿瘤标志物诊断价值、风险评分区分高低危、影像模型良恶性判别。

### 1.4 直觉理解

如果随机抽一个真阳性和一个真阴性，AUC 等于“模型给阳性打的分高于阴性”的概率。AUC=0.5 相当于抛硬币，AUC=1 表示完美区分。ROC 曲线越靠左上角越好。

## 2. 核心思想与原理

### 2.1 根本困难
分类概率必须经过阈值才变成决策，不同临床代价需要不同阈值；单个准确率把这种权衡藏起来。

### 2.2 关键洞察
按得分从高到低逐步放宽阈值，每纳入一个阳性提高 TPR，每纳入一个阴性提高 FPR。AUC 汇总排序能力，不依赖某个阈值。

### 2.3 与朴素做法对比
准确率受患病率和阈值影响；ROC-AUC 较稳定但不评价概率校准，在极不平衡数据上 PR 曲线更直接。

## 3. 数学形式

### 3.1 核心公式

在阈值 $c$ 下，由混淆矩阵定义两个坐标：

$$
\mathrm{TPR}(c)=\frac{TP}{TP+FN},\qquad \mathrm{FPR}(c)=\frac{FP}{FP+TN}
$$

ROC 曲线是 $(\mathrm{FPR}(c),\mathrm{TPR}(c))$ 随 $c$ 变化的轨迹。AUC 为曲线下面积，并等价于概率解释：

$$
\mathrm{AUC}=P\big(\hat{s}_{i}>\hat{s}_{j}\mid y_i=1,\,y_j=0\big)=\frac{U}{n_1 n_0}
$$

其中 $U$ 是 Mann-Whitney U 统计量，$n_1,n_0$ 为阳性、阴性样本数。

### 3.3 参数或统计量含义

- TPR：敏感度（sensitivity / recall），真阳性率。
- FPR：$1-$ 特异度，假阳性率。
- AUC：区分度指标，与 Mann-Whitney U、Wilcoxon 秩和及生存分析中的 C-index 同源。
- Youden 指数 $J=\mathrm{TPR}-\mathrm{FPR}$：常用于选“最优阈值”。

### 3.4 关键假设

- 需要连续评分或预测概率，以及二分类金标准。
- ROC/AUC 只衡量“排序/区分”，与预测概率是否校准无关。
- AUC 对类别患病率不敏感（是其优点也是局限，见第 8 节）。

## 4. 手把手算例

两名阳性得分 $(0.9,0.6)$，两名阴性得分 $(0.7,0.2)$。四个阳性—阴性配对中：

- 0.9 胜过 0.7、0.2，共 2 次；
- 0.6 胜过 0.2，但输给 0.7，共 1 次。

因此
$$
AUC=\frac{3}{4}=0.75
$$
它表示随机抽取一阳一阴时，模型有 75% 概率把阳性排得更高；不表示“预测准确率 75%”，也不说明 0.8 风险真的有 80% 发病。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：模型输出的连续评分或预测概率。
- 因变量类型：二分类金标准（0/1）。
- 数据结构：每行一个个体，含预测分数与真实标签。
- 是否适合高维数据：ROC 只作用于最终评分，与特征维度无关。
- 是否适合缺失较多数据：需先得到完整的预测分数。
- 是否适合删失数据：不直接适用；时间到事件请用时依 ROC 或 C-index。
- 是否适合重复测量数据：需考虑聚类，独立性被破坏时 AUC 的方差估计要调整。

### 5.2 示例表格

| 个体 | 真实标签 y | 预测概率 p |
| --- | --- | --- |
| 1 | 1 | 0.92 |
| 2 | 0 | 0.31 |
| 3 | 1 | 0.55 |
| 4 | 0 | 0.10 |
| 5 | 1 | 0.48 |

### 5.3 输入与产出

#### 输入

- 输入数据：预测概率/评分向量、真实标签向量。
- 关键变量：预测分数、二分类结局。
- 需要预处理的内容：确保分数与标签一一对应；在独立测试集或交叉验证上评估。

#### 产出

- 模型对象/统计结果：ROC 曲线坐标、AUC 值。
- 参数估计：AUC 点估计。
- 预测结果：各阈值的敏感度/特异度、Youden 最优阈值。
- 不确定性指标：AUC 的 95% CI（DeLong 法或 bootstrap）、两模型 AUC 差异检验。

## 6. 适用场景

- 适合：比较区分能力、选择阈值前理解敏感度–特异度权衡、诊断试验评价。
- 不适合：只关心某一工作点的表现、严重类别不平衡（宜看 PR 曲线）、需要评估临床净获益（宜用决策曲线）。
- 使用前需要特别检查的点：是否在独立数据上评估、类别是否严重不平衡、是否还需要校准与净获益证据。

## 7. 实现

### 7.1 Python

常用包：

- `scikit-learn`

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

clf = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:, 1]

auc = roc_auc_score(yte, p)
fpr, tpr, thr = roc_curve(yte, p)
print("AUC:", round(auc, 3))
```

### 7.2 R

常用包：

- `pROC`

```r
library(pROC)

# p: 预测概率向量; y: 0/1 标签
roc_obj <- roc(response = y, predictor = p, ci = TRUE)
print(auc(roc_obj))
print(ci.auc(roc_obj))          # DeLong 95% CI
plot(roc_obj)

# 两模型 AUC 比较
# roc.test(roc_obj1, roc_obj2, method = "delong")
```

## 8. 结果如何解释

- 核心结果看什么：AUC 点估计及其 95% CI，配合 ROC 曲线形状。
- 每个主要参数如何解释：AUC=0.80 表示随机取一对阳/阴样本，模型对阳性打分更高的概率约为 80%。
- 临床或医学意义如何表达：AUC 反映“区分”能力，常用分级仅供参考（0.7–0.8 可接受、0.8–0.9 良好）；应结合具体阈值下的敏感度与特异度报告。
- 常见误读：把 AUC 当成准确率或校准指标；AUC 高不代表在某个临床阈值下有用，也不代表预测概率可信。

## 9. 假设诊断与稳健性

- 只在独立患者测试集计算，重复测量需聚类置信区间。
- 报告 bootstrap/DeLong 置信区间及模型差异。
- 同时检查 PR-AUC、校准和临床阈值。
- 按中心、时间和亚组评价谱偏倚。
- 避免用测试集选择最佳 Youden 阈值再报同集性能。

## 10. 推荐可视化

- ROC 曲线（标注 AUC 与 95% CI），必要时叠加多模型对比。
- 精确率–召回率（PR）曲线：类别不平衡时比 ROC 更能反映性能。
- 敏感度、特异度随阈值变化的双曲线图，辅助阈值选择。

## 11. 优势、局限与常见坑

### 优势

- 阈值无关，概括整体区分能力，便于跨模型比较。
- 与 Mann-Whitney U / C-index 有清晰概率解释。
- 不随类别患病率变化，跨人群相对稳定。

### 局限

- 只反映区分度，不反映校准与临床净获益。
- 在严重类别不平衡下会过于乐观，掩盖大量假阳性。
- 对整体曲线取面积，可能包含临床上无意义的阈值区间。

### 常见坑

- 在训练集上报告 AUC，导致乐观偏倚。
- 用 AUC 单独下“模型可用”的结论，忽略校准（见 [[校准曲线（Calibration Curve）]]）与净获益（见 [[决策曲线分析（Decision Curve Analysis, DCA）]]）。
- 比较两个 AUC 时未用 DeLong 检验或配对结构。

## 12. 与相近方法的区别

- 和 [[校准曲线（Calibration Curve）]] 的区别：ROC/AUC 评“排序”，校准评“预测概率是否准”，两者互补且都需报告。
- 和 [[决策曲线分析（Decision Curve Analysis, DCA）]] 的区别：DCA 引入阈值概率权衡真假阳性，评估临床净获益，AUC 不含成本权衡。
- 和 PR 曲线的区别：PR 曲线聚焦阳性类，类别不平衡时更敏感。

## 13. 医学研究中的典型应用

- 评价新生物标志物相对既有指标的诊断增量。
- 临床风险预测模型（如术后并发症、再入院）区分度报告。
- 影像/组学分类模型在外部验证集上的区分度评估。

## 14. 关键术语

- **TPR/灵敏度**：阳性中被正确判阳的比例。
- **FPR**：阴性中被误判阳的比例。
- **区分度（Discrimination）**：阳性是否总体排在阴性前。
- **Youden 指数**：灵敏度+特异度−1。
- **谱偏倚（Spectrum bias）**：病例难度/构成改变性能。

## 15. 相关方法

- [[校准曲线（Calibration Curve）]]
- [[决策曲线分析（Decision Curve Analysis, DCA）]]
- [[Logistic回归（Logistic Regression）]]
- [[Wilcoxon秩和检验（Wilcoxon Rank-Sum Test）]]
- [[交叉验证（Cross-Validation）]]

## 16. 参考资料

- Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*. 1982;143(1):29-36.
- DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated ROC curves: a nonparametric approach. *Biometrics*. 1988;44(3):837-845.
- Steyerberg EW. *Clinical Prediction Models*. 2nd ed. Springer; 2019.
- scikit-learn. Metrics: roc_auc_score. [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html) （访问日期：2026-07-02）
