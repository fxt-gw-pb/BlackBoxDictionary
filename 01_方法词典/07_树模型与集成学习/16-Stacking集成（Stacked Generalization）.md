---
title: Stacking集成
english_name: Stacked Generalization
slug: stacked-generalization
aliases: [stacking, stacked generalization, 堆叠, "Stacking集成（Stacked Generalization）"]
category: 树模型与集成学习
subcategory: 模型融合
tags: [医学统计, 数据科学, 集成学习, 模型融合]
status: 已建
difficulty: intermediate
question_type: 多模型二层融合
data_type: [表格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [caretEnsemble, stacks]
---

# Stacking集成（Stacked Generalization）

## 1. 方法概览

### 1.1 定义

Stacking 是一种两层或多层模型融合方法。第一层训练多个基础模型，第二层用这些基础模型的交叉验证预测作为新特征，训练一个元学习器来学习如何组合它们。

### 1.2 它主要解决什么问题

- 研究问题：如何让不同类型模型的预测优势被一个元模型系统整合。
- 适用任务：二分类、多分类、连续结局预测。
- 常见医学场景：融合临床评分、机器学习模型、影像组学模型和组学模型的预测输出。

### 1.3 直觉理解

如果投票集成是“每个模型一票”，Stacking 则是让一个总模型学习“什么时候该更相信哪个模型”。关键是元学习器必须用 out-of-fold 预测训练，否则很容易发生信息泄露。

## 2. 数学形式

### 2.1 核心公式

设有 $M$ 个基础模型 $f_1,\dots,f_M$，元学习器为 $g$，最终预测为：

$$
\hat y = g(f_1(x), f_2(x), \dots, f_M(x))
$$

若元学习器为线性模型，则：

$$
\hat y=\beta_0+\sum_{m=1}^{M}\beta_m \hat y_m
$$

其中 $\hat y_m=f_m(x)$ 是第 $m$ 个基础模型的预测。实际训练时，$\hat y_m$ 应来自交叉验证的 out-of-fold 预测。

### 2.2 参数或统计量含义

- 基础模型：第一层模型，如随机森林、SVM、GBDT、Logistic 回归。
- 元学习器：第二层融合模型，如 Ridge、Logistic 回归、XGBoost。
- out-of-fold 预测：每个训练样本由未见过该样本的基础模型产生的预测。
- `cv`：生成元学习器训练数据的交叉验证折数。

### 2.3 关键假设

- 基础模型之间有互补信息。
- 元学习器训练过程严格避免信息泄露。
- 样本量足以支持多模型训练和二层融合。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：取决于基础模型，可融合多种特征来源。
- 因变量类型：二分类、多分类或连续型。
- 数据结构：监督学习数据。
- 是否适合高维数据：适合，但需要严格交叉验证和正则化。
- 是否适合缺失较多数据：需在各基础模型流程中处理。
- 是否适合删失数据：普通 Stacking 不直接适合，除非基础模型和元学习器均支持生存结局。
- 是否适合重复测量数据：需按个体分组交叉验证，避免同一患者数据泄露。

### 3.2 示例表格

以多模态住院死亡风险预测为例：

| Age | SOFA | LabScore | RadiologyScore | NoteScore | Death30d |
| --- | --- | --- | --- | --- | --- |
| 74 | 10 | 0.81 | 0.72 | 0.69 | 1 |
| 47 | 3 | 0.22 | 0.18 | 0.31 | 0 |
| 66 | 7 | 0.64 | 0.58 | 0.62 | 1 |
| 39 | 2 | 0.15 | 0.12 | 0.20 | 0 |
| 58 | 5 | 0.43 | 0.36 | 0.41 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：训练集特征、结局，以及多个基础模型配置。
- 关键变量：基础模型集合、元学习器、交叉验证方案、是否传递原始特征。
- 需要预处理的内容：分层或分组交叉验证、缺失处理、标准化、概率校准。

#### 产出

- 模型对象/统计结果：基础模型、元学习器、二层融合模型。
- 参数估计：若元学习器为线性模型，可解释为各基础模型预测的融合权重。
- 预测结果：类别、概率或连续预测值。
- 不确定性指标：交叉验证性能、外部测试性能、校准指标。

## 4. 适用场景

- 适合：多模型性能相近但错误模式不同、且预测性能优先的任务。
- 不适合：样本量小、计算资源有限、需要简单可解释模型的任务。
- 使用前需要特别检查的点：out-of-fold 预测是否正确生成、是否按患者分组拆分、元学习器是否过强。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv("multimodal_mortality.csv")
X = df[["Age", "SOFA", "LabScore", "RadiologyScore", "NoteScore"]]
y = df["Death30d"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

base_models = [
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
    ("gb", GradientBoostingClassifier(random_state=42)),
    ("svm", SVC(probability=True, random_state=42))
]

fit = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    stack_method="predict_proba",
    cv=5
)
fit.fit(X_train, y_train)

pred_prob = fit.predict_proba(X_test)[:, 1]
```

### 5.2 R

常用包：

- `caretEnsemble`

```r
library(caret)
library(caretEnsemble)

ctrl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = "final",
  classProbs = TRUE
)

models <- caretList(
  Death30d ~ Age + SOFA + LabScore + RadiologyScore + NoteScore,
  data = df_train,
  trControl = ctrl,
  methodList = c("glm", "rf", "gbm")
)

stack <- caretStack(models, method = "glm", metric = "ROC", trControl = ctrl)
pred <- predict(stack, newdata = df_test, type = "prob")
```

## 6. 结果如何解释

- 核心结果看什么：Stacking 是否优于最佳单模型、外部验证性能、概率校准。
- 每个主要参数如何解释：元学习器权重可反映基础模型预测在融合中的贡献，但不等于原始变量重要性。
- 临床或医学意义如何表达：适合描述为“融合多个数据源或模型的综合风险预测器”。
- 常见误读：把元学习器权重解释为临床变量效应，或忽略信息泄露带来的虚高性能。

## 7. 推荐可视化

- 单模型与 Stacking 模型性能对比图。
- 元学习器权重条形图。
- 不同基础模型预测概率相关性热图。

## 8. 优势、局限与常见坑

### 优势

- 能利用异质模型的互补性。
- 可融合不同模态或不同建模策略。
- 通常能提升预测性能上限。

### 局限

- 计算成本较高。
- 解释性较弱。
- 对交叉验证设计非常敏感。

### 常见坑

- 直接用基础模型在训练集上的拟合预测训练元学习器，造成信息泄露。
- 元学习器过于复杂，在小样本中严重过拟合。
- 同一患者多条记录被拆到不同折中，导致性能虚高。

## 9. 与相近方法的区别

- 和 [[投票集成（Voting Ensemble）]] 的区别：投票集成用固定规则合并预测；Stacking 用元学习器学习合并规则。
- 和 [[Bagging算法（Bootstrap Aggregating）]] 的区别：Bagging 多为同类模型加重抽样；Stacking 强调不同模型的二层融合。
- 和 Blending 的区别：Blending 通常用一个固定验证集训练元学习器，Stacking 通常使用交叉验证生成二层训练数据。

## 10. 医学研究中的典型应用

- 融合结构化 EHR、影像组学和文本模型输出。
- 对多个传统模型与机器学习模型进行二层集成。
- 疾病亚型或预后风险预测中的多数据源融合。

## 11. 相关方法

- [[投票集成（Voting Ensemble）]]
- [[Bagging算法（Bootstrap Aggregating）]]
- [[Boosting算法（Boosting）]]
- [[随机森林（Random Forest）]]

## 12. 参考资料

- Wolpert DH. Stacked generalization. *Neural Netw*. 1992;5(2):241-259.
- van der Laan MJ, Polley EC, Hubbard AE. Super learner. *Stat Appl Genet Mol Biol*. 2007;6(1):Article25.
- scikit-learn Developers. `sklearn.ensemble.StackingClassifier`. scikit-learn API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) （访问日期：2026-07-02）
