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

### 1.1 一句话本质

Stacking 把多个基础模型的 out-of-fold 预测当作新特征，再训练一个元学习器决定何时、以多大程度相信每个模型。

### 1.2 定义

Stacking 是两层或多层模型融合方法。第一层包含若干基础学习器，第二层元学习器根据第一层预测形成最终输出。训练元学习器时，必须让每个训练病例的一级预测来自未见过该病例的模型。

### 1.3 它主要解决什么问题

- 固定平均无法适应模型间系统性差异。
- 不同算法或模态在不同数据区域各有优势。
- 希望从多个预测器中学习数据驱动的组合规则。

### 1.4 直觉与类比

投票像固定会诊规则；Stacking 像训练一位总诊医生，根据过往“各专家在未见病例上的表现”学习组合。若总诊医生看的是专家对自己熟悉病例的答案，表现会被严重高估。

## 2. 核心思想与原理

### 2.1 OOF 预测是关键

将训练集分为 $K$ 折。每次用 $K-1$ 折训练基础模型，对留出折预测；拼接后，每名患者都有一个由“未见过自己”的模型产生的预测，这才是元学习器合法的训练特征。

### 2.2 为什么不能用拟合值

复杂基础模型在训练集上的概率可能接近 0 或 1。若元模型学习这些拟合值，它学到的是过拟合能力，而非新患者上的预测能力，构成二层信息泄漏。

### 2.3 元学习器选择

Logistic、Ridge 或非负线性组合通常是稳健起点。元学习器越复杂，对样本量和嵌套验证要求越高。基础模型已很复杂时，第二层往往宜简单。

## 3. 数学形式

### 3.1 二层表示

对第 $i$ 个病例，基础模型 $m$ 的 OOF 预测记为 $z_{im}$：

$$
z_{im}=f_m^{(-k(i))}(x_i)
$$

其中 $f_m^{(-k(i))}$ 未使用病例 $i$ 所在折训练。

元学习器为：

$$
\hat y_i=g(z_{i1},z_{i2},\ldots,z_{iM})
$$

### 3.2 线性元学习器

回归任务可写为：

$$
\hat y_i=\beta_0+\sum_{m=1}^{M}\beta_mz_{im}
$$

分类任务可用：

$$
\operatorname{logit}[P(Y_i=1)]
=\beta_0+\sum_{m=1}^{M}\beta_mz_{im}
$$

### 3.3 完整训练流程

1. 生成所有训练病例的 OOF 预测矩阵 $Z$。
2. 用 $(Z,y)$ 训练元学习器。
3. 在全部训练数据上重训每个基础模型。
4. 对新病例生成一级预测，再交给已训练的元学习器。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| OOF 预测严格无泄漏 | 性能虚高 | 保存每折索引与预测来源 |
| 各模型使用相同折 | 二层特征不可比 | 固定共享重采样方案 |
| 样本量支持二层学习 | 元学习器不稳定 | 重复嵌套交叉验证 |
| 患者或中心分组正确 | 同一主体跨折泄漏 | Group/cluster split |

## 4. 手把手算例

二分类任务有 4 名训练患者。模型 A、B 的 OOF 阳性概率与结局为：

| 患者 | $z_A$ | $z_B$ | $y$ |
| --- | ---: | ---: | ---: |
| 1 | 0.10 | 0.30 | 0 |
| 2 | 0.40 | 0.20 | 0 |
| 3 | 0.60 | 0.80 | 1 |
| 4 | 0.90 | 0.70 | 1 |

为便于手算，令元模型为凸组合：

$$
\hat p=wz_A+(1-w)z_B
$$

最小化平方误差时：

$$
w=
\frac{\sum_i(z_{Ai}-z_{Bi})(y_i-z_{Bi})}
{\sum_i(z_{Ai}-z_{Bi})^2}
$$

分子为 $0.04$，分母为 $0.16$，所以 $w=0.25$。元模型给 A 权重 0.25、B 权重 0.75，预测为：

$$
\hat p=(0.25,0.25,0.75,0.75)
$$

Brier 分数为：

$$
\frac{0.25^2+0.25^2+(1-0.75)^2+(1-0.75)^2}{4}
=0.0625
$$

模型 A 的 Brier 为 0.085，模型 B 为 0.065，因此这个简单 Stack 略有改善。

**关键点：** 表中的 $z_A,z_B$ 必须是 OOF 预测。若是对训练集的拟合概率，这个权重和性能都不可信。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 分类或连续结局均可，但所有基础模型必须针对同一预测时点和结局。
- 可融合不同特征模态，也可只融合不同算法。
- 缺失处理、标准化、特征选择必须在各折内部完成。

### 5.2 输入与产出

输入包括基础模型、共享交叉验证方案、元学习器及是否传递原始特征。输出包括完整基础模型、元模型和最终预测。线性元模型权重表示一级预测的组合贡献，不是临床变量效应。

## 6. 适用场景

- 多个合格模型具有互补错误模式。
- 多模态模型已分别开发，需要在预测层融合。
- 不适合样本很小、计算资源有限或无法实施严格嵌套验证的任务。

## 7. 实现

### 7.1 Python

```python
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_models = [
    ("rf", RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, random_state=42
    )),
    ("gb", GradientBoostingClassifier(random_state=42)),
    ("svm", make_pipeline(
        StandardScaler(),
        SVC(C=1, probability=True, random_state=42)
    )),
]

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=2000),
    stack_method="predict_proba",
    cv=cv,
    passthrough=False,
    n_jobs=-1,
)
stack.fit(X_train, y_train)
prob = stack.predict_proba(X_test)[:, 1]
```

### 7.2 R

```r
library(caret)
library(caretEnsemble)

set.seed(42)
fold_index <- createFolds(train$Death30d, k = 5, returnTrain = TRUE)
ctrl <- trainControl(
  method = "cv",
  index = fold_index,
  classProbs = TRUE,
  savePredictions = "final",
  summaryFunction = twoClassSummary
)

models <- caretList(
  Death30d ~ .,
  data = train,
  trControl = ctrl,
  metric = "ROC",
  methodList = c("glm", "rf", "gbm")
)

stack <- caretStack(
  models,
  method = "glm",
  metric = "ROC",
  trControl = ctrl
)
prob <- predict(stack, newdata = test, type = "prob")
```

## 8. 结果如何解释

- 先报告每个基础模型、简单平均与 Stacking 的独立测试性能。
- 元模型权重反映基础预测的条件组合，不等于对应模态或变量的因果重要性。
- 负权重可能抵消高度相关模型；是否允许应在建模方案中预先说明。
- 分类模型仍须检查概率校准与临床阈值。

## 9. 诊断与稳健性

1. 审计每个 OOF 预测的训练索引，确认患者未被看见。
2. 比较基础预测相关性和模型间分歧。
3. 重复嵌套交叉验证，观察元权重与性能稳定性。
4. 比较简单平均、受约束线性组合和复杂元学习器。
5. 进行时间外、中心外、亚组与校准评估。

## 10. 推荐可视化

- 单模型、平均模型与 Stacking 的性能对比。
- OOF 预测相关性热图和散点矩阵。
- 元学习器权重图。
- 校准曲线、决策曲线和折间性能分布。

## 11. 优势、局限与常见坑

**优势：** 能融合异质模型和模态，可学习比固定平均更灵活的组合。

**局限：** 计算成本高、解释复杂、对验证设计极敏感。

**常见坑：** 用训练拟合值训练元模型；不同模型使用不同折；同一患者跨折；元学习器过强；调参后直接报告同一交叉验证性能。

## 12. 与相近方法的区别

- [[投票集成（Voting Ensemble）]]：固定合并规则，简单且样本效率更高。
- Blending：通常只用单个留出集训练元模型，浪费部分训练数据。
- [[Bagging算法（Bootstrap Aggregating）]]：通过重抽样聚合同类模型，重点是降方差。
- 选择经验：先建立简单平均基线；只有 OOF Stacking 在严格验证中稳定改善时才采用。

## 13. 医学研究中的典型应用

- 融合结构化 EHR、影像、文本和组学模型。
- 汇总传统风险评分、统计模型与机器学习模型。
- 多中心或多设备预测器的二层融合。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| base learner | 第一层基础模型 |
| meta-learner | 学习如何组合一级预测的第二层模型 |
| out-of-fold prediction | 由未见过该病例的折外模型生成的预测 |
| passthrough | 将原始特征与一级预测一同交给元模型 |
| Super Learner | 以交叉验证风险选择或组合学习器的框架 |

## 15. 相关方法

- [[投票集成（Voting Ensemble）]]
- [[Bagging算法（Bootstrap Aggregating）]]
- [[Boosting算法（Boosting）]]
- [[随机森林（Random Forest）]]

## 16. 参考资料

- Wolpert DH. Stacked generalization. *Neural Netw*. 1992;5(2):241-259.
- van der Laan MJ, Polley EC, Hubbard AE. Super learner. *Stat Appl Genet Mol Biol*. 2007;6(1):Article25.
- scikit-learn Developers. `StackingClassifier` API Reference. [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) （访问日期：2026-07-09）
