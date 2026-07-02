---
title: SHAP
english_name: Shapley Additive Explanations, SHAP
slug: shapley-additive-explanations-shap
aliases: [SHAP, Shapley value, SHAP值, "SHAP（Shapley Additive Explanations, SHAP）"]
category: 模型评估与解释
subcategory: 模型可解释性
tags: [医学统计, 数据科学, 模型解释, 特征归因, 可解释AI]
status: 已建
difficulty: intermediate
question_type: 模型预测的特征归因解释
data_type: [表格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [shap]
r_packages: [shapviz]
---

# SHAP（Shapley Additive Explanations, SHAP）

## 1. 方法概览

### 1.1 定义

SHAP 是一种基于博弈论 Shapley 值的特征归因方法，把某次预测相对基线的偏移，公平地分配到各个特征上，给出统一的局部与全局解释。

### 1.2 它主要解决什么问题

- 研究问题：黑箱模型为什么对这个病人预测了高风险，哪些特征在起作用、方向如何。
- 适用任务：单样本预测解释、全局特征重要性、特征效应形态与交互探索。
- 常见医学场景：为树/提升模型的风险预测提供可解释报告，辅助临床沟通与模型审查。

### 1.3 直觉理解

把预测看成一场“合作博弈”，每个特征是一名玩家，预测值是总收益。SHAP 通过考察“加入某特征前后预测的平均变化”，把总收益公平地分给每个特征，正值把预测往上推、负值往下拉。

## 2. 数学形式

### 2.1 核心公式

特征 $i$ 的 Shapley 值是其在所有特征子集中的平均边际贡献：

$$
\phi_i=\sum_{S\subseteq F\setminus\{i\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\big[f(S\cup\{i\})-f(S)\big]
$$

SHAP 满足可加性，单次预测可分解为：

$$
f(x)=\phi_0+\sum_{i=1}^{p}\phi_i
$$

其中 $\phi_0=E[f(X)]$ 是基线（平均预测）。

### 2.2 参数或统计量含义

- $\phi_0$：基线值，通常为背景数据上的平均预测。
- $\phi_i$：特征 $i$ 对本次预测的贡献（带符号）。
- $F,S$：全部特征集合与子集。
- 全局重要性：$\text{mean}(|\phi_i|)$，对所有样本取绝对值均值。

### 2.3 关键假设

- 满足局部准确性、缺失性、一致性三条公理。
- 归因是关联性解释，不等于因果效应。
- 精确 Shapley 计算随特征数指数增长，实际用 TreeSHAP（树模型精确）或 KernelSHAP（近似）。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：任意特征（连续、分类编码）。
- 因变量类型：回归或分类模型的输出。
- 数据结构：训练好的模型 + 需要解释的样本 + 背景数据集。
- 是否适合高维数据：适合，但 KernelSHAP 在高维下计算昂贵。
- 是否适合缺失较多数据：以模型能处理为前提；背景数据代表参考分布。
- 是否适合删失数据：可解释生存模型（如生存梯度提升）的风险评分。
- 是否适合重复测量数据：解释针对单次预测，聚类结构需在汇总时注意。

### 3.2 示例表格

| 特征 | 本次取值 | SHAP 值 φ | 方向 |
| --- | --- | --- | --- |
| worst radius | 25.4 | +1.82 | 推高风险 |
| mean texture | 21.9 | +0.63 | 推高风险 |
| mean smoothness | 0.09 | −0.41 | 拉低风险 |
| worst symmetry | 0.28 | −0.12 | 拉低风险 |

### 3.3 输入与产出

#### 输入

- 输入数据：已训练模型、待解释样本、背景/参考数据。
- 关键变量：特征矩阵、模型预测函数。
- 需要预处理的内容：特征与训练时一致；选择合适的 explainer。

#### 产出

- 模型对象/统计结果：每样本每特征的 SHAP 值矩阵。
- 参数估计：全局特征重要性排序。
- 预测结果：单样本贡献分解、特征效应曲线。
- 不确定性指标：近似方法的估计误差、背景集选择的敏感性。

## 4. 适用场景

- 适合：解释树/提升模型（TreeSHAP 高效且精确）、需要局部与全局一致解释、审查模型是否依赖不合理特征。
- 不适合：把归因当因果、特征高度相关时逐特征解读、追求实时解释而用昂贵的 KernelSHAP。
- 使用前需要特别检查的点：背景数据是否代表目标人群、特征相关性、解释目的是理解模型还是理解现象。

## 5. 实现

### 5.1 Python

常用包：

- `shap`

```python
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
model = RandomForestClassifier(n_estimators=300, random_state=0).fit(X, y)

explainer = shap.TreeExplainer(model)
sv = explainer(X)                 # 每样本每特征的 SHAP 值
shap.plots.beeswarm(sv[:, :, 1])  # 全局解释（阳性类）
shap.plots.waterfall(sv[0, :, 1]) # 单样本解释
```

### 5.2 R

常用包：

- `shapviz`

```r
library(shapviz)
library(xgboost)

# X: 特征矩阵; model: 已训练 xgboost 模型
shp <- shapviz(model, X_pred = as.matrix(X), X = X)
sv_importance(shp)          # 全局重要性
sv_waterfall(shp, row_id = 1)  # 单样本解释
```

## 6. 结果如何解释

- 核心结果看什么：全局 mean(|SHAP|) 重要性排序、依赖图的效应方向与形状、单样本贡献分解。
- 每个主要参数如何解释：某特征 SHAP 为正表示在本次预测中把输出推高，绝对值越大影响越强。
- 临床或医学意义如何表达：说明“模型如何用这些特征”，不能直接读成“改变该特征就会改变结局”。
- 常见误读：把 SHAP 当因果效应；在强相关特征间比较重要性；忽略背景集选择对基线的影响。

## 7. 推荐可视化

- beeswarm 蜂群图：全局特征重要性与效应方向。
- dependence 依赖图：单特征效应形态及交互。
- waterfall / force 图：单样本预测的贡献分解。

## 8. 优势、局限与常见坑

### 优势

- 有博弈论公理支撑，局部与全局解释统一。
- TreeSHAP 对树模型精确且高效。
- 可加性使单样本解释直观可读。

### 局限

- 归因是关联性，不等于因果。
- 特征相关时归因会在相关特征间分摊，解读需谨慎。
- KernelSHAP 计算昂贵，高维实时场景受限。

### 常见坑

- 用 SHAP 结论指导干预（当成因果）。
- 背景数据集选得不合理，导致基线和归因失真。
- 只看全局重要性，忽略效应方向与交互。

## 9. 与相近方法的区别

- 和 [[基于树模型的特征选择（Tree-based Feature Selection）]] 的区别：树的分裂重要性偏向高基数/高方差特征，SHAP 归因更一致且带方向。
- 和置换重要性的区别：置换重要性衡量打乱特征后性能下降，SHAP 分解到单次预测；相关特征下二者可能不一致。
- 和 [[线性回归（Linear Regression）]] 系数的区别：线性系数是全局线性效应，SHAP 适用于任意复杂模型的局部归因。

## 10. 医学研究中的典型应用

- 为风险预测模型（树/提升模型）提供可解释报告，辅助临床采纳。
- 审查模型是否依赖了泄漏变量或不合理代理特征。
- 探索非线性效应与特征交互，生成后续研究假设。

## 11. 相关方法

- [[基于树模型的特征选择（Tree-based Feature Selection）]]
- [[随机森林（Random Forest）]]
- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]
- [[ROC曲线与AUC（Receiver Operating Characteristic and AUC）]]

## 12. 参考资料

- Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *NeurIPS*. 2017;30:4765-4774.
- Lundberg SM, Erion G, Chen H, et al. From local explanations to global understanding with explainable AI for trees. *Nat Mach Intell*. 2020;2(1):56-67.
- shap. Documentation. [https://shap.readthedocs.io/](https://shap.readthedocs.io/) （访问日期：2026-07-02）
