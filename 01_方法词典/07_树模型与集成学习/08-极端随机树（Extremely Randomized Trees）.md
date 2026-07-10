---
title: 极端随机树
english_name: Extremely Randomized Trees
slug: extremely-randomized-trees
aliases: [extra trees, ExtraTrees, extremely randomized trees, "极端随机树（Extremely Randomized Trees）"]
category: 树模型与集成学习
subcategory: Bagging集成
tags: [医学统计, 数据科学, 集成学习, 树模型, 随机森林]
status: 已建
difficulty: intermediate
question_type: 高随机性树集成预测
data_type: [表格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [ranger, extraTrees]
---

# 极端随机树（Extremely Randomized Trees）

## 1. 方法概览

### 1.1 一句话本质

极端随机树比随机森林更随机：每个节点不仅随机抽特征，还随机抽切分阈值，再从随机候选中挑一个较好的。

### 1.2 定义

极端随机树，也称 Extra Trees，是一种树集成方法。它与随机森林相似，都训练多棵随机树并聚合预测；区别在于 Extra Trees 在节点分裂时随机生成候选阈值，而不是像随机森林那样在候选特征上穷举寻找最佳阈值。

### 1.3 它主要解决什么问题

- 研究问题：如何进一步降低树与树之间的相关性，并加快树集成训练。
- 适用任务：二分类、多分类、连续结局预测、中高维表格数据基线模型。
- 常见医学场景：影像组学表格特征分类、肿瘤良恶性判别、复杂临床风险预测。

### 1.4 直觉与类比

随机森林像让每位专家随机看一部分指标，然后认真找最佳阈值；Extra Trees 像让专家还要随机抽几个阈值，只能在这些阈值里挑。单个专家可能更粗糙，但专家之间更不一样，平均后可能更稳、更快。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

随机森林已经降低了单棵树方差，但每个节点仍会在候选特征上寻找最优切分。若数据中有强变量，不同树仍可能选择相似阈值，相关性依然偏高。Extra Trees 解决的是：**如何用更强随机性进一步打散树之间的相似性，并减少寻找最佳阈值的计算成本**。

### 2.2 关键洞察

Extra Trees 的关键洞察是接受单棵树更粗糙，让集成更分散。随机阈值会增加单树偏差，但显著降低树间相关性和计算量；当多棵树平均后，整体方差可能下降，泛化表现反而更好。

### 2.3 与朴素/相邻做法的对比

- 相对 [[随机森林（Random Forest）]]：Extra Trees 额外随机化切分点，树间差异更大。
- 相对 [[Bagging算法（Bootstrap Aggregating）]]：Extra Trees 不只是抽样本，还随机特征和阈值。
- 相对单棵 [[决策树（Decision Tree）]]：Extra Trees 完全依赖多树集成，单棵树解释价值较低。

## 3. 数学形式

### 3.1 核心公式

分类预测仍是投票：

$$
\hat y=
\operatorname*{arg\,max}_{k}
\sum_{b=1}^{B}I\{T_b(x)=k\}
$$

回归预测为平均：

$$
\hat f(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x)
$$

在某个节点，Extra Trees 从随机特征 $j$ 的当前取值范围 $[a_j,b_j]$ 中随机抽切分点：

$$
s_j\sim U(a_j,b_j)
$$

再从这些随机候选切分中选出 Gini、熵或 MSE 改善最好的一个。

这个式子在说：阈值不是穷举搜索出来的，而是随机抽出来后再比较。

### 3.2 推导脉络

1. 对每棵树选训练样本；经典 Extra Trees 可使用整个训练集，也可配合 bootstrap。
2. 每个节点随机抽取若干候选特征。
3. 对每个候选特征，在当前样本取值范围内随机生成一个或多个阈值。
4. 计算这些随机阈值带来的不纯度或 MSE 改善。
5. 选择随机候选中最好的分裂，继续长树，最后多树投票或平均。

### 3.3 参数与统计量含义

- `n_estimators`：树的数量。
- `max_features`：每个节点随机候选特征数。
- 随机阈值：Extra Trees 与随机森林的核心差异。
- `min_samples_leaf`：叶节点最小样本数，影响概率或连续预测平滑。
- 特征重要性：可用 impurity 或 permutation importance，但需警惕偏倚。

### 3.4 关键假设(含违反后果)

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 更强随机性有益 | 降低相关性带来的收益超过偏差增加 | 欠拟合，性能不如随机森林 | 比较验证集误差 |
| 多树数量足够 | 随机阈值需靠大量树平均 | 预测不稳定 | 树数-误差曲线 |
| 叶节点不过小 | 随机分裂可能产生极小叶子 | 概率和均值噪声大 | 调大 `min_samples_leaf` |
| 训练数据代表目标人群 | 随机化不能解决分布漂移 | 外部性能下降 | 外部验证 |
| 解释以预测为主 | 单棵树随机性强 | 阈值不宜做医学规则 | 看全局解释而非单树 |

## 4. 手把手算例

用 6 名患者按 CRP 判断是否恶性，当前节点数据如下：

| 患者 | CRP | 恶性Y |
| --- | --- | --- |
| 1 | 10 | 0 |
| 2 | 20 | 0 |
| 3 | 30 | 0 |
| 4 | 60 | 1 |
| 5 | 70 | 1 |
| 6 | 90 | 1 |

随机森林会枚举多个可能阈值，如 15、25、45、65、80，找最优。Extra Trees 假设这次随机抽到两个候选阈值：25 和 75。

**Step 1：阈值 25 的 Gini。**

左侧 CRP 不超过 25：2 个非恶性，$G_L=0$。

右侧 4 人：1 个非恶性、3 个恶性：

$$
G_R=1-\left(\frac14\right)^2-\left(\frac34\right)^2=0.375
$$

加权 Gini：

$$
G_{25}=\frac26\times0+\frac46\times0.375=0.250
$$

**Step 2：阈值 75 的 Gini。**

左侧 5 人：3 个非恶性、2 个恶性：

$$
G_L=1-\left(\frac35\right)^2-\left(\frac25\right)^2=0.480
$$

右侧 1 人全恶性，$G_R=0$：

$$
G_{75}=\frac56\times0.480+\frac16\times0=0.400
$$

**Step 3：从随机候选中选择。**

Extra Trees 在这两个随机阈值中选择 Gini 更低的 25。它不会在本次节点中发现真正更好的阈值 45，除非 45 恰好被随机抽到或其他树抽到相近阈值。

**结论：** Extra Trees 的单棵树更随机、更粗糙；它靠很多棵树的平均，把这些随机阈值的误差摊开。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：连续、二分类、多分类变量均可。
- 因变量类型：二分类、多分类或连续型。
- 数据结构：宽表数据。
- 是否适合高维数据：适合中高维表格数据。
- 是否适合缺失较多数据：通常需先处理缺失。
- 是否适合删失数据：原始方法不直接适合。
- 是否适合重复测量数据：不直接适合。

### 5.2 示例表格

| Radius | Texture | Perimeter | Area | Smoothness | Malignant |
| --- | --- | --- | --- | --- | --- |
| 17.9 | 10.4 | 122.8 | 1001 | 0.118 | 1 |
| 12.3 | 15.7 | 82.6 | 477 | 0.089 | 0 |
| 19.7 | 21.3 | 130.0 | 1203 | 0.110 | 1 |
| 11.4 | 14.9 | 73.5 | 402 | 0.082 | 0 |
| 15.1 | 18.2 | 99.1 | 712 | 0.101 | 1 |

### 5.3 输入与产出

#### 输入

- 输入数据：特征矩阵和目标变量。
- 关键变量：树数、随机特征数、树深、叶节点最小样本数。
- 需要预处理的内容：缺失处理、类别编码、训练测试划分。

#### 产出

- 模型对象/统计结果：极端随机树集成、特征重要性。
- 参数估计：不提供传统回归系数。
- 预测结果：类别、概率或连续预测值。
- 不确定性指标：交叉验证性能、测试集 AUC/MSE、校准指标。

## 6. 适用场景

- 适合：需要快速训练、多特征、非线性关系明显且希望降低方差的任务。
- 不适合：强解释导向、样本极小、变量效应需要精确估计的任务。
- 使用前需要特别检查的点：是否过度随机导致欠拟合、特征重要性是否稳定、概率是否校准。

## 7. 实现

### 7.1 Python

常用包:

- `scikit-learn`

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

model = ExtraTreesClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, prob))
print(model.feature_importances_)
```

### 7.2 R

常用包:

- `ranger`

```r
library(ranger)

fit <- ranger(
  Malignant ~ Radius + Texture + Perimeter + Area + Smoothness,
  data = train_df,
  probability = TRUE,
  num.trees = 500,
  splitrule = "extratrees",
  min.node.size = 5,
  importance = "permutation"
)

pred <- predict(fit, data = test_df)$predictions[, "1"]
```

## 8. 结果如何解读

- 核心结果看什么：测试集性能、校准情况、变量重要性稳定性。
- 每个主要参数如何解释：`max_features` 控制随机候选特征数，`min_samples_leaf` 控制叶节点稳定性。
- 临床或医学意义如何表达：可作为稳健预测模型或变量筛选初探，不应把随机阈值解释为临床 cut-off。
- 常见误读：Extra Trees 训练快不代表一定比随机森林更准；需要实际验证比较。

## 9. 假设诊断与稳健性

- 与随机森林比较：同一数据上比较 AUC/RMSE、校准和训练时间。
- 树数敏感性：查看树数增加后性能是否稳定。
- 欠拟合检查：若随机阈值太粗，可增加树数、调整叶节点大小或改用随机森林。
- 重要性稳健性：用 permutation importance 和重采样评估排序。
- 概率校准：分类任务用校准曲线和 Brier score。

## 10. 推荐可视化

- 特征重要性条形图。
- ROC/PR 曲线和校准曲线。
- 树数与验证性能曲线。
- Extra Trees 与随机森林性能对比图。

## 11. 优势、局限与常见坑

### 优势

- 训练速度通常快于随机森林。
- 更强随机性可进一步降低树间相关性。
- 对非线性和交互建模能力强。

### 局限

- 单树偏差可能高于随机森林。
- 可解释性仍然有限。
- 概率输出常需校准。

### 常见坑

- 不调 `min_samples_leaf`，导致概率不稳定。
- 只看 impurity importance，忽视其对连续变量或高基数变量的偏倚。
- 在小样本中使用过多复杂树后只看内部验证结果。
- 把随机阈值解释成医学最佳阈值。

## 12. 与相近方法的区别

- 和 [[随机森林（Random Forest）]] 的区别：随机森林寻找候选特征中的最优切分点；Extra Trees 随机生成切分点后再选择。
- 和 [[Bagging算法（Bootstrap Aggregating）]] 的区别：Extra Trees 是树模型上的强随机化集成。
- 和 [[决策树（Decision Tree）]] 的区别：Extra Trees 通过多树平均或投票降低单树不稳定性。
- 和 [[随机森林回归（Random Forest Regression）]] 的关系：Extra Trees 也可用于回归，区别仍是随机阈值。
- 如何选择：若随机森林树间相关仍高或训练成本较大，可试 Extra Trees，并用验证集比较。

## 13. 医学研究中的典型应用

- 乳腺癌、肺结节等表格化影像特征分类。
- 多指标临床风险预测。
- 基因、蛋白或影像标志物初筛中的非线性模型基线。

## 14. 关键术语

- **极端随机树（Extremely Randomized Trees）**：随机特征和随机阈值共同生成的树集成。
- **随机阈值（Random Threshold）**：从特征取值范围中随机抽出的切分点。
- **树间相关性（Tree Correlation）**：不同树预测相似的程度。
- **强随机化（Strong Randomization）**：比随机森林更激进地引入随机性。
- **欠拟合（Underfitting）**：模型过于粗糙，训练和测试表现都差。
- **Permutation Importance**：打乱变量后观察性能下降的变量重要性。
- **Brier Score**：衡量概率预测误差的指标。

## 15. 相关方法

- [[随机森林（Random Forest）]]
- [[Bagging算法（Bootstrap Aggregating）]]
- [[决策树（Decision Tree）]]
- [[随机森林回归（Random Forest Regression）]]

## 16. 参考资料

- Geurts P, Ernst D, Wehenkel L. Extremely randomized trees. *Mach Learn*. 2006;63:3-42.
- Breiman L. Random forests. *Mach Learn*. 2001;45:5-32.
- Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. 2nd ed. Springer; 2009.
- Wright MN, Ziegler A. ranger: A fast implementation of random forests for high dimensional data in C++ and R. *J Stat Softw*. 2017;77(1):1-17.
