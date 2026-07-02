---
title: 基于树模型的特征选择
english_name: Tree-based Feature Selection
slug: tree-based-feature-selection
aliases: [tree-based feature selection, tree feature importance selection, "基于树模型的特征选择（Tree-based Feature Selection）"]
category: 数据预处理与特征工程
subcategory: 嵌入式特征选择
tags: [医学统计, 数据科学, 特征选择, 树模型, 集成学习]
status: 已建
difficulty: intermediate
question_type: 基于树模型重要性的变量筛选
data_type: [表格数据, 高维特征矩阵]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [ranger, randomForest, vip]
---

# 基于树模型的特征选择（Tree-based Feature Selection）

## 1. 方法概览

### 1.1 定义

基于树模型的特征选择利用决策树、随机森林、梯度提升树等模型产生的特征重要性，对特征进行排序或筛选。常用重要性包括不纯度减少、分裂次数、置换重要性和 SHAP 重要性。

### 1.2 它主要解决什么问题

- 研究问题：哪些特征在树模型预测中贡献较大。
- 适用任务：分类、回归、非线性预测、高维特征筛选。
- 常见医学场景：临床风险预测变量筛选、影像组学特征排序、生物标志物优先级排序。

### 1.3 直觉理解

树模型会反复选择能最好划分样本的特征。一个特征如果经常用于有效分裂，或打乱它会明显降低模型性能，就说明模型很依赖它。

## 2. 数学形式

### 2.1 核心公式

以不纯度减少为例，节点 $t$ 的不纯度为 $i(t)$，分裂为左右子节点后，特征 $X_j$ 在该节点带来的减少量为：

$$
\Delta i(t)=i(t)-\frac{n_L}{n_t}i(t_L)-\frac{n_R}{n_t}i(t_R)
$$

特征重要性可写为该特征在所有相关节点上的加权累计：

$$
I_j=\sum_{t:v(t)=j}\frac{n_t}{n}\Delta i(t)
$$

随机森林或梯度提升树中，通常再对多棵树的重要性求和或平均。置换重要性则比较置换某特征前后的模型性能：

$$
PI_j=\mathrm{Score}(f,X,y)-\mathrm{Score}(f,X_{\pi(j)},y)
$$

### 2.2 参数或统计量含义

- $i(t)$：节点不纯度，如 Gini、熵或回归方差。
- $\Delta i(t)$：一次分裂带来的不纯度减少。
- $I_j$：特征 $j$ 的累计重要性。
- 置换重要性：打乱某特征后模型性能下降幅度。
- 阈值：选择特征的重要性下限，如均值、中位数或累计贡献率。

### 2.3 关键假设

- 树模型适合当前数据结构。
- 特征重要性可代表模型对该变量的依赖程度。
- 重要性解释应限定在模型预测语境中，不能直接解释为因果效应。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：连续、二分类、多分类变量。
- 因变量类型：二分类、多分类或连续型。
- 数据结构：监督学习宽表数据。
- 是否适合高维数据：可用，但强相关特征会分散重要性。
- 是否适合缺失较多数据：按树模型要求处理，部分实现支持缺失。
- 是否适合删失数据：普通树模型不直接适合，需生存树或生存森林。
- 是否适合重复测量数据：需分组验证或专门处理相关结构。

### 3.2 示例表格

以住院患者恶化预测为例：

| Age | NEWS2 | Lactate | Albumin | WBC | Deterioration |
| --- | --- | --- | --- | --- | --- |
| 78 | 8 | 3.2 | 29 | 14.2 | 1 |
| 49 | 2 | 1.1 | 41 | 6.3 | 0 |
| 66 | 5 | 2.4 | 34 | 10.8 | 1 |
| 37 | 1 | 0.8 | 43 | 5.8 | 0 |
| 57 | 4 | 1.6 | 38 | 8.4 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：特征矩阵和目标变量。
- 关键变量：树模型类型、重要性类型、选择阈值。
- 需要预处理的内容：缺失处理、类别编码、训练测试划分、重采样稳定性评估。

#### 产出

- 模型对象/统计结果：树模型、特征重要性排序、保留特征集合。
- 参数估计：重要性得分。
- 预测结果：若直接使用树模型，也可输出预测。
- 不确定性指标：交叉验证性能、置换重要性分布、bootstrap 入选频率。

## 4. 适用场景

- 适合：非线性和交互明显、希望用树模型辅助筛选特征的任务。
- 不适合：强因果解释、需要无模型偏倚变量筛选、样本量很小的任务。
- 使用前需要特别检查的点：重要性指标类型、相关特征组、连续变量或高基数分类变量的重要性偏倚。

## 5. 实现

### 5.1 Python

常用包：

- `scikit-learn`

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

df = pd.read_csv("deterioration.csv")
X = df[["Age", "NEWS2", "Lactate", "Albumin", "WBC"]]
y = df["Deterioration"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

selector = SelectFromModel(rf, threshold="mean", prefit=True)
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importance)
print(X.columns[selector.get_support()].tolist())
```

### 5.2 R

常用包：

- `ranger`
- `vip`

```r
library(ranger)

fit <- ranger(
  Deterioration ~ Age + NEWS2 + Lactate + Albumin + WBC,
  data = df_train,
  probability = TRUE,
  num.trees = 500,
  importance = "permutation"
)

sort(fit$variable.importance, decreasing = TRUE)
```

## 6. 结果如何解释

- 核心结果看什么：重要性排序、不同重采样下是否稳定、筛选后模型性能。
- 每个主要参数如何解释：阈值越高保留越少；置换重要性越大说明模型越依赖该变量。
- 临床或医学意义如何表达：可以说模型预测中该变量贡献较大，不能直接说它是病因。
- 常见误读：把 impurity importance 直接当作无偏变量贡献。

## 7. 推荐可视化

- 特征重要性条形图。
- 置换重要性箱线图。
- 不同重采样下 Top-K 特征入选频率图。

## 8. 优势、局限与常见坑

### 优势

- 能捕捉非线性和交互。
- 与树模型预测目标一致。
- 可处理较多候选特征。

### 局限

- 重要性受模型和参数影响。
- 强相关特征会共享或稀释重要性。
- impurity importance 对连续变量和高基数特征可能有偏。

### 常见坑

- 只报告一次训练得到的重要性，不评估稳定性。
- 在全数据上选择特征后再评估模型。
- 把特征重要性写成因果或独立效应。

## 9. 与相近方法的区别

- 和 [[随机森林（Random Forest）]] 的区别：随机森林是建模方法；本卡强调用树模型输出做特征选择。
- 和 [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]] 的区别：GBDT 可作为基模型之一；本卡覆盖多类树模型选择流程。
- 和 [[嵌入式特征选择（Embedded Feature Selection）]] 的区别：树模型选择是嵌入式特征选择的一个分支。

## 10. 医学研究中的典型应用

- EHR 表格数据中筛选关键风险预测变量。
- 影像组学高维特征排序和降维。
- 多生物标志物组合预测中的候选特征优先级评估。

## 11. 相关方法

- [[嵌入式特征选择（Embedded Feature Selection）]]
- [[随机森林（Random Forest）]]
- [[梯度提升决策树（Gradient Boosting Decision Tree, GBDT）]]
- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]
- [[LightGBM（Light Gradient Boosting Machine）]]

## 12. 参考资料

- Breiman L. Random forests. *Mach Learn*. 2001;45:5-32.
- Strobl C, Boulesteix AL, Zeileis A, Hothorn T. Bias in random forest variable importance measures. *BMC Bioinformatics*. 2007;8:25.
- scikit-learn Developers. Feature selection using SelectFromModel. scikit-learn User Guide. [https://scikit-learn.org/stable/modules/feature_selection.html#select-from-model](https://scikit-learn.org/stable/modules/feature_selection.html#select-from-model) （访问日期：2026-07-02）
