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

### 1.1 一句话本质

树模型选择按特征在分裂中减少多少预测误差或打乱后损失多少性能来排序，能捕捉非线性与交互，但重要性会受基数和相关性偏倚。

### 1.2 定义

它用决策树、随机森林或梯度提升拟合结局，再按 impurity decrease、gain 或 permutation importance 选择特征。属于嵌入式/模型后选择，结果与树算法、超参数和重要性定义绑定。

### 1.3 它主要解决什么问题

- 非线性、阈值和交互明显的表格数据选择。
- 从大量混合类型变量中建立候选子集。
- 医学场景：EHR、组学、影像组学与临床风险预测。

### 1.4 直觉与类比

树不断问问题来把患者分得更纯。一个变量若经常让错误大幅下降，就像一项高效分诊问题；但若两个问题答案相同，树只问其中一个，另一个可能被误判为“不重要”。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

线性稀疏方法需要预先指定变换和交互。树通过阈值分裂自动发现“年龄大于 70 且乳酸高”等规则，选择标准与非线性预测直接对齐。

### 2.2 关键洞察

每个被选分裂都带来局部损失下降。把某特征在全树/全森林的下降加总，就是 MDI/gain；或在验证集打乱某列，性能下降越多，说明模型越依赖该列。

### 2.3 与朴素/相邻做法的对比

- MDI 计算快但偏爱连续、高基数特征，并在相关变量间任意分配。
- permutation importance 更贴近验证性能，但相关特征互相替代时也会低估。
- SHAP 可分解个体预测，不等同于稳定特征选择。

## 3. 数学形式

### 3.1 核心公式

节点 $t$ 的加权不纯度下降：

$$
\Delta I(t)
=\frac{N_t}{N}I(t)
-\frac{N_L}{N}I(L)
-\frac{N_R}{N}I(R)
$$

特征 $j$ 的 MDI：

$$
\operatorname{Imp}(j)
=\sum_{t:v(t)=j}\Delta I(t)
$$

置换重要性：

$$
\operatorname{PI}_j
=M(\mathbf X,y)-M(\mathbf X_{\pi(j)},y)
$$

这个式子在说：分裂时累计“纯度收益”，或打乱一列后看验证性能损失。

### 3.2 推导脉络

分类树选择最大 Gini/熵下降，回归树选择最大 SSE 下降。集成模型将许多树的贡献平均。选择阈值可用均值/中位数重要性、Top-k 或验证集性能决定。

### 3.3 参数与统计量含义

- $I(t)$：节点 Gini、熵或 SSE。
- $N_t/N$：节点覆盖的总体比例。
- MDI/gain：训练结构内部的重要性。
- PI：依赖指定验证集和性能指标的模型依赖度。
- selection threshold：把连续重要性转为保留/删除的阈值。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后果 |
| --- | --- | --- |
| 树在独立数据上泛化 | 重要性来自真实模式 | 过拟合噪声 |
| 变量机会公平 | 基数/缺失处理不偏 | MDI 偏向多切点变量 |
| 相关替代被认识 | 低重要性可能因替代 | 错删冗余组全部 |
| 选择嵌入训练折 | 测试不参与重要性 | 泄漏 |

## 4. 手把手算例

8 人结局为 $(0,0,0,0,1,1,1,1)$。父节点：

$$
Gini_P=1-(4/8)^2-(4/8)^2=0.5
$$

特征 A 的阈值把前 4 个阴性与后 4 个阳性完全分开，两个子节点 Gini 都为 0：

$$
\Delta Gini_A
=0.5-\frac48(0)-\frac48(0)=0.5
$$

若整棵树只有这个分裂，归一化重要性为 A=1、其他=0。

现在加入 B=2A。B 也能完美分裂，但贪心树先选 A 后已无需 B，于是 B 的 MDI=0。结论不是“B 无信息”，而是“在 A 已存在时树不再需要 B”。相关变量的 masking 是树选择必须报告的局限。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 数值/编码类别表格，分类或回归结局。
- 缺失处理依具体实现；scikit-learn 多数树前需插补。
- 生存结局需随机生存森林等专用模型。
- 重复测量必须患者级切分。

### 5.2 示例表格

| patient_id | marker_A | marker_B | noise | severe |
| --- | ---: | ---: | ---: | ---: |
| P01 | 1 | 2 | 0.3 | 0 |
| P02 | 2 | 4 | 0.8 | 0 |
| P05 | 5 | 10 | 0.1 | 1 |

### 5.3 输入与产出

输入训练数据、树模型、重要性类型和阈值；产出拟合模型、重要性、支持集及其验证性能。

## 6. 适用场景

- 适合：复杂非线性、交互、混合尺度表格数据。
- 不适合：需要稳定方向性系数、小样本高维且树易过拟合。
- 最好比较 MDI、置换和跨折稳定性。

## 7. 实现

### 7.1 Python

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, stratify=y, random_state=42
)
rf = RandomForestClassifier(
    n_estimators=500, min_samples_leaf=5,
    random_state=42, n_jobs=-1
).fit(Xtr, ytr)
selector = SelectFromModel(rf, threshold="median", prefit=True)
print(X.columns[selector.get_support()].tolist())
pi = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=42)
print(pi.importances_mean)
```

### 7.2 R

```r
library(ranger)

set.seed(42)
fit <- ranger(
  Species ~ ., data = iris,
  num.trees = 500,
  importance = "permutation",
  min.node.size = 5
)
sort(fit$variable.importance, decreasing = TRUE)
selected <- names(fit$variable.importance)[
  fit$variable.importance > median(fit$variable.importance)
]
selected
```

## 8. 结果如何解读

高重要性表示当前树模型依赖该特征，不是因果效应。MDI 的单位是累计不纯度下降，PI 的单位是性能指标下降，二者不可直接混读。

## 9. 假设诊断与稳健性

- 在独立验证折计算 permutation importance。
- 重复 CV/Bootstrap 报告入选频率。
- 对相关特征做分组置换或聚类后选代表。
- 比较不同深度、叶节点大小和随机种子。
- 加入随机噪声列作为偶然重要性基线。
- 检查亚组、中心和时间漂移。

## 10. 推荐可视化

- MDI 与 PI 并列排序图。
- 各折重要性箱线图。
- 相关特征组网络/热图。
- 特征数—验证性能曲线。
- PDP/SHAP 仅作行为补充。

## 11. 优势、局限与常见坑

### 优势

- 自动捕捉非线性、阈值和交互。
- 无需连续变量标准化。

### 局限

- MDI 有基数偏倚，相关变量会相互遮蔽。
- 支持集对超参数和样本波动敏感。

### 常见坑

- 用训练集 PI 选择。
- 将 gain 当因果效应。
- 忽略患者级切分。
- 一次随机森林支持集直接定稿。

## 12. 与相近方法的区别

- Lasso 偏向线性稀疏；树选择适合非线性。
- 过滤式 MI 逐列评分；树重要性可反映交互中的使用。
- SHAP 解释预测分摊，树选择决定保留列。
- 如何选择：用任务结构决定模型，并以嵌套验证比较支持集。

## 13. 医学研究中的典型应用

- EHR 风险因素筛选。
- 影像组学和多组学预测变量压缩。
- 临床评分候选变量优先级排序。

需报告重要性定义、阈值、相关组、缺失处理、患者级切分和外部稳定性。

## 14. 关键术语

- **MDI**：平均不纯度下降重要性。
- **Permutation importance**：打乱特征后的性能损失。
- **Gain**：提升树中分裂带来的目标下降。
- **Masking**：相关特征使彼此重要性被低估。
- **分组置换（Grouped permutation）**：同时打乱一组相关特征。
- **支持集（Support）**：最终保留的变量集合。

## 15. 相关方法

- [[决策树（Decision Tree）]]
- [[随机森林（Random Forest）]]
- [[XGBoost（Extreme Gradient Boosting, XGBoost）]]
- [[嵌入式特征选择（Embedded Feature Selection）]]
- [[SHAP（Shapley Additive Explanations, SHAP）]]

## 16. 参考资料

- Breiman L. Random forests. *Machine Learning*. 2001;45:5-32.
- Strobl C, et al. Bias in random forest variable importance measures. *BMC Bioinformatics*. 2007;8:25.
- scikit-learn developers. Feature importance evaluation. https://scikit-learn.org/stable/modules/permutation_importance.html
