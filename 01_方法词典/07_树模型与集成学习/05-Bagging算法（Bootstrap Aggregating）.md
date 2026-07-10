---
title: Bagging算法
english_name: Bootstrap Aggregating
slug: bootstrap-aggregating
aliases: [bagging, bootstrap aggregating, "Bagging算法（Bootstrap Aggregating）"]
category: 树模型与集成学习
subcategory: Bagging集成
tags: [医学统计, 数据科学, 集成学习, 重抽样, 树模型]
status: 已建
difficulty: basic
question_type: 降低高方差模型的预测波动
data_type: [表格数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [ipred, caret]
---

# Bagging算法（Bootstrap Aggregating）

## 1. 方法概览

### 1.1 一句话本质

Bagging 让许多个「看过不同重抽样训练集」的模型各自预测，再投票或平均，以抵消单个模型的偶然波动。

### 1.2 定义

Bagging 是 Bootstrap Aggregating 的缩写。它先对训练集做多次 bootstrap 带放回抽样，每次训练一个基学习器，然后把这些基学习器的结果聚合：分类用多数投票或概率平均，回归用预测均值。

### 1.3 它主要解决什么问题

- 研究问题：单个高方差模型受训练样本扰动影响很大，如何让预测更稳定。
- 适用任务：二分类、多分类、连续结局预测、稳定化不稳定模型。
- 常见医学场景：并发症风险预测、复发风险分类、多个高噪声生物标志物组合预测。

### 1.4 直觉与类比

Bagging 像让多位医生分别看「略有不同的病例样本」训练出来的经验，再把他们的判断投票。某一位医生可能被自己见过的病例偏到一边，但很多人的偶然偏差平均后会小得多。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

像 [[决策树（Decision Tree）]] 这样的模型很灵活，也很不稳定：训练集中多几个或少几个病例，树的第一刀可能就变了。根本困难是：**高灵活模型偏差可能低，但方差高，单次训练得到的规则容易被样本噪声牵着走**。

### 2.2 关键洞察

Bagging 的关键洞察是：如果一个估计器容易摇摆，就训练很多个摇摆方向不同的版本，然后平均。只要这些模型的错误不是完全同步，多模型平均就能降低方差。Bootstrap 抽样提供了制造「不同版本训练集」的简单办法。

### 2.3 与朴素/相邻做法的对比

- 相对单棵树：Bagging 用多棵树投票或平均，显著降低方差。
- 相对简单重复训练同一模型：Bagging 每次用 bootstrap 样本，主动制造训练差异。
- 相对 [[Boosting算法（Boosting）]]：Bagging 并行训练，主要降方差；Boosting 串行纠错，主要降偏差。

## 3. 数学形式

### 3.1 核心公式

给定训练集 $D=\{(x_i,y_i)\}_{i=1}^{n}$，第 $b$ 次 bootstrap 样本记为 $D_b$，基学习器为 $h_b(x)$。

分类投票为：

$$
\hat y=\operatorname*{arg\,max}_{k}\sum_{b=1}^{B}I\{h_b(x)=k\}
$$

回归平均为：

$$
\hat f(x)=\frac{1}{B}\sum_{b=1}^{B}h_b(x)
$$

若每个基学习器方差为 $\sigma^2$，两两相关约为 $\rho$，平均后的方差近似为：

$$
\operatorname{Var}\{\hat f(x)\}
=
\sigma^2\left(\rho+\frac{1-\rho}{B}\right)
$$

这个式子在说：树越多，独立那部分方差越被平均掉；但树之间越相关，剩下的共同波动越难消掉。

### 3.2 推导脉络

1. 对原训练集做 $B$ 次带放回抽样，得到 $D_1,\ldots,D_B$。
2. 在每个 $D_b$ 上训练同类型基学习器 $h_b$。
3. 分类时统计各类别票数；回归时取预测均值。
4. 未被某次 bootstrap 抽中的样本可作为袋外样本，用于估计 OOB 误差。

### 3.3 参数与统计量含义

- $B$：基学习器数量，越大通常越稳定，但计算更慢。
- bootstrap 抽样：有放回抽样，大小常等于原训练集。
- OOB 样本：某棵树训练时没抽到的样本，可近似验证该树。
- 基学习器：被集成的模型，常见为深决策树。
- $\rho$：基学习器预测之间的相关性，越低集成收益越大。

### 3.4 关键假设(含违反后果)

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 基模型高方差 | 单模型对样本扰动敏感 | Bagging 收益明显 | 重采样看单模型波动 |
| 基模型错误不完全相关 | 不同模型犯错不完全同步 | 相关过高则平均收益小 | 比较基模型预测相关 |
| 训练样本代表目标人群 | bootstrap 只模拟样本波动 | 外部人群偏移仍会失败 | 外部验证 |
| 样本量足够 | 每次抽样能学到信号 | 小样本 OOB 不稳定 | 学习曲线 |
| 预测目标明确 | 聚合方式与结局类型匹配 | 概率或阈值解释错误 | 分类/回归指标分开报告 |

## 4. 手把手算例

### 4.1 分类投票

3 棵 bootstrap 决策树对 5 位患者是否 30 天并发症给出预测：

| 患者 | 树1 | 树2 | 树3 | Bagging投票 |
| --- | --- | --- | --- | --- |
| A | 1 | 1 | 0 | 1 |
| B | 0 | 0 | 1 | 0 |
| C | 1 | 1 | 1 | 1 |
| D | 0 | 1 | 0 | 0 |
| E | 1 | 0 | 0 | 0 |

以患者 A 为例，3 棵树中 2 棵投 1，所以集成预测为 1。患者 E 中只有 1 棵投 1，所以预测为 0。

### 4.2 方差降低

假设每棵树的预测方差 $\sigma^2=9$，树之间相关性 $\rho=0.2$，训练 $B=10$ 棵树：

$$
\operatorname{Var}(\hat f)=
9\left(0.2+\frac{1-0.2}{10}\right)
=9(0.28)=2.52
$$

单棵树方差为 9，Bagging 后约为 2.52。

**结论：** Bagging 的核心不是让每棵树都完美，而是让多棵不完全相同的树互相抵消偶然波动。树之间相关性越低、树数越多，集成越稳。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：连续、二分类、多分类变量均可，取决于基学习器。
- 因变量类型：二分类、多分类或连续型。
- 数据结构：宽表数据。
- 是否适合高维数据：可用，但需关注基学习器和样本量。
- 是否适合缺失较多数据：通常需先处理缺失。
- 是否适合删失数据：普通 Bagging 不直接处理删失结局。
- 是否适合重复测量数据：不直接适合，需要先处理相关结构或使用专门模型。

### 5.2 示例表格

| Age | ASA | SurgeryTime | Albumin | Emergency | Complication30d |
| --- | --- | --- | --- | --- | --- |
| 72 | 3 | 145 | 32 | 1 | 1 |
| 48 | 2 | 80 | 42 | 0 | 0 |
| 64 | 3 | 130 | 35 | 0 | 1 |
| 55 | 2 | 95 | 40 | 0 | 0 |
| 77 | 4 | 160 | 29 | 1 | 1 |

### 5.3 输入与产出

#### 输入

- 输入数据：特征矩阵和目标变量。
- 关键变量：基学习器类型、基学习器数量、每次抽样比例、是否 bootstrap。
- 需要预处理的内容：缺失处理、类别编码、训练测试集划分、类别不平衡处理。

#### 产出

- 模型对象/统计结果：Bagging 集成模型、OOB 误差、各基学习器预测。
- 参数估计：通常不输出传统系数。
- 预测结果：类别、概率或连续预测值。
- 不确定性指标：OOB 误差、交叉验证指标、测试集 AUC/F1/MSE。

## 6. 适用场景

- 适合：单模型不稳定、非线性关系明显、希望降低预测方差的任务。
- 不适合：基学习器本身非常稳定或偏差很高的任务，例如简单线性模型直接 Bagging 通常收益有限。
- 使用前需要特别检查的点：基模型是否足够多样、OOB 或外部验证表现、类别不平衡下阈值选择。

## 7. 实现

### 7.1 Python

常用包:

- `scikit-learn`

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

base = DecisionTreeClassifier(max_depth=None, min_samples_leaf=10)
model = BaggingClassifier(
    estimator=base,
    n_estimators=200,
    max_samples=0.8,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print(model.oob_score_)
prob = model.predict_proba(X_test)[:, 1]
```

### 7.2 R

常用包:

- `ipred`

```r
library(ipred)

fit <- bagging(
  Complication30d ~ Age + ASA + SurgeryTime + Albumin + Emergency,
  data = train_df,
  nbagg = 200,
  coob = TRUE
)

pred <- predict(fit, newdata = test_df, type = "prob")
```

## 8. 结果如何解读

- 核心结果看什么：测试集性能、OOB 误差、预测稳定性。
- 每个主要参数如何解释：树数越多通常越稳定；抽样比例影响基模型差异；基学习器复杂度影响偏差与方差。
- 临床或医学意义如何表达：Bagging 更适合作为稳健预测工具，不适合直接解释单个变量的独立效应。
- 常见误读：OOB 误差不能替代外部验证；投票结果也不保证概率已校准。

## 9. 假设诊断与稳健性

- OOB 误差曲线：观察树数增加后误差是否趋于稳定。
- 基学习器相关性：若树之间高度相似，Bagging 收益有限。
- 外部验证：OOB 是内部估计，医学模型仍需时间外或中心外验证。
- 类别不平衡：报告敏感度、特异度、AUC、PR-AUC 和校准，不只看准确率。
- 概率校准：集成投票概率可用校准曲线和 Brier score 检查。

## 10. 推荐可视化

- OOB 误差随树数变化曲线。
- ROC 曲线、PR 曲线和校准曲线。
- 各基模型预测分布与集成预测分布对比。
- bootstrap 样本覆盖示意图。

## 11. 优势、局限与常见坑

### 优势

- 能显著降低高方差模型的预测波动。
- 基学习器可并行训练。
- 调参压力通常低于 Boosting。

### 局限

- 主要降方差，对高偏差模型帮助有限。
- 解释性弱于单个模型。
- 基模型数量多时预测成本增加。

### 常见坑

- 对稳定模型机械使用 Bagging，收益不明显。
- 忽略类别不平衡和概率校准。
- 只报告准确率，不报告医学预测中更关键的灵敏度、特异度、校准和临床阈值表现。
- 把 OOB 表现当成外部验证。

## 12. 与相近方法的区别

- 和 [[随机森林（Random Forest）]] 的区别：随机森林是 Bagging 的特化版本，额外在每个节点随机抽取特征。
- 和 [[Boosting算法（Boosting）]] 的区别：Bagging 并行训练多个模型以降方差，Boosting 串行训练模型以逐步纠错。
- 和 [[投票集成（Voting Ensemble）]] 的区别：投票集成通常组合已训练的不同模型，Bagging 通过 bootstrap 自动生成多个训练版本。
- 如何选择：若单模型很不稳定且希望稳健平均，用 Bagging；若还想降低树间相关性，用随机森林。

## 13. 医学研究中的典型应用

- 临床表格数据风险预测的稳健基线模型。
- 病理、影像或组学特征较多时的分类任务。
- 小到中等样本量下对单棵树不稳定性的修正。

## 14. 关键术语

- **Bootstrap 抽样（Bootstrap Sampling）**：从训练集有放回抽样生成新训练集。
- **聚合（Aggregating）**：把多个模型预测合并为一个预测。
- **袋外样本（Out-of-Bag, OOB）**：某次 bootstrap 中未被抽中的样本。
- **基学习器（Base Learner）**：被 Bagging 复制训练的单个模型。
- **方差降低（Variance Reduction）**：通过平均多个不完全相关模型减少预测波动。
- **多数投票（Majority Vote）**：分类任务中票数最多的类别作为预测。
- **概率校准（Probability Calibration）**：检查预测概率是否接近真实发生率。

## 15. 相关方法

- [[随机森林（Random Forest）]]
- [[极端随机树（Extremely Randomized Trees）]]
- [[投票集成（Voting Ensemble）]]
- [[Boosting算法（Boosting）]]
- [[Bootstrap重抽样（Bootstrap Resampling）]]

## 16. 参考资料

- Breiman L. Bagging predictors. *Mach Learn*. 1996;24:123-140.
- Breiman L. Random forests. *Mach Learn*. 2001;45:5-32.
- Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. 2nd ed. Springer; 2009.
- James G, Witten D, Hastie T, Tibshirani R. *An Introduction to Statistical Learning*. 2nd ed. Springer; 2021.
