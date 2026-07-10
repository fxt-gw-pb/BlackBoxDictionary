---
title: 因子分析
english_name: Factor Analysis
slug: factor-analysis
aliases: [factor analysis, exploratory factor analysis, EFA, "因子分析（Factor Analysis）"]
category: 降维与表征学习
subcategory: 潜变量模型
tags: [医学统计, 数据科学, 降维, 潜变量, 心理测量]
status: 已建
difficulty: intermediate
question_type: 潜变量提取与量表结构分析
data_type: [表格数据, 量表数据, 高维特征矩阵]
outcome_type: [无监督表征, 多种]
python_packages: [factor_analyzer, scikit-learn]
r_packages: [psych, stats]
---

# 因子分析（Factor Analysis）

## 1. 方法概览

### 1.1 一句话本质

因子分析用少数不可观测公共因子解释多个指标为何相关，并把每个指标未被公共因子解释的变异单独留作特异误差。

### 1.2 定义

因子分析是线性潜变量模型。观测变量由公共因子的线性组合与变量特异误差构成；探索性因子分析（EFA）用于发现可能的因子结构，验证性因子分析用于检验预先规定结构。

### 1.3 它主要解决什么问题

- 多个题项背后是否存在少数潜在构念。
- 如何区分共同方差与题项特异方差。
- 如何构建、精简和解释量表维度。

### 1.4 直觉与类比

疼痛、疲劳和睡眠问题经常一起变化，可能共同反映“躯体负担”；焦虑和抑郁可能共同反映“情绪困扰”。因子分析寻找这些看不见但能解释相关性的共同来源。

## 2. 核心思想与原理

### 2.1 解释共同方差

与 PCA 压缩总方差不同，因子分析认为每个变量既有公共部分，也有自身特异与测量误差。模型重点复现变量间协方差。

### 2.2 因子旋转

未旋转载荷常难解释。正交旋转保持因子不相关，斜交旋转允许因子相关。旋转改变坐标表达，不改变模型对共同结构的基本拟合。

### 2.3 因子数需要多证据

碎石图、平行分析、拟合指标、残差与理论解释应共同决定因子数。机械使用“特征值大于 1”常不可靠。

## 3. 数学形式

### 3.1 测量模型

$$
x=\mu+\Lambda f+\epsilon
$$

常设：

$$
E(f)=0,\quad\operatorname{Cov}(f)=\Phi,\quad
\operatorname{Cov}(\epsilon)=\Psi
$$

$\Psi$ 通常为对角矩阵。

### 3.2 协方差分解

$$
\Sigma=\Lambda\Phi\Lambda^\top+\Psi
$$

正交因子时 $\Phi=I$：

$$
\Sigma=\Lambda\Lambda^\top+\Psi
$$

### 3.3 共同度

正交模型中，变量 $j$ 的共同度：

$$
h_j^2=\sum_{k=1}^{q}\lambda_{jk}^2
$$

标准化变量的特异度为 $\psi_j=1-h_j^2$。

### 3.4 关键条件

| 条件 | 违反后果 | 检查方式 |
| --- | --- | --- |
| 少数因子可解释相关 | 残差相关大、结构不清 | 残差相关矩阵 |
| 题项相关估计合适 | Likert 数据 Pearson 相关失真 | polychoric 敏感性 |
| 样本量与题项质量足够 | 载荷和因子数不稳定 | 平行分析、bootstrap |
| 特异误差近似独立 | 局部依赖未建模 | modification/residual 检查 |

## 4. 手把手算例

两个标准化题项 $x_1,x_2$ 由一个方差为 1 的公共因子生成，载荷为：

$$
\Lambda=
\begin{pmatrix}
0.8\\
0.6
\end{pmatrix}
$$

为使两个题项方差都为 1，设：

$$
\Psi=
\begin{pmatrix}
0.36&0\\
0&0.64
\end{pmatrix}
$$

**Step 1：共同度。**

$$
h_1^2=0.8^2=0.64,\qquad
h_2^2=0.6^2=0.36
$$

题项 1 有 64% 方差由公共因子解释，题项 2 为 36%。

**Step 2：重构协方差。**

$$
\Lambda\Lambda^\top=
\begin{pmatrix}
0.64&0.48\\
0.48&0.36
\end{pmatrix}
$$

加上 $\Psi$：

$$
\Sigma=
\begin{pmatrix}
1&0.48\\
0.48&1
\end{pmatrix}
$$

**结论：** 两题相关 0.48 完全来自共同因子；各题剩余 0.36 与 0.64 被视作特异方差。

## 5. 数据形式与输入输出

### 5.1 数据要求

- 多个相关连续指标或有序量表题项。
- 反向题需先正确编码，缺失机制和题项分布需检查。
- 重复测量和多层数据需使用相应多层/纵向因子模型。

### 5.2 输入与产出

输入为题项矩阵或相关矩阵、因子数、提取法和旋转。输出为载荷、共同度、特异度、因子相关、因子得分及拟合信息。

## 6. 适用场景

- 量表开发、症状群与潜在构念探索。
- 多个观测指标被假定由少数共同原因驱动。
- 不适合只追求预测压缩、题项几乎不相关或结构高度非线性的任务。

## 7. 实现

### 7.1 Python

```python
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler

X_s = StandardScaler().fit_transform(X)
fa = FactorAnalyzer(
    n_factors=2,
    method="ml",
    rotation="oblimin",
)
fa.fit(X_s)

loadings = fa.loadings_
communalities = fa.get_communalities()
uniquenesses = fa.get_uniquenesses()
scores = fa.transform(X_s)
print(loadings, communalities, uniquenesses)
```

### 7.2 R

```r
library(psych)

items <- df[, item_names]
fa.parallel(items, fa = "fa", fm = "ml")

fit <- fa(
  items,
  nfactors = 2,
  rotate = "oblimin",
  fm = "ml",
  scores = "regression"
)

print(fit$loadings, cutoff = 0.30)
fit$communality
fit$uniquenesses
```

## 8. 结果如何解释

- 载荷表示变量与因子的关系，但阈值不应机械化。
- 共同度低表示所提因子难以解释该题项。
- 斜交旋转后应同时看 pattern matrix 与因子相关。
- 因子命名来自高载荷题项与理论，不由软件自动决定。

## 9. 诊断与稳健性

1. 检查相关矩阵、KMO 和 Bartlett 检验。
2. 用平行分析和拟合残差选择因子数。
3. 比较正交与斜交旋转、不同提取法。
4. bootstrap 载荷和因子一致性。
5. 在独立样本做 CFA 或结构复现。

## 10. 推荐可视化

- 观测与随机特征值的平行分析图。
- 旋转载荷热图。
- 共同度和特异度条形图。
- 因子得分与因子相关图。

## 11. 优势、局限与常见坑

**优势：** 显式区分共同与特异方差，适合潜在构念和量表结构。

**局限：** 因子数与旋转有主观性，样本和题项质量要求高，探索结果需验证。

**常见坑：** 把 PCA 当 EFA；只用特征值大于 1；强行删除交叉载荷题；把探索性结构写成已证实构念；Likert 题不检查相关类型。

## 12. 与相近方法的区别

- [[主成分分析（Principal Component Analysis, PCA）]]：PCA 压缩总方差，FA 建模共同方差和特异误差。
- [[独立成分分析（Independent Component Analysis, ICA）]]：ICA 追求独立非高斯源，FA 解释相关结构。
- [[奇异值分解（Singular Value Decomposition, SVD）]]：SVD 是代数分解，不含测量误差模型。
- 选择经验：量表构念与测量结构用 FA，纯压缩与可视化用 PCA。

## 13. 医学研究中的典型应用

- 患者报告结局和生活质量量表。
- 抑郁、焦虑、疲劳与疼痛症状维度。
- 多项生活方式或社会决定因素的潜在构念。

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| common factor | 解释多个观测变量共同变异的潜变量 |
| loading | 观测变量与因子的线性关系强度 |
| communality | 变量方差中由公共因子解释的比例 |
| uniqueness | 未被公共因子解释的特异方差 |
| rotation | 改变因子坐标以获得更清晰载荷结构 |

## 15. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[独立成分分析（Independent Component Analysis, ICA）]]
- [[奇异值分解（Singular Value Decomposition, SVD）]]
- [[线性判别分析（Linear Discriminant Analysis, LDA）]]

## 16. 参考资料

- Harman HH. *Modern Factor Analysis*. 3rd ed. University of Chicago Press; 1976.
- Fabrigar LR, Wegener DT. *Exploratory Factor Analysis*. Oxford University Press; 2011.
- Revelle W. `psych`: Procedures for Psychological, Psychometric, and Personality Research. [https://cran.r-project.org/package=psych](https://cran.r-project.org/package=psych) （访问日期：2026-07-09）
