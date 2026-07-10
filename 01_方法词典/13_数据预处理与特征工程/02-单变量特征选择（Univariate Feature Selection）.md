---
title: 单变量特征选择
english_name: Univariate Feature Selection
slug: univariate-feature-selection
aliases: [univariate feature selection, SelectKBest, "单变量特征选择（Univariate Feature Selection）"]
category: 数据预处理与特征工程
subcategory: 过滤式特征选择
tags: [医学统计, 数据科学, 特征选择, 单变量分析]
status: 已建
difficulty: basic
question_type: 按特征与结局的单独关联筛选变量
data_type: [表格数据, 高维特征矩阵]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [FSelectorRcpp, caret]
---

# 单变量特征选择（Univariate Feature Selection）

## 1. 方法概览

### 1.1 一句话本质

单变量特征选择让每个特征单独参加一次“与结局的考试”，再按得分或校正后的显著性筛选，但看不到联合效应和交互。

### 1.2 定义

它是一组监督式过滤方法：对每列独立计算统计量、P 值或互信息，再选前 $k$ 个、前若干百分比或过阈值的特征。分类可用 ANOVA F、卡方或互信息，回归可用相关/F 检验。

### 1.3 它主要解决什么问题

- 高维、小样本建模前的快速降维。
- 从成千上万候选中建立初筛列表。
- 医学场景：组学筛选、影像组学、候选生物标志物排序。

### 1.4 直觉与类比

像让每位候选人单独面试：能快速淘汰明显不合格者，却无法发现两人合作才有效，也无法识别“单独表现普通、加入模型后重要”的抑制变量。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

$p$ 很大时直接拟合复杂模型不稳定、计算昂贵。逐列评分把多变量优化拆成大量廉价的一维问题，可作为第一道监督筛选。

### 2.2 关键洞察

如果某特征单独与结局完全无关，它通常不是优先候选；但“边际无关”不等于“条件无关”。因此单变量筛选是降维启发式，而不是混杂调整或因果发现。

### 2.3 与朴素/相邻做法的对比

- 比肉眼挑变量可重复、可嵌入交叉验证。
- 比嵌入式选择更快、与最终模型无关。
- 代价是忽略特征相关性、非线性形式与联合交互。

## 3. 数学形式

### 3.1 核心公式

二分类连续特征的 ANOVA F 得分：

$$
F_j=\frac{SSB_j/(G-1)}{SSW_j/(n-G)}
$$

其中

$$
SSB_j=\sum_g n_g(\bar x_{gj}-\bar x_j)^2,\qquad
SSW_j=\sum_g\sum_{i\in g}(x_{ij}-\bar x_{gj})^2
$$

这个式子在说：组间差异相对组内噪声越大，特征得分越高。

### 3.2 推导脉络

若各组均值相同，组间平方和只反映随机波动；若均值分开而组内紧凑，F 增大。对大量特征同时检验时，原始 P 值必须做 FDR 等多重校正。

### 3.3 参数与统计量含义

- $F_j$：第 $j$ 列边际组间区分度。
- $k$：保留特征数量，是超参数。
- P 值：零假设下出现当前或更极端得分的概率，不是效应大小。
- q 值/FDR：多重检验后的错误发现控制指标。

### 3.4 关键假设（含违反后果）

| 评分 | 关键前提 | 违反风险 |
| --- | --- | --- |
| ANOVA F | 近似正态、组内方差合理 | 异常值/异方差影响排名 |
| 卡方 | 非负频数、期望频数足够 | 连续值误用、稀疏不稳 |
| 互信息 | 样本足够估计分布 | 小样本偏差大 |
| 所有方法 | 筛选只在训练折 | 测试泄漏、性能虚高 |

## 4. 手把手算例

6 人结局 $y=(0,0,0,1,1,1)$。候选 A 为 $(1,2,3,4,5,6)$，候选 B 为 $(0,1,0,1,0,1)$。

**特征 A：** 两组均值 2 和 5，总均值 3.5：

$$
SSB=3(2-3.5)^2+3(5-3.5)^2=13.5
$$

组内平方和为 $2+2=4$：

$$
F_A=\frac{13.5/1}{4/4}=13.5
$$

**特征 B：** 两组均值 $1/3$ 和 $2/3$，总均值 $1/2$：

$$
SSB=\frac16,\qquad SSW=\frac43
$$

$$
F_B=\frac{1/6}{(4/3)/4}=0.5
$$

A 排在 B 前。但这只是该小样本中的边际排序；若 A 与另一个特征完全重复，两者都会高分。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 表格/组学高维矩阵，分类或连续结局。
- 需要为特征类型与结局选择匹配评分。
- 缺失、删失和重复测量不能被普通评分自动处理。

### 5.2 示例表格

| patient_id | marker_A | marker_B | severe |
| --- | ---: | ---: | ---: |
| P01 | 1 | 0 | 0 |
| P02 | 2 | 1 | 0 |
| P04 | 4 | 1 | 1 |

### 5.3 输入与产出

输入 $X,y$、评分函数和 $k$/阈值；产出每列得分、P 值/排名和保留掩码。

## 6. 适用场景

- 适合：快速初筛、极高维矩阵、需要模型无关排名。
- 不适合：交互是核心、混杂严重、时间到事件结局未处理。
- 最终性能必须在嵌套或严格训练折流水线中评估。

## 7. 实现

### 7.1 Python

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

X = pd.DataFrame({
    "marker_A": [1, 2, 3, 4, 5, 6],
    "marker_B": [0, 1, 0, 1, 0, 1],
})
y = [0, 0, 0, 1, 1, 1]
selector = SelectKBest(score_func=f_classif, k=1).fit(X, y)
print(pd.DataFrame({
    "feature": X.columns,
    "F": selector.scores_,
    "p": selector.pvalues_,
}))
```

### 7.2 R

```r
dat <- data.frame(
  marker_A = 1:6,
  marker_B = c(0, 1, 0, 1, 0, 1),
  severe = factor(c(0, 0, 0, 1, 1, 1))
)
scores <- sapply(dat[1:2], function(x) {
  summary(aov(x ~ dat$severe))[[1]][["F value"]][1]
})
sort(scores, decreasing = TRUE)
```

## 8. 结果如何解读

高分表示单独关联更强，不表示独立贡献、因果效应或最终模型必需。P 值很小也可能效应很小；应同时看效应量、稳定性和外部验证。

## 9. 假设诊断与稳健性

- 在每个训练折重做筛选，记录入选频率。
- 比较 F、秩检验与互信息排名。
- 对多重检验控制 FDR。
- 检查异常值、类别不平衡、批次效应与亚组异质性。
- 将 $k$ 作为流水线超参数调优。

## 10. 推荐可视化

- 得分/效应量排序图。
- 火山图（效应量 vs 校正 P 值）。
- 交叉验证入选频率图。
- Top 特征按结局分组的箱线/小提琴图。

## 11. 优势、局限与常见坑

### 优势

- 快速、可扩展、与最终模型解耦。
- 每个得分易于初步解释。

### 局限

- 忽略冗余、混杂和交互。
- 小样本排名不稳定。

### 常见坑

- 全数据筛选后再交叉验证。
- 只按未校正 P 值筛选。
- 把“单变量不显著”当作绝对无用。
- 用普通检验处理删失/重复测量。

## 12. 与相近方法的区别

- 方差阈值不看结局；单变量选择看边际关联。
- 相关过滤侧重删除特征间冗余。
- 嵌入式选择在模型拟合中考虑联合条件贡献。
- 如何选择：先做数据卫生，再把单变量初筛与联合模型比较。

## 13. 医学研究中的典型应用

- 转录组/蛋白组候选标志物初筛。
- 影像组学降维。
- 大量实验室指标的风险预测预筛。

必须说明多重检验、缺失机制、批次效应、类别不平衡和筛选是否完全位于训练折。

## 14. 关键术语

- **边际关联（Marginal association）**：不控制其他变量时的关系。
- **条件贡献（Conditional contribution）**：控制其他变量后的增量信息。
- **多重检验（Multiple testing）**：同时检验大量假设。
- **FDR**：所有发现中期望假阳性比例。
- **SelectKBest**：按得分保留前 $k$ 列的策略。

## 15. 相关方法

- [[方差阈值法（Variance Threshold）]]
- [[信息增益（Information Gain）]]
- [[互信息特征选择（Mutual Information Feature Selection）]]
- [[嵌入式特征选择（Embedded Feature Selection）]]
- [[多重检验与错误率控制（Multiple Testing and Error Rate Control）]]

## 16. 参考资料

- Guyon I, Elisseeff A. An introduction to variable and feature selection. *JMLR*. 2003;3:1157-1182.
- scikit-learn developers. Univariate feature selection. https://scikit-learn.org/stable/modules/feature_selection.html
