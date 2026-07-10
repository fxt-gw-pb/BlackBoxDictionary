---
title: 信息增益
english_name: Information Gain
slug: information-gain
aliases: [information gain, IG, "信息增益（Information Gain）"]
category: 数据预处理与特征工程
subcategory: 过滤式特征选择
tags: [医学统计, 数据科学, 特征选择, 信息论, 决策树]
status: 已建
difficulty: intermediate
question_type: 分类特征的信息量评估
data_type: [表格数据, 离散特征]
outcome_type: [二分类, 多分类]
python_packages: [scikit-learn, pandas]
r_packages: [FSelectorRcpp]
---

# 信息增益（Information Gain）

## 1. 方法概览

### 1.1 一句话本质

信息增益衡量知道某特征后，类别的不确定性平均减少了多少；减少越多，特征越能把不同类别分开。

### 1.2 定义

信息增益是分类目标熵与按特征分组后的条件熵之差。它既用于 ID3 决策树选择分裂，也可作为过滤式特征排序。对连续特征需离散化、搜索阈值或改用互信息估计。

### 1.3 它主要解决什么问题

- 评价离散特征对分类结局的信息量。
- 捕捉不限于线性或均值差异的依赖。
- 医学场景：症状/检查阳性、基因型、分箱生物标志物的初筛。

### 1.4 直觉与类比

未看检查前，阳性/阴性各半像抛硬币；若看到检查结果后每组几乎都是同一结局，猜测难度大幅下降。信息增益就是“少猜了多少”。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

相关和 F 检验偏向线性或均值差异，分类关系可能是非线性、离散或多类别的。熵只依赖概率分布，可统一衡量分组前后纯度。

### 2.2 关键洞察

一个有用特征应让每个取值内的类别更纯。条件熵是各组剩余不确定性的加权平均；从原熵中减去它，便得到特征提供的信息。

### 2.3 与朴素/相邻做法的对比

- 与卡方检验都衡量离散依赖，但信息增益以 bit 表示不确定性减少。
- 与互信息在离散情形等价。
- 原始信息增益偏爱取值多的特征，患者 ID 可获得虚高满分；增益率或最小叶样本约束可缓解。

## 3. 数学形式

### 3.1 核心公式

类别熵：

$$
H(Y)=-\sum_{k=1}^{K}p_k\log_2p_k
$$

条件熵与信息增益：

$$
H(Y\mid X)=\sum_{v\in\mathcal V}
p(X=v)H(Y\mid X=v)
$$

$$
IG(Y;X)=H(Y)-H(Y\mid X)
$$

这个式子在说：特征分组后剩余不确定性越小，提供的信息越多。

### 3.2 推导脉络

熵是编码类别所需的平均最短信息量。观察 $X$ 后编码长度降为条件熵，两者之差就是互信息；因此 $IG\ge0$，且完全确定结局时等于 $H(Y)$。

### 3.3 参数与统计量含义

- $p_k$：第 $k$ 类比例。
- $H(Y)$：观察特征前的不确定性，单位 bit。
- $H(Y\mid X)$：观察特征后仍剩的不确定性。
- split information：特征自身取值熵；增益率用 $IG/H(X)$ 惩罚高基数。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后果 |
| --- | --- | --- |
| 频率能估计概率 | 各取值样本足够 | 小样本 IG 向上偏 |
| 离散化合理 | 切点保留真实结构 | 排名依赖任意分箱 |
| 训练折内估计 | 测试标签不可见 | 监督泄漏 |
| 高基数受控制 | ID/时间戳不应取巧 | 记忆样本、泛化差 |

## 4. 手把手算例

6 人结局 $Y=(0,0,0,1,1,1)$，所以

$$
H(Y)=-0.5\log_2 0.5-0.5\log_2 0.5=1\text{ bit}
$$

**强特征 A：** 取值 $(L,L,L,H,H,H)$，L 组全为 0、H 组全为 1：

$$
H(Y\mid A)=0,\qquad IG(Y;A)=1
$$

**弱特征 B：** 取值 $(0,1,0,1,0,1)$。B=0 组结局为 $(0,0,1)$，B=1 组为 $(0,1,1)$；两组熵均为

$$
H(1/3,2/3)=0.918
$$

因此

$$
IG(Y;B)=1-\left(\frac36 0.918+\frac36 0.918\right)=0.082
$$

但“患者 ID”每人一个取值，也会使每组熵为 0、IG=1。这揭示高基数偏倚：满分不一定可泛化。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 离散特征与二/多分类结局。
- 连续特征需分箱或连续互信息估计。
- 缺失可作为一类但可能把诊疗流程当信号。
- 不直接处理删失、重复测量或混杂。

### 5.2 示例表格

| patient_id | marker_group | alternating_flag | severe |
| --- | --- | ---: | ---: |
| P01 | L | 0 | 0 |
| P02 | L | 1 | 0 |
| P04 | H | 1 | 1 |

### 5.3 输入与产出

输入训练特征、分类结局和离散化规则；产出每列 IG/互信息、排名和保留掩码。

## 6. 适用场景

- 适合：离散/非线性分类关系的快速筛选。
- 不适合：极高基数 ID、小样本稀疏类别、需要条件/因果效应。
- 应配合交叉验证稳定性和高基数惩罚。

## 7. 实现

### 7.1 Python

```python
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

X = pd.DataFrame({
    "strong": [0, 0, 0, 1, 1, 1],
    "weak": [0, 1, 0, 1, 0, 1],
})
y = [0, 0, 0, 1, 1, 1]
scores = mutual_info_classif(
    X, y, discrete_features=True, random_state=42
)
print(pd.Series(scores, index=X.columns).sort_values(ascending=False))
```

### 7.2 R

```r
entropy <- function(x) {
  p <- prop.table(table(x))
  -sum(p * log2(p))
}
information_gain <- function(x, y) {
  h_cond <- sum(sapply(split(y, x), function(g) {
    length(g) / length(y) * entropy(g)
  }))
  entropy(y) - h_cond
}
y <- c(0, 0, 0, 1, 1, 1)
strong <- c("L", "L", "L", "H", "H", "H")
weak <- c(0, 1, 0, 1, 0, 1)
c(strong = information_gain(strong, y),
  weak = information_gain(weak, y))
```

## 8. 结果如何解读

IG=0 表示在样本分布下未减少类别不确定性；IG=1 bit 的含义取决于原始熵。高分不是因果效应，也可能来自过细分组、缺失模式或 ID 泄漏。

## 9. 假设诊断与稳健性

- 用交叉验证/Bootstrap 记录排名稳定性。
- 合并稀疏类别，限制最小组大小。
- 比较不同分箱与连续互信息估计。
- 排除 ID、时间戳、未来编码等高基数泄漏列。
- 对大量特征用置换基线或嵌套验证。

## 10. 推荐可视化

- IG 排序条形图。
- 特征取值 × 类别比例堆积图。
- 分箱阈值—IG 曲线。
- 各折入选频率图。
- 特征基数与 IG 散点图。

## 11. 优势、局限与常见坑

### 优势

- 能捕捉一般离散依赖，不限线性。
- 单位清晰、适合分类排序。

### 局限

- 有限样本估计偏差和高基数偏好。
- 忽略特征间冗余、交互和混杂。

### 常见坑

- 把连续变量随意分太多箱。
- 患者 ID 获得高分仍保留。
- 全数据计算 IG 后交叉验证。
- 把信息量解释为临床效应大小。

## 12. 与相近方法的区别

- 互信息是更一般名称，离散情形与 IG 等价。
- 卡方给假设检验量；IG 给不确定性减少。
- C4.5 增益率惩罚高基数，ID3 使用原始 IG。
- 如何选择：离散分类初筛可用 IG，但需控制基数并检查稳定性。

## 13. 医学研究中的典型应用

- 症状、基因型、检查阳性与诊断类别筛选。
- 分箱生物标志物的分类信息量评估。
- 决策树分裂候选比较。

必须审计类别不平衡、稀疏取值、缺失机制、患者级切分和高基数泄漏。

## 14. 关键术语

- **熵（Entropy）**：类别不确定性的平均信息量。
- **条件熵（Conditional entropy）**：知道特征后剩余的不确定性。
- **互信息（Mutual information）**：两个变量共享的信息。
- **Bit**：以 2 为对数底的信息单位。
- **高基数偏倚（High-cardinality bias）**：取值多的特征更易获得虚高增益。
- **增益率（Gain ratio）**：信息增益除以特征自身熵。

## 15. 相关方法

- [[互信息特征选择（Mutual Information Feature Selection）]]
- [[ID3决策树（ID3 Decision Tree）]]
- [[C4.5决策树（C4.5 Decision Tree）]]
- [[单变量特征选择（Univariate Feature Selection）]]
- [[Pearson卡方独立性检验（Pearson Chi-Squared Test of Independence）]]

## 16. 参考资料

- Cover TM, Thomas JA. *Elements of Information Theory*. Wiley; 2006.
- Quinlan JR. Induction of decision trees. *Machine Learning*. 1986;1:81-106.
- scikit-learn developers. Mutual information feature selection. https://scikit-learn.org/stable/modules/feature_selection.html
