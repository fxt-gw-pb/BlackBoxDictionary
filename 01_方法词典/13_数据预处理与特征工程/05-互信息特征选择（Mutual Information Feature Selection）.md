---
title: 互信息特征选择
english_name: Mutual Information Feature Selection
slug: mutual-information-feature-selection
aliases: [mutual information feature selection, mutual information, MI, 互信息法, "互信息特征选择（Mutual Information Feature Selection）"]
category: 数据预处理与特征工程
subcategory: 过滤式特征选择
tags: [医学统计, 数据科学, 特征选择, 信息论, 非线性关系]
status: 已建
difficulty: intermediate
question_type: 非线性单变量依赖筛选
data_type: [表格数据, 高维特征矩阵]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [scikit-learn]
r_packages: [FSelectorRcpp, infotheo]
---

# 互信息特征选择（Mutual Information Feature Selection）

## 1. 方法概览

### 1.1 一句话本质

互信息衡量知道一个特征后，结局的不确定性平均减少多少，因此能发现不限于线性或均值差异的边际依赖。

### 1.2 定义

互信息（MI）是两个随机变量共享的信息量。特征选择时逐列估计 $I(X_j;Y)$，按得分保留前 $k$ 个。离散分类中它等于信息增益；连续变量常用分箱、核密度或 k 近邻估计。

### 1.3 它主要解决什么问题

- 发现 U 形、阈值、单调或离散的非线性边际关系。
- 对分类和连续结局建立模型无关的初筛排名。
- 医学场景：组学、影像组学、多类型生物标志物筛选。

### 1.4 直觉与类比

相关问“两者是否沿直线一起升降”，互信息问“知道 X 能否让我更少猜 Y”。即使高低两端都危险、直线相关接近 0，只要 X 能缩小 Y 的可能范围，MI 仍可大于 0。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

Pearson/F 检验依赖特定函数形状。真实生物关系常有阈值、饱和或多峰，预先枚举变换困难。MI 直接比较联合分布与边际分布乘积，不预设线性。

### 2.2 关键洞察

若 $X,Y$ 独立，则 $p(x,y)=p(x)p(y)$，观察 X 不改变 Y 的分布，MI 为 0；偏离独立越大，MI 越高。MI 对可逆变换理论上不变，但有限样本估计会受尺度、邻居数和分箱影响。

### 2.3 与朴素/相邻做法的对比

- 比相关更一般，但不提供方向和简单效应大小。
- 比模型内选择更快，却忽略特征冗余与交互。
- mRMR 等方法在高相关候选中同时奖励结局信息、惩罚特征间 MI。

## 3. 数学形式

### 3.1 核心公式

离散变量：

$$
I(X;Y)=\sum_x\sum_y p(x,y)
\log_2\frac{p(x,y)}{p(x)p(y)}
$$

等价形式：

$$
I(X;Y)=H(Y)-H(Y\mid X)
$$

这个式子在说：观察 X 前后的不确定性差，就是 X 提供的关于 Y 的信息。

连续变量将求和换成积分：

$$
I(X;Y)=\iint p(x,y)\log
\frac{p(x,y)}{p(x)p(y)}\,dx\,dy
$$

### 3.2 推导脉络

MI 是联合分布与“独立情况下的分布”之间的 KL 散度，因此非负，且独立时为 0。有限样本需估计分布；分箱太细或邻居数太小会高方差，太粗或邻居数太大会抹平关系。

### 3.3 参数与统计量含义

- $H(Y)$：结局原始熵。
- $H(Y\mid X)$：知道特征后剩余熵。
- $k$：kNN 估计器邻居数，不是保留特征数。
- $I(X_j;Y)$：边际信息量，不含控制其他变量后的增量信息。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后果 |
| --- | --- | --- |
| 样本近似独立同分布 | 经验分布可估计目标分布 | 重复患者使 MI 虚高 |
| 样本量支持密度估计 | 稀疏区域有足够邻居 | 得分偏差/波动 |
| 离散/连续类型正确 | 估计器与变量匹配 | 得分失真 |
| 筛选只在训练折 | 测试标签不可见 | 泄漏 |

## 4. 手把手算例

8 人二元特征与结局列联表：

| | $Y=0$ | $Y=1$ |
| --- | ---: | ---: |
| $X=0$ | 3 | 1 |
| $X=1$ | 1 | 3 |

Y 两类各半，故 $H(Y)=1$ bit。每个 X 组内类别比例为 $3/4,1/4$：

$$
H(Y\mid X=0)=H(Y\mid X=1)
=-0.75\log_2 0.75-0.25\log_2 0.25
=0.811
$$

两组等大：

$$
H(Y\mid X)=0.5(0.811)+0.5(0.811)=0.811
$$

$$
I(X;Y)=1-0.811=0.189\text{ bit}
$$

观察 X 后，结局不确定性减少 0.189 bit。它有依赖，但并非完美预测；MI 不告诉我们“X=1 增加还是降低风险”，方向需看条件概率。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 离散/连续特征，分类或连续结局。
- 时间到事件需专门生存互信息；普通 MI 忽略删失。
- 重复测量需患者级抽样或专门依赖处理。
- 缺失作为一类可能捕捉诊疗流程而非生物信号。

### 5.2 示例表格

| patient_id | marker_binary | nonlinear_marker | outcome |
| --- | ---: | ---: | ---: |
| P01 | 0 | -2.1 | 0 |
| P02 | 0 | 0.1 | 1 |
| P03 | 1 | 2.0 | 1 |

### 5.3 输入与产出

输入训练 $X,y$、变量类型、估计器参数与保留数；产出 MI 得分、排名、支持掩码和稳定性结果。

## 6. 适用场景

- 适合：怀疑非线性边际关系、候选很多、最终模型尚未确定。
- 不适合：样本很小、交互才是唯一信号、需要方向/因果解释。
- 与冗余控制和嵌套验证结合使用。

## 7. 实现

### 7.1 Python

```python
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

X = pd.DataFrame({
    "informative": [0, 0, 0, 0, 1, 1, 1, 1],
    "noise": [0, 1, 0, 1, 0, 1, 0, 1],
})
y = [0, 0, 0, 1, 0, 1, 1, 1]
mi = mutual_info_classif(
    X, y, discrete_features=True, random_state=42
)
print(pd.Series(mi, index=X.columns).sort_values(ascending=False))
```

### 7.2 R

```r
library(infotheo)

dat <- data.frame(
  informative = c(0, 0, 0, 0, 1, 1, 1, 1),
  noise = c(0, 1, 0, 1, 0, 1, 0, 1),
  y = c(0, 0, 0, 1, 0, 1, 1, 1)
)
sapply(dat[1:2], function(x) mutinformation(x, dat$y))
```

## 8. 结果如何解读

得分越高，边际依赖越强；0 不证明总体独立，只表示当前估计器和样本未发现信息。不同估计器的绝对值不宜直接比较，排名也需报告稳定性。

## 9. 假设诊断与稳健性

- 多随机种子/邻居数/分箱重复估计。
- bootstrap 或交叉验证记录入选频率。
- 与 Pearson、Spearman 和单变量 F 排名对照。
- 置换 y 建立零信息基线。
- 去除批次、中心、ID 和未来信息后重算。

## 10. 推荐可视化

- MI 排序图与置换阈值。
- 特征—结局散点/条件分布图。
- 邻居数或分箱数敏感性曲线。
- 各折入选频率热图。
- MI 冗余网络图。

## 11. 优势、局限与常见坑

### 优势

- 捕捉一般非线性依赖。
- 不依赖最终预测模型。

### 局限

- 连续 MI 估计困难且无方向。
- 边际筛选忽略冗余与纯交互。

### 常见坑

- 在全数据上筛选。
- 把连续编码误标为离散或反之。
- 只看单次随机估计。
- 把 MI 当因果效应大小。

## 12. 与相近方法的区别

- 信息增益是离散分类场景中的 MI。
- 相关描述线性/单调方向，MI 更一般但无符号。
- 嵌入式选择考虑模型内联合贡献。
- 如何选择：非线性初筛用 MI，随后用冗余控制和联合模型验证。

## 13. 医学研究中的典型应用

- 非线性生物标志物、组学和影像组学筛选。
- 症状/基因型与诊断类别信息量评估。
- 多模态候选变量初筛。

需报告估计方法、缺失、批次、类别不平衡、患者级切分和选择稳定性。

## 14. 关键术语

- **互信息（Mutual information）**：两个变量共享的信息量。
- **联合分布（Joint distribution）**：变量组合取值的概率。
- **KL 散度**：一个分布相对另一个分布的差异。
- **kNN 估计器**：用近邻距离估计连续 MI。
- **mRMR**：最大相关、最小冗余选择。
- **置换基线（Permutation baseline）**：打乱结局得到的偶然得分分布。

## 15. 相关方法

- [[信息增益（Information Gain）]]
- [[单变量特征选择（Univariate Feature Selection）]]
- [[Pearson相关（Pearson Correlation）]]
- [[Spearman秩相关（Spearman Rank Correlation）]]
- [[嵌入式特征选择（Embedded Feature Selection）]]

## 16. 参考资料

- Cover TM, Thomas JA. *Elements of Information Theory*. Wiley; 2006.
- Kraskov A, Stögbauer H, Grassberger P. Estimating mutual information. *Physical Review E*. 2004;69:066138.
- scikit-learn developers. Mutual information. https://scikit-learn.org/stable/modules/feature_selection.html
