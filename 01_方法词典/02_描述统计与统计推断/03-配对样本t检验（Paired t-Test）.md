---
title: 配对样本t检验
english_name: Paired t-Test
slug: paired-t-test
aliases: [paired t-test, 配对t检验, 成对t检验, "配对样本t检验（Paired t-Test）"]
category: 描述统计与统计推断
subcategory: 均值比较
tags: [医学统计, 数据科学, 参数检验, 配对设计]
status: 已建
difficulty: basic
question_type: 配对均值比较
data_type: [表格数据]
outcome_type: [连续型]
python_packages: [scipy]
r_packages: [stats]
---

# 配对样本t检验（Paired t-Test）

## 1. 方法概览

### 1.1 定义

配对样本 t 检验比较同一批个体（或匹配对）在两种条件下的连续测量，通过检验“配对差值的均值是否为 0”来判断两条件是否有系统差异。

### 1.2 它主要解决什么问题

- 研究问题：治疗前后、左右侧、两种方法在同一对象上的测量是否有差异。
- 适用任务：前后对照、自身配对、匹配设计的均值比较。
- 常见医学场景：干预前后血压、同一患者两种测量仪器读数、配对病例对照的连续指标。

### 1.3 直觉理解

配对设计让每个人当自己的对照，先算每个个体的“后减前”差值，再看这些差值平均是否显著偏离 0。因为同一个体内部相关被消掉，配对检验通常比两独立样本检验更有效。

## 2. 数学形式

### 2.1 核心公式

对配对差 $d_i=x_{i2}-x_{i1}$，检验 $H_0:\mu_d=0$：

$$
t=\frac{\bar{d}}{s_d/\sqrt{n}}\sim t_{n-1}
$$

其中 $\bar{d}$ 为差值均值、$s_d$ 为差值标准差、$n$ 为配对数。

### 2.2 参数或统计量含义

- $\bar{d}$：平均配对差（效应估计）。
- $s_d$：配对差的标准差。
- $t$：检验统计量，自由度 $n-1$。
- 效应量 Cohen's $d_z=\bar{d}/s_d$。

### 2.3 关键假设

- 配对差值近似正态（不是原始测量正态）。
- 各配对之间相互独立。
- 测量为连续量、配对关系正确对应。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：两水平的配对条件（前/后、方法 A/B）。
- 因变量类型：连续型测量。
- 数据结构：每行一个个体，含两次测量或一列差值。
- 是否适合高维数据：单指标检验；多指标需多重比较校正。
- 是否适合缺失较多数据：缺失配对通常整对剔除或需插补。
- 是否适合删失数据：不适用。
- 是否适合重复测量数据：仅两个时间点；多时间点用重复测量/混合模型。

### 3.2 示例表格

| 个体 | 治疗前 SBP | 治疗后 SBP | 差值 d |
| --- | --- | --- | --- |
| 1 | 148 | 138 | −10 |
| 2 | 155 | 149 | −6 |
| 3 | 141 | 142 | +1 |
| 4 | 160 | 150 | −10 |

### 3.3 输入与产出

#### 输入

- 输入数据：两次配对测量或其差值。
- 关键变量：配对标识、两次测量值。
- 需要预处理的内容：核对配对对应关系、检查差值分布。

#### 产出

- 模型对象/统计结果：$t$、自由度、p 值。
- 参数估计：平均差 $\bar{d}$ 及其置信区间。
- 预测结果：无。
- 不确定性指标：均差 CI、效应量。

## 4. 适用场景

- 适合：自身前后对照、匹配对、同一对象两法比较。
- 不适合：差值明显偏态或有强离群值（用 Wilcoxon 符号秩）、多于两个时间点。
- 使用前需要特别检查的点：差值正态性、配对是否真实、是否应报告一致性（Bland-Altman）。

## 5. 实现

### 5.1 Python

常用包：

- `scipy`

```python
from scipy import stats

t, p = stats.ttest_rel(df["post"], df["pre"])
diff = df["post"] - df["pre"]
print(f"t={t:.3f}, p={p:.4f}, 平均差={diff.mean():.2f}")
```

### 5.2 R

常用包：

- `stats`

```r
t.test(df$post, df$pre, paired = TRUE)
```

## 6. 结果如何解释

- 核心结果看什么：平均差及其 95% CI、p 值。
- 每个主要参数如何解释：平均差为负（如 −8 mmHg）表示治疗后系统性下降；CI 反映效应精度。
- 临床或医学意义如何表达：优先报告有临床意义的均差与区间，而非只报 p 值。
- 常见误读：误用两独立样本 t 检验处理配对数据；只看 p 值忽视效应大小。

## 7. 推荐可视化

- 前后配对连线图（spaghetti plot）。
- 差值的直方图/箱线图配 0 参考线。
- Bland-Altman 图（两法一致性场景）。

## 8. 优势、局限与常见坑

### 优势

- 消除个体间变异，检验效率高。
- 结果直观、易于报告平均改变量。
- 计算与假设简单。

### 局限

- 依赖差值正态。
- 仅限两个条件/时间点。
- 对差值离群值敏感。

### 常见坑

- 把配对数据当独立两样本分析。
- 检验原始值正态而非差值正态。
- 忽略回归到均值效应对“前后差”的影响。

## 9. 与相近方法的区别

- 和 [[两独立样本t检验（Two-Sample t-Test）]] 的区别：后者用于独立两组，配对检验用于同一/匹配对象。
- 和 [[Wilcoxon符号秩检验（Wilcoxon Signed-Rank Test）]] 的区别：差值偏态或小样本时用后者（非参数）。
- 和 [[线性混合效应模型（Linear Mixed-Effects Model, LMM）]] 的区别：多于两个时间点或含协变量时用混合模型。

## 10. 医学研究中的典型应用

- 干预前后连续指标改变的评估。
- 同一患者两种测量方法/仪器的系统差异。
- 匹配设计中连续结局的比较。

## 11. 相关方法

- [[两独立样本t检验（Two-Sample t-Test）]]
- [[单样本t检验（One-Sample t-Test）]]
- [[Wilcoxon符号秩检验（Wilcoxon Signed-Rank Test）]]

## 12. 参考资料

- Altman DG. *Practical Statistics for Medical Research*. Chapman and Hall/CRC; 1990.
- Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. *Lancet*. 1986;327(8476):307-310.
- R Core Team. `t.test`. R Manual. [https://stat.ethz.ch/R-manual/R-devel/library/stats/html/t.test.html](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/t.test.html) （访问日期：2026-07-02）
