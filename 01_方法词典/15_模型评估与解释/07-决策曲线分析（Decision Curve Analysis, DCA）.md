---
title: 决策曲线分析
english_name: Decision Curve Analysis, DCA
slug: decision-curve-analysis-dca
aliases: [DCA, decision curve analysis, net benefit, "决策曲线分析（Decision Curve Analysis, DCA）"]
category: 模型评估与解释
subcategory: 临床效用评估
tags: [医学统计, 数据科学, 模型评估, 临床效用, 风险预测]
status: 已建
difficulty: intermediate
question_type: 预测模型临床净获益评估
data_type: [表格数据]
outcome_type: [二分类]
python_packages: [dcurves]
r_packages: [dcurves]
---

# 决策曲线分析（Decision Curve Analysis, DCA）

## 1. 方法概览

### 1.1 一句话本质

DCA 用阈值概率把漏诊与误治代价换成同一尺度，比较模型、全治和全不治谁带来更高净获益。

### 1.2 定义

决策曲线分析通过“净获益”指标，评估一个预测模型或诊断策略在不同决策阈值下相对于“全部干预”和“全部不干预”两个极端策略的临床价值。

### 1.3 它主要解决什么问题

- 研究问题：用这个模型来指导“治不治/查不查”的决策，相比一刀切策略，到底带来多少净收益。
- 适用任务：预测模型的临床效用评估、多个模型/标志物的效用比较、干预阈值区间的选择。
- 常见医学场景：是否做活检、是否给预防性治疗、是否进一步影像检查等二元临床决策。

### 1.4 直觉理解

区分度和校准都不回答“用了到底值不值”。DCA 引入阈值概率 $p_t$——它同时代表医生愿意接受的“多少假阳性换一个真阳性”的权衡。在每个 $p_t$ 下比较模型、全干预、全不干预的净获益，谁的曲线更高谁更有用。

## 2. 核心思想与原理

### 2.1 根本困难
AUC/校准不直接回答模型是否值得改变临床行动；同一性能在不同治疗代价下价值不同。

### 2.2 关键洞察
阈值 $p_t$ 隐含假阳性相对真阳性的代价比 $p_t/(1-p_t)$。用它惩罚 FP，可把决策结果换算为每患者净真阳性收益。

### 2.3 与朴素做法对比
DCA 不需要给效用赋货币值，但依赖阈值含义合理；它不是随机试验，也不能证明模型实施后真实改善结局。

## 3. 数学形式

### 3.1 核心公式

在阈值概率 $p_t$ 下，把预测概率 $\ge p_t$ 者判为“干预”，净获益定义为：

$$
\mathrm{NB}(p_t)=\frac{TP}{n}-\frac{FP}{n}\cdot\frac{p_t}{1-p_t}
$$

其中权重 $\dfrac{p_t}{1-p_t}$ 是“一个假阳性相对一个真阳性的代价”。两条参考策略：

$$
\mathrm{NB}_{\text{all}}(p_t)=\pi-(1-\pi)\cdot\frac{p_t}{1-p_t},\qquad \mathrm{NB}_{\text{none}}=0
$$

$\pi$ 为事件患病率。

### 3.3 参数或统计量含义

- $p_t$：阈值概率，反映真假阳性的相对权衡（也可换算为交换比 odds）。
- $TP,FP$：在阈值 $p_t$ 判为阳性时的真、假阳性数。
- 净获益 NB：以“每 100 人多少个净真阳性”为单位，可直接比较策略。
- treat-all / treat-none：两条基准，模型必须优于二者才有增量价值。

### 3.4 关键假设

- 需要预测概率（或可换算为概率的评分）与二分类结局。
- 阈值概率区间应覆盖临床上合理的偏好范围。
- 预测概率的校准会直接影响净获益，故 DCA 建议与校准一起报告。

## 4. 手把手算例

100 人中模型在阈值 $p_t=0.20$ 下得到 TP=20、FP=10：

$$
NB=\frac{20}{100}-\frac{10}{100}\frac{0.20}{0.80}
=0.20-0.025=0.175
$$

即每 100 人净获益相当于 17.5 个不增加误治的真阳性。若患病率 25%，全治策略：

$$
NB_{all}=0.25-0.75\frac{0.20}{0.80}=0.0625
$$

全不治净获益为 0；在该阈值下模型优于两种默认策略。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：一个或多个模型的预测概率。
- 因变量类型：二分类结局。
- 数据结构：每行含预测概率与真实标签。
- 是否适合高维数据：只作用于最终概率。
- 是否适合缺失较多数据：需先得到完整预测概率。
- 是否适合删失数据：有针对时间到事件的生存版 DCA（基于特定时间点风险）。
- 是否适合重复测量数据：聚类需在不确定性估计中考虑。

### 5.2 示例表格

| 阈值概率 pt | 模型净获益 | 全干预净获益 | 全不干预净获益 |
| --- | --- | --- | --- |
| 0.10 | 0.171 | 0.140 | 0 |
| 0.20 | 0.150 | 0.075 | 0 |
| 0.30 | 0.121 | 0.014 | 0 |
| 0.40 | 0.088 | −0.060 | 0 |
| 0.50 | 0.052 | −0.160 | 0 |

### 5.3 输入与产出

#### 输入

- 输入数据：预测概率、真实标签。
- 关键变量：预测概率、二分类结局、临床合理的阈值区间。
- 需要预处理的内容：在独立/验证数据评估；概率最好先校准。

#### 产出

- 模型对象/统计结果：各阈值下的净获益曲线。
- 参数估计：模型相对基准策略的净获益增量。
- 预测结果：可换算为“每千人避免的无效干预数”。
- 不确定性指标：bootstrap 净获益区间。

## 6. 适用场景

- 适合：将预测模型落到二元临床决策、比较多个模型的临床价值、论证模型“值得用”。
- 不适合：非决策型输出、阈值区间无法从临床上界定的场景。
- 使用前需要特别检查的点：阈值区间是否临床合理、概率是否校准、是否在验证集上评估。

## 7. 实现

### 7.1 Python

常用包：

- `dcurves`

```python
import pandas as pd
from dcurves import dca

# df 含真实结局列 outcome(0/1) 与模型预测概率列 model_prob
res = dca(
    data=df,
    outcome="outcome",
    modelnames=["model_prob"],
    thresholds=[i / 100 for i in range(1, 51)],
)
print(res.head())   # 各阈值下 net_benefit
```

### 7.2 R

常用包：

- `dcurves`

```r
library(dcurves)

# df 含结局 outcome(0/1) 与预测概率 model_prob
dca(outcome ~ model_prob,
    data = df,
    thresholds = seq(0.01, 0.50, by = 0.01)) |>
  plot(smooth = TRUE)
```

## 8. 结果如何解释

- 核心结果看什么：在临床相关阈值区间内，模型曲线是否稳定高于 treat-all 与 treat-none。
- 每个主要参数如何解释：净获益差可换算为“每 100（或 1000）人中，在不增加假阳性代价的前提下多找出的真阳性数”。
- 临床或医学意义如何表达：说明模型在“医生愿意接受的权衡范围”内是否带来净收益，而非仅统计显著。
- 常见误读：只看某一个阈值；忽略校准不良会使净获益被高估；把 treat-all 当成 0 基线。

## 9. 假设诊断与稳健性

- 阈值范围必须来自真实临床选择，而非挑选模型占优区间。
- 使用独立/验证预测，必要时校正删失。
- bootstrap 给净获益置信带。
- 同时展示 treat-all 与 treat-none。
- 检查不同中心、亚组和患病率下稳健性。
- 区分模型决策效用与实施效果。

## 10. 推荐可视化

- 决策曲线：净获益对阈值概率，含模型、treat-all、treat-none 三条线。
- 多模型净获益对比图。
- 可选的“干预避免数”曲线（net reduction in interventions）。

## 11. 优势、局限与常见坑

### 优势

- 直接回答临床“值不值得用”，弥补区分度与校准的空白。
- 无需外部指定成本效用，用阈值概率隐含权衡。
- 可同时比较多个模型与基准策略。

### 局限

- 依赖概率校准；校准差会扭曲净获益。
- 阈值区间的选择需要临床判断。
- 净获益的抽样不确定性常被忽略。

### 常见坑

- 阈值区间设得过宽，包含临床无意义区域。
- 用未校准的概率直接算 DCA。
- 只报模型曲线，不与 treat-all/treat-none 对比。

## 12. 与相近方法的区别

- 和 [[ROC曲线与AUC（Receiver Operating Characteristic and AUC）]] 的区别：AUC 评区分度且不含成本权衡，DCA 引入阈值权衡评净获益。
- 和 [[校准曲线（Calibration Curve）]] 的区别：校准评概率是否准，DCA 评这些概率用于决策是否有净收益；二者应配套。
- 和传统成本效益分析的区别：DCA 用阈值概率替代显式成本估计，更易在缺乏精确成本时应用。

## 13. 医学研究中的典型应用

- 论证新风险模型或生物标志物在临床决策中的增量价值。
- 比较“加入新标志物”前后的模型是否值得采用。
- 指导筛查/活检/预防用药阈值的设定。

## 14. 关键术语

- **阈值概率（Threshold probability）**：愿意采取行动的最低风险。
- **净获益（Net benefit）**：真阳性收益减加权假阳性损失。
- **Treat-all**：所有人都采取干预。
- **Treat-none**：无人采取干预。
- **临床影响曲线**：各阈值下高风险人数与真阳性人数。

## 15. 相关方法

- [[ROC曲线与AUC（Receiver Operating Characteristic and AUC）]]
- [[校准曲线（Calibration Curve）]]
- [[Logistic回归（Logistic Regression）]]

## 16. 参考资料

- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making*. 2006;26(6):565-574.
- Vickers AJ, van Calster B, Steyerberg EW. Net benefit approaches to the evaluation of prediction models, molecular markers, and diagnostic tests. *BMJ*. 2016;352:i6.
- dcurves. Decision Curve Analysis. [https://mskcc-epi-bio.github.io/dcurves/](https://mskcc-epi-bio.github.io/dcurves/) （访问日期：2026-07-02）
