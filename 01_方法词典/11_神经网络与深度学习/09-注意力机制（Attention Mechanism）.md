---
title: 注意力机制
english_name: Attention Mechanism
slug: attention-mechanism
aliases: [attention, attention mechanism, "注意力机制（Attention Mechanism）"]
category: 神经网络与深度学习
subcategory: 注意力与序列建模
tags: [医学统计, 数据科学, 深度学习, 序列建模]
status: 已建
difficulty: advanced
question_type: 依赖关系加权建模
data_type: [序列数据, 文本数据, 图像特征, 图数据]
outcome_type: [上下文表征, 二分类, 多分类, 连续型]
python_packages: [torch, tensorflow, keras]
r_packages: [keras, torch]
---

# 注意力机制（Attention Mechanism）

## 1. 方法概览

### 1.1 一句话本质

注意力机制先计算“当前查询与每条信息有多相关”，再把相关性归一化成权重，对各条信息做数据依赖的加权汇总。

### 1.2 定义

注意力不是一种完整网络，而是一类信息检索与聚合运算。它把输入表示成查询（query）、键（key）和值（value）：查询与键产生相关性分数，softmax 将分数变成权重，值按权重求和得到上下文表示。查询与键值来自同一序列时称自注意力，来自不同来源时称交叉注意力。

### 1.3 它主要解决什么问题

- 研究问题：输入很长且不同位置的重要性随当前任务变化时，如何动态选择信息。
- 适用任务：序列编码、机器翻译、文本分类、图像区域聚合、时序预测和多模态对齐。
- 常见医学场景：病历文本摘要、长期检验序列风险预测、影像—文本融合、病理图像块聚合。

### 1.4 直觉与类比

查询像医生当前要回答的问题，键像病历条目的索引标签，值像条目的实际内容。若问题是“近期感染证据”，白细胞、体温和培养记录的键会与查询更匹配；最终摘要不是挑一条，而是按相关程度混合多个值。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

普通 RNN 要把整个历史压进固定长度状态，远处信息可能被遗忘；平均池化又让所有位置权重相同。实际任务中，重要位置因样本和查询而异。注意力让模型在需要时直接回看全部候选位置，避免先把所有信息不可逆压成一个摘要。

### 2.2 关键洞察

注意力把“找谁”和“取什么”分开：

- query 表示当前需求；
- key 表示每条信息可被什么需求命中；
- value 承载真正要汇总的内容。

同一 value 可因 query 不同而获得不同权重，因此注意力是条件化的动态加权。softmax 保证权重非负且和为 1，使输出成为值的加权平均；温度、缩放和 mask 决定权重有多尖锐、哪些位置可参与。

### 2.3 与朴素/相邻做法的对比

- 相比平均池化：注意力权重随样本和查询变化。
- 相比最大池化：注意力能软性组合多条证据并保持可微。
- 相比 RNN 最后状态：注意力提供到任意位置的短路径。
- 相比特征重要性：注意力权重是内部信息路由，不等同于因果贡献或忠实解释。

## 3. 数学形式

### 3.1 核心公式

缩放点积注意力为

$$
\operatorname{Attention}(\mathbf Q,\mathbf K,\mathbf V)
=
\operatorname{softmax}\!\left(
\frac{\mathbf Q\mathbf K^\top}{\sqrt{d_k}}+\mathbf M
\right)\mathbf V
$$

对单个 query：

$$
s_i=\frac{\mathbf q^\top\mathbf k_i}{\sqrt{d_k}},
\qquad
\alpha_i=\frac{\exp(s_i)}
{\sum_j\exp(s_j)},
\qquad
\mathbf c=\sum_i\alpha_i\mathbf v_i
$$

这个式子在说：先用 query–key 相似度给每个位置打分，再把分数变成和为 1 的权重，最后加权汇总 values。mask $\mathbf M$ 在禁止位置放一个极大负数，使其 softmax 权重接近 0。

### 3.2 推导脉络

最朴素的检索是找相似度最大的 key，取对应 value，但 argmax 不连续且只保留一个位置。用 softmax 代替硬选择，就得到可微的软检索。高维点积方差随 $d_k$ 增大，可能把 softmax 推入饱和区，因此除以 $\sqrt{d_k}$ 保持尺度稳定。加性注意力则用小网络计算 $s_i$，思想相同。

多头注意力让不同投影子空间分别执行注意力：

$$
\operatorname{head}_h
=\operatorname{Attention}
(\mathbf Q\mathbf W_h^Q,\mathbf K\mathbf W_h^K,\mathbf V\mathbf W_h^V)
$$

再拼接各头并线性投影。

### 3.3 参数与统计量含义

- $\mathbf Q$：queries，每一行代表一个当前需求。
- $\mathbf K$：keys，用于与 query 匹配。
- $\mathbf V$：values，被实际加权汇总的内容。
- $d_k$：key/query 维度，决定缩放因子。
- $s_i$：未归一化相关性分数。
- $\alpha_i$：softmax 权重，非负且和为 1。
- $\mathbf M$：padding、因果或业务规则 mask。
- $\mathbf c$：上下文向量，是 values 的加权组合。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| key–query 相似度能表达相关性 | 投影空间中内积有任务意义 | 权重无法找到有效位置 | 消融与检索案例审查 |
| 候选信息可被加权组合 | value 混合保留任务信息 | 平均后丢失离散结构 | 比较硬选择/池化 |
| mask 正确 | padding、未来位置被排除 | 泄漏或补齐伪影 | 检查权重和置换测试 |
| 训练覆盖查询—信息组合 | 学到的匹配规则能外推 | 新中心/术语下失效 | 外部验证、错例分析 |
| 权重不被误当作因果解释 | 注意力只是内部路由 | 产生错误机制结论 | 梯度/遮挡/反事实对照 |

## 4. 手把手算例

设一个查询和三个键均为二维向量：

$$
\mathbf q=(1,0)
$$

$$
\mathbf k_1=(1,0),\quad
\mathbf k_2=(0,1),\quad
\mathbf k_3=(1,1)
$$

对应 values 为

$$
\mathbf v_1=(1,0),\quad
\mathbf v_2=(0,2),\quad
\mathbf v_3=(3,1)
$$

**第 1 步：算缩放点积分数。** 因 $d_k=2$：

$$
(s_1,s_2,s_3)
=\frac{(1,0,1)}{\sqrt2}
=(0.707,0,0.707)
$$

**第 2 步：softmax。** $\exp(0.707)=2.028$，总和为 $2.028+1+2.028=5.056$：

$$
(\alpha_1,\alpha_2,\alpha_3)
=(0.401,0.198,0.401)
$$

**第 3 步：加权 values。**

$$
\begin{aligned}
\mathbf c
&=0.401(1,0)+0.198(0,2)+0.401(3,1)\\
&=(0.401,0)+(0,0.396)+(1.203,0.401)\\
&=(1.604,0.797)
\end{aligned}
$$

查询在第一维为 1，因此与 $\mathbf k_1,\mathbf k_3$ 同样匹配，两者各拿约 40.1% 权重；正交的 $\mathbf k_2$ 仍保留 19.8%，体现 softmax 是软选择。若将第二位置 mask，其分数视为负无穷，剩余两项重新归一化为 0.5/0.5。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：序列表示、图像块、图节点、多模态 token。
- 因变量类型：注意力本身无固定结局，由下游分类/回归/生成任务决定。
- 数据结构：batch × length × embedding，并配有效位置 mask。
- 是否适合高维数据：适合，但全局自注意力的时间/内存常随长度平方增长。
- 是否适合缺失较多数据：可用 mask 排除缺失，但信息性缺失仍需建模。
- 是否适合删失数据：注意力不自动处理删失。
- 是否适合重复测量数据：可聚合时序记录，但需表达实际时间间隔并按患者切分。

### 5.2 示例表格

| patient_id | time | token_type | value | valid_mask |
| --- | ---: | --- | ---: | ---: |
| P01 | 0 | temperature | 38.6 | 1 |
| P01 | 2 | WBC | 15.2 | 1 |
| P01 | 4 | padding | 0 | 0 |

### 5.3 输入与产出

#### 输入

- query、key、value 张量与 padding/causal mask。
- 投影维度、头数、dropout 和下游损失。
- 位置或时间编码，若顺序/间隔重要。

#### 产出

- 上下文表示和可选注意力矩阵。
- 下游预测由后续层产生。
- 权重可用于诊断信息路由，不自动等于特征贡献。
- 不产出传统效应量、置信区间或删失校正。

## 6. 适用场景

- 适合：候选信息多、重要位置随查询变化、需跨较远位置聚合。
- 不适合：序列极长且平方复杂度不可承受、数据很少、简单池化已足够或要求严格因果解释。
- 使用前检查：mask、位置编码、序列长度、权重饱和、未来泄漏和外部域偏移。

## 7. 实现

### 7.1 Python

核心计算只需 NumPy；随后给出 PyTorch 多头注意力的张量形状。

```python
import numpy as np
import torch

q = np.array([[1.0, 0.0]])          # 1 query
k = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])          # 3 keys
v = np.array([[1.0, 0.0],
              [0.0, 2.0],
              [3.0, 1.0]])          # 3 values

scores = q @ k.T / np.sqrt(k.shape[1])
weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)
context = weights @ v
print(weights, context)

# 可训练的多头自注意力
mha = torch.nn.MultiheadAttention(
    embed_dim=4, num_heads=2, batch_first=True
)
x = torch.randn(2, 5, 4)  # batch=2, length=5, embedding=4
context_t, attention_t = mha(x, x, x, need_weights=True)
print(context_t.shape, attention_t.shape)
```

### 7.2 R

核心缩放点积注意力可直接用矩阵运算实现。

```r
q <- matrix(c(1, 0), nrow = 1)
k <- rbind(c(1, 0), c(0, 1), c(1, 1))
v <- rbind(c(1, 0), c(0, 2), c(3, 1))

scores <- q %*% t(k) / sqrt(ncol(k))
softmax <- function(x) {
  e <- exp(x - max(x))
  e / sum(e)
}
weights <- softmax(as.numeric(scores))
context <- matrix(weights, nrow = 1) %*% v

print(weights)
print(context)
```

正式训练时可使用 keras3 的多头注意力层或 torch 的 nn_multihead_attention，并同时传入 padding/causal mask。

## 8. 结果如何解读

- 每行权重和为 1，表示针对一个 query 在候选位置间如何分配聚合比例。
- 权重高不保证该位置对最终预测的因果贡献大：value 大小、残差连接、后续层都可能改变影响。
- 不同头可能学到冗余或不同关系，但“某头专门代表某临床概念”需独立验证。
- 平均所有层/头的热图会隐藏差异，报告时需说明聚合方式。
- 若用于实时预测，任何未来位置权重非零都提示因果 mask 失败。

## 9. 假设诊断与稳健性

- mask 审计：检查 padding 与未来位置权重是否严格为 0。
- 权重置换/遮挡：移除高权重位置，观察预测是否真的变化。
- 替代解释：与梯度、integrated gradients、SHAP 和反事实遮挡比较。
- 稳定性：多随机种子与输入小扰动下比较权重分布。
- 长度偏倚：按序列长度和有效 token 数分层评价。
- 域偏移：检查新中心术语、设备或采样频率下注意力是否转向伪影。

## 10. 推荐可视化

- query × key 注意力热图，同时显示 mask。
- 单病例时间线叠加注意力权重与预测风险。
- 多头并列图，避免只展示最“好看”的头。
- 遮挡前后预测变化与注意力权重散点图。
- 按亚组和中心汇总的权重分布。

## 11. 优势、局限与常见坑

### 优势

- 动态聚合，重要位置随查询和样本改变。
- 任意两位置间梯度路径短。
- 可用于序列、图像、图和多模态。

### 局限

- 全局注意力常有平方时间/内存成本。
- 权重可能弥散、冗余或不稳定。
- 注意力权重不是天然可信解释。

### 常见坑

- 忘记 padding 或 causal mask。
- 没有位置编码却期望模型理解顺序。
- 把所有头平均后作强机制结论。
- 用注意力热图替代性能、校准和外部验证。
- 实时任务误用双向/全序列上下文。

## 12. 与相近方法的区别

- 与平均/最大池化：注意力是数据依赖的软加权。
- 与 RNN：注意力可直接访问远处位置，RNN 通过状态递推。
- 与 Transformer：注意力是一种运算；Transformer 是以多头自注意力为核心的完整架构。
- 与 SHAP：SHAP 是预测归因框架，注意力是模型内部计算；二者都不自动给因果解释。
- 如何选择：简单固定汇总够用时先用池化；重要位置随任务变化且数据量足够时再用注意力。

## 13. 医学研究中的典型应用

- 长病历文本中聚合与当前诊断相关的句子。
- 纵向 EHR 中选择关键检验和就诊时点。
- 多实例病理学习中聚合图像块。
- 影像、文本和结构化数据的跨模态对齐。

必须明确 query 的临床含义、时间 mask、缺失机制、患者级切分和权重解释边界。注意力不自动处理混杂、删失或类别不平衡。

## 14. 关键术语

- **查询（Query）**：表示当前要寻找什么的信息向量。
- **键（Key）**：用于与查询匹配的索引向量。
- **值（Value）**：被实际加权汇总的内容向量。
- **注意力分数（Attention score）**：query 与 key 的未归一化相关性。
- **Softmax 权重**：非负且和为 1 的软选择比例。
- **自注意力（Self-attention）**：query、key、value 来自同一序列。
- **交叉注意力（Cross-attention）**：query 与 key/value 来自不同序列或模态。
- **多头注意力（Multi-head attention）**：在多个投影子空间并行执行注意力。
- **因果 mask（Causal mask）**：禁止当前位置查看未来位置的遮罩。

## 15. 相关方法

- [[Transformer（Transformer）]]
- [[循环神经网络（Recurrent Neural Network, RNN）]]
- [[长短期记忆网络（Long Short-Term Memory, LSTM）]]
- [[卷积神经网络（Convolutional Neural Network, CNN）]]
- [[SHAP（Shapley Additive Explanations, SHAP）]]

## 16. 参考资料

- Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate. *ICLR*. 2015. https://arxiv.org/abs/1409.0473
- Luong MT, Pham H, Manning CD. Effective approaches to attention-based neural machine translation. *EMNLP*. 2015:1412-1421. https://doi.org/10.18653/v1/D15-1166
- Vaswani A, et al. Attention is all you need. *NeurIPS*. 2017;30.
- Jain S, Wallace BC. Attention is not explanation. *NAACL*. 2019:3543-3556. https://doi.org/10.18653/v1/N19-1357
