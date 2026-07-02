---
title: Transformer
english_name: Transformer
slug: transformer
aliases: [transformer, "Transformer（Transformer）"]
category: 神经网络与深度学习
subcategory: 注意力与序列建模
tags: [医学统计, 数据科学, 深度学习, 序列建模]
status: 已建
difficulty: advanced
question_type: 全局依赖并行序列建模
data_type: [序列数据, 文本数据, 图像块序列, 多模态数据]
outcome_type: [二分类, 多分类, 连续型, 序列输出]
python_packages: [torch, tensorflow, keras]
r_packages: [keras, torch]
---

# Transformer（Transformer）

## 1. 方法概览

### 1.1 定义

Transformer 是以多头注意力为核心的序列建模架构。它不依赖传统递归结构，而是通过自注意力和位置编码并行建模序列中的全局依赖。

### 1.2 它主要解决什么问题

- 研究问题：当序列很长、远距离依赖重要且并行训练效率关键时，如何更高效地学习上下文关系。
- 适用任务：文本理解与生成、时间序列建模、图像 patch 建模、多模态学习。
- 常见医学场景：临床文本分类、长病历摘要、时序监测数据预测、多模态诊疗辅助建模。

### 1.3 直觉理解

Transformer 不再一步一步“读”序列，而是让每个位置同时查看所有其他位置，决定哪些信息最相关，再把这些相关信息聚合到当前位置表示里。

## 2. 数学形式

### 2.1 核心公式

Transformer 的核心组件是多头注意力：

$$
\operatorname{head}_i = \operatorname{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

$$
\operatorname{MHA}(Q,K,V) = \operatorname{Concat}(\operatorname{head}_1,\dots,\operatorname{head}_h)W^O
$$

编码器层常写为：

$$
H' = \operatorname{LayerNorm}(H + \operatorname{MHA}(H,H,H))
$$

$$
H^{+} = \operatorname{LayerNorm}(H' + \operatorname{FFN}(H'))
$$

### 2.2 参数或统计量含义

- 头数 $h$：表示并行关注的子空间数。
- $d_{\text{model}}$：隐藏表示维度。
- 位置编码：注入顺序信息。
- FFN：对每个位置独立进行非线性变换。

### 2.3 关键假设

- 全局依赖对任务很重要。
- 数据量和算力足以支撑较大模型训练。
- 顺序信息可以通过位置编码补充。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：文本 token 序列、时间序列窗口、图像 patch、跨模态嵌入。
- 因变量类型：分类、回归、序列生成、掩码预测。
- 数据结构：嵌入序列张量。
- 是否适合高维数据：适合，但长序列时显存开销较大。
- 是否适合缺失较多数据：需通过 mask 或缺失标记处理。
- 是否适合删失数据：不直接处理删失。
- 是否适合重复测量数据：适合，尤其是长程依赖明显的重复测量。

### 3.2 示例表格

以出院小结文本分类为例：

| NoteID | TokenSequence | Department | Outcome |
| --- | --- | --- | --- |
| N001 | t1, t2, ... | cardiology | readmit |
| N002 | t1, t2, ... | oncology | no_readmit |
| N003 | t1, t2, ... | nephrology | readmit |

### 3.3 输入与产出

#### 输入

- 输入数据：token 或时间步嵌入序列。
- 关键变量：层数、头数、隐藏维度、前馈维度、最大序列长度。
- 需要预处理的内容：token 化、padding、mask、位置编码、数据切分。

#### 产出

- 模型对象/统计结果：编码器/解码器参数和验证指标。
- 参数估计：注意力投影矩阵、前馈网络参数。
- 预测结果：类别概率、连续值、生成序列或上下文表征。
- 不确定性指标：外部验证性能、校准、时间外或中心外泛化。

## 4. 适用场景

- 适合：长序列、多模态、需要全局上下文建模和并行计算的任务。
- 不适合：数据量很小、算力受限、或简单时序结构即可解释的问题。
- 使用前需要特别检查的点：序列长度、显存开销、mask 正确性、预训练与微调策略。

## 5. 实现

### 5.1 Python

常用包：

- `torch`

```python
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,
    nhead=4,
    dim_feedforward=256,
    batch_first=True
)
model = nn.TransformerEncoder(encoder_layer, num_layers=2)
```

### 5.2 R

常用包：

- `keras`

```r
# R 中通常通过 keras / torch 调用多头注意力与前馈层组合，
# 构建简化版 Transformer 编码器。
```

## 6. 结果如何解释

- 核心结果看什么：验证性能、长序列上的稳健性、是否优于 RNN/LSTM 基线。
- 每个主要参数如何解释：层数和维度决定容量，头数影响多视角关联学习。
- 临床或医学意义如何表达：适合强调复杂上下文建模能力，但仍需外部验证和错误案例审查。
- 常见误读：注意力权重图不等于完整解释，也不能替代临床机制分析。

## 7. 推荐可视化

- 训练/验证损失曲线。
- 注意力热图。
- 不同序列长度或不同模型架构的性能对比图。

## 8. 优势、局限与常见坑

### 优势

- 并行训练效率高。
- 擅长长程依赖建模。
- 易于扩展到预训练和多模态框架。

### 局限

- 对数据量和算力要求较高。
- 长序列自注意力复杂度较高。
- 结构复杂，可解释性有限。

### 常见坑

- 训练样本不足却使用过大模型。
- 只看内部验证，不做时间外或中心外评估。
- 忽略 token 化和位置编码细节对结果的影响。

## 9. 与相近方法的区别

- 和 [[注意力机制（Attention Mechanism）]] 的区别：注意力是组件，Transformer 是完整架构。
- 和 [[循环神经网络（Recurrent Neural Network, RNN）]] 的区别：Transformer 并行建模全局依赖，RNN 递归顺序处理。
- 和 [[长短期记忆网络（Long Short-Term Memory, LSTM）]] 的区别：LSTM 更适合中短序列和较小模型，Transformer 在大规模长序列上通常更强。

## 10. 医学研究中的典型应用

- 临床文本理解、编码和摘要。
- 多变量监测序列预测。
- 医学图像 patch 建模和多模态诊疗模型。

## 11. 相关方法

- [[注意力机制（Attention Mechanism）]]
- [[循环神经网络（Recurrent Neural Network, RNN）]]
- [[长短期记忆网络（Long Short-Term Memory, LSTM）]]
- [[卷积神经网络（Convolutional Neural Network, CNN）]]

## 12. 参考资料

- Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. *Advances in Neural Information Processing Systems*. 2017;30.
- Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: transformers for image recognition at scale. *arXiv*. 2020. [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
