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

### 1.1 定义

注意力机制是一类根据“相关性权重”聚合信息的计算方式。它允许模型在处理当前目标时，对输入中的不同位置分配不同关注程度。

### 1.2 它主要解决什么问题

- 研究问题：当输入中只有部分位置对当前预测最关键时，如何动态聚焦这些信息。
- 适用任务：机器翻译、文本分类、时间序列建模、图像描述、跨模态对齐。
- 常见医学场景：长病历文本摘要、实验室指标序列风险预测、多模态病理与文本联合建模。

### 1.3 直觉理解

注意力机制像阅读病历时的“重点标注”。不同于把所有信息平均对待，它会自动给更相关的词、时间点或图像区域更高权重。

## 2. 数学形式

### 2.1 核心公式

缩放点积注意力可写为：

$$
\operatorname{Attention}(Q, K, V) =
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中 $Q$、$K$、$V$ 分别为查询、键和值矩阵，$d_k$ 为键向量维度。

### 2.2 参数或统计量含义

- $Q$：当前要“问什么”。
- $K$：各位置提供的“索引信息”。
- $V$：各位置真正携带的内容。
- 注意力权重：表示当前位置对其他位置的关注程度。

### 2.3 关键假设

- 输入元素之间存在可学习的相关性结构。
- 不同位置对预测的重要性并不相同。
- 训练数据足够支持权重模式学习。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：文本 token 序列、时序特征序列、图像 patch 表征、多模态嵌入。
- 因变量类型：分类、回归、生成、序列标注。
- 数据结构：嵌入矩阵或特征序列。
- 是否适合高维数据：适合，但计算和显存开销可能较大。
- 是否适合缺失较多数据：需结合 mask 机制处理。
- 是否适合删失数据：不直接针对删失。
- 是否适合重复测量数据：适合，只要可组织成顺序或集合表示。

### 3.2 示例表格

以门诊就诊序列风险建模为例：

| PatientID | VisitOrder | DiagnosisCode | LabEmbedding | RiskLabel |
| --- | --- | --- | --- | --- |
| P201 | 1 | I10 | v1 | 0 |
| P201 | 2 | E11 | v2 | 0 |
| P201 | 3 | N18 | v3 | 1 |
| P202 | 1 | J18 | v4 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：嵌入表示序列和可选 mask。
- 关键变量：注意力头数、维度、是否使用 self-attention 或 cross-attention。
- 需要预处理的内容：token 化、嵌入构造、padding、mask。

#### 产出

- 模型对象/统计结果：注意力模块权重和输出表示。
- 参数估计：投影矩阵与输出线性层。
- 预测结果：上下文向量、类别概率、连续输出。
- 不确定性指标：验证集性能、注意力稳定性、外部泛化表现。

## 4. 适用场景

- 适合：长序列、跨位置依赖明显、不同位置重要性差异大的任务。
- 不适合：序列很短且简单、或计算资源非常受限的场景。
- 使用前需要特别检查的点：注意力矩阵规模、mask 设计、是否真正带来性能增益。

## 5. 实现

### 5.1 Python

常用包：

- `torch`

```python
import math
import torch

def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V
```

### 5.2 R

常用包：

- `keras`

```r
# R 中通常通过 keras 的多头注意力层或 torch 自定义模块实现注意力机制。
```

## 6. 结果如何解释

- 核心结果看什么：是否提升预测性能，以及模型是否更稳定地捕捉长距离依赖。
- 每个主要参数如何解释：头数影响并行关注子空间，投影维度影响表示容量。
- 临床或医学意义如何表达：注意力图可作为辅助解释线索，但不能直接等同于因果重要性。
- 常见误读：高注意力权重不必然代表临床因果贡献。

## 7. 推荐可视化

- 注意力热图。
- 预测性能对比图。
- 不同头部关注模式的示意图。

## 8. 优势、局限与常见坑

### 优势

- 能动态建模远距离依赖。
- 对不同信息位置进行自适应加权。
- 是 Transformer 等现代模型的核心组件。

### 局限

- 计算复杂度可较高。
- 注意力权重的解释不总是稳定。
- 单独使用时仍需合适的骨架网络承载。

### 常见坑

- 序列很长却不控制显存和复杂度。
- 把注意力热图当作充分解释证据。
- 忽略 padding 和 mask 的实现细节。

## 9. 与相近方法的区别

- 和 [[循环神经网络（Recurrent Neural Network, RNN）]] 的区别：RNN 依赖递归状态传递，注意力直接比较任意位置相关性。
- 和 [[Transformer（Transformer）]] 的区别：注意力机制是组件，Transformer 是以多头注意力为核心的完整架构。
- 和 [[卷积神经网络（Convolutional Neural Network, CNN）]] 的区别：CNN 更强调局部感受野，注意力可学习全局相关性。

## 10. 医学研究中的典型应用

- 长病历文本风险分类与摘要。
- 多次就诊事件序列建模。
- 医学影像与文本报告的跨模态对齐。

## 11. 相关方法

- [[循环神经网络（Recurrent Neural Network, RNN）]]
- [[Transformer（Transformer）]]
- [[卷积神经网络（Convolutional Neural Network, CNN）]]
- [[长短期记忆网络（Long Short-Term Memory, LSTM）]]

## 12. 参考资料

- Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate. *arXiv*. 2014. [https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
- Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. *Advances in Neural Information Processing Systems*. 2017;30.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
