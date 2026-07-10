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

### 1.1 一句话本质

Transformer 让序列中每个位置用自注意力直接汇总其他位置，再通过残差、归一化和逐位置前馈网络反复加工，从而不依赖递归也能建模全局上下文。

### 1.2 定义

Transformer 是以多头注意力为核心的神经网络架构。一个编码器块通常包含多头自注意力、残差连接、层归一化和前馈网络；解码器还使用因果 mask 与编码器—解码器交叉注意力。由于各位置可并行计算，Transformer 很适合大规模预训练。

### 1.3 它主要解决什么问题

- 研究问题：序列很长、远距离关系重要、递归计算限制并行时，如何学习上下文表示。
- 适用任务：文本理解/生成、序列分类、时间序列预测、图像块建模和多模态融合。
- 常见医学场景：临床文本编码、纵向 EHR 风险预测、医学影像 Transformer、多模态诊疗模型。

### 1.4 直觉与类比

RNN 像按页阅读并不断改写一张摘要；Transformer 像把所有病历页铺在桌面上，每一页都能同时标记与自己最相关的其他页。多头注意力相当于让多位审阅者分别关注用药、检验、时间和症状关系，前馈层再对每页的综合信息做非线性加工。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

递归网络中，位置 1 的信息到位置 100 要经过约 99 次状态传递，路径长且不能完全并行。卷积可并行，但要堆叠多层才能扩大感受野。自注意力让任意两位置在一层内直接交互，将远距离依赖路径缩短为一步；代价是标准全局注意力对序列长度 $T$ 需要约 $T^2$ 的分数矩阵。

### 2.2 关键洞察

Transformer 的本质不只是“注意力”，而是四项组合：

1. 多头自注意力负责跨位置通信。
2. 位置/时间编码补回注意力本身缺少的顺序信息。
3. 逐位置前馈网络负责非线性特征变换。
4. 残差连接与层归一化让深层堆叠可训练。

注意力负责“从哪里取信息”，前馈网络负责“取到后如何变换”；两者交替才构成完整块。

### 2.3 与朴素/相邻做法的对比

- 相比 RNN/LSTM：Transformer 并行且远程路径短，但全局注意力内存更高。
- 相比 CNN：Transformer 全局交互直接、局部先验较弱，通常更依赖数据或预训练。
- 相比单独注意力层：Transformer 加入位置、残差、归一化、FFN 和层堆叠，是完整架构。
- 相比传统 EHR 特征汇总：Transformer 保留事件级顺序和上下文，但解释、校准和数据需求更高。

## 3. 数学形式

### 3.1 核心公式

单头缩放点积自注意力：

$$
\mathbf Q=\mathbf X\mathbf W^Q,\qquad
\mathbf K=\mathbf X\mathbf W^K,\qquad
\mathbf V=\mathbf X\mathbf W^V
$$

$$
\mathbf A
=\operatorname{softmax}\!\left(
\frac{\mathbf Q\mathbf K^\top}{\sqrt{d_k}}+\mathbf M
\right),\qquad
\mathbf Z=\mathbf A\mathbf V
$$

多头注意力：

$$
\operatorname{MHA}(\mathbf X)
=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)
\mathbf W^O
$$

编码器块常写为

$$
\mathbf R
=\operatorname{LayerNorm}
\left(\mathbf X+\operatorname{MHA}(\mathbf X)\right)
$$

$$
\mathbf Y
=\operatorname{LayerNorm}
\left(\mathbf R+\operatorname{FFN}(\mathbf R)\right)
$$

$$
\operatorname{FFN}(\mathbf r)
=\phi(\mathbf r\mathbf W_1+\mathbf b_1)\mathbf W_2+\mathbf b_2
$$

这些式子在说：每个位置先从全序列汇总信息，再独立通过同一个小型 MLP；残差保留原表示，归一化稳定深层训练。

### 3.2 推导脉络

自注意力若只看内容，对输入置换具有等变性，不能区分“先用药后发热”和“先发热后用药”。因此先把 token embedding 与位置/时间编码相加：

$$
\mathbf x_t=\mathbf e_t+\mathbf p_t
$$

然后在每层重复“跨位置通信—逐位置计算”。分类任务可取特殊分类 token、最后位置或池化表示；生成任务用上三角因果 mask 禁止查看未来。预训练通过遮盖预测、下一 token 预测或对比目标学习通用表示，再在医学任务上微调。

### 3.3 参数与统计量含义

- $\mathbf X$：加入位置/时间信息的 token 表示矩阵。
- $\mathbf W^Q,\mathbf W^K,\mathbf W^V$：query/key/value 投影。
- $\mathbf A$：每个 query 对所有 keys 的注意力矩阵。
- $H$：注意力头数；各头在不同投影子空间工作。
- $\mathbf M$：padding 或 causal mask。
- FFN：对每个位置独立应用、参数共享的两层网络。
- residual：把模块输入直接加回输出。
- LayerNorm：在单个 token 的特征维上标准化。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| token 化保留临床信息 | 事件/词/图像块表示合理 | 关键信息在输入前已丢失 | 审核词表、时间窗和 OOV |
| 位置/时间编码表达真实顺序 | 模型能区分先后与间隔 | 顺序或间隔被误解 | 时间打乱、间隔分层 |
| mask 与任务一致 | 实时预测不看未来，padding 被排除 | 严重泄漏或伪影 | 权重与特征时间审计 |
| 数据量/预训练支撑容量 | 高参数模型有足够信息 | 过拟合或不稳定 | 学习曲线、简单基线 |
| 训练分布覆盖应用域 | 术语、中心、设备与流程可迁移 | 外部性能和校准下降 | 跨中心/时间验证 |

## 4. 手把手算例

用两个 token、二维表示手算一个简化编码器块。设

$$
\mathbf X=
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix},
\qquad
\mathbf W^Q=\mathbf W^K=\mathbf W^V=\mathbf I
$$

因此 $\mathbf Q=\mathbf K=\mathbf V=\mathbf X$，且 $d_k=2$。

**第 1 步：计算所有 token 两两分数。**

$$
\frac{\mathbf Q\mathbf K^\top}{\sqrt2}
=
\begin{pmatrix}
0.707&0\\
0&0.707
\end{pmatrix}
$$

**第 2 步：逐行 softmax。** 因 $\exp(0.707)=2.028$：

$$
\mathbf A
=
\begin{pmatrix}
0.670&0.330\\
0.330&0.670
\end{pmatrix}
$$

每个 token 对自身权重 0.670，也从另一个 token 汇入 0.330。

**第 3 步：加权 values。**

$$
\mathbf Z=\mathbf A\mathbf V
=
\begin{pmatrix}
0.670&0.330\\
0.330&0.670
\end{pmatrix}
$$

**第 4 步：残差连接。** 暂时略去 LayerNorm 以便手算：

$$
\mathbf R=\mathbf X+\mathbf Z
=
\begin{pmatrix}
1.670&0.330\\
0.330&1.670
\end{pmatrix}
$$

**第 5 步：逐位置 FFN。** 为简化，设所有元素为正且 $\operatorname{FFN}(\mathbf r)=0.5\mathbf r$：

$$
\mathbf Y=\mathbf R+0.5\mathbf R
=
\begin{pmatrix}
2.505&0.495\\
0.495&2.505
\end{pmatrix}
$$

这个例子展示了完整逻辑：自注意力让两个位置交换信息，残差保留原身份，FFN 再逐位置加工。若不加入位置编码，交换两个输入 token 只会交换两行输出，模型无法知道哪个是“先”发生的。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：文本 token、医疗事件、时间片、图像 patch 或多模态 token。
- 因变量类型：分类、回归、序列生成、遮盖重构。
- 数据结构：batch × sequence length × embedding，配 mask 与位置/时间编码。
- 是否适合高维数据：适合，但参数和显存需求高。
- 是否适合缺失较多数据：可显式编码 mask/缺失 token；信息性缺失仍需分析。
- 是否适合删失数据：标准目标不处理删失，需生存损失。
- 是否适合重复测量数据：适合事件序列预测，但需按患者切分并表达真实间隔。

### 5.2 示例表格

| patient_id | event_time | token | value | delta_t | valid_mask |
| --- | --- | --- | ---: | ---: | ---: |
| P01 | 08:00 | heart_rate | 82 | 0 | 1 |
| P01 | 10:00 | lactate | 3.2 | 2 | 1 |
| P01 | NA | PAD | 0 | 0 | 0 |

### 5.3 输入与产出

#### 输入

- token/patch/事件序列、位置或时间编码、padding/causal mask。
- 模型维度、头数、层数、FFN 宽度、dropout 和预训练权重。
- 患者级、时间前向的训练/验证/测试划分。

#### 产出

- 每位置上下文表示或池化后的样本表示。
- 分类概率、连续预测或生成序列。
- 注意力矩阵与中间表示，可用于诊断但非天然解释。
- 普通 Transformer 不自动给出传统效应量、删失校正或因果关系。

## 6. 适用场景

- 适合：大规模文本/事件/图像数据、远距离依赖重要、可利用预训练。
- 不适合：样本很少且无预训练、序列极长而资源有限、实时流式延迟严格，或简单模型已充分。
- 使用前检查：token 化、时间/位置编码、未来泄漏、padding、序列截断、亚组和外部校准。

## 7. 实现

### 7.1 Python

常用包：PyTorch。示例用一层 Transformer 编码合成序列并预测最后三步之和是否为正。

```python
import numpy as np
import torch
from torch import nn

torch.manual_seed(42)
rng = np.random.default_rng(42)
X = rng.normal(size=(800, 12, 1)).astype("float32")
y = (X[:, -3:, 0].sum(axis=1) > 0).astype("float32")

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(1, 8)
        self.position = nn.Parameter(torch.zeros(1, 12, 8))
        layer = nn.TransformerEncoderLayer(
            d_model=8, nhead=2, dim_feedforward=16,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        h = self.input_proj(x) + self.position[:, :x.size(1)]
        h = self.encoder(h)
        return self.output(h.mean(dim=1)).squeeze(-1)

model = TinyTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()
x_train = torch.tensor(X[:600])
y_train = torch.tensor(y[:600])

for _ in range(100):
    optimizer.zero_grad()
    loss = loss_fn(model(x_train), y_train)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    prob = torch.sigmoid(model(torch.tensor(X[600:])))
    accuracy = ((prob >= 0.5) == torch.tensor(y[600:])).float().mean()
print(float(accuracy))
```

### 7.2 R

常用包：keras3。这里把归一化时间位置作为第二输入通道，显式提供顺序。

```r
library(keras3)

set_random_seed(42)
value <- array(rnorm(800 * 12), dim = c(800, 12, 1))
y <- as.numeric(apply(value[, 10:12, 1], 1, sum) > 0)
position <- array(
  rep(seq(0, 1, length.out = 12), each = 800),
  dim = c(800, 12, 1)
)
x <- array(c(value, position), dim = c(800, 12, 2))
idx <- sample(seq_len(800), 600)

inputs <- layer_input(shape = c(12, 2))
h <- inputs |> layer_dense(8)
attn_layer <- layer_multi_head_attention(num_heads = 2, key_dim = 4)
attn <- attn_layer(query = h, value = h)
h1 <- layer_add(list(h, attn)) |> layer_layer_normalization()
ffn <- h1 |>
  layer_dense(16, activation = "relu") |>
  layer_dense(8)
h2 <- layer_add(list(h1, ffn)) |> layer_layer_normalization()
outputs <- h2 |>
  layer_global_average_pooling_1d() |>
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)
model |> compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = metric_auc(name = "auc")
)
model |> fit(
  x[idx, , , drop = FALSE], y[idx],
  validation_split = 0.2, epochs = 30, batch_size = 32,
  callbacks = callback_early_stopping(
    patience = 4, restore_best_weights = TRUE
  ),
  verbose = 0
)
model |> evaluate(x[-idx, , , drop = FALSE], y[-idx])
```

## 8. 结果如何解读

- 与其他预测模型一样，最终看患者级外部测试表现、校准、阈值和临床效用，不看预训练损失就下结论。
- 注意力图展示信息路由，不等于忠实或因果解释；残差和 FFN 也会影响输出。
- 预训练模型的高性能可能来自数据重叠、中心线索或编码习惯，需做去重与时间审计。
- 截断长病历会丢失早期信息，应报告最大长度、截断方向和覆盖比例。
- 生成式 Transformer 可能产生流畅但错误内容，临床使用需检索、约束、置信与人工复核。

## 9. 假设诊断与稳健性

- 泄漏审计：核对预训练/微调/测试患者、文档和时间是否重叠。
- mask 测试：实时任务检查未来位置权重和特征是否严格不可见。
- 位置消融：打乱顺序、移除时间间隔或改变位置编码，验证模型确实使用时序。
- 长度稳健性：按序列长度、截断程度和稀疏度分层评估。
- 基线比较：同切分下与 Logistic/GBDT、RNN/LSTM、CNN 比较。
- 外部验证：跨医院、设备、术语系统和时间段评估性能与校准。

## 10. 推荐可视化

- token/事件时间线与 padding/causal mask。
- 多层多头注意力热图，配合遮挡验证。
- 不同序列长度和截断比例下的性能曲线。
- 预训练—微调学习曲线和校准曲线。
- 按中心、亚组、设备和时间的性能森林图。

## 11. 优势、局限与常见坑

### 优势

- 远距离位置可在一层直接交互。
- 训练高度并行，适合大规模预训练。
- 同一基本架构可处理文本、图像、时间序列和多模态。

### 局限

- 标准全局注意力时间和内存约随长度平方增长。
- 参数多、数据和计算需求高。
- 位置、mask、token 化与预训练数据问题会显著影响结果。

### 常见坑

- 未加入位置/时间信息却要求理解顺序。
- 在线预测使用了双向上下文或未来事件。
- 同一患者文档跨预训练、微调和测试。
- 把注意力图当成临床机制证据。
- 只因模型更大就忽略简单强基线。

## 12. 与相近方法的区别

- 与注意力机制：注意力是模块；Transformer 是注意力、位置、残差、归一化和 FFN 的架构。
- 与 RNN/LSTM/GRU：Transformer 并行且全局路径短，循环模型更自然地流式更新。
- 与 CNN：CNN 有强局部和平移先验；Transformer 全局灵活但通常更依赖数据/预训练。
- 与 MLP：MLP 不共享序列位置间的内容依赖；Transformer 的注意力显式建立 token 间交互。
- 如何选择：小到中等数据先比较树/RNN/CNN；有大规模预训练、长程依赖和算力时再采用 Transformer。

## 13. 医学研究中的典型应用

- 临床文本分类、信息抽取、摘要和问答。
- 纵向 EHR 中诊断、检验、处方序列预测。
- Vision Transformer 用于医学图像分类和分割。
- 影像、文本、组学与结构化 EHR 的多模态融合。

必须报告数据时间范围、预训练来源、患者去重、token 化、截断、缺失、类别不平衡和外部验证。时间到事件结局仍需删失感知损失；Transformer 不自动解决混杂或幻觉。

## 14. 关键术语

- **Token**：序列中的基本建模单位，如词、诊断事件或图像块。
- **Embedding**：把离散 token 或连续输入映射为稠密向量。
- **位置编码（Positional encoding）**：向表示加入顺序/位置的信息。
- **自注意力（Self-attention）**：同一序列内部各位置相互聚合。
- **残差连接（Residual connection）**：把模块输入直接加回输出。
- **层归一化（Layer normalization）**：在单个 token 的特征维上标准化。
- **前馈网络（Feed-forward network, FFN）**：对每个位置独立应用的共享 MLP。
- **编码器（Encoder）**：把输入转成上下文表示的 Transformer 堆栈。
- **解码器（Decoder）**：在因果 mask 下逐步生成输出的堆栈。
- **预训练（Pretraining）**：先在大规模任务上学习通用参数，再微调下游任务。

## 15. 相关方法

- [[注意力机制（Attention Mechanism）]]
- [[循环神经网络（Recurrent Neural Network, RNN）]]
- [[长短期记忆网络（Long Short-Term Memory, LSTM）]]
- [[门控循环单元（Gated Recurrent Unit, GRU）]]
- [[卷积神经网络（Convolutional Neural Network, CNN）]]

## 16. 参考资料

- Vaswani A, et al. Attention is all you need. *NeurIPS*. 2017;30.
- Devlin J, Chang MW, Lee K, Toutanova K. BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*. 2019:4171-4186. https://doi.org/10.18653/v1/N19-1423
- Dosovitskiy A, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*. 2021. https://arxiv.org/abs/2010.11929
- Li Y, et al. BEHRT: Transformer for electronic health records. *Scientific Reports*. 2020;10:7155. https://doi.org/10.1038/s41598-020-62922-y
