---
title: 循环神经网络
english_name: Recurrent Neural Network, RNN
slug: recurrent-neural-network-rnn
aliases: [RNN, recurrent neural network, "循环神经网络（Recurrent Neural Network, RNN）"]
category: 神经网络与深度学习
subcategory: 循环神经网络
tags: [医学统计, 数据科学, 深度学习, 序列建模]
status: 已建
difficulty: intermediate
question_type: 序列依赖建模
data_type: [序列数据, 时间序列数据, 文本数据]
outcome_type: [二分类, 多分类, 连续型, 序列输出]
python_packages: [torch, tensorflow, keras]
r_packages: [keras, torch]
---

# 循环神经网络（Recurrent Neural Network, RNN）

## 1. 方法概览

### 1.1 一句话本质

RNN 用一个随时间递推的隐藏状态充当“压缩记忆”，让当前预测同时依赖当前输入和此前序列。

### 1.2 定义

循环神经网络是一类面向有序数据的人工神经网络。它在每个时间步复用同一组参数，将当前输入与上一步隐藏状态合并为新状态；可用于序列到标签、序列到数值、序列到序列等任务。本卡的“普通 RNN”特指 tanh/ReLU 型简单循环单元，而不是 LSTM、GRU 等门控变体。

### 1.3 它主要解决什么问题

- 研究问题：顺序和上下文会改变当前信息含义时，如何让历史参与预测。
- 适用任务：序列分类、逐时点预测、时间序列回归、文本和事件序列建模。
- 常见医学场景：生命体征预警、按就诊顺序建模 EHR、ECG/EEG 序列分类、临床文本编码。

### 1.4 直觉与类比

RNN 像医生一边翻阅病历一边维护一张“当前印象”便笺。每读到一个新时间点，模型把新观察和旧便笺合并成新便笺，再传到下一步。便笺长度固定，所以它不是完整保存历史，而是学习哪些历史信息值得压缩进状态。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

把每个时间点独立建模会丢掉顺序；把固定长度序列全部展平又要求所有病例等长，参数随时间窗增长，并且同一模式在不同时间位置要重复学习。RNN 用同一状态转移函数扫描任意长度序列：参数量不随序列长度增长，顺序通过递推路径进入预测。

### 2.2 关键洞察

RNN 的关键是“参数共享的状态机”。当前状态

$$
\mathbf h_t=f(\mathbf x_t,\mathbf h_{t-1})
$$

既是到当前为止的摘要，也是下一步的上下文。把循环网络沿时间展开后，它像一个层数等于序列长度、但各层共享权重的深网络。正因为很深，反向传播要连续乘许多 Jacobian；乘积趋近 0 产生梯度消失，过大则产生梯度爆炸。

### 2.3 与朴素/相邻做法的对比

- 相比独立时间点模型：RNN 保留顺序和上下文。
- 相比滞后特征回归：RNN 自动压缩多个滞后，但系数不再直观。
- 相比 LSTM/GRU：普通 RNN 参数少、短依赖够用，却更容易遗忘远处信息。
- 相比 Transformer：RNN 计算按时间顺序进行，长序列不易并行；Transformer 可并行看全局，但内存成本通常更高。

## 3. 数学形式

### 3.1 核心公式

最简单的 RNN 为

$$
\mathbf a_t
=\mathbf W_x\mathbf x_t
+\mathbf W_h\mathbf h_{t-1}
+\mathbf b_h,\qquad
\mathbf h_t=\tanh(\mathbf a_t)
$$

$$
\hat{\mathbf y}_t
=g(\mathbf W_y\mathbf h_t+\mathbf b_y)
$$

这个式子在说：每一步都用同一套输入权重和循环权重更新记忆，再从记忆生成当前输出。序列分类常只用最后状态 $\mathbf h_T$，逐时预测则在每个 $t$ 输出。

时间反向传播（BPTT）中的一段梯度包含

$$
\frac{\partial\mathbf h_t}{\partial\mathbf h_{t-k}}
=
\prod_{j=t-k+1}^{t}
\left[
\operatorname{diag}\!\left(1-\mathbf h_j^2\right)\mathbf W_h
\right]
$$

### 3.2 推导脉络

若希望任意长度序列都能使用同一模型，就把“读一个输入并更新状态”的函数在各时间点重复。前向传播按 $1\rightarrow T$ 递推；训练时将递推图展开，累计各时间点损失，再从 $T\rightarrow1$ 用链式法则传梯度。共享参数在每一步都会收到梯度贡献，因此总梯度是所有时间位置贡献之和。

梯度连乘解释了长程困难：当循环权重和 tanh 导数的有效模小于 1，早期信号指数衰减；大于 1 时可能爆炸。LSTM/GRU 用门控的加性记忆路径缓解了这一问题。

### 3.3 参数与统计量含义

- $\mathbf x_t$：时间点 $t$ 的输入向量。
- $\mathbf h_t$：隐藏状态，即截至 $t$ 的学习型摘要。
- $\mathbf W_x$：输入到状态的权重。
- $\mathbf W_h$：上一步状态到新状态的循环权重。
- $\mathbf W_y$：状态到输出的权重。
- $g$：由任务决定的输出函数，如 sigmoid、softmax 或恒等函数。
- $T$：有效序列长度；批训练时需 padding 与 mask 区分真实时点和补齐位置。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 顺序有信息 | 调换时间点会改变目标关系 | RNN 复杂度无收益 | 与打乱顺序/静态基线比较 |
| 状态转移近似稳定 | 同一参数可跨时间复用 | 不同阶段机制差异被混合 | 分阶段评估、加入时间特征 |
| 时间间隔已表达 | 相邻步的实际间隔可比较或显式输入 | 把 1 小时和 30 天当成同一步 | 输入 delta_t，做间隔分层 |
| padding 被正确屏蔽 | 补零不是真实观测 | 模型学习序列长度伪影 | 检查 mask 与长度分布 |
| 患者间切分独立 | 同一患者的序列不跨集合 | 严重泄漏 | patient_id 分组切分 |

## 4. 手把手算例

用一个只有一个隐藏单元的 RNN 读取三步序列 $x=(1,2,0)$。设

$$
h_t=\tanh(0.5x_t+0.8h_{t-1}),\qquad h_0=0
$$

最后用 $\hat p=\sigma(h_3-0.2)$ 预测阳性。

**第 1 步：读入 $x_1=1$。**

$$
a_1=0.5\times1+0.8\times0=0.5,\qquad
h_1=\tanh(0.5)=0.462
$$

**第 2 步：读入 $x_2=2$。**

$$
a_2=0.5\times2+0.8\times0.462=1.370
$$

$$
h_2=\tanh(1.370)=0.879
$$

**第 3 步：当前输入为 0，但历史仍在状态中。**

$$
a_3=0.5\times0+0.8\times0.879=0.703
$$

$$
h_3=\tanh(0.703)=0.606
$$

**第 4 步：输出概率。**

$$
\hat p=\sigma(0.606-0.2)=0.600
$$

即使最后一个输入是 0，模型仍因前两步历史给出 60.0% 阳性概率。

**第 5 步：亲眼看梯度衰减。** 每一步的循环局部导数是

$$
\frac{\partial h_t}{\partial h_{t-1}}
=0.8(1-h_t^2)
$$

三步约为 $0.629,\ 0.182,\ 0.506$，从 $h_0$ 传到 $h_3$ 的乘积只有

$$
0.629\times0.182\times0.506=0.058
$$

早期状态影响只剩约 5.8%。长序列继续连乘会更小，这就是普通 RNN 难学长程依赖的数值根源。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：按时间/位置排序的连续、二元或嵌入向量。
- 因变量类型：序列级分类/回归，或每时间点的分类/回归。
- 数据结构：样本 × 时间 × 特征的张量，附有效长度或 mask。
- 是否适合高维数据：可先用嵌入、CNN 或降维压缩每步输入。
- 是否适合缺失较多数据：需插补并加入缺失 mask、距上次观测时间等信息。
- 是否适合删失数据：标准损失不处理删失，需生存 RNN 或离散时间风险损失。
- 是否适合重复测量数据：适合预测型序列建模，但不是传统群体平均/个体随机效应推断的替代。

### 5.2 示例表格

| patient_id | hour | heart_rate | spo2 | lab_missing | deterioration_next_6h |
| --- | ---: | ---: | ---: | ---: | ---: |
| P01 | 0 | 82 | 97 | 0 | 0 |
| P01 | 1 | 96 | 94 | 1 | 0 |
| P01 | 2 | 108 | 91 | 0 | 1 |
| P02 | 0 | 76 | 98 | 0 | 0 |

建模前还需定义预测锚点，保证每个输入时间窗严格早于结局窗口。

### 5.3 输入与产出

#### 输入

- 变长序列、时间戳、mask 与标签。
- 隐藏维度、层数、单向/双向、截断长度和优化设置。
- 训练集拟合的标准化参数与患者级切分。

#### 产出

- 最后隐藏状态或全部时间状态。
- 序列级/时间点级预测。
- 训练损失、验证指标和可选嵌入。
- 普通 RNN 不自动提供个体随机效应、可解释系数或删失校正。

## 6. 适用场景

- 适合：短到中等长度序列，顺序重要，需要参数共享的动态预测。
- 不适合：长程依赖是核心、时间间隔极不规则且未建模、主要目标是纵向参数解释，或简单滞后模型已足够。
- 使用前检查：预测时点、未来信息泄漏、变长 mask、采样频率、缺失机制、序列长度和类别不平衡。

## 7. 实现

### 7.1 Python

常用包：TensorFlow/Keras。合成任务的标签由序列最后三步之和决定。

```python
import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
rng = np.random.default_rng(42)
X = rng.normal(size=(800, 12, 1)).astype("float32")
y = (X[:, -3:, 0].sum(axis=1) > 0).astype("float32")

idx = rng.permutation(len(X))
train, test = idx[:600], idx[600:]

model = tf.keras.Sequential([
    tf.keras.Input(shape=(12, 1)),
    tf.keras.layers.SimpleRNN(16),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)
model.fit(
    X[train], y[train],
    validation_split=0.2, epochs=30, batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        patience=4, restore_best_weights=True
    )],
    verbose=0,
)
model.evaluate(X[test], y[test])
```

真实不等长序列可先 padding，再在输入中用 Masking 层；但数值 0 必须确实代表补齐值而非有效观测。

### 7.2 R

常用包：keras3。

```r
library(keras3)

set.seed(42)
x <- array(rnorm(800 * 12), dim = c(800, 12, 1))
y <- as.numeric(apply(x[, 10:12, 1], 1, sum) > 0)
idx <- sample(seq_len(800), 600)

model <- keras_model_sequential() |>
  layer_simple_rnn(units = 16, input_shape = c(12, 1)) |>
  layer_dense(units = 1, activation = "sigmoid")

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

- 序列级分类仍要解读患者级概率、区分度和校准；时间点级预警还要考虑每患者报警次数和提前量。
- 隐藏状态是任务驱动的历史摘要，不等同于临床疾病状态，也不保证可辨识。
- 双向 RNN 同时使用过去与未来，只适合离线序列标注；若用于实时预测会泄漏未来。
- 每时点标签高度相关，不能把所有时间点当作独立样本计算过窄置信区间。
- 评估应按真实部署回放：在每个预测时点只提供当时可获得的数据。

## 9. 假设诊断与稳健性

- 长程诊断：按依赖距离或序列长度分层评估，比较 SimpleRNN、LSTM/GRU 和非序列基线。
- 梯度诊断：记录梯度范数；爆炸时使用 gradient clipping，消失时缩短 BPTT 或换门控单元。
- 时间泄漏：做严格时间截断，审计插补、聚合和标签生成是否使用未来值。
- 缺失稳健性：加入 mask 与 delta_t，比较不同插补，检查观测频率是否成了护理强度代理。
- 采样敏感性：改变重采样频率和窗口长度，结果应有合理稳定性。
- 随机性：多随机种子重复并报告波动。

## 10. 推荐可视化

- 序列时间线：同时标出观测、缺失、预测时点和结局窗口。
- 随时间变化的预测风险曲线与临床事件标记。
- 训练/验证损失和梯度范数曲线。
- 按序列长度、预测提前量和中心分层的性能图。
- 隐藏状态 PCA/UMAP 轨迹：仅用于探索状态演化。

## 11. 优势、局限与常见坑

### 优势

- 参数量不随序列长度线性增长。
- 能处理变长序列并保留顺序信息。
- 结构简单，可作为序列神经网络基线。

### 局限

- 梯度消失/爆炸使长程依赖难学。
- 按时间顺序计算，训练并行性较差。
- 对不规则间隔、信息性缺失和纵向推断没有自动解决方案。

### 常见坑

- padding 没有 mask，模型把补零当数据。
- 双向 RNN 用于实时预测，偷看未来。
- 将上一次观测向前填充到结局后，形成泄漏。
- 按时间点随机切分，让同一患者跨训练与测试。
- 只报告时间点 AUC，不报告患者级误报负担和提前量。

## 12. 与相近方法的区别

- 与 LSTM：LSTM 有显式细胞状态和门控，长依赖通常更稳，但参数更多。
- 与 GRU：GRU 用更紧凑的门控状态，常是效率与性能之间的折中。
- 与 Transformer：Transformer 用注意力直接连接远距离位置并易并行，RNN 则以固定状态逐步压缩历史。
- 与混合效应模型/GEE：后两者面向纵向效应估计和相关性处理；RNN 主要面向灵活预测。
- 如何选择：短序列先用普通 RNN 建基线；长依赖优先比较 GRU/LSTM，超长序列和大数据再考虑 Transformer。

## 13. 医学研究中的典型应用

- ICU 生命体征与实验室序列的短期恶化预警。
- 按就诊顺序编码诊断、处方和操作事件。
- ECG/EEG 波形分类与逐段标注。
- 临床文本中的序列标注和上下文编码。

必须报告采样频率、观察窗口、预测时点、缺失机制、重复测量、患者级切分和报警评价。若有删失或竞争风险，普通二分类 RNN 损失会产生偏差，需要专门生存目标。

## 14. 关键术语

- **序列（Sequence）**：有明确顺序、位置交换会改变含义的一组观测。
- **隐藏状态（Hidden state）**：RNN 对截至当前历史的学习型压缩摘要。
- **循环权重（Recurrent weight）**：将上一状态映射到新状态的共享参数。
- **时间展开（Unrolling）**：把循环图按时间复制成等价的前馈计算图。
- **时间反向传播（BPTT）**：在展开图上从后向前计算梯度。
- **梯度消失（Vanishing gradient）**：连续链式相乘后，远处梯度趋近于 0。
- **梯度爆炸（Exploding gradient）**：连续相乘后梯度异常增大。
- **Mask**：标记哪些时间步是真实观测、哪些只是 padding。
- **双向 RNN（Bidirectional RNN）**：同时从前向后和从后向前读取序列的离线结构。

## 15. 相关方法

- [[长短期记忆网络（Long Short-Term Memory, LSTM）]]
- [[门控循环单元（Gated Recurrent Unit, GRU）]]
- [[Transformer（Transformer）]]
- [[人工神经网络（Artificial Neural Network, ANN）]]
- [[广义线性混合效应模型（Generalized Linear Mixed-Effects Model, GLMM）]]

## 16. 参考资料

- Elman JL. Finding structure in time. *Cognitive Science*. 1990;14(2):179-211. https://doi.org/10.1207/s15516709cog1402_1
- Rumelhart DE, Hinton GE, Williams RJ. Learning representations by back-propagating errors. *Nature*. 1986;323:533-536. https://doi.org/10.1038/323533a0
- Bengio Y, Simard P, Frasconi P. Learning long-term dependencies with gradient descent is difficult. *IEEE Transactions on Neural Networks*. 1994;5(2):157-166. https://doi.org/10.1109/72.279181
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016. https://www.deeplearningbook.org/
