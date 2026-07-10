---
title: 长短期记忆网络
english_name: Long Short-Term Memory, LSTM
slug: long-short-term-memory-lstm
aliases: [LSTM, long short-term memory, "长短期记忆网络（Long Short-Term Memory, LSTM）"]
category: 神经网络与深度学习
subcategory: 循环神经网络
tags: [医学统计, 数据科学, 深度学习, 序列建模]
status: 已建
difficulty: advanced
question_type: 长程序列依赖建模
data_type: [序列数据, 时间序列数据, 文本数据]
outcome_type: [二分类, 多分类, 连续型, 序列输出]
python_packages: [torch, tensorflow, keras]
r_packages: [keras, torch]
---

# 长短期记忆网络（Long Short-Term Memory, LSTM）

## 1. 方法概览

### 1.1 一句话本质

LSTM 在普通 RNN 之外增加一条近似加性的细胞状态，并用门控制“忘什么、写什么、读什么”，让有用信息和梯度能跨越更长时间。

### 1.2 定义

长短期记忆网络是门控循环神经网络的经典结构。每个时间步维护细胞状态 $\mathbf c_t$ 和隐藏状态 $\mathbf h_t$：遗忘门控制旧记忆保留比例，输入门控制新候选内容写入比例，输出门控制记忆向外暴露比例。它缓解普通 RNN 的梯度消失，但并不保证无限长期记忆。

### 1.3 它主要解决什么问题

- 研究问题：关键线索与结局相隔较远、短期噪声又很多时，如何选择性保存和更新历史。
- 适用任务：长序列分类、时间序列预测、逐时点风险、文本与医疗事件序列建模。
- 常见医学场景：ICU 多小时监测预警、长期随访事件预测、处方/诊断序列编码、临床文本建模。

### 1.4 直觉与类比

把细胞状态想成一条传送带，信息可以在上面向后传。三个门像比例阀：遗忘门擦掉已过时内容，输入门决定新观察写多少，输出门决定当前要把多少记忆交给预测层。普通 RNN 每步都重写整张便笺；LSTM 则允许“保留大部分，只做小幅增删”。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

普通 RNN 的历史影响要穿过连续的权重矩阵和激活导数，长序列上容易指数衰减或爆炸。临床序列中，数小时前的低血压、数月前的用药或病历开头的否定词都可能仍然重要。LSTM 让细胞状态通过

$$
\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\cdots
$$

以加法方式更新，梯度沿这条路径主要乘遗忘门，而不必每步都穿过完整的 tanh 变换。

### 2.2 关键洞察

“记忆”不应在每一步全部重算，而应允许受控复制和局部修改。门值由 sigmoid 产生，位于 0 到 1：接近 1 表示通过，接近 0 表示阻断。由于门本身也由当前输入和旧状态计算，模型能针对不同病例、不同时间点动态决定信息流。

若遗忘门长期接近 1，早期细胞状态对后来状态的导数

$$
\frac{\partial \mathbf c_t}{\partial \mathbf c_{t-k}}
\approx
\prod_{j=t-k+1}^{t}\mathbf f_j
$$

可以保持较大；这就是 LSTM 比普通 RNN 更容易学习长依赖的核心。

### 2.3 与朴素/相邻做法的对比

- 相比普通 RNN：LSTM 的状态和门更多，参数约增至四组，但长程训练更稳定。
- 相比 GRU：LSTM 分离 $\mathbf c_t$ 与 $\mathbf h_t$、有三类门；GRU 更紧凑，性能需实证比较。
- 相比 Transformer：LSTM 递归扫描、内存随序列长度较温和；Transformer 更易并行并直接连接远距离位置。
- 相比传统时间序列模型：LSTM 能学复杂非线性，但较难解释，也不自动保证趋势、季节性或不规则时间处理正确。

## 3. 数学形式

### 3.1 核心公式

令 $\mathbf s_t=[\mathbf h_{t-1};\mathbf x_t]$ 为旧隐藏状态与当前输入的拼接：

$$
\mathbf f_t=\sigma(\mathbf W_f\mathbf s_t+\mathbf b_f)
$$

$$
\mathbf i_t=\sigma(\mathbf W_i\mathbf s_t+\mathbf b_i),
\qquad
\tilde{\mathbf c}_t=\tanh(\mathbf W_c\mathbf s_t+\mathbf b_c)
$$

$$
\mathbf c_t
=\mathbf f_t\odot\mathbf c_{t-1}
+\mathbf i_t\odot\tilde{\mathbf c}_t
$$

$$
\mathbf o_t=\sigma(\mathbf W_o\mathbf s_t+\mathbf b_o),
\qquad
\mathbf h_t=\mathbf o_t\odot\tanh(\mathbf c_t)
$$

这个式子在说：先按比例保留旧记忆，再按比例写入候选记忆，最后只把需要的部分暴露为当前隐藏状态。$\odot$ 表示逐元素相乘。

### 3.2 推导脉络

普通 RNN 只有 $\mathbf h_t=\tanh(\cdots)$，旧状态每步都被重新压缩。LSTM 把内部长期存储 $\mathbf c_t$ 与对外工作状态 $\mathbf h_t$ 分开：

1. 用遗忘门决定旧记忆剩多少。
2. 生成候选内容，并用输入门决定写多少。
3. 两部分相加形成新细胞状态。
4. 用输出门决定当前读出多少。

四组仿射变换可在一次矩阵乘法中并行计算。训练仍用 BPTT；门控改变的是梯度流的路径，不是优化器的基本原理。

### 3.3 参数与统计量含义

- $\mathbf f_t$：遗忘门，逐维控制旧细胞状态保留比例。
- $\mathbf i_t$：输入门，逐维控制候选记忆写入比例。
- $\tilde{\mathbf c}_t$：候选记忆，范围通常为 $-1$ 到 $1$。
- $\mathbf c_t$：细胞状态，承担较长期的信息传递。
- $\mathbf o_t$：输出门，控制细胞状态对当前隐藏状态的暴露。
- $\mathbf h_t$：对下一时间步和输出层可见的工作状态。
- 若输入维为 $d$、隐藏维为 $m$，标准 LSTM 参数量为 $4m(d+m+1)$。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 远程历史确有增量信息 | 早期观测在控制近期信息后仍有用 | 复杂门控不带来收益 | 与短窗口和普通 RNN 比较 |
| 同一动态机制可跨步共享 | 参数共享近似合理 | 阶段性机制被平均 | 加时间/阶段特征，分层评估 |
| 时间间隔和缺失已表达 | 步长含义一致或显式输入 | 门把观测次数当真实时间 | 加 delta_t 与 missing mask |
| 序列切分无泄漏 | 患者和未来信息严格隔离 | 性能虚高 | 患者级、时间前向验证 |
| 序列长度足以训练门控 | 有足够有效序列和结局 | 过拟合且门值不稳定 | 学习曲线、多随机种子 |

## 4. 手把手算例

沿用 RNN 卡的三步序列思路，考虑一个只有一维细胞状态的简化 LSTM。为突出门控更新，假设各步仿射变换已经给出相同门值

$$
f_t=0.8,\qquad i_t=0.6,\qquad o_t=0.7
$$

三个时间点产生的候选记忆分别为

$$
\tilde c_1=0.50,\qquad
\tilde c_2=-0.25,\qquad
\tilde c_3=0.75
$$

初始 $c_0=0$。

**第 1 步：写入第一条信息。**

$$
c_1=0.8\times0+0.6\times0.50=0.300
$$

$$
h_1=0.7\tanh(0.300)=0.204
$$

**第 2 步：保留旧信息，同时写入一条负向证据。**

$$
c_2=0.8\times0.300+0.6\times(-0.25)
=0.240-0.150=0.090
$$

$$
h_2=0.7\tanh(0.090)=0.063
$$

负向候选抵消了一部分旧记忆，但没有把它全部覆盖。

**第 3 步：加入新的正向证据。**

$$
c_3=0.8\times0.090+0.6\times0.75
=0.072+0.450=0.522
$$

$$
h_3=0.7\tanh(0.522)=0.335
$$

**追踪第一步信息。** $c_1$ 中的 0.300 经过两次遗忘门后仍贡献

$$
0.300\times0.8\times0.8=0.192
$$

也就是保留原贡献的 64%。作为对照，RNN 卡的普通循环链经过三次局部导数连乘只剩约 5.8%。这不是说 LSTM 永不遗忘，而是说明当模型把 $f_t$ 学到接近 1 时，信息可沿加性状态通道更稳定地传播。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：按时间或位置排序的多变量序列。
- 因变量类型：序列级分类/回归，或逐时间点输出。
- 数据结构：样本 × 时间 × 特征，配合长度、mask、时间间隔。
- 是否适合高维数据：可与 embedding、CNN 或降维层结合。
- 是否适合缺失较多数据：需插补、mask 和 delta_t；缺失本身可能信息性。
- 是否适合删失数据：标准 LSTM 分类/回归损失不处理删失，需专用生存损失。
- 是否适合重复测量数据：适合动态预测，但若目标是群体平均效应或随机效应推断，应使用纵向统计模型。

### 5.2 示例表格

| patient_id | hour | heart_rate | spo2 | lactate | lactate_missing | delta_t | deterioration_6h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P01 | 0 | 82 | 97 | 1.4 | 0 | 0 | 0 |
| P01 | 2 | 98 | 94 | NA | 1 | 2 | 0 |
| P01 | 5 | 112 | 90 | 3.2 | 0 | 3 | 1 |

将表格转为张量时，应保留实际间隔、缺失指示和有效长度；预测窗口内或之后的信息不能进入输入。

### 5.3 输入与产出

#### 输入

- 变长序列、时间戳/间隔、mask 和结局。
- LSTM 隐藏维度、层数、dropout、单向/双向和截断长度。
- 严格按患者与时间建立的训练/验证/测试集。

#### 产出

- 最后或全部时间点的 $\mathbf h_t$，以及内部 $\mathbf c_t$。
- 序列级或逐时点预测。
- 训练历史和验证指标。
- 门值可用于诊断，但不能直接当成临床可解释因果权重。

## 6. 适用场景

- 适合：短期与长期线索共同作用、序列中有选择性遗忘需求、数据量足以训练门控。
- 不适合：序列很短、静态汇总已充分、需要可解释纵向效应、存在未处理删失或超长序列需高并行。
- 使用前检查：时间泄漏、实际间隔、缺失机制、padding mask、患者级切分、序列长度和报警评价。

## 7. 实现

### 7.1 Python

常用包：TensorFlow/Keras。为便于和 RNN 卡比较，使用相同的合成任务。

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
    tf.keras.layers.LSTM(16, dropout=0.1),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0),
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

若序列以 0 padding，可在 LSTM 前加入 Masking 层；必须确认真实数据中的全零时间点不会被误当成 padding。

### 7.2 R

常用包：keras3。

```r
library(keras3)

set.seed(42)
x <- array(rnorm(800 * 12), dim = c(800, 12, 1))
y <- as.numeric(apply(x[, 10:12, 1], 1, sum) > 0)
idx <- sample(seq_len(800), 600)

model <- keras_model_sequential() |>
  layer_lstm(units = 16, dropout = 0.1, input_shape = c(12, 1)) |>
  layer_dense(units = 1, activation = "sigmoid")

model |> compile(
  optimizer = optimizer_adam(learning_rate = 1e-3, clipnorm = 1),
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

- 先按真实部署单元评价：患者级风险、事件级灵敏度、每患者误报数和预警提前量。
- LSTM 优于 RNN 只说明门控结构在当前验证方案中有预测收益，不证明发现了“长期因果记忆”。
- 门值是高维、条件依赖的内部计算量；单独展示某门接近 1 不能证明某临床变量重要。
- 双向 LSTM 会使用未来上下文，只适用于离线分类/标注；实时预测必须使用单向结构。
- 概率需独立校准；不同中心和时间段可能需要再校准或重训。

## 9. 假设诊断与稳健性

- 基线对照：与静态 Logistic/GBDT、普通 RNN、GRU 和不同窗口长度比较，确认长历史确有增益。
- 时间消融：遮掉早期、中期或近期窗口，观察性能下降是否符合临床预期。
- 泄漏审计：确保插补、标准化、标签和特征聚合不使用预测时点后的信息。
- 门与梯度诊断：检查遗忘门是否长期饱和在 0/1、梯度范数是否异常；必要时调整初始化、归一化和 clipping。
- 不规则采样：加入 delta_t/mask，并与时间衰减 LSTM 或神经 ODE 类方法做敏感性比较。
- 外部稳健性：跨中心、设备、采样频率和人群亚组评价性能与校准。

## 10. 推荐可视化

- 观测窗口—预测时点—结局窗口时间轴。
- 患者级动态风险曲线，叠加用药、检验和事件。
- RNN、GRU、LSTM 在不同依赖距离/序列长度下的性能曲线。
- 训练/验证损失与梯度范数。
- 遗忘门/输入门随时间的摘要热图：作为诊断，不作为因果解释。

## 11. 优势、局限与常见坑

### 优势

- 比普通 RNN 更容易学习长程依赖。
- 门控允许选择性保留、写入和读出信息。
- 可处理变长、多变量序列并与 CNN/embedding 联合。

### 局限

- 参数和计算量高于普通 RNN/GRU。
- 仍是顺序计算，长序列训练不易并行。
- 不自动解决不规则采样、信息性缺失、删失和因果解释。

### 常见坑

- 因“LSTM 会记忆”而无限延长窗口，不做消融。
- 忽略 padding mask 或把合法 0 值屏蔽。
- 双向 LSTM 用于在线预警。
- 把同一患者不同窗口随机分到不同集合。
- 从内部门值直接得出临床机制结论。

## 12. 与相近方法的区别

- 与普通 RNN：LSTM 用细胞状态和门控建立更稳定的长程路径，代价是更多参数。
- 与 GRU：GRU 只有更新门和重置门，不分离细胞/隐藏状态；小数据或速度敏感时值得先试。
- 与 Transformer：Transformer 直接用注意力连接全序列并高度并行；LSTM 以固定状态递归，流式推理自然。
- 与时间序列统计模型：ARIMA/状态空间模型结构更明确、可解释；LSTM 更灵活但需要严格基线比较。
- 如何选择：有明确长依赖且数据量中等时比较 GRU/LSTM；若短依赖足够用简单 RNN，若超长序列和大规模预训练则比较 Transformer。

## 13. 医学研究中的典型应用

- ICU/病房连续监测中的动态恶化与脓毒症预警。
- 长期 EHR 中诊断、检验、处方和操作序列的风险预测。
- ECG、EEG 和可穿戴设备波形分类。
- 临床文本和病历段落的序列编码。

医学应用必须明确预测锚点、输入可用时点、缺失机制、实际时间间隔、重复窗口、类别不平衡和患者级切分。对死亡/复发等时间到事件结局，普通固定窗二分类会忽视删失，应使用专门生存目标并考虑竞争风险。

## 14. 关键术语

- **细胞状态（Cell state）**：LSTM 内部承担较长期信息传递的状态 $\mathbf c_t$。
- **遗忘门（Forget gate）**：控制上一细胞状态保留比例的门。
- **输入门（Input gate）**：控制候选记忆写入比例的门。
- **输出门（Output gate）**：控制细胞状态向隐藏状态暴露比例的门。
- **候选记忆（Cell candidate）**：当前时间步可能写入的新内容 $\tilde{\mathbf c}_t$。
- **门饱和（Gate saturation）**：sigmoid 门值长期接近 0 或 1，梯度可能很小。
- **长程依赖（Long-term dependency）**：相距较远的位置仍存在对预测有用的关系。
- **梯度裁剪（Gradient clipping）**：限制梯度范数或数值以防爆炸。
- **截断 BPTT（Truncated BPTT）**：只在有限时间段内反传梯度以节省计算。

## 15. 相关方法

- [[循环神经网络（Recurrent Neural Network, RNN）]]
- [[门控循环单元（Gated Recurrent Unit, GRU）]]
- [[注意力机制（Attention Mechanism）]]
- [[Transformer（Transformer）]]
- [[自回归模型（Autoregressive Model, AR）]]

## 16. 参考资料

- Hochreiter S, Schmidhuber J. Long short-term memory. *Neural Computation*. 1997;9(8):1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735
- Gers FA, Schmidhuber J, Cummins F. Learning to forget: Continual prediction with LSTM. *Neural Computation*. 2000;12(10):2451-2471. https://doi.org/10.1162/089976600300015015
- Greff K, Srivastava RK, Koutník J, Steunebrink BR, Schmidhuber J. LSTM: A search space odyssey. *IEEE Transactions on Neural Networks and Learning Systems*. 2017;28(10):2222-2232. https://doi.org/10.1109/TNNLS.2016.2582924
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016. https://www.deeplearningbook.org/
