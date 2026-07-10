---
title: 门控循环单元
english_name: Gated Recurrent Unit, GRU
slug: gated-recurrent-unit-gru
aliases: [GRU, gated recurrent unit, "门控循环单元（Gated Recurrent Unit, GRU）"]
category: 神经网络与深度学习
subcategory: 循环神经网络
tags: [医学统计, 数据科学, 深度学习, 序列建模]
status: 已建
difficulty: advanced
question_type: 高效门控序列建模
data_type: [序列数据, 时间序列数据, 文本数据]
outcome_type: [二分类, 多分类, 连续型, 序列输出]
python_packages: [torch, tensorflow, keras]
r_packages: [keras, torch]
---

# 门控循环单元（Gated Recurrent Unit, GRU）

## 1. 方法概览

### 1.1 一句话本质

GRU 用更新门在“保留旧状态”和“写入新候选”之间做逐维插值，再用重置门决定生成候选时要参考多少历史，以更紧凑的结构缓解普通 RNN 的长程梯度问题。

### 1.2 定义

门控循环单元是一类门控循环神经网络。它只维护一个隐藏状态，不像 LSTM 那样另设细胞状态；每一步通过更新门和重置门控制信息流。GRU、LSTM 都是为普通 RNN 难学长依赖而设计，二者没有对所有数据都成立的性能高下。

### 1.3 它主要解决什么问题

- 研究问题：序列中既要保留历史又要快速接纳新信息时，如何以较少参数学习门控记忆。
- 适用任务：序列分类、动态风险预测、时间序列回归、文本与事件序列建模。
- 常见医学场景：生命体征预警、处方/诊断序列编码、ECG/EEG 分类、随访轨迹预测。

### 1.4 直觉与类比

把隐藏状态想成一份会滚动更新的临床摘要。更新门像“修订幅度”：接近 1 时沿用旧摘要，接近 0 时大幅采用新版本；重置门像“写新摘要前要不要翻旧病历”，接近 0 时暂时忽略历史，接近 1 时充分参考。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

普通 RNN 每一步都用 tanh 重新改写状态，早期信息和梯度要反复穿过非线性变换，长序列上容易消失。LSTM 为此引入细胞状态和三类门，但参数与计算更多。GRU 的关键问题是：能否保留一条近似恒等的状态通道，同时把门控压缩成更少部件？

### 2.2 关键洞察

GRU 把新状态写成旧状态与候选状态的凸组合：

$$
\mathbf h_t
=\mathbf z_t\odot\mathbf h_{t-1}
+(1-\mathbf z_t)\odot\tilde{\mathbf h}_t
$$

当更新门 $\mathbf z_t$ 接近 1，状态几乎原样复制，梯度也可沿复制路径向前传；接近 0 时，模型快速改用当前候选。重置门只影响候选的生成，让模型能在检测到新阶段时暂时清空旧背景。

### 2.3 与朴素/相邻做法的对比

- 相比普通 RNN：GRU 增加门控，参数更多，但更容易保留长程信息。
- 相比 LSTM：GRU 不分离细胞状态与隐藏状态，只有两类门，通常更快、更省内存。
- 相比 Transformer：GRU 适合流式递推且单步内存固定；Transformer 更易并行并可直接连接任意位置。
- 相比纵向统计模型：GRU 强在预测，不提供群体平均系数、随机效应或传统置信区间。

## 3. 数学形式

### 3.1 核心公式

本卡采用“$z_t$ 越大越保留旧状态”的常见约定：

$$
\mathbf z_t
=\sigma(\mathbf W_z\mathbf x_t+\mathbf U_z\mathbf h_{t-1}+\mathbf b_z)
$$

$$
\mathbf r_t
=\sigma(\mathbf W_r\mathbf x_t+\mathbf U_r\mathbf h_{t-1}+\mathbf b_r)
$$

$$
\tilde{\mathbf h}_t
=\tanh\!\left(
\mathbf W_h\mathbf x_t
+\mathbf U_h(\mathbf r_t\odot\mathbf h_{t-1})
+\mathbf b_h
\right)
$$

$$
\mathbf h_t
=\mathbf z_t\odot\mathbf h_{t-1}
+(1-\mathbf z_t)\odot\tilde{\mathbf h}_t
$$

这个式子在说：先决定生成候选时参考多少旧信息，再决定最终状态保留旧版还是采用候选。部分文献或软件把 $z_t$ 与 $1-z_t$ 的命名对调，阅读实现时应核对约定。

### 3.2 推导脉络

从普通 RNN 的“完全覆盖”

$$
\mathbf h_t=\tanh(\mathbf W_x\mathbf x_t+\mathbf U_h\mathbf h_{t-1})
$$

出发，加入一条直接复制 $\mathbf h_{t-1}$ 的路径，再用数据驱动的门选择复制比例。为了让候选状态能在“延续旧过程”和“开启新阶段”之间切换，又在候选计算中加入重置门。三组仿射变换分别生成更新门、重置门和候选，因此参数量为

$$
3m(d+m+1)
$$

其中 $d$ 是输入维、$m$ 是隐藏维；标准 LSTM 对应 $4m(d+m+1)$。

### 3.3 参数与统计量含义

- $\mathbf z_t$：更新门；在本卡约定中越接近 1，旧状态保留越多。
- $\mathbf r_t$：重置门；越接近 0，候选越少参考历史。
- $\tilde{\mathbf h}_t$：结合当前输入和门控历史得到的候选状态。
- $\mathbf h_t$：最终隐藏状态，同时承担记忆与对外表示。
- $\mathbf W_*$：当前输入到各门/候选的权重。
- $\mathbf U_*$：旧状态到各门/候选的循环权重。
- $\odot$：逐元素乘法，使不同隐藏维可以有不同保留比例。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 顺序与历史有增量信息 | 静态汇总不足以完成任务 | 门控复杂度没有收益 | 与静态和打乱顺序基线比较 |
| 同一转移函数可跨时间共享 | 动态机制在不同步近似稳定 | 阶段效应被混合 | 加时间/阶段特征，分层评估 |
| 实际时间间隔已表达 | 相邻步含义一致或有 delta_t | 把 1 小时与 30 天等同 | 加间隔与采样频率敏感性分析 |
| padding 和缺失被正确标记 | 补齐值不是临床观测 | 学到长度与缺失伪影 | 检查 mask、缺失模式 |
| 患者级切分无泄漏 | 同一患者窗口不跨数据集 | 性能虚高 | group split 与时间前向验证 |

## 4. 手把手算例

用一个一维 GRU 看两步状态更新。初始状态 $h_0=0.20$，候选公式设为

$$
\tilde h_t=\tanh(x_t+0.8r_t h_{t-1})
$$

**第 1 步：模型认为旧状态仍重要。** 设 $x_1=0.60$、更新门 $z_1=0.75$、重置门 $r_1=0.50$。

$$
\tilde h_1
=\tanh(0.60+0.8\times0.50\times0.20)
=\tanh(0.68)=0.592
$$

$$
h_1
=0.75\times0.20+(1-0.75)\times0.592
=0.150+0.148=0.298
$$

虽然候选是 0.592，但高更新门让状态主要沿用旧值。

**第 2 步：出现强新证据，模型允许快速改写。** 设 $x_2=1.00$、$z_2=0.20$、$r_2=0.90$。

$$
\tilde h_2
=\tanh(1.00+0.8\times0.90\times0.298)
=\tanh(1.215)=0.838
$$

$$
h_2
=0.20\times0.298+0.80\times0.838
=0.060+0.670=0.730
$$

状态从 0.298 快速升到 0.730。若第 2 步的 $z_2$ 仍为 0.75，则 $h_2$ 只有

$$
0.75\times0.298+0.25\times0.838=0.433
$$

这清楚展示：更新门不是“是否更新”的硬开关，而是旧状态与候选之间的连续插值。重置门则在生成候选时决定历史参与程度。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：按时间/位置排序的多变量序列。
- 因变量类型：序列级或时间点级分类、回归。
- 数据结构：样本 × 时间 × 特征，附长度、mask 和时间间隔。
- 是否适合高维数据：可与 embedding、CNN 或降维层联合。
- 是否适合缺失较多数据：需插补并加入 missing mask、delta_t。
- 是否适合删失数据：标准损失不处理删失，需生存 GRU 或离散时间风险损失。
- 是否适合重复测量数据：适合预测型序列建模，不替代 LMM/GEE 的效应推断。

### 5.2 示例表格

| patient_id | hour | heart_rate | spo2 | lactate | missing_lactate | delta_t | event_next_6h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P01 | 0 | 82 | 97 | 1.4 | 0 | 0 | 0 |
| P01 | 2 | 98 | 94 | NA | 1 | 2 | 0 |
| P01 | 5 | 112 | 90 | 3.2 | 0 | 3 | 1 |

### 5.3 输入与产出

#### 输入

- 变长序列、时间戳/间隔、mask 和标签。
- 隐藏维度、层数、dropout、方向和截断长度。
- 患者级训练/验证/测试划分及训练集拟合的预处理。

#### 产出

- 最后状态或全部时间点状态。
- 序列级/逐时点预测及训练历史。
- 门值可作行为诊断，但不自动具有临床或因果含义。
- 普通 GRU 不自动给出传统置信区间、随机效应或删失校正。

## 6. 适用场景

- 适合：序列有中长程依赖、希望比 LSTM 更紧凑、需要在线逐步更新。
- 不适合：序列极短、静态基线已充分、主要目标是纵向效应解释，或超长序列需要高度并行。
- 使用前检查：时间泄漏、不规则间隔、mask、患者级切分、类别不平衡和预警误报负担。

## 7. 实现

### 7.1 Python

常用包：TensorFlow/Keras。示例与 RNN、LSTM 卡使用同一合成任务，便于公平比较。

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
    tf.keras.layers.GRU(16, dropout=0.1),
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

### 7.2 R

常用包：keras3。

```r
library(keras3)

set.seed(42)
x <- array(rnorm(800 * 12), dim = c(800, 12, 1))
y <- as.numeric(apply(x[, 10:12, 1], 1, sum) > 0)
idx <- sample(seq_len(800), 600)

model <- keras_model_sequential() |>
  layer_gru(units = 16, dropout = 0.1, input_shape = c(12, 1)) |>
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

- 以真实部署单元评价：患者级概率、事件灵敏度、每患者误报数和提前量，而非只看时间点准确率。
- GRU 优于 LSTM/RNN 说明在当前数据、调参预算和验证方案中预测更好，不代表其门对应真实生理机制。
- 更新门高表示模型在该隐藏维上更多保留旧状态，不等于某个临床变量“更重要”。
- 双向 GRU 使用未来上下文，只适合离线编码；实时预警必须单向。
- 分类概率仍需外部校准与阈值决策，softmax/sigmoid 数值不是完整不确定性。

## 9. 假设诊断与稳健性

- 与静态模型、普通 RNN、LSTM 和不同窗口长度做同一切分下的基线比较。
- 遮掉早期/近期时间段，确认模型是否真的使用长历史。
- 检查梯度范数、训练/验证曲线和多随机种子波动。
- 审计插补、前向填充和标签生成是否使用预测时点后的信息。
- 加入 missing mask 与 delta_t，并改变采样频率做敏感性分析。
- 按中心、设备、序列长度和人群亚组做外部稳健性与校准评估。

## 10. 推荐可视化

- 观察窗—预测点—结局窗时间轴。
- 患者级风险轨迹，叠加用药、检验与事件。
- GRU/RNN/LSTM 随序列长度或依赖距离变化的性能曲线。
- 训练/验证损失和梯度范数。
- 更新门、重置门的时间热图：仅用于行为诊断。

## 11. 优势、局限与常见坑

### 优势

- 比普通 RNN 更容易保留较长依赖。
- 参数量通常少于同隐藏维 LSTM。
- 单一状态适合流式推理和资源受限场景。

### 局限

- 仍按时间顺序计算，长序列并行性有限。
- 不规则采样、信息性缺失、删失需额外建模。
- 门值和隐藏维难以直接解释。

### 常见坑

- 不核对公式/软件中更新门的相反命名约定。
- padding 无 mask 或把合法全零时间点屏蔽。
- 双向 GRU 用于实时预测。
- 同一患者的滑动窗口跨训练与测试。
- 只因参数少就假定 GRU 一定优于 LSTM。

## 12. 与相近方法的区别

- 与普通 RNN：GRU 增加更新与重置门，长程梯度更稳定。
- 与 LSTM：GRU 不分离细胞状态，结构更紧凑；LSTM 的三门与细胞状态提供更细粒度控制。
- 与 Transformer：GRU 流式、单步内存固定；Transformer 并行且远距离交互直接。
- 与 GEE/LMM：后者用于纵向效应估计和相关结构推断，GRU 主要用于非线性预测。
- 如何选择：短序列先用简单 RNN；中长序列同时比较 GRU/LSTM，以相同验证和调参预算决定。

## 13. 医学研究中的典型应用

- ICU 与可穿戴连续监测的动态恶化预警。
- 纵向 EHR 的诊断、处方和检验事件序列编码。
- ECG/EEG 等一维信号分类。
- 医疗文本或编码序列的上下文表示。

必须说明预测锚点、输入可用时点、采样间隔、缺失机制、重复窗口、类别不平衡和患者级切分。固定窗二分类 GRU 不处理删失或竞争风险，时间到事件任务需专用损失。

## 14. 关键术语

- **更新门（Update gate）**：控制旧状态与候选状态混合比例的门。
- **重置门（Reset gate）**：控制生成候选时参考多少旧状态的门。
- **候选状态（Candidate state）**：当前时间步拟议写入的新状态。
- **凸组合（Convex combination）**：非负权重且和为 1 的加权平均。
- **门控（Gating）**：用 0 到 1 的数据驱动系数调节信息流。
- **流式推理（Streaming inference）**：数据逐步到达时只更新当前状态，不重算全部历史。
- **参数共享（Parameter sharing）**：所有时间步复用同一组转移参数。
- **梯度消失（Vanishing gradient）**：长链式相乘使早期梯度趋近 0。

## 15. 相关方法

- [[循环神经网络（Recurrent Neural Network, RNN）]]
- [[长短期记忆网络（Long Short-Term Memory, LSTM）]]
- [[Transformer（Transformer）]]
- [[广义线性混合效应模型（Generalized Linear Mixed-Effects Model, GLMM）]]
- [[广义估计方程（Generalized Estimating Equations, GEE）]]

## 16. 参考资料

- Cho K, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*. 2014:1724-1734. https://doi.org/10.3115/v1/D14-1179
- Chung J, Gülçehre Ç, Cho K, Bengio Y. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv:1412.3555; 2014. https://arxiv.org/abs/1412.3555
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016. https://www.deeplearningbook.org/
- TensorFlow developers. GRU layer. https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU
