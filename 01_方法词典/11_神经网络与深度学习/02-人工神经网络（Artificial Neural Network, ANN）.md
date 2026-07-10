---
title: 人工神经网络
english_name: Artificial Neural Network, ANN
slug: artificial-neural-network-ann
aliases: [ANN, artificial neural network, "人工神经网络（Artificial Neural Network, ANN）"]
category: 神经网络与深度学习
subcategory: 神经网络基础
tags: [医学统计, 数据科学, 神经网络, 深度学习]
status: 已建
difficulty: basic
question_type: 通用非线性映射建模
data_type: [表格数据, 图像向量, 序列数据]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [tensorflow, keras, torch]
r_packages: [keras, torch]
---

# 人工神经网络（Artificial Neural Network, ANN）

## 1. 方法概览

### 1.1 一句话本质

人工神经网络把复杂函数拆成许多可学习的小变换，并通过组合、非线性和梯度优化，从数据中共同学出“如何表示”与“如何预测”。

### 1.2 定义

人工神经网络是由节点和带权连接组成的函数模型总称，不是一种单一算法。MLP、CNN、RNN、LSTM、Transformer 都属于这个家族；它们的区别主要在连接方式和对数据结构的归纳偏置。一个基本神经元执行“加权求和 + 偏置 + 激活”，多个神经元按层或图结构连接后形成网络。

### 1.3 它主要解决什么问题

- 研究问题：输入与结局之间存在难以预先写出的非线性、交互和高阶结构时，如何直接从数据学习映射。
- 适用任务：分类、回归、表征学习、生成、序列预测和多模态融合。
- 常见医学场景：EHR 风险预测、影像识别、病理切片分类、生理信号分析和影像—文本联合建模。

### 1.4 直觉与类比

神经网络像一条可改造的流水线。每个工位只做简单运算，但训练会同时调整所有工位的参数，让早期工位产出对最终任务有用的中间零件。与传统“先人工做特征、再建模型”不同，神经网络常把特征学习和预测合并在一个可微系统中。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

许多真实规律不是单个线性公式能表达的。朴素办法是人工创造平方项、分段、交互和图像纹理特征，但可能组合爆炸，也很依赖专家经验。ANN 的解法是：给模型一组可组合的基础变换，让中间表示也成为待学习对象，再用一个端到端损失统一指导所有层。

### 2.2 关键洞察

有三条核心洞察：

1. **组合性**：复杂函数可由许多简单函数复合而成。
2. **非线性**：激活函数让网络能表达弯曲边界；没有激活，多层线性层仍只是一个线性层。
3. **可微优化**：计算图记录每一步依赖关系，反向传播用链式法则高效得到全部参数的梯度。

通用逼近定理说明，在适当条件下，含足够隐藏单元的网络可以逼近广泛的连续函数；它说明“有表达能力”，却不保证有限数据下一定学得到、学得稳或能外推。

### 2.3 与朴素/相邻做法的对比

- 线性/广义线性模型预先规定函数形状，ANN 则学习中间特征，灵活性更高但推断和解释更困难。
- 传统机器学习常先人工提取特征；深度网络可从像素、波形或词元端到端学习。
- 神经网络架构不是越通用越好：CNN 对空间局部性、RNN 对顺序、Transformer 对全局依赖加入了不同的归纳偏置。

## 3. 数学形式

### 3.1 核心公式

第 $\ell$ 层的统一写法是：

$$
\mathbf a^{(\ell)}
=\mathbf W^{(\ell)}\mathbf h^{(\ell-1)}
+\mathbf b^{(\ell)},\qquad
\mathbf h^{(\ell)}=\phi^{(\ell)}\!\left(\mathbf a^{(\ell)}\right)
$$

其中 $\mathbf h^{(0)}=\mathbf x$。经过 $L$ 层得到预测 $\hat{\mathbf y}=f_\theta(\mathbf x)$，再最小化

$$
\hat\theta
=\underset{\theta}{\operatorname{arg\,min}}\;
\frac{1}{n}\sum_{i=1}^{n}
\mathcal L\!\left(y_i,f_\theta(\mathbf x_i)\right)
+\lambda\Omega(\theta)
$$

这个式子在说：网络逐层生成表示，训练寻找能让预测损失与复杂度惩罚之和最小的一组参数。

### 3.2 推导脉络

对单个神经元，线性预测子 $a=\mathbf w^\top\mathbf x+b$ 汇总证据，激活 $h=\phi(a)$ 决定输出。把许多神经元并行排列得到一层，再把层复合得到深度网络。若损失为 $\mathcal L$，某个早期权重 $w$ 的梯度按链式法则分解：

$$
\frac{\partial\mathcal L}{\partial w}
=
\frac{\partial\mathcal L}{\partial\hat y}
\frac{\partial\hat y}{\partial h}
\frac{\partial h}{\partial a}
\frac{\partial a}{\partial w}
$$

反向传播从右到左复用这些局部导数；优化器再按学习率更新权重。mini-batch 是用一小批样本的平均梯度近似全数据梯度，在速度、内存和噪声之间折中。

### 3.3 参数与统计量含义

- $\mathbf W^{(\ell)}$：第 $\ell$ 层权重矩阵；行数是本层单元数，列数是上一层单元数。
- $\mathbf b^{(\ell)}$：偏置向量，平移每个单元的激活阈值。
- $\phi$：激活函数；ReLU 常用于隐藏层，sigmoid/softmax 常用于分类输出。
- $\theta$：全部可训练参数的集合。
- $\mathcal L$：任务损失；分类常用交叉熵，回归常用均方误差。
- $\Omega(\theta)$：正则项；常见为权重的 $L_2$ 范数。
- $\lambda$：拟合与复杂度之间的权衡强度。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 训练样本覆盖目标人群 | 学习到的模式在应用人群仍成立 | 数据集偏移导致性能骤降 | 外部、时间和中心验证 |
| 输入表示保留有效信息 | 预处理没有丢掉预测所需结构 | 再大网络也无法恢复信息 | 审核特征时间窗和数据质量 |
| 复杂度与信息量相称 | 参数量、任务难度与有效样本量匹配 | 记忆训练数据 | 学习曲线、重复训练 |
| 损失与临床目标一致 | 优化指标反映实际代价 | 数学性能好但临床无用 | 阈值分析、决策曲线 |
| 标签和切分可信 | 无未来信息、重复患者泄漏和系统性错标 | 产生虚高性能或错误模式 | 数据谱系审计、患者级切分 |

## 4. 手把手算例

用最小网络解释“为什么非线性层能解决 XOR”。两个二元输入 $x_1,x_2$ 表示两项条件，目标是“恰好满足一项时输出 1”：

| $x_1$ | $x_2$ | XOR 目标 |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

单条直线无法把两个阳性点与两个阴性点分开。现在构造两个 ReLU 隐藏单元：

$$
h_1=\operatorname{ReLU}(x_1+x_2),\qquad
h_2=\operatorname{ReLU}(x_1+x_2-1)
$$

输出层取

$$
\hat y=h_1-2h_2
$$

逐行计算：

| $(x_1,x_2)$ | $h_1$ | $h_2$ | $\hat y=h_1-2h_2$ |
| --- | ---: | ---: | ---: |
| $(0,0)$ | 0 | 0 | 0 |
| $(0,1)$ | 1 | 0 | 1 |
| $(1,0)$ | 1 | 0 | 1 |
| $(1,1)$ | 2 | 1 | 0 |

网络准确实现了 XOR。真正训练时这些权重不是手工指定，而是从随机初始化出发，由损失梯度逐步学习。这个例子揭示：隐藏层不是神秘“黑箱”，它是在构造新的坐标；非线性让原空间中不可线性分割的问题，在新表示中可由简单输出层解决。

参数量也可手算：若输入 2 维、隐藏层 2 个单元、输出 1 个单元，则参数数为

$$
(2\times2+2)+(2\times1+1)=9
$$

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：经数值化的表格、图像、序列、文本或多模态输入。
- 因变量类型：连续、二分类、多分类、序列或重构目标。
- 数据结构：由架构决定；全连接网络用矩阵，CNN 用张量，RNN/Transformer 用序列张量。
- 是否适合高维数据：适合，但需要架构先验、正则化和足够样本。
- 是否适合缺失较多数据：网络本身通常不接受缺失值；需插补、掩码或专门结构。
- 是否适合删失数据：只有采用生存似然或风险集损失的专用网络才适合。
- 是否适合重复测量数据：需按患者分组并用序列/层级结构表达相关性。

### 5.2 示例表格

| patient_id | time | heart_rate | spo2 | medication | deterioration_6h |
| --- | ---: | ---: | ---: | ---: | ---: |
| P01 | 0 | 82 | 97 | 0 | 0 |
| P01 | 1 | 96 | 94 | 0 | 0 |
| P02 | 0 | 110 | 91 | 1 | 1 |
| P02 | 1 | 118 | 88 | 1 | 1 |

同一患者的多行记录存在相关性，不能拆到不同数据集；若使用全连接 ANN，还需先汇总成固定长度特征。

### 5.3 输入与产出

#### 输入

- 数据张量与监督标签，或自监督/生成目标。
- 网络结构、损失函数、优化器和正则化设置。
- 明确的训练、验证、测试切分与预处理流水线。

#### 产出

- 训练后的参数和计算图。
- 分类概率、连续预测、序列表征或生成样本。
- 训练历史和验证指标。
- 中间层表示；其几何结构可探索，但不自动具有临床语义。

## 6. 适用场景

- 适合：输入维度高、结构复杂、非线性强，有足够数据与计算资源，任务偏预测或表征学习。
- 不适合：小样本且需稳定参数推断、可由简单模型充分解决、数据质量或标签定义尚未厘清。
- 使用前检查：预测时点、泄漏、样本单位、类别/中心分布、缺失机制、损失与临床目标是否一致。

## 7. 实现

### 7.1 Python

常用包：TensorFlow/Keras。下面用可重复的合成回归数据演示最小 ANN。

```python
import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
rng = np.random.default_rng(42)
X = rng.normal(size=(400, 3)).astype("float32")
y = (2 * X[:, 0] - X[:, 1] ** 2 + 0.5 * X[:, 2]).astype("float32")

idx = rng.permutation(len(X))
train_idx, test_idx = idx[:300], idx[300:]

normalizer = tf.keras.layers.Normalization()
normalizer.adapt(X[train_idx])  # 只在训练集拟合

model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    normalizer,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(
    X[train_idx], y[train_idx],
    validation_split=0.2, epochs=100, batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )],
    verbose=0,
)
print(model.evaluate(X[test_idx], y[test_idx], verbose=0))
```

### 7.2 R

常用包：keras3。

```r
library(keras3)

set.seed(42)
x <- matrix(rnorm(400 * 3), ncol = 3)
y <- 2 * x[, 1] - x[, 2]^2 + 0.5 * x[, 3]
idx <- sample(seq_len(nrow(x)), 300)

# 只用训练集估计缩放参数
mu <- colMeans(x[idx, ])
sdv <- apply(x[idx, ], 2, sd)
x_scaled <- sweep(sweep(x, 2, mu, "-"), 2, sdv, "/")

model <- keras_model_sequential() |>
  layer_dense(units = 16, activation = "relu", input_shape = c(3)) |>
  layer_dense(units = 8, activation = "relu") |>
  layer_dense(units = 1)

model |> compile(optimizer = "adam", loss = "mse", metrics = "mae")
model |> fit(
  x_scaled[idx, ], y[idx],
  validation_split = 0.2, epochs = 100, batch_size = 32,
  callbacks = callback_early_stopping(
    patience = 10, restore_best_weights = TRUE
  ),
  verbose = 0
)
model |> evaluate(x_scaled[-idx, ], y[-idx], verbose = 0)
```

## 8. 结果如何解读

- 分类网络的输出先看独立测试集区分度、校准和阈值性能；回归网络看 MAE/RMSE、误差分布和临床可接受范围。
- 训练损失表示优化目标，不等于泛化性能，也不直接等于临床效用。
- 隐藏单元和单个权重通常没有唯一语义：交换两个隐藏单元仍可表示同一函数。
- 相同总体指标可能掩盖亚组失败，应按年龄、性别、中心、设备和疾病严重程度检查。
- 若模型用于决策，需报告预测时点、阈值、错误代价以及对现有流程的增量价值。

## 9. 假设诊断与稳健性

- 拟合诊断：同时画训练/验证曲线，检查欠拟合、过拟合和异常震荡。
- 优化诊断：更换随机种子、学习率和初始化，报告均值与波动。
- 消融实验：移除某模态、时间窗或模块，确认性能收益来自预期信息。
- 负对照与泄漏审计：加入不应有预测力的特征或时间打乱实验，排查未来信息。
- 外部稳健性：跨时间、医院、设备和人群验证，并重新评估校准。
- 不确定性：可用 bootstrap、深度集成或 conformal prediction；单次 softmax 概率不是完整不确定性。

## 10. 推荐可视化

- 网络计算图与各层张量形状：解释信息如何流动。
- 训练/验证损失曲线：定位过拟合和优化不稳定。
- 分类的 ROC、PR、校准曲线或回归的预测—真实散点与残差图。
- 中间表示的 PCA/UMAP 图：仅作探索，不据此宣称生物亚型。
- 亚组与外部中心森林图：展示性能异质性。

## 11. 优势、局限与常见坑

### 优势

- 函数表达灵活，可联合学习特征与预测。
- 架构可适配图像、序列、图和多模态数据。
- 可通过迁移学习利用大规模预训练知识。

### 局限

- 通常需要更多数据、计算和调参。
- 训练是非凸优化，不同随机运行可能得到不同参数与性能。
- 可解释性、校准和跨中心泛化需要额外验证。

### 常见坑

- 把“通用逼近”误解为“小数据也一定能学好”。
- 先看测试集再反复调参，使测试集变成验证集。
- 忽略同一患者、同一切片或同一检查的分组关系。
- 用网络复杂度代替数据质量和研究设计。
- 把高维嵌入的漂亮分群当成已证实的临床亚型。

## 12. 与相近方法的区别

- 与 MLP：ANN 是上位概念，MLP 是由全连接前馈层构成的具体网络。
- 与深度学习：深度学习通常指具有多层表征学习的 ANN 方法集合。
- 与广义线性模型：后者函数形状和系数解释更明确；ANN 更灵活但统计推断更弱。
- 与树模型：树对表格数据和非线性很强，通常应作为深度表格模型的基线。
- 如何选择：先按输入结构选归纳偏置——表格从简单模型/MLP 起步，图像用 CNN，序列用 RNN/LSTM/Transformer；最终以严格外部验证决定。

## 13. 医学研究中的典型应用

- 诊断：从影像、病理或多指标输出疾病分类概率。
- 预后：从历史 EHR 预测恶化、再入院或资源需求。
- 表征：从无标签数据学习患者、图像块或基因表达嵌入。
- 多模态：融合影像、文本、检验和人口学信息。

医学应用要特别区分预测与因果问题；说明结局类型、缺失机制、删失、重复测量、类别不平衡和数据采集偏倚。网络不会自动消除混杂，也不会自动处理删失。

## 14. 关键术语

- **神经元（Neuron）**：执行加权求和、加偏置和激活的基本计算单元。
- **层（Layer）**：并行组织多个神经元或结构化运算的模块。
- **深度（Depth）**：从输入到输出经历的可学习变换层数。
- **表示学习（Representation learning）**：让模型自动学习对任务有用的中间特征。
- **归纳偏置（Inductive bias）**：架构预先偏好的结构，如 CNN 偏好局部和平移共享。
- **计算图（Computational graph）**：记录变量和运算依赖关系的图，用于前向计算和自动求导。
- **梯度（Gradient）**：损失对参数的偏导数组，指示局部最快上升方向。
- **优化器（Optimizer）**：根据梯度更新参数的规则，如 SGD、Adam。
- **正则化（Regularization）**：限制有效复杂度以改善泛化的方法。

## 15. 相关方法

- [[多层感知机分类器（Multilayer Perceptron Classifier, MLP）]]
- [[卷积神经网络（Convolutional Neural Network, CNN）]]
- [[循环神经网络（Recurrent Neural Network, RNN）]]
- [[Transformer（Transformer）]]
- [[自编码器（Autoencoder）]]

## 16. 参考资料

- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016. https://www.deeplearningbook.org/
- Hornik K, Stinchcombe M, White H. Multilayer feedforward networks are universal approximators. *Neural Networks*. 1989;2(5):359-366. https://doi.org/10.1016/0893-6080(89)90020-8
- LeCun Y, Bengio Y, Hinton G. Deep learning. *Nature*. 2015;521:436-444. https://doi.org/10.1038/nature14539
- TensorFlow developers. Keras guide. https://www.tensorflow.org/guide/keras
