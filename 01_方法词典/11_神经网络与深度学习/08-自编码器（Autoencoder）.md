---
title: 自编码器
english_name: Autoencoder
slug: autoencoder
aliases: [AE, autoencoder, "自编码器（Autoencoder）"]
category: 神经网络与深度学习
subcategory: 表征学习与重构
tags: [医学统计, 数据科学, 深度学习, 无监督学习]
status: 已建
difficulty: intermediate
question_type: 无监督表征学习与重构
data_type: [表格数据, 图像数据, 高维特征矩阵]
outcome_type: [重构输出, 低维表征, 异常分数]
python_packages: [tensorflow, keras, torch]
r_packages: [keras]
---

# 自编码器（Autoencoder）

## 1. 方法概览

### 1.1 一句话本质

自编码器强迫网络把输入经过受限的潜在表示再还原；若仍能重构，瓶颈中就必须保留数据的主要结构而舍弃部分冗余或噪声。

### 1.2 定义

自编码器是一类以输入本身为学习目标的神经网络，由编码器 $f_\theta$ 和解码器 $g_\phi$ 组成。编码器将输入压成潜在表示，解码器从表示重构输入，训练最小化重构误差。它通常属于自监督表征学习，而非严格意义上“没有目标”的学习，因为目标就是原输入或被破坏前的输入。

### 1.3 它主要解决什么问题

- 研究问题：没有人工标签时，如何学习低维表示、去噪或发现偏离常态的样本。
- 适用任务：降维、压缩、去噪、异常检测、预训练和缺失重构。
- 常见医学场景：多组学表征、医学影像去噪、监护数据异常检测、病理图像嵌入。

### 1.4 直觉与类比

把编码器想成写“病历摘要”，解码器则只看摘要尝试还原原病历。若摘要字数受限，还能还原主要内容，说明摘要抓住了共性结构。若摘要容量毫无限制，模型可能逐字抄写而没有学到有用规律；所以瓶颈、噪声、稀疏或正则约束是方法的灵魂。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

高维观测中常有冗余：相邻像素相关、多个基因共表达、生命体征共同变化。PCA 只能用线性子空间压缩，复杂数据可能位于弯曲流形。自编码器用非线性编码和解码学习“如何压缩、如何展开”，但必须限制信息通道，否则学到恒等映射就失去意义。

### 2.2 关键洞察

重构任务本身制造了监督信号。模型不能只记住一个标签，而要保存足以解释输入的结构。三种常见约束分别表达不同思想：

- 欠完备自编码器：潜在维度小于输入维度，形成几何瓶颈。
- 去噪自编码器：输入人为加噪，目标仍是干净输入，逼迫模型学习稳定结构。
- 稀疏/正则自编码器：限制隐藏激活或编码器导数，阻止简单复制。

### 2.3 与朴素/相邻做法的对比

- 相比 PCA：线性自编码器在特定条件下恢复 PCA 子空间；非线性自编码器能表示弯曲结构。
- 相比普通 MLP：MLP 用外部标签预测，自编码器用输入自身作为目标。
- 相比 VAE：普通自编码器给确定性编码，不保证潜空间连续可采样；VAE 约束潜变量分布。
- 相比 GAN：自编码器优化逐样本重构，GAN 用判别器匹配生成分布。

## 3. 数学形式

### 3.1 核心公式

编码、解码和经验风险为

$$
\mathbf z=f_\theta(\mathbf x),\qquad
\hat{\mathbf x}=g_\phi(\mathbf z)
$$

$$
\min_{\theta,\phi}
\frac{1}{n}\sum_{i=1}^{n}
\mathcal L\!\left(
\mathbf x_i,
g_\phi(f_\theta(\mathbf x_i))
\right)
+\lambda\Omega(\theta,\phi)
$$

连续型、已标准化输入常用均方误差：

$$
\mathcal L(\mathbf x,\hat{\mathbf x})
=\|\mathbf x-\hat{\mathbf x}\|_2^2
$$

这个式子在说：把输入压缩后再还原，训练让重构尽量接近原输入，同时用瓶颈或正则避免无意义复制。

去噪自编码器改为

$$
\min_{\theta,\phi}
\mathbb E_{\tilde{\mathbf x}\sim q(\tilde{\mathbf x}\mid\mathbf x)}
\mathcal L\!\left(
\mathbf x,g_\phi(f_\theta(\tilde{\mathbf x}))
\right)
$$

### 3.2 推导脉络

若只最小化重构误差且网络容量无限，令 $g(f(x))=x$ 就能得到平凡解。于是必须限制：

1. 降低潜在维度，让网络选择保留哪些方向。
2. 或破坏输入，让逐像素复制不再可行。
3. 或惩罚复杂权重、隐藏激活和局部敏感性。

异常检测则建立在一个额外逻辑上：只用“正常”样本训练后，正常模式容易重构，未见异常重构误差可能更大。但高容量网络也可能把异常重构得很好，所以这一逻辑必须实证验证。

### 3.3 参数与统计量含义

- $\mathbf x$：原始输入，通常需按变量尺度预处理。
- $\mathbf z$：潜在编码或瓶颈表示。
- $\hat{\mathbf x}$：解码器重构。
- $f_\theta,g_\phi$：编码器和解码器。
- $\mathcal L$：按数据类型选择的重构损失；连续用 MSE，二元可用交叉熵。
- $\Omega$：权重、稀疏性或平滑性正则。
- 重构误差：可作样本级异常分数，但阈值必须在独立数据上验证。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 数据存在可压缩结构 | 低维表示能保留主要信息 | 重构差或编码无用 | 比较 PCA、不同瓶颈 |
| 损失与重要信息一致 | 像素/变量误差反映任务价值 | 重构好却丢掉临床关键细节 | 下游任务与专家检查 |
| 正常训练集较纯 | 异常检测时训练样本代表正常 | 异常被学成正常 | 审核训练集、稳健训练 |
| 容量受到有效限制 | 模型不能轻易复制所有输入 | 恒等映射、异常也低误差 | 容量/正则消融 |
| 训练与应用预处理一致 | 尺度、设备、批次可比 | 重构误差反映域差异而非异常 | 外部中心和批次分层 |

## 4. 手把手算例

用一个线性、单维瓶颈自编码器说明“投影—重构—异常误差”。正常数据沿直线 $x_2=0.5x_1$，其单位方向为

$$
\mathbf w
=\frac{1}{\sqrt5}
\begin{pmatrix}2\\1\end{pmatrix}
=(0.894,0.447)^\top
$$

编码器与解码器设为

$$
z=\mathbf w^\top\mathbf x,\qquad
\hat{\mathbf x}=z\mathbf w
$$

**正常点 $\mathbf x=(2,1)^\top$。**

$$
z=\frac{2\times2+1\times1}{\sqrt5}
=\sqrt5=2.236
$$

$$
\hat{\mathbf x}
=2.236(0.894,0.447)^\top
=(2,1)^\top
$$

重构平方误差为 0。

**偏离正常方向的点 $\mathbf x=(2,2)^\top$。**

$$
z=\frac{2\times2+1\times2}{\sqrt5}
=\frac6{\sqrt5}=2.683
$$

$$
\hat{\mathbf x}
=2.683(0.894,0.447)^\top
=(2.4,1.2)^\top
$$

$$
\|\mathbf x-\hat{\mathbf x}\|_2^2
=(2-2.4)^2+(2-1.2)^2
=0.16+0.64=0.80
$$

若正常验证集误差的 95% 分位数为 0.10，这个点会被标记为异常。这个线性例子等价于投影到主方向；非线性自编码器把直线推广为可弯曲的低维流形。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：连续/二元表格、高维组学、图像或序列张量。
- 因变量类型：输入自身、干净版本或被遮盖部分。
- 数据结构：样本 × 特征或结构化张量。
- 是否适合高维数据：适合，但样本量和正则必须支撑网络容量。
- 是否适合缺失较多数据：可设计掩码重构；普通 MSE 不能直接处理 NA。
- 是否适合删失数据：普通自编码器不处理删失。
- 是否适合重复测量数据：需序列/层级编码器并按患者切分。

### 5.2 示例表格

| patient_id | heart_rate_z | spo2_z | lactate_z | split |
| --- | ---: | ---: | ---: | --- |
| P01 | -0.3 | 0.5 | -0.4 | train-normal |
| P02 | 0.1 | 0.2 | 0.0 | train-normal |
| P03 | 2.8 | -2.5 | 3.1 | test |

异常检测训练集应尽量由正常样本组成；标准化参数只能从训练集估计。

### 5.3 输入与产出

#### 输入

- 训练数据与可选噪声/掩码机制。
- 编码器、瓶颈维度、解码器、损失和正则。
- 训练集拟合的缩放及患者级切分。

#### 产出

- 潜在表示 $\mathbf z$ 和重构 $\hat{\mathbf x}$。
- 样本级或变量级重构误差。
- 可用于下游预测的编码器。
- 不自动产出可辨识因子、概率密度或因果表征。

## 6. 适用场景

- 适合：无标签高维数据、存在非线性结构、需要去噪/压缩/预训练。
- 不适合：样本太少、临床解释是首要目标、异常与正常可被简单规则区分，或缺失/删失机制未建模。
- 使用前检查：容量是否过大、缩放、批次效应、潜在维稳定性、异常阈值和外部验证。

## 7. 实现

### 7.1 Python

常用包：TensorFlow/Keras。示例只用正常训练数据拟合，再以重构误差检测异常。

```python
import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
rng = np.random.default_rng(42)
normal = rng.normal(size=(1000, 2)).astype("float32")
normal[:, 1] = 0.5 * normal[:, 0] + rng.normal(
    0, 0.1, size=1000
)
train, valid = normal[:800], normal[800:]

encoder = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, name="latent"),
])
decoder = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(2),
])
inputs = tf.keras.Input(shape=(2,))
autoencoder = tf.keras.Model(inputs, decoder(encoder(inputs)))
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(
    train, train, validation_data=(valid, valid),
    epochs=100, batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        patience=8, restore_best_weights=True
    )],
    verbose=0,
)

valid_hat = autoencoder.predict(valid, verbose=0)
threshold = np.quantile(np.mean((valid - valid_hat) ** 2, axis=1), 0.95)
test = np.array([[2.0, 2.0]], dtype="float32")
score = np.mean((test - autoencoder.predict(test, verbose=0)) ** 2)
print("threshold:", threshold, "score:", score, "anomaly:", score > threshold)
```

### 7.2 R

常用包：keras3。

```r
library(keras3)

set_random_seed(42)
normal <- matrix(rnorm(1000 * 2), ncol = 2)
normal[, 2] <- 0.5 * normal[, 1] + rnorm(1000, 0, 0.1)
train <- normal[1:800, ]
valid <- normal[801:1000, ]

encoder <- keras_model_sequential() |>
  layer_dense(4, activation = "relu", input_shape = c(2)) |>
  layer_dense(1, name = "latent")
decoder <- keras_model_sequential() |>
  layer_dense(4, activation = "relu", input_shape = c(1)) |>
  layer_dense(2)

inputs <- layer_input(shape = c(2))
autoencoder <- keras_model(inputs, decoder(encoder(inputs)))
autoencoder |> compile(optimizer = "adam", loss = "mse")
autoencoder |> fit(
  train, train, validation_data = list(valid, valid),
  epochs = 100, batch_size = 32,
  callbacks = callback_early_stopping(
    patience = 8, restore_best_weights = TRUE
  ),
  verbose = 0
)

valid_hat <- autoencoder |> predict(valid, verbose = 0)
valid_error <- rowMeans((valid - valid_hat)^2)
threshold <- unname(quantile(valid_error, 0.95))
test <- matrix(c(2, 2), nrow = 1)
score <- mean((test - (autoencoder |> predict(test, verbose = 0)))^2)
c(threshold = threshold, score = score, anomaly = score > threshold)
```

## 8. 结果如何解读

- 重构误差低只表示模型能按训练分布还原输入，不代表编码具有临床意义。
- 潜在维不是天然“疾病因子”；旋转、缩放或重新排列潜在坐标仍可能得到同样重构。
- 异常分数高可能代表疾病，也可能只是设备、中心、批次、缺失模式或预处理差异。
- 阈值应在独立正常验证集上设定，并在有标签测试集报告灵敏度、特异度和阳性预测值。
- 下游使用编码时必须在训练折内拟合自编码器，避免把测试分布泄露给表征学习。

## 9. 假设诊断与稳健性

- 容量诊断：改变瓶颈宽度和层数；若过宽模型连异常也完美重构，需加强约束。
- 与 PCA 比较：检查非线性结构是否带来真实增益。
- 重构残差：按变量、空间区域、中心和亚组查看，而非只看总 MSE。
- 稳定性：多随机种子后比较潜在距离、聚类和下游性能。
- 域偏移：外部中心的高误差可能是设备差异；先分层再解释。
- 异常评价：使用已知异常的独立集，避免只用漂亮重构图自证。

## 10. 推荐可视化

- 原始—重构—残差三联图。
- 样本重构误差分布和验证阈值。
- 二维潜在空间散点，按已知标签/中心着色。
- 瓶颈维度与重构/下游性能曲线。
- 去噪任务的含噪输入与清洁重构对照。

## 11. 优势、局限与常见坑

### 优势

- 无需人工标签即可学习非线性表示。
- 同一框架可做压缩、去噪、异常检测和预训练。
- 编码器可迁移到下游任务。

### 局限

- 重构目标未必与临床任务一致。
- 高容量网络可能学习恒等映射。
- 潜空间不保证可解释、连续或适合生成。

### 常见坑

- 输入未标准化，MSE 被大尺度变量主导。
- 在全数据上先训练编码器，再做交叉验证。
- 用异常污染严重的数据训练“正常”模型。
- 把 UMAP/PCA 后潜在分群直接命名为疾病亚型。
- 用训练误差设阈值并报告同一数据性能。

## 12. 与相近方法的区别

- 与 PCA：PCA 线性、解唯一性和方差解释更明确；自编码器更灵活但更难审计。
- 与 VAE：VAE 学习潜变量分布并加 KL 正则，普通自编码器是确定性映射。
- 与 GAN：GAN 关注生成分布逼真，自编码器关注逐样本重构。
- 与 UMAP/t-SNE：后两者主要用于低维可视化，不提供解码重构。
- 如何选择：线性结构和小样本先用 PCA；确有非线性、数据量足够且需要重构时再用自编码器。

## 13. 医学研究中的典型应用

- 多组学和高维检验数据的低维表示。
- CT/MRI/病理图像的去噪、压缩与预训练。
- 监护波形、设备数据的异常检测。
- 缺失模态或遮盖区域的重构。

必须说明训练集是否仅含正常、输入尺度、批次/中心效应、缺失机制、重复测量和异常阈值。普通自编码器不处理删失，也不能从重构关系推出因果机制。

## 14. 关键术语

- **编码器（Encoder）**：把输入映射到潜在表示的网络。
- **解码器（Decoder）**：从潜在表示重构输入的网络。
- **瓶颈（Bottleneck）**：限制信息容量的中间表示。
- **潜在表示（Latent representation）**：模型学习的低维或受约束编码。
- **重构误差（Reconstruction error）**：输入与重构之间的差异。
- **欠完备（Undercomplete）**：潜在维度小于输入维度。
- **去噪自编码器（Denoising autoencoder）**：从被破坏输入恢复干净目标的自编码器。
- **恒等映射（Identity mapping）**：输入原样复制到输出的平凡解。
- **异常分数（Anomaly score）**：用于衡量样本偏离训练常态的数值。

## 15. 相关方法

- [[主成分分析（Principal Component Analysis, PCA）]]
- [[人工神经网络（Artificial Neural Network, ANN）]]
- [[生成对抗网络（Generative Adversarial Network, GAN）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]
- [[多重插补（Multiple Imputation）]]

## 16. 参考资料

- Hinton GE, Salakhutdinov RR. Reducing the dimensionality of data with neural networks. *Science*. 2006;313(5786):504-507. https://doi.org/10.1126/science.1127647
- Vincent P, et al. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. *Journal of Machine Learning Research*. 2010;11:3371-3408.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016. https://www.deeplearningbook.org/
- TensorFlow developers. Intro to autoencoders. https://www.tensorflow.org/tutorials/generative/autoencoder
