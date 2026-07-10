---
title: 生成对抗网络
english_name: Generative Adversarial Network, GAN
slug: generative-adversarial-network-gan
aliases: [GAN, generative adversarial network, "生成对抗网络（Generative Adversarial Network, GAN）"]
category: 神经网络与深度学习
subcategory: 生成式神经网络
tags: [医学统计, 数据科学, 深度学习, 生成模型]
status: 已建
difficulty: advanced
question_type: 数据生成与分布拟合
data_type: [图像数据, 高维特征矩阵, 序列数据]
outcome_type: [生成样本, 表征学习]
python_packages: [torch, tensorflow, keras]
r_packages: [keras]
---

# 生成对抗网络（Generative Adversarial Network, GAN）

## 1. 方法概览

### 1.1 一句话本质

GAN 让生成器与判别器进行一场可微博弈：判别器学习识别真假，生成器借判别器的反馈把随机噪声逐步推成类似真实数据的样本。

### 1.2 定义

生成对抗网络是一类隐式生成模型，由生成器 $G$ 和判别器 $D$ 组成。生成器把潜在噪声 $z$ 映射成合成样本，判别器输出输入来自真实数据而非生成器的概率。两者交替优化，使生成分布逼近真实数据分布，但通常不给出可直接计算的概率密度。

### 1.3 它主要解决什么问题

- 研究问题：复杂高维分布没有易写出的似然时，如何学习并生成新样本。
- 适用任务：图像合成、域转换、超分辨率、缺失模态生成、条件生成和数据增强。
- 常见医学场景：医学影像合成、病理染色风格转换、低剂量影像增强、稀有类别辅助增强。

### 1.4 直觉与类比

生成器像“仿制者”，判别器像“鉴定师”。鉴定师越能指出伪品破绽，仿制者得到的改进信号越清楚；仿制者进步后，鉴定师也必须升级。理想平衡时，生成样本和真实样本分布一致，鉴定师对任何样本都只能猜 50%。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

传统概率生成模型往往要指定似然并计算归一化常数，高维影像分布很难显式表达。直接最小化“每个生成样本与哪个真实样本的距离”也不合理，因为生成并没有一一配对真值。GAN 把“分布是否相似”转化为一个分类问题：若判别器仍能区分真假，说明分布还有可利用差异。

### 2.2 关键洞察

对固定生成器，最优判别器为

$$
D^*(x)=\frac{p_{\text{data}}(x)}
{p_{\text{data}}(x)+p_g(x)}
$$

把它代回极小极大目标后，生成器实际在缩小真实分布与生成分布之间的 Jensen–Shannon 散度。也就是说，判别器不是最终产品，而是一个可学习的“分布差异测量器”，把高维分布差异转成生成器可反传的梯度。

### 2.3 与朴素/相邻做法的对比

- 相比自编码器：GAN 直接约束生成分布的真实性，自编码器主要优化逐样本重构。
- 相比变分自编码器：VAE 有显式概率目标、潜空间较规整，但样本可能更平滑；GAN 常更锐利但训练更不稳定。
- 相比扩散模型：扩散模型训练通常更稳定、覆盖更好，但采样往往需要多步；GAN 可一次前向快速采样。
- 相比简单过采样：GAN 能产生新组合，但也可能复制训练样本或放大偏倚。

## 3. 数学形式

### 3.1 核心公式

原始 GAN 的极小极大目标是

$$
\min_G\max_D V(D,G)
=
\mathbb E_{\mathbf x\sim p_{\text{data}}}
[\log D(\mathbf x)]
+
\mathbb E_{\mathbf z\sim p_z}
[\log(1-D(G(\mathbf z)))]
$$

这个式子在说：判别器希望真实样本得分高、生成样本得分低；生成器希望让第二项变小，从而使生成样本被判为真实。

实践中常用非饱和生成器损失：

$$
\mathcal L_G
=-\mathbb E_{\mathbf z\sim p_z}
[\log D(G(\mathbf z))]
$$

判别器损失为

$$
\mathcal L_D
=-\mathbb E_{\mathbf x\sim p_{\text{data}}}\log D(\mathbf x)
-\mathbb E_{\mathbf z\sim p_z}\log(1-D(G(\mathbf z)))
$$

### 3.2 推导脉络

1. 从易采样的先验 $p_z$ 抽取噪声。
2. 生成器用 $G_\theta(z)$ 把噪声推到数据空间，诱导出 $p_g$。
3. 判别器把真假识别写成二分类交叉熵。
4. 固定 $G$ 更新 $D$，再固定 $D$ 更新 $G$，形成交替优化。
5. 理论平衡是 $p_g=p_{\text{data}}$、$D(x)=1/2$。

非饱和损失不改变理想平衡，但在生成样本很差、$D(G(z))$ 接近 0 时提供更强梯度。WGAN 则用 Wasserstein 距离及 Lipschitz 约束改善分布支撑不重叠时的训练信号。

### 3.3 参数与统计量含义

- $\mathbf z$：潜在噪声，常来自标准正态或均匀分布。
- $G(\mathbf z)$：生成样本。
- $D(\mathbf x)$：判别器估计“来自真实数据”的概率。
- $p_{\text{data}}$：未知的真实数据分布。
- $p_g$：生成器诱导的分布，通常没有显式密度。
- $\mathcal L_G,\mathcal L_D$：两个网络各自的训练损失，不能单独作为生成质量结论。
- 条件 GAN 还输入标签或协变量 $y$，学习 $G(z,y)$。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 训练数据代表目标分布 | 样本覆盖真实变异和亚组 | 生成器复制偏倚、遗漏人群 | 分层覆盖与外部专家审查 |
| 生成器与判别器能力平衡 | 双方都能持续提供学习信号 | 一方碾压导致无梯度或伪收敛 | 联合看损失、样本与多样性 |
| 潜在维度和架构足够 | 能表达目标分布主要模式 | 欠拟合或模式遗漏 | 多指标、潜空间插值 |
| 训练样本隐私可控 | 模型不记忆可识别病例 | 成员推断或近复制风险 | 最近邻、攻击与去标识审计 |
| 评价指标覆盖真实性与多样性 | 单一分数不会掩盖失败 | 只生成少数逼真模式 | 质量、覆盖、临床效用并评 |

## 4. 手把手算例

考虑最小离散世界 $x\in\{0,1\}$。真实分布为

$$
p_{\text{data}}(0)=0.8,\qquad p_{\text{data}}(1)=0.2
$$

当前生成器还不准确：

$$
p_g(0)=0.5,\qquad p_g(1)=0.5
$$

**第 1 步：求当前生成器下的最优判别器。**

$$
D^*(0)=\frac{0.8}{0.8+0.5}=0.615
$$

$$
D^*(1)=\frac{0.2}{0.2+0.5}=0.286
$$

因为 0 在真实数据中更常见，见到 0 时判别器更倾向判真；1 被生成器过度生成，因此判真概率更低。

**第 2 步：代入极小极大目标。**

$$
\begin{aligned}
V(D^*,G)
&=0.8\log(0.615)+0.2\log(0.286)\\
&\quad+0.5\log(1-0.615)+0.5\log(1-0.286)\\
&=-0.388-0.251-0.478-0.168\\
&=-1.285
\end{aligned}
$$

**第 3 步：看理想平衡。** 若生成器学到真实比例 $(0.8,0.2)$，则两个取值上都有 $D^*=0.5$：

$$
V(D^*,G) = \log(0.5)+\log(0.5)=-\log 4=-1.386
$$

在判别器最优的前提下，生成器要把目标从 $-1.285$ 降到 $-1.386$，也就是让真假越来越不可分。这个例子还说明：判别器准确率降到 50% 只有在双方都充分训练且样本覆盖良好时才可能是好事；若判别器太弱，也会“瞎猜 50%”。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：图像、波形、表格向量、序列或条件标签。
- 因变量类型：无监督生成，或条件标签/配对域作为条件。
- 数据结构：与生成器输出形状一致的张量。
- 是否适合高维数据：适合，但训练和评价难度随维度上升。
- 是否适合缺失较多数据：可专门设计用于插补，普通 GAN 不自动识别缺失机制。
- 是否适合删失数据：普通 GAN 不处理删失，需把时间到事件机制写入目标。
- 是否适合重复测量数据：需序列生成结构并以患者为单位切分。

### 5.2 示例表格

| patient_id | image_path | modality | site | diagnosis |
| --- | --- | --- | --- | --- |
| P01 | img_001.png | CT | A | benign |
| P02 | img_002.png | CT | B | malignant |
| P03 | img_003.png | CT | A | malignant |

训练条件 GAN 时可把 diagnosis、site 等作为条件，但敏感属性和中心伪影也可能被原样复制。

### 5.3 输入与产出

#### 输入

- 真实样本、潜在噪声和可选条件标签。
- 生成器/判别器架构、学习率、更新比例和正则化。
- 患者级切分以及明确的隐私与用途边界。

#### 产出

- 可采样的生成器与训练用判别器。
- 合成样本和潜在空间。
- 训练轨迹、质量/多样性指标与下游效用。
- 不直接产出真实样本的似然或隐私保证。

## 6. 适用场景

- 适合：需要高保真快速生成、成像域转换、条件合成，且有足够数据和严格评价。
- 不适合：小数据却要求隐私安全、训练不稳定不可接受、需要精确似然，或合成数据将直接用于高风险临床决策。
- 使用前检查：模式崩塌、训练样本近复制、亚组覆盖、伪影、隐私攻击和下游真实数据验证。

## 7. 实现

### 7.1 Python

常用包：TensorFlow/Keras。下面生成二维高斯点，仅用于展示交替训练。

```python
import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
rng = np.random.default_rng(42)
latent_dim = 4
batch_size = 64

generator = tf.keras.Sequential([
    tf.keras.Input(shape=(latent_dim,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(2),
])
discriminator = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(2e-4),
    loss="binary_crossentropy",
)

discriminator.trainable = False
z_input = tf.keras.Input(shape=(latent_dim,))
gan = tf.keras.Model(z_input, discriminator(generator(z_input)))
gan.compile(
    optimizer=tf.keras.optimizers.Adam(2e-4),
    loss="binary_crossentropy",
)

for step in range(1000):
    real = rng.normal(2.0, 0.5, size=(batch_size, 2)).astype("float32")
    z = rng.normal(size=(batch_size, latent_dim)).astype("float32")
    fake = generator.predict(z, verbose=0)
    x_batch = np.vstack([real, fake])
    y_batch = np.vstack([
        np.ones((batch_size, 1)),
        np.zeros((batch_size, 1)),
    ])
    discriminator.train_on_batch(x_batch, y_batch)

    z = rng.normal(size=(batch_size, latent_dim)).astype("float32")
    gan.train_on_batch(z, np.ones((batch_size, 1)))

samples = generator.predict(
    rng.normal(size=(1000, latent_dim)), verbose=0
)
print(samples.mean(axis=0), samples.std(axis=0))
```

### 7.2 R

常用包：keras3。

```r
library(keras3)

set_random_seed(42)
latent_dim <- 4
batch_size <- 64

generator <- keras_model_sequential() |>
  layer_dense(16, activation = "relu", input_shape = c(latent_dim)) |>
  layer_dense(2)

discriminator <- keras_model_sequential() |>
  layer_dense(16, activation = "relu", input_shape = c(2)) |>
  layer_dense(1, activation = "sigmoid")

discriminator |> compile(
  optimizer = optimizer_adam(2e-4),
  loss = "binary_crossentropy"
)

discriminator$trainable <- FALSE
z_input <- layer_input(shape = c(latent_dim))
gan <- keras_model(z_input, discriminator(generator(z_input)))
gan |> compile(
  optimizer = optimizer_adam(2e-4),
  loss = "binary_crossentropy"
)

for (step in seq_len(1000)) {
  real <- matrix(rnorm(batch_size * 2, 2, 0.5), ncol = 2)
  z <- matrix(rnorm(batch_size * latent_dim), ncol = latent_dim)
  fake <- generator |> predict(z, verbose = 0)
  x_batch <- rbind(real, fake)
  y_batch <- rbind(
    matrix(1, batch_size, 1),
    matrix(0, batch_size, 1)
  )
  discriminator |> train_on_batch(x_batch, y_batch)

  z <- matrix(rnorm(batch_size * latent_dim), ncol = latent_dim)
  gan |> train_on_batch(z, matrix(1, batch_size, 1))
}
```

## 8. 结果如何解读

- 不能只看生成器/判别器损失：对抗损失不是单调的，双方此消彼长。
- 图像“看起来逼真”只覆盖保真度，不代表疾病谱、亚组比例和临床变量关系正确。
- FID 等指标衡量特征空间分布差异，但依赖特征提取器；通用图像网络未必适合医学影像。
- 下游效用应以“真实训练/真实测试”为基准，合成数据增强后仍必须在独立真实患者上验证。
- 合成不等于匿名；近复制、成员推断和属性推断都可能泄露训练信息。

## 9. 假设诊断与稳健性

- 模式崩塌：检查不同噪声是否只生成少数模板；比较类内/类间多样性与最近邻。
- 训练平衡：联合观察两方损失、梯度范数、判别输出和固定噪声样本序列。
- 真实性：由领域专家盲评病理解剖合理性和伪影，而非只看视觉锐度。
- 覆盖性：按年龄、性别、中心、设备、疾病亚型比较真实与合成分布。
- 隐私：做训练集最近邻、重复检测、成员推断与属性泄露测试。
- 稳健性：多随机种子、不同架构和 WGAN-GP 等目标做敏感性分析。

## 10. 推荐可视化

- 固定潜在噪声随 epoch 的生成样本网格。
- 真实样本与生成样本的最近邻并排图。
- 潜空间插值序列，观察变化是否连续且医学合理。
- 特征空间真实/生成分布图和亚组覆盖图。
- 质量—多样性、隐私—效用双轴图。

## 11. 优势、局限与常见坑

### 优势

- 可学习复杂高维分布并一次前向快速采样。
- 常能产生清晰样本，支持条件生成和域转换。
- 判别器提供数据驱动的分布差异信号。

### 局限

- 极小极大优化不稳定，容易模式崩塌。
- 缺少易解释的显式似然，评价困难。
- 可能复制训练样本、放大偏倚或产生医学不可能结构。

### 常见坑

- 只挑最好看的样本展示。
- 把判别器 50% 准确率自动解释为收敛。
- 用合成样本同时进入训练和测试。
- 将合成数据称为“天然匿名”。
- 仅用通用 FID 宣称医学有效性。

## 12. 与相近方法的区别

- 与自编码器：自编码器按样本重构，GAN 通过判别博弈匹配整体分布。
- 与 VAE：VAE 优化变分下界并有显式潜变量概率结构；GAN 不需显式似然。
- 与扩散模型：扩散训练更稳定、覆盖常更好；GAN 采样更快。
- 与 SMOTE：SMOTE 在局部线性插值，GAN 可学复杂分布但风险更高。
- 如何选择：若首要是稳定覆盖与可审计性，先试简单增强/VAE/扩散；需要低延迟高保真生成并能承担严格评价时再考虑 GAN。

## 13. 医学研究中的典型应用

- 合成胸片、CT、MRI、病理或皮肤镜图像用于算法开发。
- 跨设备/染色风格的图像到图像转换。
- 低剂量到常规剂量、低分辨率到高分辨率的重建。
- 稀有病变类别的辅助数据增强。

医学应用必须评估解剖与病理真实性、亚组覆盖、设备伪影、训练样本近复制、类别不平衡和隐私风险。若涉及时间到事件或缺失数据，生成目标还需尊重删失与缺失机制，不能直接套用普通 GAN。

## 14. 关键术语

- **生成器（Generator）**：把潜在噪声映射成合成样本的网络。
- **判别器（Discriminator）**：估计输入来自真实数据概率的网络。
- **潜在空间（Latent space）**：生成器输入噪声所在的低维空间。
- **隐式分布（Implicit distribution）**：可以采样但通常不能直接计算密度的分布。
- **纳什均衡（Nash equilibrium）**：双方单独改变策略都不能改善目标的博弈状态。
- **模式崩塌（Mode collapse）**：生成器只覆盖少数模式，缺乏多样性。
- **非饱和损失（Non-saturating loss）**：实践中为生成器提供更强早期梯度的替代损失。
- **Jensen–Shannon 散度**：衡量两个概率分布差异的对称量。
- **成员推断（Membership inference）**：判断某个样本是否参与模型训练的隐私攻击。

## 15. 相关方法

- [[自编码器（Autoencoder）]]
- [[人工神经网络（Artificial Neural Network, ANN）]]
- [[卷积神经网络（Convolutional Neural Network, CNN）]]
- [[UMAP（Uniform Manifold Approximation and Projection）]]
- [[主成分分析（Principal Component Analysis, PCA）]]

## 16. 参考资料

- Goodfellow I, et al. Generative adversarial nets. *NeurIPS*. 2014;27:2672-2680. https://papers.nips.cc/paper/5423-generative-adversarial-nets
- Arjovsky M, Chintala S, Bottou L. Wasserstein generative adversarial networks. *ICML*. 2017:214-223. https://proceedings.mlr.press/v70/arjovsky17a.html
- Gulrajani I, et al. Improved training of Wasserstein GANs. *NeurIPS*. 2017;30. https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans
- Yi X, Walia E, Babyn P. Generative adversarial network in medical imaging: A review. *Medical Image Analysis*. 2019;58:101552. https://doi.org/10.1016/j.media.2019.101552
