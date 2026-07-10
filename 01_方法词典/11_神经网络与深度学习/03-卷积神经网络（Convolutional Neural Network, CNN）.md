---
title: 卷积神经网络
english_name: Convolutional Neural Network, CNN
slug: convolutional-neural-network-cnn
aliases: [CNN, convolutional neural network, "卷积神经网络（Convolutional Neural Network, CNN）"]
category: 神经网络与深度学习
subcategory: 卷积神经网络
tags: [医学统计, 数据科学, 深度学习, 图像分析]
status: 已建
difficulty: intermediate
question_type: 局部空间模式识别
data_type: [图像数据, 栅格数据, 信号矩阵]
outcome_type: [二分类, 多分类, 连续型]
python_packages: [tensorflow, keras, torch, torchvision]
r_packages: [keras]
---

# 卷积神经网络（Convolutional Neural Network, CNN）

## 1. 方法概览

### 1.1 一句话本质

CNN 用同一个小型滤波器在整幅图像上滑动，把“局部模式可重复出现”写进网络结构，从而用远少于全连接层的参数学习空间特征。

### 1.2 定义

卷积神经网络是一类利用局部连接、权重共享和分层感受野处理网格数据的人工神经网络。典型 CNN 由卷积、激活、下采样/池化与任务输出层组成；浅层常响应边缘和纹理，深层把局部响应组合为更大的形状与语义结构。

### 1.3 它主要解决什么问题

- 研究问题：像素数很多、病灶可能出现在不同位置时，如何高效识别可重复的局部空间模式。
- 适用任务：二维/三维图像分类、检测、分割、定位，以及一维生理信号建模。
- 常见医学场景：胸片异常识别、病理切片分型、CT/MRI 分割、眼底分级、ECG 波形分类。

### 1.4 直觉与类比

卷积核像一块会学习的“小窗口检查表”。它在图像每个位置问同一个问题，例如“这里有没有由暗到亮的垂直边缘？”只学习一套检查表却能全图复用。后续层再把边缘组合成纹理、轮廓和病灶；感受野随层数增大，模型从局部逐渐看到整体。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

把一张 $512\times512$ 灰度图展平并连接到 100 个隐藏单元，需要约 2620 万个权重，而且模型必须分别学习“左上角边缘”和“右下角边缘”。这忽略了图像的两个事实：相邻像素关系重要，同一模式可能在不同位置重复。CNN 用局部连接减少参数，用权重共享让一个探测器在全图复用。

### 2.2 关键洞察

- **局部性**：早期视觉模式通常可由小邻域判断。
- **权重共享**：同一个卷积核在所有空间位置使用相同参数。
- **层级组合**：深层卷积扩大有效感受野，把局部模式组合成高级模式。
- **平移等变性**：输入平移时特征图相应平移；经过全局池化等操作后，可获得更强的平移不变性。

CNN 并非天然对旋转、缩放、设备差异或解剖方向不变，这些需要增强、配准或专门结构。

### 2.3 与朴素/相邻做法的对比

- 相比展平后 MLP：CNN 保留空间邻接，参数更少，图像任务通常更有效。
- 相比人工影像组学：CNN 可端到端学习特征，但更依赖数据量，也更难审计。
- 相比 Vision Transformer：CNN 的局部归纳偏置强、样本效率常更好；Transformer 更直接建模全局关系。

## 3. 数学形式

### 3.1 核心公式

二维多通道卷积层在位置 $(i,j)$、输出通道 $k$ 的计算为

$$
z_{i,j,k}
=b_k+
\sum_{c=1}^{C_{\mathrm{in}}}
\sum_{u=0}^{K_h-1}
\sum_{v=0}^{K_w-1}
W_{u,v,c,k}\,X_{i+u,j+v,c}
$$

$$
H_{i,j,k}=\phi(z_{i,j,k})
$$

这个式子在说：取输入中的一个局部块，与卷积核逐元素相乘后求和，再经过激活函数，得到特征图的一个像素。多数深度学习库实际计算的是互相关，即不把核翻转；参数学习不受这项命名差异影响。

若输入边长为 $N$，核大小为 $K$，padding 为 $P$，stride 为 $S$，dilation 为 $D$，输出边长为

$$
N_{\mathrm{out}}
=
\left\lfloor
\frac{N+2P-D(K-1)-1}{S}+1
\right\rfloor
$$

### 3.2 推导脉络

全连接层对每个输入位置学习独立权重。若假设“同一种局部模式在任意位置含义相近”，就可以约束不同位置共用权重；把共用核在输入上平移，便得到卷积。多个核学习多个模式，ReLU 组合出非线性响应，pooling 或 stride 压缩空间尺寸。堆叠两层 $3\times3$、stride 1 的卷积后，一个单元已能间接看到 $5\times5$ 区域，这就是感受野逐层扩大。

### 3.3 参数与统计量含义

- $X$：输入张量，常写为高度 × 宽度 × 通道；三维影像还多一个深度维。
- $W$：卷积核权重，形状为核高 × 核宽 × 输入通道 × 输出通道。
- $K_h,K_w$：核大小，控制单层局部视野。
- stride：窗口每次移动的步长，越大下采样越强。
- padding：边缘补值方式，常用于保持空间尺寸。
- dilation：核元素间隔，可在不大幅增参时扩大感受野。
- feature map：某个卷积核在所有位置的响应图。
- 参数量：$K_hK_wC_{\mathrm{in}}C_{\mathrm{out}}+C_{\mathrm{out}}$，与图像宽高无关。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 邻近像素有局部结构 | 局部模式具有预测意义 | CNN 的局部偏置收益下降 | 比较 MLP/Transformer 基线 |
| 同一模式可跨位置复用 | 权重共享在医学上合理 | 绝对位置极关键时表达受限 | 加位置通道或分区评估 |
| 图像方向与尺度可比 | 预处理保留一致空间语义 | 模型学到体位、标尺或边框 | 配准、裁剪并做伪影审计 |
| 训练与部署成像域接近 | 扫描器、重建核、染色流程相容 | 跨中心性能下降 | 按设备/中心外部验证 |
| 切分以患者为单位 | 同一患者图像不跨集合 | 相似图像泄漏导致虚高 | patient_id 分组去重 |

## 4. 手把手算例

用一个 $4\times4$ 灰度图和一个 $2\times2$ 卷积核，stride 为 1、无 padding：

$$
X=
\begin{pmatrix}
0&0&1&1\\
0&0&1&1\\
1&1&0&0\\
1&1&0&0
\end{pmatrix},
\qquad
W=
\begin{pmatrix}
1&-1\\
1&-1
\end{pmatrix}
$$

这个核计算局部块“左列像素和减右列像素和”，响应垂直亮度变化。

**第 1 步：算左上位置。**

$$
0\times1+0\times(-1)+0\times1+0\times(-1)=0
$$

**第 2 步：向右移动一格。**

$$
0\times1+1\times(-1)+0\times1+1\times(-1)=-2
$$

**第 3 步：在全部九个位置重复，得到特征图。**

$$
Z=
\begin{pmatrix}
0&-2&0\\
0&0&0\\
0&2&0
\end{pmatrix}
$$

**第 4 步：经过 ReLU。**

$$
H=\max(0,Z)=
\begin{pmatrix}
0&0&0\\
0&0&0\\
0&2&0
\end{pmatrix}
$$

ReLU 保留了“左亮右暗”方向的强响应 2，压掉相反方向的 $-2$。若再做全局最大池化，整张图得到特征值 2，表示“某处出现了目标方向的垂直边缘”。若还想识别相反方向，可再学习一个符号相反的卷积核。

参数量只有 $2\times2+1=5$；若同一 $4\times4$ 输入直接连接到 9 个输出，则需 $16\times9+9=153$ 个参数。规模越大，权重共享的优势越明显。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：二维图像、三维体数据、一维波形或其他规则网格张量。
- 因变量类型：图像级类别/连续结局、像素/体素级分割标签、目标框或关键点。
- 数据结构：保留空间维与通道维，不把图像过早展平。
- 是否适合高维数据：适合，但显存、样本量和过拟合是限制。
- 是否适合缺失较多数据：局部遮挡可增强；系统性缺层/缺模态需专门处理。
- 是否适合删失数据：标准 CNN 不处理删失，可将 CNN 表征接生存损失。
- 是否适合重复测量数据：图像级 CNN 不建模同一患者的时序相关；需患者级切分并与时序模型结合。

### 5.2 示例表格

| patient_id | image_path | view | scanner | label |
| --- | --- | --- | --- | ---: |
| P01 | P01_pa.png | PA | A | 0 |
| P02 | P02_ap.png | AP | B | 1 |
| P02 | P02_lateral.png | LAT | B | 1 |

元数据表之外，实际输入是图像张量。P02 的两张图必须进入同一数据分区。

### 5.3 输入与产出

#### 输入

- 规范化图像张量与标签，必要时包含设备、视角和掩码元数据。
- 核大小、通道数、stride、padding、网络深度与增强策略。
- 患者级训练/验证/测试划分。

#### 产出

- 分类概率、连续预测、分割概率图或检测框。
- 卷积特征图和深层嵌入。
- 训练/验证损失与任务指标。
- 普通 CNN 不自动给出可靠不确定性或因果解释。

## 6. 适用场景

- 适合：局部空间结构有意义、图像量足够、希望端到端学习。
- 不适合：样本非常少且无迁移学习、图像空间关系已被破坏、目标主要依赖长程全局关系却网络感受野不足。
- 使用前检查：DICOM 去标识、窗宽窗位、方向、分辨率、裁剪、重复图像、患者级切分、设备/中心偏倚和类别不平衡。

## 7. 实现

### 7.1 Python

常用包：TensorFlow/Keras。示例使用 MNIST 说明最小二维 CNN；医学图像应替换为患者级数据加载器。

```python
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train[..., None] / 255.0).astype("float32")
x_test = (x_test[..., None] / 255.0).astype("float32")

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    x_train, y_train, validation_split=0.1,
    epochs=10, batch_size=128,
    callbacks=[tf.keras.callbacks.EarlyStopping(
        patience=2, restore_best_weights=True
    )],
)
model.evaluate(x_test, y_test)
```

### 7.2 R

常用包：keras3。

```r
library(keras3)

mnist <- dataset_mnist()
x_train <- array_reshape(mnist$train$x / 255, c(-1, 28, 28, 1))
x_test <- array_reshape(mnist$test$x / 255, c(-1, 28, 28, 1))

model <- keras_model_sequential() |>
  layer_conv_2d(
    filters = 16, kernel_size = c(3, 3),
    padding = "same", activation = "relu",
    input_shape = c(28, 28, 1)
  ) |>
  layer_max_pooling_2d() |>
  layer_conv_2d(
    filters = 32, kernel_size = c(3, 3),
    padding = "same", activation = "relu"
  ) |>
  layer_global_average_pooling_2d() |>
  layer_dense(units = 10, activation = "softmax")

model |> compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)
model |> fit(
  x_train, mnist$train$y,
  validation_split = 0.1, epochs = 10, batch_size = 128,
  callbacks = callback_early_stopping(
    patience = 2, restore_best_weights = TRUE
  )
)
model |> evaluate(x_test, mnist$test$y)
```

## 8. 结果如何解读

- 分类看患者级外部测试集的 AUC/PR-AUC、灵敏度、特异度、校准与阈值；分割看 Dice/IoU，同时检查小病灶和边界误差。
- 高图像级准确率不代表高患者级性能，多视角或多切片需先按患者聚合。
- 热图显示“改变某区域会影响输出”或“网络对某区域响应”，不等于该区域是病因，也不保证解释忠实。
- 模型可能利用文字标记、床位、设备或裁剪方式等捷径，应结合遮挡、伪影分层和外部验证排查。
- 报告绝对性能之外，还应与放射科医生、临床基线或现行流程比较增量价值。

## 9. 假设诊断与稳健性

- 数据泄漏：用图像哈希/近重复检索和 patient_id 分组检查同源图像。
- 域偏移：按医院、扫描器、协议、视角和时间分层，报告性能与校准。
- 捷径学习：遮挡边框/文字、只保留肺野或病灶区、打乱背景，观察预测变化。
- 增强合理性：医学上不允许的旋转、翻转或颜色变换不能只因“常用”而采用。
- 鲁棒性：对噪声、压缩、分辨率、窗宽窗位与轻微配准误差做压力测试。
- 小样本：优先迁移学习、冻结部分骨干、强正则和重复交叉验证；测试集保持独立。

## 10. 推荐可视化

- 原图、预处理后图与增强样本并排图：验证没有破坏医学语义。
- 卷积核和浅层特征图：观察边缘、纹理与伪影响应。
- Grad-CAM、遮挡敏感性图：作为模型行为诊断，而非因果解释。
- 训练/验证曲线与混淆矩阵。
- 按中心、设备、视角、病灶大小分层的性能森林图。

## 11. 优势、局限与常见坑

### 优势

- 利用局部连接和共享权重，参数效率高。
- 分层学习从低级纹理到高级语义的空间表示。
- 可覆盖分类、检测、分割和迁移学习。

### 局限

- 对数据量、标注质量、域偏移和计算资源敏感。
- 不天然具有旋转、尺度或跨设备不变性。
- 可能学习非病理捷径，解释图也不能完全证明依据正确。

### 常见坑

- 按图像或切片随机切分，而不是按患者切分。
- 训练和测试中出现同一检查的近重复图。
- 用不合医学语义的数据增强。
- 只看总体 AUC，忽略校准、小病灶和设备亚组。
- 把二维切片当独立样本，虚增有效样本量。

## 12. 与相近方法的区别

- 与 MLP：CNN 保留空间结构并共享局部权重；MLP 展平输入后失去这些先验。
- 与影像组学：影像组学先计算人工特征，CNN 自动学习特征；前者小样本时更易审计。
- 与 Vision Transformer：CNN 强调局部性，ViT 通过注意力连接远距离图像块，通常更依赖预训练。
- 与 ResNet：ResNet 是加入残差连接、便于训练很深 CNN 的具体架构。
- 如何选择：中小型医学图像任务优先以预训练 CNN 建立强基线；只有在数据规模和全局依赖证据支持时再比较 ViT。

## 13. 医学研究中的典型应用

- 胸片、眼底、皮肤镜或病理图像的疾病分类和分级。
- CT/MRI 中器官、肿瘤和病灶的语义分割。
- 超声/内镜视频帧识别与辅助定位。
- ECG、EEG 等一维波形的局部模式检测。

研究报告应说明结局/标注单位、重复切片、患者级切分、设备与中心分布、数据增强、类别不平衡和外部验证。若结局为时间到事件，需另接处理删失的生存损失。

## 14. 关键术语

- **卷积核（Kernel/Filter）**：在输入上滑动、学习局部模式的一组共享权重。
- **特征图（Feature map）**：卷积核在各空间位置的响应。
- **通道（Channel）**：输入或特征图的并行层，如 RGB 三通道。
- **步幅（Stride）**：卷积核每次移动的像素数。
- **填充（Padding）**：在边缘补值以控制输出尺寸。
- **池化（Pooling）**：对局部区域做最大或平均汇总以降采样。
- **感受野（Receptive field）**：某层一个单元能间接看到的输入区域。
- **平移等变（Translation equivariance）**：输入平移时特征图随之平移。
- **捷径学习（Shortcut learning）**：模型利用与目标相关但非预期的伪影或流程线索。

## 15. 相关方法

- [[人工神经网络（Artificial Neural Network, ANN）]]
- [[多层感知机分类器（Multilayer Perceptron Classifier, MLP）]]
- [[残差网络（Residual Network, ResNet）]]
- [[DenseNet（Densely Connected Convolutional Network, DenseNet）]]
- [[Transformer（Transformer）]]

## 16. 参考资料

- LeCun Y, Bottou L, Bengio Y, Haffner P. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*. 1998;86(11):2278-2324. https://doi.org/10.1109/5.726791
- Krizhevsky A, Sutskever I, Hinton GE. ImageNet classification with deep convolutional neural networks. *NeurIPS*. 2012;25.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016. https://www.deeplearningbook.org/
- Litjens G, et al. A survey on deep learning in medical image analysis. *Medical Image Analysis*. 2017;42:60-88. https://doi.org/10.1016/j.media.2017.07.005
