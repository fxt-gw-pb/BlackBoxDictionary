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

### 1.1 定义

卷积神经网络是一类专门处理图像或局部空间结构数据的神经网络。它通过卷积核、激活函数和池化层，从原始像素中自动提取边缘、纹理和高级语义特征。

### 1.2 它主要解决什么问题

- 研究问题：当输入具有局部邻域结构时，如何高效学习位置相关又具一定平移鲁棒性的特征。
- 适用任务：图像分类、目标检测、医学影像分型、波形或二维矩阵识别。
- 常见医学场景：胸片异常识别、病理切片分类、眼底图像分级、ECG/EEG 图样识别。

### 1.3 直觉理解

CNN 像一组在图像上滑动的小型“特征探测器”。浅层先识别边缘和纹理，深层再把这些基础模式组合成器官、病灶或其他更复杂的视觉结构。

## 2. 数学形式

### 2.1 核心公式

对输入特征图 $X$ 和卷积核 $K$，二维卷积可写为：

$$
Y_{i,j} = \sum_{m}\sum_{n} X_{i+m, j+n} K_{m,n} + b
$$

经过激活函数后得到：

$$
A_{i,j} = \phi(Y_{i,j})
$$

最大池化常写为：

$$
P_{i,j} = \max_{(m,n)\in \mathcal{W}_{i,j}} A_{m,n}
$$

最终特征经展平和全连接层输出分类或回归结果。

### 2.2 参数或统计量含义

- 卷积核大小：决定局部感受野范围。
- 通道数：决定可学习的特征图数量。
- stride：控制滑动步长。
- padding：控制边界处理与输出尺寸。
- pooling：用于下采样和增强鲁棒性。

### 2.3 关键假设

- 数据存在局部空间相关结构。
- 相同模式可能在不同位置重复出现。
- 样本量、增强策略和正则化足以支撑深层训练。

## 3. 数据形式与输入输出

### 3.1 适合的数据形式

- 自变量类型：二维或三维图像张量，也可扩展到一维信号和三维体数据。
- 因变量类型：分类、回归、分割标签。
- 数据结构：通常为 `batch × height × width × channel` 或 `batch × channel × height × width`。
- 是否适合高维数据：适合高维像素输入，但需显存和样本支持。
- 是否适合缺失较多数据：对系统性缺失不鲁棒，需先处理。
- 是否适合删失数据：不直接处理删失。
- 是否适合重复测量数据：若重复测量表现为图像序列，通常需结合时序结构。

### 3.2 示例表格

以胸部 X 线分类为例：

| PatientID | ImagePath | View | Age | Pneumonia |
| --- | --- | --- | --- | --- |
| P001 | img_001.png | PA | 67 | 1 |
| P002 | img_002.png | AP | 45 | 0 |
| P003 | img_003.png | PA | 72 | 1 |
| P004 | img_004.png | AP | 39 | 0 |

### 3.3 输入与产出

#### 输入

- 输入数据：图像张量及标签。
- 关键变量：卷积层数、卷积核大小、池化策略、学习率、batch size。
- 需要预处理的内容：缩放、标准化、裁剪、数据增强、类别不平衡处理。

#### 产出

- 模型对象/统计结果：训练好的 CNN 权重和验证指标。
- 参数估计：卷积核和全连接层参数。
- 预测结果：类别概率、连续值或像素级输出。
- 不确定性指标：AUC、F1、校准、外部验证表现。

## 4. 适用场景

- 适合：图像和局部结构信号任务、需要自动特征提取的场景。
- 不适合：样本极少且图像差异弱、主要依赖人工定义少量结构化变量的问题。
- 使用前需要特别检查的点：标签质量、数据泄露、中心差异、图像增强是否合理。

## 5. 实现

### 5.1 Python

常用包：

- `tensorflow`
- `keras`

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(128, 128, 1)),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])
```

### 5.2 R

常用包：

- `keras`

```r
library(keras)

model <- keras_model_sequential() |>
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu",
                input_shape = c(128, 128, 1)) |>
  layer_max_pooling_2d() |>
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") |>
  layer_max_pooling_2d() |>
  layer_flatten() |>
  layer_dense(units = 64, activation = "relu") |>
  layer_dense(units = 2, activation = "softmax")
```

## 6. 结果如何解释

- 核心结果看什么：验证集区分度、混淆矩阵、外部数据泛化能力。
- 每个主要参数如何解释：卷积核和中间特征图通常难逐个解释，更常看总体性能和热图。
- 临床或医学意义如何表达：需要结合 Grad-CAM、显著性图或错误案例复核，判断模型是否关注合理解剖区域。
- 常见误读：高准确率不等于学到病灶，模型可能只利用设备差异、标记文字或背景伪特征。

## 7. 推荐可视化

- 训练/验证损失和准确率曲线。
- 混淆矩阵。
- 热图解释，如 Grad-CAM。

## 8. 优势、局限与常见坑

### 优势

- 自动提取图像层级特征。
- 对局部平移和形变较稳健。
- 能迁移利用大量预训练模型。

### 局限

- 对标注质量和样本量要求高。
- 训练成本较高。
- 可解释性通常弱于结构化模型。

### 常见坑

- 病人级数据未分层，导致同一患者图像泄露到测试集。
- 只看内部验证，不做设备或医院外验证。
- 过度依赖图像增强，产生不真实分布。

## 9. 与相近方法的区别

- 和 [[多层感知机分类器（Multilayer Perceptron Classifier, MLP）]] 的区别：MLP 直接处理展平向量，CNN 显式利用局部空间结构和参数共享。
- 和 [[AlexNet（AlexNet）]] 的区别：AlexNet 是经典深层 CNN 架构之一。
- 和 [[残差网络（Residual Network, ResNet）]] 的区别：ResNet 通过残差连接支持更深网络训练。

## 10. 医学研究中的典型应用

- 胸片、CT、MRI 的疾病识别。
- 数字病理图像分类和分型。
- 眼底图像和皮肤病变分级。

## 11. 相关方法

- [[人工神经网络（Artificial Neural Network, ANN）]]
- [[AlexNet（AlexNet）]]
- [[残差网络（Residual Network, ResNet）]]
- [[DenseNet（Densely Connected Convolutional Network, DenseNet）]]

## 12. 参考资料

- LeCun Y, Bengio Y, Hinton G. Deep learning. *Nature*. 2015;521(7553):436-444.
- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016.
- Chollet F. *Deep Learning with Python*. 2nd ed. Manning; 2021.
