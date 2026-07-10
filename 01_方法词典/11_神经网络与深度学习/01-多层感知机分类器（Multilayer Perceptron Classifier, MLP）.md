---
title: 多层感知机分类器
english_name: Multilayer Perceptron Classifier, MLP
slug: multilayer-perceptron-classifier-mlp
aliases: [MLP, multilayer perceptron classifier, 前馈神经网络, "多层感知机分类器（Multilayer Perceptron Classifier, MLP）"]
category: 神经网络与深度学习
subcategory: 前馈神经网络
tags: [医学统计, 数据科学, 神经网络, 深度学习, 分类]
status: 已建
difficulty: intermediate
question_type: 神经网络分类
data_type: [表格数据, 图像向量]
outcome_type: [二分类, 多分类]
python_packages: [scikit-learn]
r_packages: [nnet]
---

# 多层感知机分类器（Multilayer Perceptron Classifier, MLP）

## 1. 方法概览

### 1.1 一句话本质

MLP 把多次“线性组合 + 非线性激活”串起来，将原本不能用一条直线分开的类别，在逐层变换后的表示空间中分开。

### 1.2 定义

多层感知机分类器是一类前馈人工神经网络：信息从输入层经一个或多个隐藏层流向输出层，不形成时间上的反馈环。每个神经元先对上一层输出做加权求和，再经过激活函数；模型用反向传播计算梯度，用梯度下降更新全部权重。

### 1.3 它主要解决什么问题

- 研究问题：多个指标以复杂、非线性和交互方式共同决定类别时，如何学习灵活的判别边界。
- 适用任务：二分类、多分类和多标签分类。
- 常见医学场景：基于结构化 EHR 的重症风险分层、多个生物标志物联合诊断、向量化影像特征分类。

### 1.4 直觉与类比

把隐藏层想成几组会自动学习的“组合指标”。第一层可能把年龄、血压和肌酐组合成若干风险模式；下一层再组合这些模式；输出层才把最终表示翻译成疾病概率。关键不在“层数多”，而在激活函数打破纯线性：若每层都没有非线性，无论堆多少层，整体仍等价于一次线性变换。

## 2. 核心思想与原理

### 2.1 它到底在解决什么根本困难

Logistic 回归只有一个线性预测子，天然偏好平面型决策边界。现实中“高龄且高肌酐才危险”“某指标只在女性中有效”都属于弯曲边界或交互效应。手工枚举所有交互和非线性变换既困难又容易漏掉；MLP 让隐藏单元从数据中学习这些中间表示。

### 2.2 关键洞察

一层隐藏单元各自切出一个简单区域，多层网络再把许多简单区域拼成复杂边界。训练时，链式法则把最终预测误差逐层传回去：某个权重对损失影响越大，更新幅度就越大。这就是反向传播的本质——高效重复使用中间导数，而不是一种新的统计准则。

### 2.3 与朴素/相邻做法的对比

- 相比 Logistic 回归：MLP 自动学习非线性和交互，但参数更多、解释更弱，也更需要样本量与正则化。
- 相比单层感知机：MLP 的隐藏层和可微激活能处理非线性可分问题，并可输出连续概率。
- 相比树模型：MLP 的函数通常更平滑且便于端到端学习；树对表格数据、缺失模式和小样本往往更省心。

## 3. 数学形式

### 3.1 核心公式

以一层隐藏层的二分类 MLP 为例：

$$
\mathbf h=\phi(\mathbf W_1\mathbf x+\mathbf b_1),\qquad
z=\mathbf w_2^\top\mathbf h+b_2,\qquad
\hat p=\sigma(z)=\frac{1}{1+\exp(-z)}
$$

$$
\mathcal L=-\left[y\log(\hat p)+(1-y)\log(1-\hat p)\right]
$$

这个式子在说：先把输入变成隐藏表示，再把表示压成一个 logit，最后用 sigmoid 得到阳性概率，并用交叉熵惩罚与真实标签不一致的预测。

多分类时通常使用 softmax：

$$
\hat p_k=\frac{\exp(z_k)}{\sum_{j=1}^{K}\exp(z_j)}
$$

### 3.2 推导脉络

1. 线性组合 $\mathbf W_1\mathbf x+\mathbf b_1$ 汇总输入证据。
2. 激活函数 $\phi$ 引入非线性；ReLU 的形式是 $\max(0,a)$。
3. 二分类需要 $0$ 到 $1$ 的输出，因此用 sigmoid；多分类用 softmax 让各类概率和为 1。
4. 对 Bernoulli 或 categorical 似然取负对数，就得到二元或多类交叉熵。
5. 由链式法则，输出层的关键误差信号为 $\partial\mathcal L/\partial z=\hat p-y$，再逐层乘局部导数得到每个权重的梯度。

### 3.3 参数与统计量含义

- $\mathbf x$：一个样本的输入特征向量，训练前通常需要标准化。
- $\mathbf W_1,\mathbf w_2$：连接权重，决定信息以多大方向和强度传播。
- $\mathbf b_1,b_2$：偏置，使激活阈值不必经过原点。
- $\mathbf h$：隐藏表示，不是直接观测到的临床变量。
- $\hat p$：模型估计的类别概率；它是否可信还取决于校准。
- $\mathcal L$：交叉熵损失；训练优化的是平均损失加正则项，不是直接优化准确率。

### 3.4 关键假设（含违反后果）

| 假设 | 含义 | 违反后会怎样 | 如何粗查 |
| --- | --- | --- | --- |
| 训练与应用同分布 | 未来患者与训练患者的采集流程、谱系近似 | 外部性能和校准下降 | 按时间、中心和设备做外部验证 |
| 切分后样本独立 | 同一患者的多次记录不能跨训练集和测试集 | 数据泄漏，性能虚高 | 按 patient_id 分组切分 |
| 标签定义可靠 | 阳性和阴性的判定标准一致 | 网络会拟合标注噪声 | 抽样复核、报告一致性 |
| 信息量足以支撑复杂度 | 有效样本量与参数量、类别数相匹配 | 严重过拟合 | 比较训练/验证学习曲线 |
| 缺失机制被妥善处理 | 缺失值不能未经处理直接输入 | 训练失败或学到就医流程偏差 | 绘制缺失模式并做敏感性分析 |

## 4. 手把手算例

设患者有两个已标准化特征 $\mathbf x=(2,1)^\top$，真实标签 $y=1$。一个有两个 ReLU 隐藏单元的网络参数为：

$$
\mathbf W_1=
\begin{pmatrix}
0.5&0.5\\
-0.5&1
\end{pmatrix},
\quad
\mathbf b_1=
\begin{pmatrix}
-1\\
0.5
\end{pmatrix},
\quad
\mathbf w_2=
\begin{pmatrix}
1.2\\
-0.8
\end{pmatrix},
\quad b_2=0.1
$$

**第 1 步：隐藏层前向传播。**

$$
a_1=0.5\times2+0.5\times1-1=0.5
$$

$$
a_2=-0.5\times2+1\times1+0.5=0.5
$$

两者都大于 0，所以 ReLU 后 $\mathbf h=(0.5,0.5)^\top$。

**第 2 步：输出概率。**

$$
z=1.2\times0.5-0.8\times0.5+0.1=0.3
$$

$$
\hat p=\frac{1}{1+\exp(-0.3)}=0.574
$$

模型给出 57.4% 的阳性概率。

**第 3 步：计算损失。**

$$
\mathcal L=-\log(0.574)=0.555
$$

**第 4 步：看一次反向传播。** 输出误差信号为

$$
\delta_z=\hat p-y=0.574-1=-0.426
$$

因此输出权重梯度为

$$
\frac{\partial\mathcal L}{\partial\mathbf w_2}
=\delta_z\mathbf h
=(-0.213,-0.213)^\top
$$

若学习率为 0.1，第一条输出权重从 1.2 更新为 $1.2-0.1(-0.213)=1.2213$；朝着提高阳性概率的方向移动。这个例子把“前向得到预测、损失衡量错误、反向把错误分给权重”连成了一条完整链。

## 5. 数据形式与输入输出

### 5.1 适合的数据形式

- 自变量类型：连续、二元或编码后的类别变量；量纲通常需要统一。
- 因变量类型：二分类、多分类或多标签结局。
- 数据结构：每行一个独立分析单位，每列一个输入特征。
- 是否适合高维数据：可以，但需正则化、降维或足够样本量。
- 是否适合缺失较多数据：不直接适合，需插补并保留缺失指示。
- 是否适合删失数据：普通分类损失不处理删失；需改用生存神经网络。
- 是否适合重复测量数据：可将序列汇总为特征；原始时序更适合 RNN、LSTM 或 Transformer。

### 5.2 示例表格

| patient_id | age_z | creatinine_z | diabetes | severe_30d |
| --- | ---: | ---: | ---: | ---: |
| P01 | -0.8 | -0.4 | 0 | 0 |
| P02 | 0.2 | 0.1 | 1 | 0 |
| P03 | 1.1 | 1.8 | 1 | 1 |
| P04 | 0.7 | -0.2 | 0 | 1 |

### 5.3 输入与产出

#### 输入

- 输入数据：数值矩阵 $\mathbf X$ 与类别标签 $\mathbf y$。
- 关键设置：隐藏层宽度、层数、激活函数、学习率、批大小和正则化强度。
- 预处理：按训练集参数标准化、编码类别变量、处理缺失并按患者切分。

#### 产出

- 模型对象：已学习的权重、偏置与预处理流程。
- 预测结果：每类概率与按阈值得到的类别。
- 训练过程：训练/验证损失、迭代次数和早停点。
- 不确定性：普通 MLP 不自动给出置信区间，可用 bootstrap、深度集成或贝叶斯近似。

## 6. 适用场景

- 适合：关系明显非线性、交互复杂、样本量足够且重点是预测。
- 不适合：样本极少、主要目标是参数的因果/临床解释、存在未处理删失，或表格数据上简单模型已足够。
- 使用前特别检查：患者级切分、类别不平衡、标准化、过拟合、概率校准和外部可迁移性。

## 7. 实现

### 7.1 Python

常用包：scikit-learn。

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, log_loss

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        alpha=1e-3,
        early_stopping=True,
        max_iter=1000,
        random_state=42,
    ),
)
model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, prob))
print("log loss:", log_loss(y_test, prob))
```

真实 EHR 中应把随机切分替换成按患者、中心或时间的分组切分。

### 7.2 R

常用包：nnet。

```r
library(nnet)

set.seed(42)
idx <- sample(seq_len(nrow(iris)), size = 0.75 * nrow(iris))
train <- iris[idx, ]
test <- iris[-idx, ]

# 仅用训练集均值和标准差缩放，避免信息泄漏
mu <- sapply(train[, 1:4], mean)
sdv <- sapply(train[, 1:4], sd)
train_x <- scale(train[, 1:4], center = mu, scale = sdv)
test_x <- scale(test[, 1:4], center = mu, scale = sdv)

fit <- nnet(
  x = train_x, y = class.ind(train$Species),
  size = 8, softmax = TRUE, decay = 1e-3,
  maxit = 1000, trace = FALSE
)
prob <- predict(fit, test_x, type = "raw")
pred <- colnames(prob)[max.col(prob)]
mean(pred == test$Species)
```

## 8. 结果如何解读

- 首先看独立验证集的区分度、校准和临床效用，而不是训练准确率。
- 例如 AUC 为 0.86 说明随机抽取一名阳性和一名阴性时，阳性风险分数更高的概率约为 86%；它不说明概率已经校准。
- 概率 0.70 表示模型在当前训练分布下给出 70% 风险，不等于该患者“有 70% 的确定性患病”。
- 阈值应根据漏诊与误诊代价、患病率和应用流程预先选择，不能默认使用 0.5。
- 单个权重通常不能像回归系数那样独立解释；可辅以置换重要性、SHAP、部分依赖和敏感性分析，但解释仍是关联性的。

## 9. 假设诊断与稳健性

- 学习曲线：训练损失下降而验证损失上升提示过拟合，可早停、增大 weight decay、dropout 或缩小网络。
- 优化稳定性：用多个随机种子重复训练；结果差异很大说明结论依赖初始化。
- 概率校准：画校准曲线并报告 Brier score；必要时在独立校准集上做 Platt 或 isotonic calibration。
- 类别不平衡：同时报告 PR-AUC、灵敏度和阳性预测值；可用类别权重或重采样，但测试集保持原患病率。
- 分布漂移：按时间、医院、设备和人群亚组评估；外部验证优先于内部随机切分。
- 缺失稳健性：比较合理插补方案，并检查“缺失指示”是否只是代理了就医强度。

## 10. 推荐可视化

- 训练与验证损失曲线：观察收敛、过拟合和早停位置。
- ROC、PR 与校准曲线：分别观察区分度、不平衡场景表现和概率可信度。
- 不同阈值的混淆矩阵或决策曲线：连接模型性能与临床代价。
- 特征归因图：用于提出模型行为假设，不将其当作因果解释。

### 10.1 图像示例

下图展示训练损失随迭代下降；若同时绘制验证损失，更容易发现过拟合拐点。

![](../../04_示例图像/mlp_training_loss_curve.png)

## 11. 优势、局限与常见坑

### 优势

- 能自动学习复杂非线性与交互。
- 可统一处理二分类、多分类和多标签任务。
- 可与图像、文本等表征端到端连接。

### 局限

- 对缩放、超参数、样本量和优化过程较敏感。
- 参数可辨识性和直接解释性弱。
- 在中小型表格数据上不一定优于正则化回归或梯度提升树。

### 常见坑

- 标准化在全数据上拟合，造成测试信息泄漏。
- 同一患者的多条记录跨数据集切分。
- 只报告准确率，忽略患病率、校准和阈值代价。
- 把高维、少样本网络的训练拟合当作泛化能力。
- 把解释算法给出的重要性误写成因果效应。

## 12. 与相近方法的区别

- 与人工神经网络：ANN 是总称，MLP 是其中最经典的前馈全连接结构；本卡聚焦分类。
- 与 Logistic 回归：当线性和少量预设交互足够时，Logistic 更稳定、更易审计。
- 与 CNN：图像有局部空间结构时，CNN 的局部连接和权重共享比展平后输入 MLP 更高效。
- 与树集成：结构化表格、中小样本或缺失较多时，应把 XGBoost/随机森林作为强基线。
- 如何选择：先建立正则化 Logistic 与树模型基线；只有在样本量、非线性收益和验证证据都支持时再采用 MLP。

## 13. 医学研究中的典型应用

- 用人口学、检验和生命体征预测 30 天再入院或院内恶化。
- 用多组学或影像组学特征进行肿瘤分型。
- 将多个已提取的影像、文本和结构化特征做晚期融合。

必须说明结局定义、预测时点、特征可用时点、缺失处理、类别不平衡、患者级切分和外部验证；普通 MLP 不处理时间到事件结局中的删失。

## 14. 关键术语

- **前馈（Feedforward）**：信息只从输入向输出传播，不把后一步状态传回前一步。
- **隐藏层（Hidden layer）**：位于输入和输出之间、学习中间表示的神经元层。
- **激活函数（Activation function）**：为线性组合加入非线性的函数，如 ReLU、sigmoid。
- **Logit**：概率的对数优势 $\log[p/(1-p)]$，二分类输出层在 sigmoid 前的数值。
- **交叉熵（Cross-entropy）**：衡量预测概率与真实类别不一致程度的损失。
- **反向传播（Backpropagation）**：用链式法则从输出向前高效计算每个参数梯度。
- **Epoch**：模型完整看过一次训练集。
- **批大小（Batch size）**：每次参数更新所用的样本数。
- **早停（Early stopping）**：验证性能不再改善时停止训练，以减少过拟合。

## 15. 相关方法

- [[人工神经网络（Artificial Neural Network, ANN）]]
- [[感知机（Perceptron）]]
- [[Logistic回归（Logistic Regression）]]
- [[卷积神经网络（Convolutional Neural Network, CNN）]]
- [[交叉验证（Cross-Validation）]]

## 16. 参考资料

- Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press; 2016. https://www.deeplearningbook.org/
- Rumelhart DE, Hinton GE, Williams RJ. Learning representations by back-propagating errors. *Nature*. 1986;323:533-536. https://doi.org/10.1038/323533a0
- Pedregosa F, et al. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*. 2011;12:2825-2830.
- scikit-learn developers. Neural network models (supervised). https://scikit-learn.org/stable/modules/neural_networks_supervised.html
