import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from wordcloud import WordCloud

# 读取数据集
file_path = "dataset/010/spam.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# 查看数据前几行
df.head()

# 查看数据集的基本信息
df.info()

# 查看数据的统计摘要
df.describe(include="all")

# 只保留有用的列
df = df[['v1', 'v2']]

# 重命名列
df.columns = ['label', 'message']

# 再次检查数据
df.head()

# 检查缺失值
missing_values = df.isnull().sum()
print(missing_values)

# 由于本数据集无缺失值，若有可使用以下方法填充：
df = df.dropna()  # 删除缺失值
# df = df.fillna('未知')  # 或者填充特定值

# 检查重复值
duplicate_count = df.duplicated().sum()
print(f"重复值数量: {duplicate_count}")

# 删除重复值
df = df.drop_duplicates()

# 统计各类别的数量
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label', palette="coolwarm")

# 设置标题
plt.title("Distribution of Messages", fontsize=14)
plt.xlabel("Message Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()

# 添加短信长度列
df["message_length"] = df["message"].apply(len)

# 绘制短信长度分布
plt.figure(figsize=(8, 5))
sns.histplot(df["message_length"], bins=50, kde=True, color='royalblue')

# 设置标题和标签
plt.title("Message Length Distribution", fontsize=14)
plt.xlabel("Message Length", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()

# 生成垃圾短信和正常短信的文本
spam_text = " ".join(df[df["label"] == "spam"]["message"])
ham_text = " ".join(df[df["label"] == "ham"]["message"])

# 绘制词云
plt.figure(figsize=(12, 5))

# 垃圾短信词云
plt.subplot(1, 2, 1)
spam_wc = WordCloud(width=400, height=300, background_color="black", colormap="Reds").generate(spam_text)
plt.imshow(spam_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Spam Message WordCloud", fontsize=14)

# 正常短信词云
plt.subplot(1, 2, 2)
ham_wc = WordCloud(width=400, height=300, background_color="black", colormap="Blues").generate(ham_text)
plt.imshow(ham_wc, interpolation="bilinear")
plt.axis("off")
plt.title("Ham Message WordCloud", fontsize=14)

plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='label', y='message_length', palette="coolwarm")

# 设置标题
plt.title("Message Length vs Spam/Ham", fontsize=14)
plt.xlabel("Message Type", fontsize=12)
plt.ylabel("Message Length", fontsize=12)
plt.show()

# 使用 LabelEncoder 进行标签编码
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# 查看编码结果
df[["label", "label_encoded"]].head(10)

# 创建 BoW 特征提取器
bow_vectorizer = CountVectorizer(stop_words='english', max_features=3000)

# 进行特征转换
X_bow = bow_vectorizer.fit_transform(df["message"])

# 查看特征矩阵形状
print(f"BoW 特征矩阵维度: {X_bow.shape}")

# 创建 TF-IDF 特征提取器
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

# 进行特征转换
X_tfidf = tfidf_vectorizer.fit_transform(df["message"])

# 查看特征矩阵形状
print(f"TF-IDF 特征矩阵维度: {X_tfidf.shape}")

# 选择 TF-IDF 特征
X = X_tfidf  # 也可以替换为 X_bow
y = df["label_encoded"]

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 查看训练集和测试集的样本数量
print(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# 构建线性SVM模型（适合高维、稀疏文本数据）
svm_model = LinearSVC(random_state=42, C=1.0)  # C为正则化参数

# 训练模型（这里选择了TF-IDF特征，也可以尝试BoW特征）
svm_model.fit(X_train, y_train)

# 预测测试集
y_pred = svm_model.predict(X_test)

# 输出分类报告
print(classification_report(y_test, y_pred))

# 输出混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.show()

# 降维到二维（只用于可视化，不参与模型训练）
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train.toarray())
X_test_pca = pca.transform(X_test.toarray())

# 重新训练一个在PCA空间下的SVM模型（用于可视化目的）
svm_model_pca = LinearSVC(random_state=42, C=1.0)
svm_model_pca.fit(X_train_pca, y_train)

# 绘制散点图及决策边界
plt.figure(figsize=(8, 6))
# 绘制测试集散点图
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm', edgecolor='k', s=50, alpha=0.7)

# 绘制决策边界
# 创建网格数据
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = svm_model_pca.decision_function(grid)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], alpha=0.2, colors=['blue', 'red'])
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

plt.title("SVM Decision Boundary (PCA Projection)", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.show()

# 1. 模型训练与超参数调优
# 定义初始的 LinearSVC 模型
svm_model = LinearSVC(random_state=42)

# 定义超参数网格，用于调优 C 和 max_iter 参数
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'max_iter': [1000, 2000, 5000]
}

# 使用 GridSearchCV 进行超参数调优，评价指标选择 F1 分数，采用 5 折交叉验证
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳超参数
print("Best Parameters:", grid_search.best_params_)

# 使用最佳超参数构建模型
best_svm_model = grid_search.best_estimator_

# 2. 模型评估
# 在测试集上进行预测
y_pred = best_svm_model.predict(X_test)

# 计算主要评价指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# 输出详细的分类报告
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 3. 可视化展示
# 3.1 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Spectral', cbar=True)
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.show()

# 3.2 绘制 ROC 曲线
# LinearSVC 没有 predict_proba 方法，使用 decision_function 得到连续的分数
y_scores = best_svm_model.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.show()