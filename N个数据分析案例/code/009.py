import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 读取数据
file_path = "./dataset/009/creditcard.csv"  # 数据文件路径
df = pd.read_csv(file_path)

# 显示数据集的基本信息
print("数据集基本信息：")
print(df.info())

# 显示数据的前几行
print("\n数据集前几行：")
print(df.head())

# 统计数据的基本描述信息
print("\n数据集统计描述：")
print(df.describe())

# 检查缺失值
print("\n数据集中缺失值统计：")
print(df.isnull().sum())

# 检查是否存在无穷大 (inf) 数据
print("\n数据集中是否包含无穷大 (inf) 值：")
print(np.isinf(df.values).sum())

# 检查重复数据
print("\n数据集中重复行的数量：", df.duplicated().sum())

# 如果有重复数据，则删除
df = df.drop_duplicates()
print("去除重复数据后数据集大小：", df.shape)

# 交易金额的分布
plt.figure(figsize=(10, 5))
sns.histplot(df['Amount'], bins=50, kde=True, color='blue')
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.title("Distribution of Transaction Amount")
plt.show()

# 交易时间的分布
plt.figure(figsize=(10, 5))
sns.histplot(df['Time'], bins=50, kde=True, color='green')
plt.xlabel("Transaction Time (Seconds)")
plt.ylabel("Frequency")
plt.title("Distribution of Transaction Time")
plt.show()

# 类别分布
plt.figure(figsize=(6, 6))
labels = ["Non-Fraud (0)", "Fraud (1)"]
sizes = df["Class"].value_counts().values
colors = ["#3498db", "#e74c3c"]
plt.pie(sizes, labels=labels, autopct='%1.2f%%', colors=colors, startangle=140, explode=(0, 0.1))
plt.title("Class Distribution")
plt.show()

# 计算相关性矩阵
corr_matrix = df.corr()

# 绘制相关性热力图
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Correlation Heatmap of Features")
plt.show()

# 显示与 Class 相关性最高的前 5 个特征
corr_with_target = corr_matrix["Class"].abs().sort_values(ascending=False)
print("与 Class 相关性最高的前 5 个特征：\n", corr_with_target.head(6))

# 选取与 Class 相关性较高的特征
important_features = ["V10", "V12", "V14", "V17", "V18"]

plt.figure(figsize=(15, 8))
for i, feature in enumerate(important_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df["Class"], y=df[feature], palette=["blue", "red"])
    plt.xlabel("Class")
    plt.ylabel(feature)
    plt.title(f"Boxplot of {feature}")
plt.tight_layout()
plt.show()

# 去除无关特征 'Time'
df = df.drop(columns=['Time'])

# 归一化交易金额 'Amount'
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# 查看数据处理后的信息
print("数据集处理后信息：")
print(df.head())

from imblearn.under_sampling import RandomUnderSampler

X = df.drop(columns=['Class'])  # 选择特征
y = df['Class']  # 目标变量

# 进行欠采样，使欺诈交易和正常交易数量相等
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# 重新合并成 DataFrame
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Class'])],
                         axis=1)

# 重新检查类别分布
print("欠采样后数据类别分布：")
print(df_resampled['Class'].value_counts())

from sklearn.model_selection import train_test_split

# 划分训练集和测试集，80% 作为训练集，20% 作为测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42,
                                                    stratify=y_resampled)

# 输出数据集划分结果
print(f"训练集样本数：{X_train.shape[0]}")
print(f"测试集样本数：{X_test.shape[0]}")

from xgboost import XGBClassifier

# 计算类别权重，解决类别不均衡问题
fraud_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 定义 XGBoost 模型
model = XGBClassifier(
    scale_pos_weight=fraud_weight,  # 处理类别不均衡
    n_estimators=50,  # 50 棵树
    max_depth=5,  # 树的最大深度
    learning_rate=0.15,  # 学习率
    subsample=0.8,  # 采样比例
    colsample_bytree=0.8,  # 选择部分特征进行训练
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix

# 预测测试集
y_pred = model.predict(X_test)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 打印分类报告
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))

import seaborn as sns
import matplotlib.pyplot as plt

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制混淆矩阵热力图
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=['Non-Fraud', 'Fraud'],
            yticklabels=['Non-Fraud', 'Fraud'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import roc_curve, auc

# 计算预测概率
y_scores = model.predict_proba(X_test)[:, 1]

# 计算 FPR 和 TPR
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

from sklearn.model_selection import RandomizedSearchCV

# 定义超参数搜索范围
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.02, 0.05, 0.08, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# 进行随机搜索
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(scale_pos_weight=fraud_weight, random_state=42),
    param_distributions=param_dist,
    n_iter=20,  # 进行 20 轮搜索
    scoring='f1',  # 选择 F1-score 作为优化目标
    cv=5,  # 3 折交叉验证
    verbose=2,
    random_state=42,
    n_jobs=-1  # 并行计算
)

# 训练模型
random_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters:", random_search.best_params_)

# 使用最优参数训练最终模型
best_model = random_search.best_estimator_

# 预测测试集
y_pred_best = best_model.predict(X_test)

# 计算评估指标
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)

print(f"Optimized Accuracy: {accuracy_best:.4f}")
print(f"Optimized Precision: {precision_best:.4f}")
print(f"Optimized Recall: {recall_best:.4f}")
print(f"Optimized F1 Score: {f1_best:.4f}")