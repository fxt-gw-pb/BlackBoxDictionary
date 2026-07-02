# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置Seaborn风格
sns.set(style="whitegrid", palette="pastel")

# 读取数据
file_path = "./dataset/007/data.csv"
df = pd.read_csv(file_path)

# 显示数据基本信息
print("数据基本信息：")
print(df.info())  # 查看数据类型、缺失值情况等

# 显示数据前5行
print("\n数据预览：")
print(df.head())

# 统计描述
print("\n数值特征统计信息：")
print(df.describe())

# 删除无关列
df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# 检查缺失值
missing_values = df.isnull().sum()
print("\n缺失值情况：")
print(missing_values[missing_values > 0])  # 只显示有缺失值的列

# 由于"Unnamed: 32"列已删除，数据集实际上没有缺失值

# 检查重复行
print("\n重复数据条数：", df.duplicated().sum())

# 删除重复值（如果有的话）
df.drop_duplicates(inplace=True)

numeric_df = df.select_dtypes(include=[np.number])

# 计算异常值范围（1.5倍IQR原则）
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

# 设定异常值范围
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 统计异常值情况
outliers = ((numeric_df < lower_bound) | (numeric_df > upper_bound)).sum()
print("\n各列异常值数量：")
print(outliers[outliers > 0])

# 直接删除异常值（仅适用于异常值极端的情况）
df = df[~((numeric_df < lower_bound) | (numeric_df > upper_bound)).any(axis=1)]

# 统计良性（B）和恶性（M）样本数量
plt.figure(figsize=(6, 4))
sns.countplot(x=df["diagnosis"], palette=["#FF9999", "#66B2FF"])

# 设置标题和标签
plt.title("Distribution of Diagnosis", fontsize=14)
plt.xlabel("Diagnosis", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()

# 选取部分关键变量
features = ["radius_mean", "texture_mean", "area_mean"]
plt.figure(figsize=(12, 4))

for i, feature in enumerate(features):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[feature], kde=True, bins=30, color=np.random.rand(3,))
    plt.title(f"Distribution of {feature}")

plt.tight_layout()
plt.show()

# 只选择数值型特征进行相关性计算
numeric_df = df.select_dtypes(include=['number'])

# 计算特征相关性矩阵
corr_matrix = numeric_df.corr()

# 绘制热力图
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5)

# 设置标题
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()

# 将 'diagnosis' 转换为二进制变量（M=1，B=0）
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 确保转换成功
print(df['diagnosis'].value_counts())


import seaborn as sns
import numpy as np

# 计算相关性矩阵
corr_matrix = df.corr().abs()

# 设定阈值（如 0.9），高于该阈值的特征将被移除
threshold = 0.9

# 找到相关性高的特征对
high_corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] > threshold:
            feature_name = corr_matrix.columns[i]
            high_corr_features.add(feature_name)

print("高相关特征（将被移除）：", high_corr_features)

# 删除高相关特征
df.drop(columns=high_corr_features, inplace=True)


from sklearn.feature_selection import SelectKBest, f_classif

# 选择前 10 个最重要的特征
X = df.drop(columns=['diagnosis'])  # 特征
y = df['diagnosis']  # 目标变量

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 获取被选中的特征名称
selected_features = X.columns[selector.get_support()]
print("被选中的 10 个关键特征：", selected_features)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)  # 只对数值特征进行标准化

# 转换为 DataFrame（方便后续分析）
X_scaled = pd.DataFrame(X_scaled, columns=selected_features)

from sklearn.model_selection import train_test_split

# 划分数据集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 查看数据形状
print("训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# 初始化 XGBoost 分类器（初始参数）
model = XGBClassifier(
    n_estimators=100,  # 决策树数量
    learning_rate=0.1,  # 学习率
    max_depth=4,  # 树的最大深度
    subsample=0.8,  # 采样比例
    colsample_bytree=0.8,  # 列采样
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)


# 预测测试集
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy:.4f}")

# 显示分类报告
print("\nClassification Report:\n", classification_report(y_test, y_pred))


from sklearn.model_selection import GridSearchCV

# 设定超参数搜索范围
param_grid = {
    'n_estimators': [50, 100, 200],  # 决策树数量
    'max_depth': [3, 4, 5],  # 树的深度
    'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    'subsample': [0.8, 1.0],  # 采样比例
    'colsample_bytree': [0.8, 1.0]  # 列采样
}

# 进行网格搜索
grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters:", grid_search.best_params_)

# 训练最佳模型
best_model = grid_search.best_estimator_


import matplotlib.pyplot as plt
import seaborn as sns

# 获取特征重要性
importances = best_model.feature_importances_

# 绘制柱状图
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=X_train.columns, palette="coolwarm")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# 计算预测概率
y_prob = best_model.predict_proba(X_test)[:, 1]  # 取"恶性"的概率

# 绘制直方图
plt.figure(figsize=(8, 4))
sns.histplot(y_prob, bins=20, kde=True, color="purple")
plt.title("Prediction Probability Distribution")
plt.xlabel("Probability of Malignant Cancer")
plt.ylabel("Frequency")
plt.show()

from sklearn.metrics import roc_curve, auc

# 计算 ROC 曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()