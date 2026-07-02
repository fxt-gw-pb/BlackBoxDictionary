# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据集
file_path = './dataset/004/Churn_Modelling.csv'
df = pd.read_csv(file_path)

# 查看数据的基本信息
print(df.info())  # 显示数据的列信息和数据类型
print(df.describe())  # 显示数值型列的统计信息
print(df.head())  # 显示前几行数据，检查数据读取是否正常

# 检查是否存在缺失值
print(df.isnull().sum())  # 输出每列缺失值的数量

# 检查是否有重复值
print(f"重复值数量: {df.duplicated().sum()}")  # 查看重复值的数量

# 删除重复值（如果有的话）
df.drop_duplicates(inplace=True)

# 绘制数值列的箱线图，检查异常值
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']])
plt.title('Boxplot of Numeric Columns')
plt.show()

# 过滤年龄在合理范围之外的异常值
df = df[(df['Age'] > 18) & (df['Age'] < 100)]

# 对其他列也可以根据业务逻辑进行处理，如余额
df = df[df['Balance'] >= 0]

# 设置图形风格
sns.set(style='whitegrid', palette='bright')

# 绘制数值列的直方图
df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']].hist(bins=20, figsize=(12, 8), color='skyblue')
plt.suptitle('Histograms of Numeric Columns', fontsize=16)
plt.show()

# 绘制客户年龄和账户余额的散点图，观察数据分布趋势
plt.figure(figsize=(8, 6))
plt.scatter(df['Age'], df['Balance'], alpha=0.6, c=df['Exited'], cmap='coolwarm')
plt.title('Scatterplot of Age vs Balance')
plt.xlabel('Age')
plt.ylabel('Balance')
plt.colorbar(label='Exited')
plt.show()

# 只选择数值型特征进行相关性计算
numeric_df = df.select_dtypes(include=['number'])

# 计算相关性矩阵
corr_matrix = numeric_df.corr()

# 绘制热图展示相关性
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# 最终检查是否还有缺失值和重复值
print(f"最终缺失值数量: {df.isnull().sum().sum()}")
print(f"最终重复值数量: {df.duplicated().sum()}")

# 保存清洗后的数据
df.to_csv('./dataset/004/Churn_Modelling_cleaned.csv', index=False)

# 丢弃不相关的列
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# 检查特征选择后的数据
print(df.head())

# 使用独热编码处理类别特征
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# 查看编码后的数据集
print(df.head())

# 创建新特征：账户余额与预估工资的比率
df['BalanceSalaryRatio'] = df['Balance'] / df['EstimatedSalary']

# 检查新特征的分布
print(df[['Balance', 'EstimatedSalary', 'BalanceSalaryRatio']].head())

from sklearn.preprocessing import StandardScaler

# 选择需要标准化的数值特征
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio']

# 初始化标准化器
scaler = StandardScaler()

# 对数值特征进行标准化
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# 查看标准化后的特征
print(df.head())

from sklearn.model_selection import train_test_split

# 定义特征和目标变量
X = df.drop('Exited', axis=1)  # 特征
y = df['Exited']  # 目标变量

# 划分训练集和测试集，80%训练集，20%测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 输出训练集和测试集的形状
print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")


# 导入必要的库
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV

# 构建并训练XGBoost分类器
xgb_model = XGBClassifier(
    n_estimators=100,  # 初始树数量
    learning_rate=0.1,  # 初始学习率
    max_depth=4,  # 树的最大深度
    scale_pos_weight=3,  # 处理类别不平衡问题
    random_state=42  # 固定随机数种子
)

# 训练模型
xgb_model.fit(X_train, y_train)

# 测试集预测
y_pred = xgb_model.predict(X_test)

# 输出混淆矩阵、分类报告和AUC值
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 计算AUC分数
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]  # 获取流失客户的预测概率
auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC Score: {auc:.2f}")

# 定义超参数网格
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'scale_pos_weight': [1, 2, 3]
}

# 网格搜索
grid_search = GridSearchCV(estimator=XGBClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='roc_auc',
                           cv=3,
                           verbose=1)

# 训练网格搜索模型
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)

# 使用最佳参数训练最终模型
best_xgb_model = grid_search.best_estimator_
y_best_pred = best_xgb_model.predict(X_test)

# 输出最佳模型的评估结果
print(classification_report(y_test, y_best_pred))

# 导入绘图库
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.show()

# 绘制ROC曲线
def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="orange", label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curve", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# 绘制混淆矩阵和ROC曲线
plot_confusion_matrix(y_test, y_best_pred)
plot_roc_curve(y_test, best_xgb_model.predict_proba(X_test)[:, 1])




