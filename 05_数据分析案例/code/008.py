import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 读取数据
file_path = "./dataset/008/Mall_Customers.csv"
df = pd.read_csv(file_path)

# 查看数据基本信息
print(df.info())
print(df.describe())

# 查看前 5 行数据
df.head()

# 检查缺失值
print("缺失值情况：\n", df.isnull().sum())

# 检查重复数据
print("重复值情况：", df.duplicated().sum())

# 统计每一列的唯一值数量
print("唯一值数量：\n", df.nunique())

# 检查极端值（利用 IQR 方法）
Q1 = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].quantile(0.25)
Q3 = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].quantile(0.75)
IQR = Q3 - Q1

# 计算异常值范围
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 查找异常值
outliers = ((df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] < lower_bound) |
            (df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']] > upper_bound)).sum()
print("异常值统计：\n", outliers)

# 设置 Seaborn 风格
sns.set(style="whitegrid")

# 年龄、年收入、消费评分的分布
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df['Age'], bins=20, kde=True, color='royalblue', ax=axes[0])
axes[0].set_title('Age Distribution')
axes[0].set_xlabel('Age')

sns.histplot(df['Annual Income (k$)'], bins=20, kde=True, color='tomato', ax=axes[1])
axes[1].set_title('Annual Income Distribution')
axes[1].set_xlabel('Annual Income (k$)')

sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, color='seagreen', ax=axes[2])
axes[2].set_title('Spending Score Distribution')
axes[2].set_xlabel('Spending Score')

plt.show()

# 计算相关性矩阵
corr_matrix = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()

# 可视化相关性
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 三个变量的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Age', size='Age',
                palette='viridis', sizes=(20, 200), data=df)
plt.title("Annual Income vs Spending Score (Age as Hue)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend(title="Age")
plt.show()

# 删除无关列
df_cluster = df.drop(columns=['CustomerID', 'Gender'])

# 归一化数据
scaler = MinMaxScaler()
df_cluster_scaled = pd.DataFrame(scaler.fit_transform(df_cluster), columns=df_cluster.columns)

# 查看归一化后的数据
df_cluster_scaled.head()

# 将数据集划分为训练集（80%）和测试集（20%）
train_data, test_data = train_test_split(df_cluster_scaled, test_size=0.2, random_state=42)

# 输出数据形状
print("训练集大小:", train_data.shape)
print("测试集大小:", test_data.shape)

# 选择 K 值（使用肘部法则）
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_cluster_scaled)
    wcss.append(kmeans.inertia_)

# 绘制肘部法则图
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='royalblue')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal K")
plt.show()

# 训练 K-Means
optimal_k = 5  # 例如通过肘部法则确定为5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_cluster_scaled['KMeans_Cluster'] = kmeans.fit_predict(df_cluster_scaled)

# 训练 GMM
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
df_cluster_scaled['GMM_Cluster'] = gmm.fit_predict(df_cluster_scaled)

# 训练 K-Means 模型
optimal_k = 5  # 通过肘部法则确定的最佳 K 值
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_cluster_scaled['KMeans_Cluster'] = kmeans.fit_predict(df_cluster_scaled)

# 查看聚类结果
print(df_cluster_scaled[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'KMeans_Cluster']].head())

# 训练 GMM 模型
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
df_cluster_scaled['GMM_Cluster'] = gmm.fit_predict(df_cluster_scaled)

# 查看聚类结果
print(df_cluster_scaled[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'GMM_Cluster']].head())

# 计算 K-Means 和 GMM 的轮廓系数
kmeans_score = silhouette_score(df_cluster_scaled[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']],
                                df_cluster_scaled['KMeans_Cluster'])
gmm_score = silhouette_score(df_cluster_scaled[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']],
                             df_cluster_scaled['GMM_Cluster'])

print(f"K-Means Silhouette Score: {kmeans_score:.4f}")
print(f"GMM Silhouette Score: {gmm_score:.4f}")

# 计算 K-Means 的 WCSS（聚类内平方误差）
wcss = kmeans.inertia_
print(f"K-Means WCSS: {wcss:.4f}")

# 定义参数范围
param_grid = {
    'n_clusters': [3, 4, 5, 6, 7, 8, 9, 10]  # K 值范围
}

# 使用 GridSearchCV 进行超参数调优
grid_search = GridSearchCV(estimator=KMeans(random_state=42), param_grid=param_grid, cv=3,
                           scoring='neg_mean_squared_error')
grid_search.fit(df_cluster_scaled[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# 输出最佳参数
print(f"Best K for K-Means: {grid_search.best_params_['n_clusters']}")

# 定义参数范围
param_grid_gmm = {
    'n_components': [3, 4, 5, 6],  # 簇数
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],  # 协方差类型
    'max_iter': [100, 200]  # 最大迭代次数
}

# 使用 GridSearchCV 进行超参数调优
grid_search_gmm = GridSearchCV(estimator=GaussianMixture(), param_grid=param_grid_gmm, cv=3)
grid_search_gmm.fit(df_cluster_scaled[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# 输出最佳参数
print(f"Best GMM Parameters: {grid_search_gmm.best_params_}")

# K-Means 聚类结果的可视化
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='KMeans_Cluster', data=df_cluster_scaled,
                palette='Set1', s=100, marker='o')
plt.title('K-Means Clustering (5 clusters)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

# GMM 聚类结果的可视化
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='GMM_Cluster', data=df_cluster_scaled,
                palette='Set2', s=100, marker='o')
plt.title('GMM Clustering (5 components)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

# 对比 K-Means 和 GMM 的聚类效果
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# K-Means 可视化
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='KMeans_Cluster', data=df_cluster_scaled,
                palette='Set1', s=100, marker='o', ax=axes[0])
axes[0].set_title('K-Means Clustering')
axes[0].set_xlabel('Annual Income (k$)')
axes[0].set_ylabel('Spending Score (1-100)')

# GMM 可视化
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='GMM_Cluster', data=df_cluster_scaled,
                palette='Set2', s=100, marker='o', ax=axes[1])
axes[1].set_title('GMM Clustering')
axes[1].set_xlabel('Annual Income (k$)')
axes[1].set_ylabel('Spending Score (1-100)')

plt.show()