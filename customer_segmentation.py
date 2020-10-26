# CUSTOMER SEGMENTATION

'''
In this project, we will cluster the customers according to their Recency, Frequency and Monetary metrics.

Methods to use:
    - K-Means-Clustering
    - Hierarchical Clustering
'''

# Import dependencies
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# to display all columns and rows:
pd.set_option('display.max_columns', None);
pd.set_option('display.max_rows', None);
# digits after comma --> 2
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# Load the dataset
df_2010_2011 = pd.read_excel(r"C:\Users\yakup\PycharmProjects\dsmlbc\datasets\online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = df_2010_2011.copy()
df.head()
df.shape

# Check if there are any missing values
df.isnull().values.any()
# Missing values for each variable/column
df.isnull().sum().sort_values(ascending=False)

# We need total spending for each customer, in order to make RFM Analysis.
# So, let's calculate the total price for each row by simply multiplying quantity and price
df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()

# Remove the cancelled transactions from the dataset by using ~ (Tilda).
df = df[~df["Invoice"].str.contains("C", na=False)]

df.shape

# For simplicity just drop the  Null values for this dataset. Filling missing values will be handled in another notebook.
df.dropna(inplace=True)
df.isnull().sum()

# New shape of the dataset
df.shape


# RECENCY

# We will accept the last transaction date in our dataset as today's date to be able to calculate Recency logically.
import datetime as dt
today_date = dt.datetime(2011, 12, 9)

# Time passed after the last transaction. Save this as temporary df
temp_df = (today_date - df.groupby("Customer ID").agg({"InvoiceDate":"max"}))
temp_df.head()

# Rename "InvoiceDate" as "Recency"
temp_df.rename(columns={"InvoiceDate":"Recency"}, inplace = True)
temp_df.head()

# Remove unneccessary parts, but days! For his purpose, we can use apply / lambda structure.
recency_df = temp_df["Recency"].apply(lambda x: x.days)
recency_df.head()


# FREQUENCY

# We can calculate Fequency by simply finding number of unique values for each customer
freq_df = df.groupby("Customer ID").agg({"InvoiceDate":"nunique"})
freq_df.head()

# And, finally rename the column as 'Frequency'
freq_df.rename(columns={"InvoiceDate": "Frequency"}, inplace = True)
freq_df.head()


# MONETARY

# We need one more variable for RFM analysis--> Monetary: How much money did each customer spent?.
# Let's bring 'TotalPrice' for each customer.
monetary_df = df.groupby("Customer ID").agg({"TotalPrice": "sum"})
monetary_df.head()

# And rename the column
monetary_df.rename(columns={"TotalPrice":"Monetary"}, inplace=True)
monetary_df.head()


# RFM METRICS

# See the shapes of recency_df, freq_df and monetary_df
print(recency_df.shape, freq_df.shape, monetary_df.shape)

# Concatenate these seperate dataframes and make one --> rfm. Show the first 5 rows.
rfm = pd.concat([recency_df, freq_df, monetary_df],  axis=1)
rfm.head()


# OUTLIER ANALYSIS

# if interested, outliers can be determined for RFM scores
for feature in ["Recency", "Frequency", "Monetary"]:
    Q1 = rfm[feature].quantile(0.05)
    Q3 = rfm[feature].quantile(0.95)
    IQR = Q3-Q1
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR
    if rfm[(rfm[feature] > upper) | (rfm[feature] < lower)].any(axis=None):
        print(feature,"yes")
        print(rfm[(rfm[feature] > upper) | (rfm[feature] < lower)].shape[0])
    else:
        print(feature, "no")


# STANDARDIZATION

# MinMaxScaler
min_max_scaler = MinMaxScaler((0, 1))
cols = rfm.columns
index = rfm.index
scaled_rfm = min_max_scaler.fit_transform(rfm)
scaled_rfm = pd.DataFrame(scaled_rfm, columns=cols, index=index)
scaled_rfm.head()


# MODELING


# K-MEANS_CLUSTERING

kmeans = KMeans(n_clusters=10)
k_fit = kmeans.fit(scaled_rfm)

k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_


# Visualization of Clusters

k_means = KMeans(n_clusters=10).fit(scaled_rfm)
clusters = k_means.labels_
type(df)
rfm = pd.DataFrame(scaled_rfm)

plt.scatter(rfm.iloc[:, 0], rfm.iloc[:, 1], c=clusters, s=50, cmap="viridis")
plt.show()

# Show centorids
centroids = k_means.cluster_centers_
plt.scatter(rfm.iloc[:, 0], rfm.iloc[:, 1], c=clusters, s=50, cmap="viridis")
plt.scatter(centroids[:, 0], centroids[:, 1], c="black", s=200, alpha=0.5)
plt.show()


# 3-D Visualization

from mpl_toolkits.mplot3d import Axes3D

kmeans = KMeans(n_clusters=10)
k_fit = kmeans.fit(scaled_rfm)
clusters = k_fit.labels_
centroids = kmeans.cluster_centers_

plt.rcParams['figure.figsize'] = (16, 9)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(scaled_rfm.iloc[:, 0], scaled_rfm.iloc[:, 1], scaled_rfm.iloc[:, 2])
plt.show()

# Show centorids
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(scaled_rfm.iloc[:, 0], scaled_rfm.iloc[:, 1], scaled_rfm.iloc[:, 2], c=clusters)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', c='#050505', s=1000);
plt.show()


# Determining the Optimum Number of Clusters
kmeans = KMeans(n_clusters=10)
k_fit = kmeans.fit(scaled_rfm)
ssd = []

K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(scaled_rfm)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Distance Residual Sums Versus Different k Values")
plt.title("Elbow method for Optimum number of clusters")
plt.show()

# An automized method using yellowbrick library
kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k=(2, 20))
visu.fit(scaled_rfm)
visu.show()


# Creating a dataframe

kmeans = KMeans(n_clusters=6).fit(scaled_rfm)
clusters = kmeans.labels_
pd.DataFrame({"Customer ID": rfm.index, "Clusters": clusters})
rfm["ClusterKMeans"] = clusters
rfm["ClusterKMeans"] = rfm["ClusterKMeans"] + 1
rfm.groupby("ClusterKMeans").agg({"ClusterKMeans": "count"})

rfm.head()
rfm.groupby("ClusterKMeans").agg({"mean", "count"})


# Comparison of groups visually

km_clusters_amount = pd.DataFrame(rfm.groupby(["ClusterKMeans"]).Monetary.mean())
km_clusters_frequency = pd.DataFrame(rfm.groupby(["ClusterKMeans"]).Frequency.mean())
km_clusters_recency = pd.DataFrame(rfm.groupby(["ClusterKMeans"]).Recency.mean())
df_comparison = pd.concat([pd.Series([1, 2, 3, 4, 5, 6]), km_clusters_amount, km_clusters_frequency, km_clusters_recency], axis=1)
df_comparison.columns = ["ClusterKMeans", "Monetary_mean", "Frequency_mean", "Recency_mean"]
df_comparison.info()

# See the cluster means for Recency for each group
sns.barplot(x=df_comparison.ClusterKMeans, y=df_comparison.Recency_mean)
plt.show()
# See the cluster means for Frequency for each group
sns.barplot(x=df_comparison.ClusterKMeans, y=df_comparison.Frequency_mean)
plt.show()
# See the cluster means for Monetary for each group
sns.barplot(x=df_comparison.ClusterKMeans, y=df_comparison.Monetary_mean)
plt.show()


# HIERARCHICAL CLUSTERING

hc_complete = linkage(rfm, "complete")
hc_average = linkage(rfm, "average")

# Plotting dendogram
plt.figure(figsize=(15, 10))
plt.title("Hierarchical Cluster Dendrogram")
plt.xlabel("Observation Unit")
plt.ylabel("Distance")
dendrogram(hc_complete, truncate_mode="lastp", p=10, show_contracted=True, leaf_font_size=10);
plt.show()

cluster_labels = cut_tree(hc_complete, n_clusters=6).reshape(-1, )
rfm['ClusterHC'] = cluster_labels
rfm['ClusterHC'] = rfm['ClusterHC'] + 1
rfm.groupby("ClusterHC").agg(np.mean)

sns.boxplot(x='ClusterHC', y='Monetary', data=rfm);
plt.show()

sns.boxplot(x='ClusterHC', y='Frequency', data=rfm);
plt.show()

sns.boxplot(x='ClusterHC', y='Recency', data=rfm);
plt.show()

