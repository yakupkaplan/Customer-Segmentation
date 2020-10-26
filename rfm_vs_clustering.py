
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
data = pd.read_csv(r"C:\Users\yakup\PycharmProjects\dsmlbc\projects\customer_segmentation\rfm_results.csv")
df = data.copy()
df.head()
df.info()
df.shape

# Check if there are any missing values
df.isnull().values.any()

# MinMaxScaler for Recency, Frequency and Monetary columns
min_max_scaler = MinMaxScaler((0, 1))
cols_for_scaling = ['Recency', 'Frequency', 'Monetary']
scaled_rfm = min_max_scaler.fit_transform(df[cols_for_scaling])
scaled_rfm[0:5]

# Create the dataframe again
scaled_rfm = pd.DataFrame(scaled_rfm, columns=['RecencyScaled', 'FrequencyScaled', 'MonetaryScaled'], index=df.index)
scaled_rfm.head()


# K-Means-Clustering

kmeans = KMeans(n_clusters=10)
k_fit = kmeans.fit(scaled_rfm)

k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_


# Creating a dataframe

kmeans = KMeans(n_clusters=6).fit(scaled_rfm)
clusters = kmeans.labels_
pd.DataFrame({"Customer ID": scaled_rfm.index, "Clusters": clusters})
scaled_rfm["ClusterKMeans"] = clusters
scaled_rfm["ClusterKMeans"] = scaled_rfm["ClusterKMeans"] + 1
scaled_rfm.head()

# Take a detailed look at the results
scaled_rfm.groupby("ClusterKMeans").agg({"mean", "count"})


# HIERARCHICAL CLUSTERING

hc_complete = linkage(scaled_rfm, "complete")
hc_average = linkage(scaled_rfm, "average")


cluster_labels = cut_tree(hc_complete, n_clusters=10).reshape(-1, )
scaled_rfm['ClusterHierarchical'] = cluster_labels
scaled_rfm['ClusterHierarchical'] = scaled_rfm['ClusterHierarchical'] + 1
scaled_rfm.head()


df.head()
scaled_rfm.head()

# Dimension control
print(df.shape, '=', scaled_rfm.shape)

# Concatenate two dataframes
rfm_vs_clustering = pd.concat([df, scaled_rfm], axis=1, )
rfm_vs_clustering.index = df['Customer ID']
df.drop('Customer ID', axis=1, ).head()
rfm_vs_clustering.head()

# Create a csv file
rfm_vs_clustering.to_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\customer_segmentation\rfm_vs_clustering.csv', index=False)

# Now, it is time to compare the methods
rfm_vs_clustering[['Recency', 'Frequency', 'Monetary', 'Segment', 'ClusterKMeans', 'ClusterHierarchical']].head(20)


'''
Comments:
    - RFM is based on the metrics of Recency and Frequency (Monetary).
    - K-Means-Clustering takes into consideration the variables that we define and makes the segmentation accordingly.
    -> There are differences between RF(M) and K-Means-Clustering because of this dimension difference.
    -> By RFM, there are fixed, clear rules, for example, metrics, classes, etc.
    -> K-Means-Clustering is more flexible.
    
    -> In addition,there are differences between K-Means-Clustering and Hierarchical Clustering due to the algoritm, logic difference.
    
'''

