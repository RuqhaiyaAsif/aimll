import pandas as pd
import numpy as np
df=pd.read_csv("/content/CC GENERAL.csv")
print(df.shape)
df.head()
df.describe().T
df=df.drop("CUST_ID", axis=1)
df.isnull().sum()
df.fillna(df.mean(), inplace=True)
df.isnull().sum()
df.shape


import matplotlib.pyplot as plt 
import seaborn as sns
plt.figure (figsize = (16, 10))
heatmap = sns.heatmap (df.corr (), vmin = -1, vmax = 1, annot = True, cmap = 'BrBG') 
plt.figure (figsize = (16, 10))
mask = np.triu (np.ones_like (df.corr (), dtype = np.bool))
heatmap = sns.heatmap (df.corr (), mask = mask, vmin = -1, vmax = 1, annot = True, cmap = 'BrBG')
cor=df.corr()
rf=cor[cor>0.5]
rf
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
scale = StandardScaler()

copiadata = scale.fit_transform(df)
data = pd.DataFrame(copiadata, columns= df.columns)
scaler = StandardScaler() 
scaled_df = scaler.fit_transform(df) 
normalized_df = normalize(scaled_df)   
normalized_df = pd.DataFrame(normalized_df) 
pca = PCA(n_components = 2) 
data = pca.fit_transform(normalized_df) 
data = pd.DataFrame(data) 
data.columns = ['P1', 'P2'] 
data.head()
import scipy.cluster.hierarchy as shc
plt.figure(figsize =(10, 10)) 
plt.title('Customer Dendrograms') 
Dendrogram = shc.dendrogram((shc.linkage(data, method ='ward')))
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering 
silhouette_scores = [] 

for n_cluster in range(2, 8):
    silhouette_scores.append( 
        silhouette_score(data, AgglomerativeClustering(n_clusters = n_cluster).fit_predict(data))) 
k = [2, 3, 4, 5, 6,7] 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show() 
agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
labels=agg.fit(data)
labels
plt.scatter(data['P1'], data['P2'],  
           c = AgglomerativeClustering(n_clusters = 3).fit_predict(data), cmap =plt.cm.winter) 
plt.show() 
df.head()

sns.scatterplot(data=normalized_df, x=normalized_df[2], y=normalized_df[3],  
           hue = AgglomerativeClustering(n_clusters = 3).fit_predict(normalized_df)) 
plt.show() 
sns.scatterplot(data=normalized_df, x=normalized_df[6], y=normalized_df[8],  
           hue = AgglomerativeClustering(n_clusters = 3).fit_predict(normalized_df)) 
plt.show()
agg = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
agg.fit(data)
plt.scatter(data['P1'], data['P2'],  
           c = AgglomerativeClustering(n_clusters = 4).fit_predict(data), cmap =plt.cm.winter) 
plt.show() 
sns.scatterplot(data=normalized_df, x=normalized_df[2], y=normalized_df[3],  
           hue = AgglomerativeClustering(n_clusters = 4).fit_predict(normalized_df)) 
plt.show()
sns.scatterplot(data=normalized_df, x=normalized_df[6], y=normalized_df[8],  
           hue = AgglomerativeClustering(n_clusters = 4).fit_predict(normalized_df)) 
plt.show()
sns.scatterplot(data=normalized_df, x=normalized_df[2], y=normalized_df[3],  
           hue = AgglomerativeClustering(n_clusters = 7).fit_predict(normalized_df)) 
plt.show()

from sklearn.cluster import KMeans
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
silhouette_scores = [] 

for n_cluster in range(2, 8):
    silhouette_scores.append( 
        silhouette_score(data, KMeans(n_clusters = n_cluster).fit_predict(data))) 
k = [2, 3, 4, 5, 6,7] 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show()
