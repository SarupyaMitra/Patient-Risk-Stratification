import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from PCA import perform_PCA,normalize_data
from EDA import file_name
from preprocess import preprocess_data
import matplotlib.pyplot as plt

def perform_best_Kmeans(no_clusters):
    df_clean = preprocess_data(file_name)
    X,y = normalize_data(df_clean)
    new_X,evr = perform_PCA(X)
    kmeans = KMeans(n_clusters=no_clusters,random_state=3,n_init=10)
    kmeans.fit(new_X)
    y_kmeans = kmeans.predict(new_X)   # tells me the cluster index in which a particular point belongs to
    centers = kmeans.cluster_centers_

    # Checking which cluster is associated with which risk group
    check = pd.DataFrame({
        'Cluster': y_kmeans,
        'Actual_Diabetes': y.to_numpy().flatten()
    })

    print(check.groupby('Cluster')['Actual_Diabetes'].mean().round(2))

    # Cluster 0 has mean Diabetes = 0.54 ; cluster 1 has 0.16 and cluster 2 has 0.50 

    label_map = {0:'High Risk',1:'Low Risk',2:'Moderate Risk'}
    return y_kmeans,centers,label_map



if __name__ == '__main__':
    df_clean = preprocess_data(file_name)
    X,y = normalize_data(df_clean)
    new_X,evr = perform_PCA(X)

    inertia = []
    sil_scores = []
    K = range(2,9)
    for k in K:
        km = KMeans(n_clusters=k,random_state=3,n_init=10)
        km.fit(new_X)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(new_X,km.labels_))

    fig,axes = plt.subplots(1,2,figsize=(12,4))
    fig.canvas.manager.set_window_title('KMeans Results')
    axes[0].plot(K, inertia, 'bo-')
    axes[0].set_title('Elbow Method')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia')

    axes[1].plot(K, sil_scores, 'ro-')
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Score')

    plt.tight_layout()
    plt.show()

    ## After analysis we found out that its best to cluster the data into 3 clusters.

    y,centres,_ = perform_best_Kmeans(no_clusters=3)
    plt.scatter(centres[:, 0], centres[:, 1], c='black', s=200, alpha=0.5, marker='^')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    plt.title("K-Means Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

