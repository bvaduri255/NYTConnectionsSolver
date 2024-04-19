import pickle
import numpy as np
from sklearn.cluster import KMeans

with open("TestEmbeddings/InputWord2Vec.pkl", "rb") as file:
    word_embeds = pickle.load(file)

def k_means_clustering(embeddings, num_clusters):
    kmeans = KMeans(n_clusters = num_clusters, random_state=42)

    kmeans.fit(embeddings)

    return kmeans.labels_

num_clusters = 4

for embed_list in word_embeds:
    cluster_labels = k_means_clustering(embed_list, num_clusters)
    print("Cluster Labels:")
    print(cluster_labels)
    break