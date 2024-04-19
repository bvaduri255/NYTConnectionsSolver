import pickle
import numpy as np
from k_means_constrained import KMeansConstrained

with open("TestEmbeddings/InputWord2Vec.pkl", "rb") as file:
    word_embeds = pickle.load(file)

def k_means_clustering(embeddings, num_clusters):
    kmeans = KMeansConstrained(n_clusters = num_clusters, size_min = 4, size_max = 4, random_state=42)

    kmeans.fit_predict(embeddings)

    return kmeans.labels_

def scoreClusters(model_label, actual_label):
    return 0

num_clusters = 4

for embed_list in word_embeds:
    cluster_labels = k_means_clustering(embed_list, num_clusters)
    print(cluster_labels)
    break