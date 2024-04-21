import pickle
import numpy as np
from k_means_constrained import KMeansConstrained
from sklearn.metrics import adjusted_rand_score, mutual_info_score, fowlkes_mallows_score

with open("TestEmbeddings/InputWord2Vec.pkl", "rb") as file:
    word_embeds = pickle.load(file)

with open("TestEmbeddings/TestDataNumerical.pkl", "rb") as file:
    true_labels = pickle.load(file)

def k_means_clustering(embeddings, num_clusters):
    kmeans = KMeansConstrained(n_clusters = num_clusters, size_min = 4, size_max = 4, random_state=42)

    kmeans.fit_predict(embeddings)

    return kmeans.labels_

def clustering_score(true_labels, predicted_labels):
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    mi_score = mutual_info_score(true_labels, predicted_labels)
    fmi_score = fowlkes_mallows_score(true_labels, predicted_labels)
    return (ari_score, mi_score, fmi_score)

num_clusters = 4
total_ARI_score = 0
total_MI_score = 0
total_FMI_score = 0

for i in range(len(word_embeds)):
    predicted_labels = k_means_clustering(word_embeds[i], num_clusters)
    cluster_scores = clustering_score(true_labels[i], predicted_labels)
    #print(predicted_labels)
    #print(true_labels[i])
    #print(cluster_score)
    
    total_ARI_score += cluster_scores[0]
    total_MI_score += cluster_scores[1]
    total_FMI_score += cluster_scores[2]

print("Average ARI Score:")
print(total_ARI_score / len(word_embeds))
print("Average Mutual Information Score:")
print(total_MI_score / len(word_embeds))
print("Average Fowlkes-Mallows Index:")
print(total_FMI_score / len(word_embeds))
