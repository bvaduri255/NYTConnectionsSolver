import numpy as np
import pickle

with open("TestEmbeddings/InputGloveTwitter.pkl", "rb") as file:
    test_word_embeds = pickle.load(file)

if isinstance(test_word_embeds, list):
    test_word_embeds = np.array(test_word_embeds)

print("Embeddings shape:", test_word_embeds.shape)

with open("TestEmbeddings/TestDataOneHot.pkl", "rb") as file:
    test_hot_vectors = pickle.load(file)

if isinstance(test_hot_vectors, list):
    test_hot_vectors = np.array(test_hot_vectors)

print("Labels shape:", test_hot_vectors.shape)