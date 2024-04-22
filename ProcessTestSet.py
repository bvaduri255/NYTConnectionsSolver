import pandas as pd
import numpy as np
import random
import ast
import gensim
import gensim.downloader as api
import pickle
from transformers import BertModel, BertTokenizer
import torch
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from DictionaryQuery import get_definitions

nltk.download('punkt')
nltk.download('stopwords')

random.seed(42)

test_df = pd.read_csv('Data/test_dataset.csv')

puzzle_list = []

for index, row in test_df.iterrows():
    temp_list = []
    for i in range(1,5):
        temp_list.append(ast.literal_eval(row[i]))
    
    puzzle_list.append(temp_list)

shuffled_puzzle_list = []
shuffled_target_list = []

for puzzle in puzzle_list:
    temp_target = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
    temp_puzzle = []
    for group in puzzle:
        temp_puzzle += group
    
    zip_puzzle = list(zip(temp_puzzle, temp_target.copy()))
    random.shuffle(zip_puzzle)

    shuffled_puzzle, shuffled_target = zip(*zip_puzzle)

    shuffled_puzzle_list.append(shuffled_puzzle)
    shuffled_target_list.append(shuffled_target)

def onehotify(target_list):
    """Takes target list consisting of 0s, 1s, 2s, and 3s and converts the labels to a one-hot format"""
    res = []
    for target in target_list:
        temp_res = []

        for element in target:
            temp = [0 for i in range(4)]
            temp[element] = 1

            temp_res.append(temp)
        
        res.append(temp_res)
    
    return res

# with open("TestEmbeddings/TestDataNumerical.pkl", "wb") as file:
#     pickle.dump(shuffled_target_list, file)

# with open("TestEmbeddings/TestDataOneHot.pkl","wb") as file:
#     pickle.dump(onehotify(shuffled_target_list), file)

print("Saved Target Data")
# Other Embeddings Options
# glove-wiki-gigaword-300
# glove-twitter-200

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def getBERTEmbeddings(puzzle_list, file_path):
    embedded_puzzles = []
    
    for puzzle in puzzle_list:
        temp_puzzle = []

        for word in puzzle:
            word = word.lower()
            token = tokenizer.tokenize(word)

            word_id = tokenizer.convert_tokens_to_ids(token)
            word_id = torch.tensor(word_id).unsqueeze(0)
            outputs = model(word_id)
            word_embedding = outputs.last_hidden_state[0]

            temp_puzzle.append(word_embedding.detach().numpy())
        
        embedded_puzzles.append(temp_puzzle)
    
    with open(file_path, "wb") as file:
        pickle.dump(embedded_puzzles, file)


def getInputEmbeddings(puzzle_list, file_path, model_str):
    word_vectors = api.load(model_str)

    embedded_puzzles = []
    test_vec = word_vectors["apple"]

    for puzzle in puzzle_list:
        temp_puzzle = []

        for word in puzzle:
            word = word.lower()
            try:
                temp_vector = word_vectors[word]
            
            except:
                temp_vector = np.zeros_like(test_vec)

            temp_puzzle.append(temp_vector)
        
        embedded_puzzles.append(temp_puzzle)
    
    with open(file_path, "wb") as file:
        pickle.dump(embedded_puzzles, file)

def preprocess_text_for_vectorization(text):
    tokens = word_tokenize(text)
    
    tokens = [word.lower() for word in tokens]
    
    table = str.maketrans('', '', string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens
     
def getInputEmbeddingsDict(puzzle_list, file_path, model_str):
    word_vectors = api.load(model_str)

    embedded_puzzles = []
    test_vec = word_vectors["apple"]

    for puzzle in puzzle_list:
        temp_puzzle = []

        for word in puzzle:
            word = word.lower()
            dict_str = get_definitions(word)
            count = 0
            temp_sum = np.zeros_like(test_vec)
            for token in dict_str:
                try:
                    temp_vector = word_vectors[token]
                
                except:
                    temp_vector = np.zeros_like(test_vec)
                
                temp_sum += temp_vector
                count += 1
            
            avg_vector = temp_sum / count

            temp_puzzle.append(avg_vector)
        
        embedded_puzzles.append(temp_puzzle)
    
    with open(file_path, "wb") as file:
        pickle.dump(embedded_puzzles, file)

def getBERTEmbeddingsDict(puzzle_list, file_path):
    embedded_puzzles = []
    
    for puzzle in puzzle_list:
        temp_puzzle = []

        for word in puzzle:
            word = word.lower()
            word = get_definitions(word)
            token = tokenizer.tokenize(word)

            word_id = tokenizer.convert_tokens_to_ids(token)
            word_id = torch.tensor(word_id).unsqueeze(0)
            outputs = model(word_id)
            word_embedding = outputs.last_hidden_state[0]

            temp_puzzle.append(np.mean(word_embedding.detach().numpy(), axis = 0))
        
        embedded_puzzles.append(temp_puzzle)
    
    with open(file_path, "wb") as file:
        pickle.dump(embedded_puzzles, file)


getInputEmbeddingsDict(shuffled_puzzle_list, "TestEmbeddingsv2/InputWord2Vec.pkl", "word2vec-google-news-300")
print("Saved Word2Vec Embeddings")
getInputEmbeddingsDict(shuffled_puzzle_list, "TestEmbeddingsv2/InputGloveWiki.pkl", "glove-wiki-gigaword-300")
print("Saved Glove Wiki Embeddings")
getInputEmbeddingsDict(shuffled_puzzle_list, "TestEmbeddingsv2/InputGloveTwitter.pkl", "glove-twitter-200")
print("Saved Glove twitter embeddings")
getBERTEmbeddingsDict(shuffled_puzzle_list, "TestEmbeddingsv2/InputBERT.pkl")
print("Saved BERT Embeddings")

#getBERTEmbeddings(shuffled_puzzle_list, "TestEmbeddings/InputBERT.pkl")
#print("Saved BERT Embeddings")
#getInputEmbeddings(shuffled_puzzle_list, "TestEmbeddings/InputWord2Vec.pkl", "word2vec-google-news-300")
#print("Saved Word2Vec Embeddings")
#getInputEmbeddings(shuffled_puzzle_list, "TestEmbeddings/InputGloveWiki.pkl", "glove-wiki-gigaword-300")
#print("Saved Glove Wiki Embeddings")
#getInputEmbeddings(shuffled_puzzle_list, "TestEmbeddings/InputGloveTwitter.pkl", "glove-twitter-200")
#print("Saved Glove twitter embeddings")