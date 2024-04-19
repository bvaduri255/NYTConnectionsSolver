import pandas as pd
import numpy as np
import random
import ast
import gensim
import gensim.downloader as api
import pickle


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

with open("TestEmbeddings/TestDataNumerical.pkl", "wb") as file:
    #pickle.dump(shuffled_target_list, file)
    pass

with open("TestEmbeddings/TestDataOneHot.pkl","wb") as file:
    #pickle.dump(onehotify(shuffled_target_list), file)
    pass

print("Saved Target Data")
# Other Embeddings Options
# glove-wiki-gigaword-300
# glove-twitter-200

def getInputEmbeddings(puzzle_list, file_path, model_str = "word2vec-google-news-300"):
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
    

#getInputEmbeddings(shuffled_puzzle_list, "TestEmbeddings/InputWord2Vec.pkl", "word2vec-google-news-300")
#print("Saved Word2Vec Embeddings")
getInputEmbeddings(shuffled_puzzle_list, "TestEmbeddings/InputGloveWiki.pkl", "glove-wiki-gigaword-300")
print("Saved Glove Wiki Embeddings")
getInputEmbeddings(shuffled_puzzle_list, "TestEmbeddings/InputGloveTwitter.pkl", "glove-twitter-200")
print("Saved Glove twitter embeddings")