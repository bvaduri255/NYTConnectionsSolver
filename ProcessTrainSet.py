import json
import random
import pickle
import gensim.downloader as api
import numpy as np

random.seed(42)

train_file = open("Data/syn_data.json")
train_data = json.load(train_file)

def str_to_list(row):
    row1 = row.split(":")
    row2 = row1[1].split(",")

    for i in range(len(row2)):
        temp = row2[i]
        temp = temp.strip()
        row2[i] = temp

    return row2

#print(str_to_list("Types of Conflict: External, Internal, Interpersonal, Ideological"))
    
puzzle_list = []

for i in range(0, len(train_data), 4):
    puzzle = [str_to_list(train_data[i + j]) for j in range(4)]

    puzzle_list.append(puzzle)
    print(i)

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

with open("TrainEmbeddings/TrainDataNumerical.pkl", "wb") as file:
    pickle.dump(shuffled_target_list, file)

with open("TrainEmbeddings/TrainDataOneHot.pkl","wb") as file:
    pickle.dump(onehotify(shuffled_target_list), file)

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
    

getInputEmbeddings(shuffled_puzzle_list, "TrainEmbeddings/InputWord2Vec.pkl", "word2vec-google-news-300")
print("Saved Word2Vec Embeddings")
getInputEmbeddings(shuffled_puzzle_list, "TrainEmbeddings/InputGloveWiki.pkl", "glove-wiki-gigaword-300")
print("Saved Glove Wiki Embeddings")
getInputEmbeddings(shuffled_puzzle_list, "TrainEmbeddings/InputGloveTwitter.pkl", "glove-twitter-200")
print("Saved Glove twitter embeddings")