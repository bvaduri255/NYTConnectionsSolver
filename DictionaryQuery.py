import nltk

nltk.download('wordnet')

from nltk.corpus import wordnet

def get_definitions(word):
    synsets = wordnet.synsets(word)

    definitions = ""

    for synset in synsets:
        definitions += f"{synset.name().split('.')[0]}: {synset.definition()}\n"

    return definitions.strip()

word = "apple"

try:
    definitions = get_definitions(word)
    print(f"Definitions for '{word}':\n{definitions}")
except Exception as e:
    print(f"An error occurred: {e}")