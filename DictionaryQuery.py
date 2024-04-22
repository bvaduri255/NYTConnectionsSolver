import requests

def get_definitions(word):
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"

    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()

        definitions = []

        if isinstance(data, list):
            for entry in data:
                if 'meanings' in entry:
                    meanings = entry['meanings']

                    for meaning in meanings:
                        if 'definitions' in meaning:
                            for temp in meaning['definitions']:
                                definitions.append(temp['definition'])

        if definitions:
            return ''.join(definitions)
        
        else:
            return "No definitions found for this word."
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching data: {e}")

word = "model"

try:
    definitions = get_definitions(word)
    print(f"Definitions for '{word}':\n{definitions}")
except Exception as e:
    print(f"An error occurred: {e}")