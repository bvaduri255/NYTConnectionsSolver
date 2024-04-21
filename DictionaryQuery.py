import requests

def get_definitions(word):
    url = f"https://api.dictionary.com/api/v3/references/learners/json/{word}"
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()

        if isinstance(data, list):
            definitions = []
            for entry in data:
                if 'shortdef' in entry:
                    short_defs = entry['shortdef']
                    definitions.extend(short_defs)

            return '\n'.join(definitions)
        
        else:
            raise ValueError("No definitions found for this word.")
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching data: {e}")


# Example Usage
# word = input("Enter a word: ")
# try:
#     definitions = get_definitions(word)
#     print(f"Definitions for '{word}':\n{definitions}")
# except Exception as e:
#     print(f"An error occurred: {e}")