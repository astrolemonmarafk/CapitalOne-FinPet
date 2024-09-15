import os
import requests
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# DATASET STUFF
file_path = os.path.join(os.path.dirname(__file__), 'datasetai.csv')
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print("The file 'datasetai.csv' was not found.")
    exit()
except pd.errors.EmptyDataError:
    print("The file 'datasetai.csv' is empty.")
    exit()
except pd.errors.ParserError:
    print("Error parsing the file 'datasetai.csv'.")
    exit()

count_id = "f4890389cdc1dd5005eafe76b069ec85"
workers_key = 'Bearer 8ZMfDXqje9krSJJHLoGUctzMz_KMiWKLQt5gw--u'

def preprocess_data(df):
    return df[['Personality', 'Catchphrase', 'Favorite Saying', 'Style 1', 'Style 2']]
preprocessed_data = preprocess_data(data)

def find_personality(user_description, data):
    vectorizer = TfidfVectorizer()
    combined_texts = data['Personality'].tolist() + [user_description]
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    closest_index = cosine_sim.argmax()
    return data.iloc[closest_index]['Personality']

def fav_saying(personality, data):
    phrase = data[data['Personality'] == personality]['Favorite Saying'].tolist()
    return random.choice(phrase)

# MODEL
pet_model = "@hf/mistral/mistral-7b-instruct-v0.2"
API_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{count_id}/ai/run/"

def run(model, inputs):
    headers = {"Authorization": workers_key}
    input_data = {
        "messages": inputs,
        "temperature": .9,  # creativity
        "max_tokens": 30
    }
    try:
        response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input_data)
        response.raise_for_status()
        print(f"Status code: {response.status_code}")
        print(f"Response text: {response.text}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def create_prompt(user_description, name, data):
    personality = find_personality(user_description, data)
    fav_phrase = fav_saying(personality, data)
    user_pet_description = user_description + " " + fav_phrase

    prompt = f"Act as if you are the pet with the personality: {personality}. Your name is {name}. Do not mention that you are an AI or that you are not real. Simply respond as if you've just been born. Keep your response within 30 tokens and complete your sentence."

    inputs = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_pet_description}
    ]
    return inputs

def main():
    print("Legend has it that many years ago, humans were born with their own companion creature to accompany them on their journey through life; as humans care for it, the creature grows. What is the personality of your dream companion? (Describe it: personality, etc.)")
    user_description = str(input())
    name = str(input("Name: "))

    inputs = create_prompt(user_description, name, preprocessed_data)
    output = run(pet_model, inputs)
    print(output)

if __name__ == "__main__":
    main()