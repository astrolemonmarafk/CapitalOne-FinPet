import os
from dotenv import load_dotenv
import requests
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

count_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
workers_key = f"Bearer {os.getenv('CLOUDFLARE_API_KEY')}"

# DATASET STUFF
file_path = r'db\datasetai.csv'
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

def pre_data(df):
    return df[['Personality', 'Catchphrase', 'Favorite Saying', 'Style 1', 'Style 2']]
preprocessed_data = pre_data(data)

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

#MODEL
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
    
def prompt(description, name, data):
    match = find_personality(description, data)
    fav_phrase = fav_saying(match, data)
    user_pet = match + fav_phrase
    prompt = f"Act as if you are the pet with the personality: {match}. Your name is {name}. Do not mention that you are an AI or that you are not real. Simply respond as if youve just been born. Keep your response within 30 tokens and complete your sentence."
    return prompt

def main(description, name, data):
    global user_pet
    prompt = prompt(description, name, data)
    inputs = [
        { "role": "system", "content": f"{prompt}" },
        { "role": "user", "content": f"{user_pet}" }
    ]
    output = run(pet_model, inputs)
    return output

if __name__ == "__main__":
    main()
