import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def zeroshot_classification(product, cost, job, salary, hobbies):
    output = query({
        "inputs": f"Classify the purchase of {product} costing {cost} by a {job} with a salary of {salary}, whose hobbies include {hobbies}, as either 'Appropriate' or 'Irresponsible' based on their financial situation.",
        "parameters": {"candidate_labels": ["Appropiate", "Irresponsible"]},
    })

    return output