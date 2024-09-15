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
        "inputs": f"You are tasked with determining whether a transaction is considered an appropiate or irresponsible financial decision. The transaction in question involves purchasing {product} for {cost}, made by a {job} with a {salary} salary and hobbies such as {hobbies}. Keep in mind that not all non-income-generating expenses are inherently bad. Based on this context, classify the transaction. Give us a score for both 'good' and 'bad' decisions.",
        "parameters": {"candidate_labels": ["good", "bad"]},
    })       
    return output

if __name__ == "__main__":
    print(zeroshot_classification("a car", "10000", "teacher", "50000", "reading and gardening")) 
    # {'sequence': 'You are tasked with determining whether a transaction is considered an appropiate or irresponsible financial decision. The transaction in question involves purchasing a car for 10000, made by a teacher with a 50000 salary and hobbies such as reading and gardening. Keep in mind that not all non-income-generating expenses are inherently bad. Based on this context, classify the transaction. Give us a score for both "good" and "bad" decisions.', 'labels': ['good', 'bad'], 'scores': [0.9999998807907104, 1.191022724122366e-07]}