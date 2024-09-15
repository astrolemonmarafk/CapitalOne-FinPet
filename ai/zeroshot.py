import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def zeroshot_classification(cost, job, salary, hobbies):
    output = query({
        "inputs": "You are tasked with determining whether a transaction is considered an appropiate or irresponsible financial decision. The transaction in question involves purchasing laptop for {cost}, made by a {job} with a {salary} salary and hobbies such as {hobbies}. Keep in mind that not all non-income-generating expenses are inherently bad. Based on this context, classify the transaction.",
        "parameters": {"candidate_labels": ["good", "bad"]},
    })

    return output
