from flask import Flask, request, jsonify
from pymongo import MongoClient
# from dotenv import load_dotenv
from bson.objectid import ObjectId
# Load environment variables from .env file
# load_dotenv()

import os

# Initialize Flask app
app = Flask(__name__)

# MongoDB Initialization
# Use environment variables for credentials (e.g., MongoDB connection URI, Firebase config)
mongo_uri = 'mongodb+srv://maxzermeno03:123@clusterhackmty2024.wec2n.mongodb.net/?retryWrites=true&w=majority&appName=ClusterHackMTY2024'
client = MongoClient(mongo_uri)
db = client['HackMTY2024']
users_collection = db['users']


# Add new user with Firebase Auth and MongoDB
@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.json
    name = data['name']
    email = data['email']

    user_data = {

        'name': name,
        'email': email
    }

    # Insert user data into MongoDB
    users_collection.insert_one(user_data)
    return jsonify({"message": "User added successfully"}), 200

# Get user data
@app.route('/get_user/<id>', methods=['GET'])
def get_user(id):
    # Retrieve user data from MongoDB
    user = users_collection.find_one({'_id': id})
    if user:
        return jsonify(user), 200
    else:
        return jsonify({"error": "User not found"}), 404

# Update bank balance for user
@app.route('/update_balance', methods=['PATCH'])
def update_balance():

    data = request.json
    username = data['username']
    new_balance = data['bank_balance']

    # Update user's bank balance in MongoDB
    users_collection.update_one({'username': username}, {'$set': {'bank_balance': new_balance}})
    return jsonify({"message": "Bank balance updated"}), 200

# Update pet's experience for authenticated user
@app.route('/update_pet_experience', methods=['PATCH'])
def update_pet_experience():
    data = request.json
    petid = data['_id']
    new_exp = int(data['exp'])
    

    # Define the filter and update operations
    filter_query = {"_id": petid}
    update_query = {"$inc": {"exp": new_exp}}

    # Perform the update operation
    result = users_collection.update_one(filter_query, update_query)

    # Check if the update was successful
    if result.matched_count > 0:
        print("Update successful")
    else:
        print("No document matched the query")

    return jsonify({"message": "Pet experience updated"}), 200

# Delete user from MongoDB
@app.route('/delete_user/<userid>', methods=['DELETE'])
def delete_user(userid):

    # Delete the user from MongoDB
    users_collection.delete_one({'_id': userid})
    return jsonify({"message": "User deleted"}), 200

from flask_cors import CORS

def create_app(test_config=None ):

    app = Flask(__name__)

    app.config.from_object('config')  # Import things from config

    CORS(app)

    # CORS Headers 
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,true')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response


from ai.pet_model import prompt, run, pre_data
@app.route('/get_pet_response', methods=['POST'])

def def_pet_bdy():
    data = request.json
    description = data.get('description', '')
    name = data.get('name', '')

    prompt_text = prompt(description, name, pre_data)
    inputs = [
        {"role": "system", "content": f"{prompt_text}"},
        {"role": "user", "content": f"{description}"}
    ]
    
    output = run("@hf/mistral/mistral-7b-instruct-v0.2", inputs)
    return jsonify({"response": output}), 200

from ai.zeroshot import zeroshot_classification
@app.route('/get_transaction_classification', methods=['POST'])
def get_transaction_classification():
    data = request.json
    product = data.get('product', '')
    cost = data.get('cost', '')
    job = data.get('job', '')
    salary = data.get('salary', '')
    hobbies = data.get('hobbies', '')

    result = zeroshot_classification(product, cost, job, salary, hobbies)
    return jsonify({"classification": result}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=3000)


#por si se ocupa
# from pymongo.mongo_client import MongoClient
# from pymongo.server_api import ServerApi

# uri = "mongodb+srv://maxzermeno03:LLuEnvePdc166AIf@clusterhackmty2024.wec2n.mongodb.net/?retryWrites=true&w=majority&appName=ClusterHackMTY2024"

# # Create a new client and connect to the server
# client = MongoClient(uri, server_api=ServerApi('1'))

# # Send a ping to confirm a successful connection
# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)

# curl -X POST http://localhost:5000/add_user \
# -H "Content-Type: application/json" \
# -d '{"username": "john_doe", "email": "john.doe@example.com", "password": "securePassword123"}'

# curl -X PATCH http://localhost:5000/update_pet_experience \
# -H "Content-Type: application/json" \
# -d '{"_id": "your_pet_id", "exp": 10}'


# curl -X GET http://localhost:5000/get_user/your_user_id
