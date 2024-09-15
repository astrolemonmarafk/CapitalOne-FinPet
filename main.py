from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ai.zeroshot import zeroshot_classification
from ai.personality import create_prompt

from pydantic import BaseModel
from typing import List, Dict

class PetPersonalityData(BaseModel):
    Personality: List[str]
    Catchphrase: List[str]
    Favorite_Saying: List[str]
    Style_1: List[str]
    Style_2: List[str]

class RequestBody(BaseModel):
    description: str
    name: str
    data: PetPersonalityData

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/zeroshot")
def predict(cost, job, salary, hobbies):
    return zeroshot_classification(cost, job, salary, hobbies)

@app.post("/petpersonality")
def generate(request: RequestBody):
    data_dict = request.data.dict()  # Convert Pydantic model to dictionary
    return create_prompt(request.description, request.name, data_dict)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")