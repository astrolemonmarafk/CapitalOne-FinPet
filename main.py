from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ai.zeroshot import zeroshot_classification
from ai.personality import create_prompt

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
def generate(description: str, name: str, data: dict):
    return create_prompt(description, name, data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")