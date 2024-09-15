from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import zeroshot_classification from ai.zeroshot
import create_prompt from ai.personality

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
def predict(data: dict):
    return zeroshot_classification(data["cost"], data["job"], data["salary"], data["hobbies"])

@app.post("/petpersonality")
def generate(data: dict):
    return create_prompt(data["user_description"], data["name"], data["data"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)