from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/zeroshot")
def predict(cost, job, salary, hobbies):
    return zeroshot_classification(cost, job, salary, hobbies)


@app.post("/petpersonality")
def generate(request: RequestBody):
    try:
        description = request.description
        name = request.name
        data = request.data.dict()

        response_text = create_prompt(description, name, data)
        return {"response": response_text}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"KeyError: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")