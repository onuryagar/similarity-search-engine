from fastapi import FastAPI
from pydantic import BaseModel
from . import similarity_engine


app = FastAPI()

SIMILARITY_ENGINE = None

class Query(BaseModel):
    query: dict


@app.on_event("startup")
def on_startup():
    global SIMILARITY_ENGINE
    SIMILARITY_ENGINE = similarity_engine.SimilarityEngine()

@app.post("/predict")
def predict(q: Query):
    global SIMILARITY_ENGINE
    result = SIMILARITY_ENGINE.predict_similarity(q.query)
    return {"result": float(result)}