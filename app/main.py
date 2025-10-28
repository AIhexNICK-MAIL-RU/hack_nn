from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict


class PredictRequest(BaseModel):
    name: str = Field(..., description="Наименование от производителя")
    manufacturer: str = Field(..., description="Производитель")
    article: str = Field(..., description="Артикул от производителя")


class PredictResponse(BaseModel):
    analogs: List[str]


app = FastAPI(title="Analog Matcher", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    # stub implementation for now; will wire extractor + matcher next
    if not req.name or not req.manufacturer or not req.article:
        raise HTTPException(status_code=400, detail="Missing required fields")
    return PredictResponse(analogs=[])


