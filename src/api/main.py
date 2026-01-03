from fastapi import FastAPI
from src.api.schemas import DiabetesRequest, DiabetesResponse
from src.api.utils import predict_diabetes

app = FastAPI(
    title="Diabetes Detection API",
    description="Production-grade ML inference API",
    version="1.0"
)

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=DiabetesResponse)
def predict(request: DiabetesRequest):
    prediction, probability = predict_diabetes(request)
    return {
        "prediction": int(prediction),
        "probability": round(float(probability), 4)
    }

