from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200

def test_prediction():
    payload = {
        "pregnancies": 1,
        "glucose": 120,
        "blood_pressure": 70,
        "skin_thickness": 20,
        "insulin": 79,
        "bmi": 25.0,
        "diabetes_pedigree_function": 0.3,
        "age": 35
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

