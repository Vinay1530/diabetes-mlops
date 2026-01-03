import joblib
import numpy as np
import os

MODEL_PATH = "models/diabetes_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Train the model first.")
    return joblib.load(MODEL_PATH)

model = load_model()

def predict_diabetes(data):
    features = np.array([[
        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.diabetes_pedigree_function,
        data.age
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return prediction, probability

