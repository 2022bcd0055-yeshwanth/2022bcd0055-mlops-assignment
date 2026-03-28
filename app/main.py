from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("app/model.pkl")

@app.get("/")
def health():
    return {
        "Name": "Yeshwanth",
        "Roll No": "2022bcd0056"
    }

@app.post("/predict")
def predict(data: dict):
    values = list(data.values())
    prediction = model.predict([values])[0]

    return {
        "prediction": int(prediction),
        "Name": "Yeshwanth",
        "Roll No": "2022bcd0055"
    }