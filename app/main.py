from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("app/model.pkl")

@app.get("/")
def health():
    return {
        "Name": "Yeshwanth",
        "Roll No": "2022bcd0055"
    }

@app.post("/predict")
def predict(data: dict):
    try:
        import pandas as pd

        df = pd.DataFrame([data])

        # Apply same preprocessing
        df = pd.get_dummies(df)

        # Align with training features
        model_features = model.feature_names_in_
        df = df.reindex(columns=model_features, fill_value=0)

        prediction = model.predict(df)[0]

        return {
            "prediction": int(prediction),
            "Name": "Yeshwanth",
            "Roll No": "2022bcd0055"
        }

    except Exception as e:
        return {"error": str(e)}