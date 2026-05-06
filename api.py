from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("churn_model.pkl")

@app.get("/")
def home():
    return {"message": "API is working"}

@app.post("/predict")
def predict(data: dict):
    try:
        features = [[
            float(data["tenure"]),
            float(data["MonthlyCharges"]),
            float(data["TotalCharges"])
        ]]

        pred = model.predict(features)[0]

        return {"churn": int(pred)}

    except Exception as e:
        return {"error": str(e)}
