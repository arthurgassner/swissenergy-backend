from fastapi import FastAPI
import lightgbm as lgb
import joblib
import numpy as np

app = FastAPI(title="Energy Forecaster Model")

# Load model
reg = joblib.load('model.joblib')

@app.post("/predict")
async def get_prediction(features: list[int]) -> int:
    return reg.predict(np.array(features).reshape(1,-1))
