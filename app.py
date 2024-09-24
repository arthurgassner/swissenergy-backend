from fastapi import FastAPI
app = FastAPI(title="Energy Forecaster Model")

@app.get("/predict")
async def get_prediction(features: list[int]) -> int:
    return -1
