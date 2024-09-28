from fastapi import BackgroundTasks, FastAPI
import pandas as pd
from datetime import datetime
import lightgbm as lgb

from .model import Model

app = FastAPI(title="Data preparation")

model = Model(model_filepath='data/model.joblib')

def train():
    """Train the model
    """
    model.train(Xy_filepath='data/gold/df.parquet', n_estimators=100)

def test():
    """Backtest the model
    """
    _, mape = model.backtest(
        Xy_filepath='data/gold/df.parquet',
        starting_ts=pd.Timestamp(datetime.now() - pd.Timedelta(30, 'd'), tz='Europe/Zurich'),
        use_every_nth_ts=100,
    )
    print(f'Backtested MAPE: {mape:.2f}%')

@app.get("/train")
async def get_train(background_tasks: BackgroundTasks):
    background_tasks.add_task(train)
    return {"message": "Model training started"} 

@app.get("/test")
async def get_test(background_tasks: BackgroundTasks):
    background_tasks.add_task(test)
    return {"message": "Model testing started"} 

@app.get("/predict")
async def get_predict():
    yhat = model.predict(
        in_df_filepath='data/gold/df.parquet', 
        out_yhat_filepath='data/yhat.parquet',
    )

    return {
        "timestamps": list(yhat.index), 
        "predicted_24h_later_load": list(yhat['predicted_24h_later_load']),
    } 
