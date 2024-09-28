from fastapi import BackgroundTasks, FastAPI
import pandas as pd
from datetime import datetime
import lightgbm as lgb
import joblib

from .model_trainer import ModelTrainer

app = FastAPI(title="Data preparation")

def train():
    # Train model
    reg = lgb.LGBMRegressor(n_estimators=100, force_row_wise=True, verbose=0)
    _, mape = ModelTrainer.backtest(
        Xy_filepath='data/gold/df.parquet',
        model=reg,
        starting_ts=pd.Timestamp(datetime.now() - pd.Timedelta(30, 'd'), tz='Europe/Zurich'),
        use_every_nth_ts=100,
    )
    joblib.dump(reg, 'model.joblib')
    print(f'Backtested MAPE: {mape:.2f}%')

@app.get("/train")
async def train(background_tasks: BackgroundTasks):
    background_tasks.add_task(train)
    return {"message": "Model training started"} 