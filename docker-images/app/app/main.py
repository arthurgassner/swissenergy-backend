from fastapi import BackgroundTasks, FastAPI
from datetime import datetime
import joblib
from dotenv import load_dotenv
import os
import pandas as pd
import lightgbm as lgb

from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer

load_dotenv(dotenv_path='nb-prod/.env')

app = FastAPI(title="Data preparation")

def update_model(enstoe_api_key: str):

    # Update the bronze-layer data
    data_loader = DataLoader(enstoe_api_key=enstoe_api_key)
    data_loader.update_df(out_df_filepath="data/bronze/df.parquet")

    # Clean the bronze-layer data
    DataCleaner.clean(
        in_df_filepath='data/bronze/df.parquet',
        out_df_filepath='data/silver/df.parquet',
    )

    # Extract features
    FeatureExtractor.extract_features(
        in_df_filepath='data/silver/df.parquet',
        out_df_filepath='data/gold/df.parquet',
    )

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

@app.get("/update-model")
async def get_update_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(update_model, enstoe_api_key=os.getenv('ENTSOE_API_KEY'))
    return {"message": "Data preparation started"} 