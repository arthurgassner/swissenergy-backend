from fastapi import BackgroundTasks, FastAPI
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime
from pathlib import Path 

from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from .feature_extractor import FeatureExtractor
from .model import Model

app = FastAPI(title="Data preparation")

def update_forecast(entsoe_api_key: str):
    # Update the bronze-layer data
    data_loader = DataLoader(entsoe_api_key=entsoe_api_key)
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

    # Train 
    model = Model(model_filepath='data/model.joblib')
    model.train(Xy_filepath='data/gold/df.parquet', n_estimators=100)

    # Backtest model
    _, mape = model.backtest(
        Xy_filepath='data/gold/df.parquet',
        starting_ts=pd.Timestamp(datetime.now() - pd.Timedelta(30, 'd'), tz='Europe/Zurich'),
        use_every_nth_ts=100,
    )
    print(f'Backtested MAPE: {mape:.2f}%')

    # Predict    
    model.train(Xy_filepath='data/gold/df.parquet', n_estimators=100)
    model.predict(
        in_df_filepath='data/gold/df.parquet', 
        out_yhat_filepath='data/yhat.parquet',
    )


@app.get("/update-forecast")
async def get_update_forecast(background_tasks: BackgroundTasks):
    load_dotenv()
    background_tasks.add_task(update_forecast, entsoe_api_key=os.getenv('ENTSOE_API_KEY'))
    return {"message": "Forecasting task started..."} 

@app.get("/latest-forecast")
async def get_latest_forecast():
    # Load latest forecast
    yhat = pd.read_parquet('data/yhat.parquet')
    return {
        "timestamps": yhat.index.tolist(),
        "predicted_24h_later_load": yhat["predicted_24h_later_load"].tolist(),
    }

@app.get("/entsoe-loads")
async def get_entsoe_loads():
    # Load past loads
    silver_df = pd.read_parquet('data/silver/df.parquet')

    # TODO make it a POST request instead of hardcoding a delay
    silver_df = silver_df[silver_df.index > pd.Timestamp(datetime.now(), tz='Europe/Zurich') - pd.Timedelta(7, 'd')]

    return {
        "timestamps": silver_df.index.tolist(),
        "24h_later_load": silver_df["24h_later_load"].fillna('NaN').tolist(),
        "24h_later_forecast": silver_df["24h_later_forecast"].fillna('NaN').tolist(),
    }