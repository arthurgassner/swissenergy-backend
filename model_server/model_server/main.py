from fastapi import BackgroundTasks, FastAPI
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime

from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from .feature_extractor import FeatureExtractor
from .model import Model

app = FastAPI(title="Data preparation")

def get_update_forecast(entsoe_api_key: str):
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
    background_tasks.add_task(get_update_forecast, entsoe_api_key=os.getenv('ENTSOE_API_KEY'))
    return {"message": "Forecasting task started..."} 