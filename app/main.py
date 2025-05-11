import os
from datetime import datetime, timedelta
from pathlib import Path
from random import sample

import joblib
import pandas as pd
from entsoe import EntsoePandasClient
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from app.core.config import settings
from app.core.model import Model
from app.schemas.EntsoeLoads import EntsoeLoadsRequest
from app.services import (
    data_cleaning_service,
    data_loading_service,
    feature_extraction_service,
    performance_measure_service,
)

logger.add(settings.LOGS_FILEPATH, level="INFO", rotation="10 MB", retention="365 days")

app = FastAPI(title="[Swiss Energy Forcasting] ML Backend")


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


def update_forecast():

    # Data ingestion -> bronze data
    logger.info("Start downloading data from the ENTSO-E service...")
    entsoe_client = EntsoePandasClient(api_key=settings.ENTSOE_API_KEY)
    data_loading_service.fetch_df(entsoe_client, settings.BRONZE_DF_FILEPATH)
    logger.info("Data downloaded")

    # [bronze -> silver] Data cleaning
    logger.info("Start cleaning the downloaded data...")
    data_cleaning_service.clean(
        df=pd.read_pickle(settings.BRONZE_DF_FILEPATH),
        out_df_filepath=settings.SILVER_DF_FILEPATH,
    )
    logger.info("Data cleaned.")

    # Measure the performance of the official model
    logger.info("Start computing the official model's MAPE")
    mape_df = performance_measure_service.compute_mape(
        y_true_col="Actual Load",
        y_pred_col="Forecasted Load",
        data=pd.read_pickle(settings.BRONZE_DF_FILEPATH),
        timedeltas=[
            timedelta(hours=1),
            timedelta(hours=24),
            timedelta(weeks=1),
            timedelta(weeks=4),
        ],
    )
    mape = {
        "1h": mape_df.mape.iloc[0],
        "24h": mape_df.mape.iloc[1],
        "7d": mape_df.mape.iloc[2],
        "4w": mape_df.mape.iloc[3],
    }
    joblib.dump(mape, settings.ENTSOE_MAPE_FILEPATH)
    logger.info(f"ENTSO-E MAPE: {mape}")
    logger.info("Official model's MAPE computed")

    # [silver -> gold] Extract features
    logger.info("Start extracting features...")
    feature_extraction_service.extract_features(
        df=pd.read_pickle(settings.SILVER_DF_FILEPATH),
        out_df_filepath=settings.GOLD_DF_FILEPATH,
    )
    logger.info("Features extracted.")

    # Walk-forward validate the model
    logger.info("Start walk-forward validation of the model...")
    model = Model(n_estimators=int(os.getenv("MODEL_N_ESTIMATORS")))
    latest_load_ts = pd.read_pickle(settings.GOLD_DF_FILEPATH).dropna(subset=("24h_later_load")).index.max()

    # Figure out ranges to timestamps to test on
    past_24h_ts = latest_load_ts - timedelta(hours=23)
    past_1w_ts = latest_load_ts - timedelta(weeks=1)
    past_4w_ts = latest_load_ts - timedelta(weeks=4)

    past_24h_timestamps = pd.date_range(start=past_24h_ts, end=latest_load_ts, freq="h").to_list()
    past_1w_timestamps = pd.date_range(start=past_1w_ts, end=past_24h_ts, freq="h").to_list()
    past_4w_timestamps = pd.date_range(start=past_4w_ts, end=past_1w_ts, freq="h").to_list()

    # Estimate the MAPE off 10% (17 and 50) of the points for the past week/month
    # To avoid heavy computations
    walkforward_yhat = model.train_predict(
        Xy=pd.read_pickle(settings.GOLD_DF_FILEPATH),
        query_timestamps=past_24h_timestamps + sample(past_1w_timestamps, 17) + sample(past_4w_timestamps, 50),
        out_yhat_filepath=settings.WALKFORWARD_YHAT_FILEPATH,
    )
    walkforward_y = pd.read_pickle(settings.GOLD_DF_FILEPATH)[["24h_later_load"]]
    mape_df = performance_measure_service.compute_mape(
        y_true_col="24h_later_load",
        y_pred_col="predicted_24h_later_load",
        data=walkforward_yhat.join(walkforward_y, how="left"),
        timedeltas=[
            timedelta(hours=1),
            timedelta(hours=24),
            timedelta(weeks=1),
            timedelta(weeks=4),
        ],
    )
    mape = {
        "1h": mape_df.mape.iloc[0],
        "24h": mape_df.mape.iloc[1],
        "7d": mape_df.mape.iloc[2],
        "4w": mape_df.mape.iloc[3],
    }
    joblib.dump(mape, settings.OUR_MODEL_MAPE_FILEPATH)
    logger.info(f"MAPE: {mape}")
    logger.info("Walk-forward validation done.")

    # Train-predict
    logger.info("Start train-predicting the model...")
    model.train_predict(
        Xy=pd.read_pickle(settings.GOLD_DF_FILEPATH),
        query_timestamps=[pd.Timestamp(latest_load_ts) + timedelta(hours=i) for i in range(1, 25)],
        out_yhat_filepath=settings.YHAT_FILEPATH,
    )
    logger.info("Train-predict done.")


@app.get("/")
async def get_root():
    logger.info(f"Received GET /")
    return {"message": "Welcome to the swissenergy-backend!"}


@app.get("/update-forecast")
async def get_update_forecast(background_tasks: BackgroundTasks):
    logger.info(f"Received GET /update-forecast")
    background_tasks.add_task(update_forecast)
    return {"message": "Forecast updating task started..."}


@app.get("/latest-forecast")
async def get_latest_forecast():
    logger.info("Received GET /latest-forecast")

    # Load latest forecast
    timestamps, predicted_24h_later_load = [], []
    if settings.YHAT_FILEPATH.is_file():
        yhat = pd.read_pickle(settings.YHAT_FILEPATH)
        timestamps = yhat.index.tolist()
        predicted_24h_later_load = yhat["predicted_24h_later_load"].fillna("NaN").tolist()

    latest_forecasts = {
        "timestamps": timestamps,
        "predicted_24h_later_load": predicted_24h_later_load,
    }

    logger.info(
        f"Ready to send back: {len(latest_forecasts['timestamps'])} timestamps [{min(latest_forecasts['timestamps'])} -> {max(latest_forecasts['timestamps'])}]"
    )

    return latest_forecasts


@app.post("/entsoe-loads")
async def post_entsoe_loads(request: EntsoeLoadsRequest):
    logger.info(f"Received POST /entsoe-loads : {request}")

    # Load past loads
    silver_df = pd.read_pickle(settings.SILVER_DF_FILEPATH)

    # Figure out till when the records should be sent
    end_ts = silver_df.index.max()
    cutoff_ts = end_ts - request.delta_time

    # Only keep the data till
    silver_df = silver_df[silver_df.index > cutoff_ts]

    entsoe_loads = {
        "timestamps": silver_df.index.tolist(),
        "24h_later_load": silver_df["24h_later_load"].fillna("NaN").tolist(),
        "24h_later_forecast": silver_df["24h_later_forecast"].fillna("NaN").tolist(),
    }

    if len(entsoe_loads["timestamps"]):
        logger.info(f"Ready to send back: {len(entsoe_loads['timestamps'])} timestamps between {cutoff_ts} -> {end_ts}")
    else:
        logger.warning(
            f"Ready to send empty dict: {len(entsoe_loads['timestamps'])} timestamps between {cutoff_ts} -> {end_ts}"
        )

    return entsoe_loads


@app.get("/latest-forecast-ts")
async def get_latest_forecast_ts():
    logger.info(f"Received GET /latest-forecast-ts")

    if not settings.YHAT_FILEPATH.is_file():
        logger.warning("No forecast has been created. Sending back -1")
        return {"latest_forecast_ts": -1}

    creation_ts = os.path.getctime(settings.YHAT_FILEPATH)  # since epoch
    logger.info(
        f"Ready to send back the creation timestamp of {settings.YHAT_FILEPATH}: {creation_ts} ({datetime.fromtimestamp(creation_ts)})"
    )
    return {"latest_forecast_ts": creation_ts}


@app.get("/latest-mape")
async def get_latest_mape():
    logger.info(f"Received GET /latest-mape")

    # Figure out the ENTSO-E MAPE
    entsoe_filepath = Path(settings.ENTSOE_MAPE_FILEPATH)
    entsoe_mape = {}
    if entsoe_filepath.is_file():
        entsoe_mape = joblib.load(entsoe_filepath)

    # Figure out our model's MAPE
    our_model_filepath = Path(settings.OUR_MODEL_MAPE_FILEPATH)
    our_model_mape = {}
    if our_model_filepath.is_file():
        our_model_mape = joblib.load(our_model_filepath)

    mape = {
        "entsoe_model": entsoe_mape,
        "our_model": our_model_mape,
    }

    logger.info(f"Ready to send back the MAPE: {mape}")
    return mape
