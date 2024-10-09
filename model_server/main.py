import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .data_cleaner import DataCleaner
from .data_loader import DataLoader
from .feature_extractor import FeatureExtractor
from .model import Model
from .performance_measurer import PerformanceMeasurer

load_dotenv()


class DeltaTime(BaseModel):
    n_days: int = 0
    n_hours: int = 1


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler("model_server.log"),  # Log to a file
        logging.StreamHandler(),  # Also log to the shell
    ],
)
logger = logging.getLogger(__name__)
app = FastAPI(title="[Swiss Energy Forcasting] ML Backend")


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


def update_forecast(entsoe_api_key: str):

    # Update the bronze-layer data
    logger.info("Start downloading data from the ENTSO-E service...")

    data_loader = DataLoader(entsoe_api_key=entsoe_api_key)
    data_loader.update_df(out_df_filepath="data/bronze/df.pickle")

    logger.info("Data downloaded")

    # Measure the performance of the official model
    logger.info("Start computing the official model's MAPE")
    mape_df = PerformanceMeasurer.mape(
        y_true_col="Actual Load",
        y_pred_col="Forecasted Load",
        data=pd.read_pickle("data/bronze/df.pickle"),
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
    joblib.dump(mape, "entsoe_mape.joblib")
    logger.info(f"ENTSO-E MAPE: {mape}")
    logger.info("Official model's MAPE computed")

    # Clean the bronze-layer data
    logger.info("Start cleaning the downloaded data...")

    DataCleaner.clean(
        df=pd.read_pickle("data/bronze/df.pickle"),
        out_df_filepath="data/silver/df.pickle",
    )

    logger.info("Data cleaned.")

    # Extract features
    logger.info("Start extracting features...")
    FeatureExtractor.extract_features(
        df=pd.read_pickle("data/silver/df.pickle"),
        out_df_filepath="data/gold/df.pickle",
    )
    logger.info("Features extracted.")

    # Backtest model
    logger.info("Start back-testing the model...")
    model = Model(n_estimators=10_000)
    latest_load_ts = (
        pd.read_pickle("data/gold/df.pickle")
        .dropna(subset=("24h_later_load"))
        .index.max()
    )
    yhat_backtest = model.train_predict(
        Xy=pd.read_pickle("data/gold/df.pickle"),
        query_timestamps=[
            pd.Timestamp(latest_load_ts) - timedelta(hours=23) + timedelta(hours=i)
            for i in range(24)
        ],
    )
    y_backtest = pd.read_pickle("data/gold/df.pickle")[["24h_later_load"]]
    mape_df = PerformanceMeasurer.mape(
        y_true_col="24h_later_load",
        y_pred_col="predicted_24h_later_load",
        data=yhat_backtest.join(y_backtest, how="left"),
        timedeltas=[
            timedelta(hours=1),
            timedelta(hours=24),
        ],
    )
    mape = {
        "1h": mape_df.mape.iloc[0],
        "24h": mape_df.mape.iloc[1],
    }
    joblib.dump(mape, "our_model_mape.joblib")
    logger.info(f"MAPE: {mape}")
    logger.info("Back-testing done.")

    # Train-predict
    logger.info("Start train-predicting the model...")
    model.train_predict(
        Xy=pd.read_pickle("data/gold/df.pickle"),
        query_timestamps=[
            pd.Timestamp(latest_load_ts) + timedelta(hours=i) for i in range(1, 25)
        ],
        out_yhat_filepath="data/yhat.pickle",
    )
    logger.info("Train-predict done.")


@app.get("/update-forecast")
async def get_update_forecast(background_tasks: BackgroundTasks):
    logger.info(f"Received GET /update-forecast")
    background_tasks.add_task(
        update_forecast, entsoe_api_key=os.getenv("ENTSOE_API_KEY")
    )
    return {"message": "Forecast updating task started..."}


@app.get("/latest-forecast")
async def get_latest_forecast():

    logger.info("Received GET /latest-forecast")

    # Load latest forecast
    yhat = pd.read_pickle("data/yhat.pickle")
    latest_forecasts = {
        "timestamps": yhat.index.tolist(),
        "predicted_24h_later_load": yhat["predicted_24h_later_load"].tolist(),
    }

    logger.info(
        f"Ready to send back: {len(latest_forecasts['timestamps'])} timestamps [{min(latest_forecasts['timestamps'])} -> {max(latest_forecasts['timestamps'])}]"
    )

    return latest_forecasts


@app.post("/entsoe-loads")
async def get_entsoe_loads(delta_time: DeltaTime):

    logger.info(f"Received POST /entsoe-loads : {delta_time}")

    # Load past loads
    silver_df = pd.read_pickle("data/silver/df.pickle")

    # Figure out till when the records should be sent
    end_ts = silver_df.index.max()
    cutoff_ts = end_ts - pd.Timedelta(days=delta_time.n_days, hours=delta_time.n_hours)

    # Only keep the data till
    silver_df = silver_df[silver_df.index > cutoff_ts]

    entsoe_loads = {
        "timestamps": silver_df.index.tolist(),
        "24h_later_load": silver_df["24h_later_load"].fillna("NaN").tolist(),
        "24h_later_forecast": silver_df["24h_later_forecast"].fillna("NaN").tolist(),
    }

    if len(entsoe_loads["timestamps"]):
        logger.info(
            f"Ready to send back: {len(entsoe_loads['timestamps'])} timestamps between {cutoff_ts} -> {end_ts}"
        )
    else:
        logger.warning(
            f"Ready to send empty dict: {len(entsoe_loads['timestamps'])} timestamps between {cutoff_ts} -> {end_ts}"
        )

    return entsoe_loads


@app.get("/latest-forecast-ts")
async def get_latest_forecast_ts():
    logger.info(f"Received GET /latest-forecast-ts")

    yhat_filepath = Path("data/yhat.pickle")
    if not yhat_filepath.is_file():
        logger.warning("No forecast has been created. Sending back -1")
        return {"latest_forecast_ts": -1}

    creation_ts = os.path.getctime(yhat_filepath)  # since epoch
    logger.info(
        f"Ready to send back the creation timestamp of {yhat_filepath.as_posix()}: {creation_ts} ({datetime.fromtimestamp(creation_ts)})"
    )
    return {"latest_forecast_ts": creation_ts}


@app.get("/latest-mape")
async def get_latest_mape():
    logger.info(f"Received GET /latest-mape")

    # Figure out the ENTSO-E MAPE
    entsoe_filepath = Path("entsoe_mape.joblib")
    entsoe_mape = {}
    if entsoe_filepath.is_file():
        entsoe_mape = joblib.load(entsoe_filepath)

    # Figure out our model's MAPE
    our_model_filepath = Path("our_model_mape.joblib")
    our_model_mape = {}
    if our_model_filepath.is_file():
        our_model_mape = joblib.load(our_model_filepath)

    mape = {
        "entsoe_model": entsoe_mape,
        "our_model": our_model_mape,
    }

    logger.info(f"Ready to send back the MAPE: {mape}")
    return mape
