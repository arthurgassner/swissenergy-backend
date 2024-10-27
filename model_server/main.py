import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from random import sample

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

BRONZE_DF_FILEPATH = "data/bronze/df.pickle"
SILVER_DF_FILEPATH = "data/silver/df.pickle"
GOLD_DF_FILEPATH = "data/gold/df.pickle"
WALKFORWARD_YHAT_FILEPATH = "data/walkforward_yhat.pickle"
YHAT_FILEPATH = "data/yhat.pickle"
OUR_MODEL_MAPE_FILEPATH = "data/our_model_mape.joblib"
ENTSOE_MAPE_FILEPATH = "data/entsoe_mape.joblib"

load_dotenv()


class DeltaTime(BaseModel):
    n_days: int = 0
    n_hours: int = 1


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
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

    # Data ingestion -> bronze data
    logger.info("Start downloading data from the ENTSO-E service...")
    data_loader = DataLoader(entsoe_api_key=entsoe_api_key)
    data_loader.fetch_df(out_df_filepath=BRONZE_DF_FILEPATH)
    logger.info("Data downloaded")

    # [bronze -> silver] Data cleaning
    logger.info("Start cleaning the downloaded data...")

    DataCleaner.clean(
        df=pd.read_pickle(BRONZE_DF_FILEPATH),
        out_df_filepath=SILVER_DF_FILEPATH,
    )

    logger.info("Data cleaned.")

    # [silver -> gold] Extract features
    logger.info("Start extracting features...")
    FeatureExtractor.extract_features(
        df=pd.read_pickle(SILVER_DF_FILEPATH),
        out_df_filepath=GOLD_DF_FILEPATH,
    )
    logger.info("Features extracted.")

    # Measure the performance of the official model
    logger.info("Start computing the official model's MAPE")
    mape_df = PerformanceMeasurer.mape(
        y_true_col="Actual Load",
        y_pred_col="Forecasted Load",
        data=pd.read_pickle(SILVER_DF_FILEPATH),
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
    joblib.dump(mape, ENTSOE_MAPE_FILEPATH)
    logger.info(f"ENTSO-E MAPE: {mape}")
    logger.info("Official model's MAPE computed")

    # Walk-forward validate the model
    logger.info("Start walk-forward validation of the model...")
    model = Model(n_estimators=int(os.getenv("MODEL_N_ESTIMATORS")))
    latest_load_ts = (
        pd.read_pickle(GOLD_DF_FILEPATH).dropna(subset=("24h_later_load")).index.max()
    )

    # Figure out ranges to timestamps to test on
    past_24h_ts = latest_load_ts - timedelta(hours=23)
    past_1w_ts = latest_load_ts - timedelta(weeks=1)
    past_4w_ts = latest_load_ts - timedelta(weeks=4)

    past_24h_timestamps = pd.date_range(
        start=past_24h_ts, end=latest_load_ts, freq="h"
    ).to_list()
    past_1w_timestamps = pd.date_range(
        start=past_1w_ts, end=past_24h_ts, freq="h"
    ).to_list()
    past_4w_timestamps = pd.date_range(
        start=past_4w_ts, end=past_1w_ts, freq="h"
    ).to_list()

    # Estimate the MAPE off 10% (17 and 50) of the points for the past week/month
    # To avoid heavy computations
    walkforward_yhat = model.train_predict(
        Xy=pd.read_pickle(GOLD_DF_FILEPATH),
        query_timestamps=past_24h_timestamps
        + sample(past_1w_timestamps, 17)
        + sample(past_4w_timestamps, 50),
        out_yhat_filepath=WALKFORWARD_YHAT_FILEPATH,
    )
    walkforward_y = pd.read_pickle(GOLD_DF_FILEPATH)[["24h_later_load"]]
    mape_df = PerformanceMeasurer.mape(
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
    joblib.dump(mape, OUR_MODEL_MAPE_FILEPATH)
    logger.info(f"MAPE: {mape}")
    logger.info("Walk-forward validation done.")

    # Train-predict
    logger.info("Start train-predicting the model...")
    model.train_predict(
        Xy=pd.read_pickle(GOLD_DF_FILEPATH),
        query_timestamps=[
            pd.Timestamp(latest_load_ts) + timedelta(hours=i) for i in range(1, 25)
        ],
        out_yhat_filepath=YHAT_FILEPATH,
    )
    logger.info("Train-predict done.")


@app.get("/")
async def get_root():
    logger.info(f"Received GET /")
    return {"message": "Welcome to the swissenergy-backend!"}


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
    yhat_filepath = Path(YHAT_FILEPATH)
    timestamps, predicted_24h_later_load = [], []
    if yhat_filepath.is_file():
        yhat = pd.read_pickle(yhat_filepath)
        timestamps = yhat.index.tolist()
        predicted_24h_later_load = (
            yhat["predicted_24h_later_load"].fillna("NaN").tolist()
        )

    latest_forecasts = {
        "timestamps": timestamps,
        "predicted_24h_later_load": predicted_24h_later_load,
    }

    logger.info(
        f"Ready to send back: {len(latest_forecasts['timestamps'])} timestamps [{min(latest_forecasts['timestamps'])} -> {max(latest_forecasts['timestamps'])}]"
    )

    return latest_forecasts


@app.post("/entsoe-loads")
async def get_entsoe_loads(delta_time: DeltaTime):

    logger.info(f"Received POST /entsoe-loads : {delta_time}")

    # Load past loads
    silver_df = pd.read_pickle(SILVER_DF_FILEPATH)

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

    yhat_filepath = Path(YHAT_FILEPATH)
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
    entsoe_filepath = Path(ENTSOE_MAPE_FILEPATH)
    entsoe_mape = {}
    if entsoe_filepath.is_file():
        entsoe_mape = joblib.load(entsoe_filepath)

    # Figure out our model's MAPE
    our_model_filepath = Path(OUR_MODEL_MAPE_FILEPATH)
    our_model_mape = {}
    if our_model_filepath.is_file():
        our_model_mape = joblib.load(our_model_filepath)

    mape = {
        "entsoe_model": entsoe_mape,
        "our_model": our_model_mape,
    }

    logger.info(f"Ready to send back the MAPE: {mape}")
    return mape
