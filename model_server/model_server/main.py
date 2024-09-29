import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from .data_cleaner import DataCleaner
from .data_loader import DataLoader
from .feature_extractor import FeatureExtractor
from .model import Model


class WeekCounter(BaseModel):
    n_weeks_ago: int = 1


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
app = FastAPI(title="Data preparation")


def update_forecast(entsoe_api_key: str):

    # Update the bronze-layer data
    logger.info("Start downloading data from the ENTSO-E service...")

    data_loader = DataLoader(entsoe_api_key=entsoe_api_key)
    data_loader.update_df(out_df_filepath="data/bronze/df.parquet")

    logger.info("Data downloaded.")

    # Clean the bronze-layer data
    logger.info("Start cleaning the downloaded data...")

    DataCleaner.clean(
        in_df_filepath="data/bronze/df.parquet",
        out_df_filepath="data/silver/df.parquet",
    )

    logger.info("Data cleaned.")

    # Extract features
    logger.info("Start extracting features...")
    FeatureExtractor.extract_features(
        in_df_filepath="data/silver/df.parquet",
        out_df_filepath="data/gold/df.parquet",
    )
    logger.info("Features extracted.")

    # Train
    logger.info("Start training model...")

    model = Model(model_filepath="data/model.joblib")
    model.train(Xy_filepath="data/gold/df.parquet", n_estimators=1_000)

    logger.info("Training done.")

    # Backtest model
    logger.info("Start back-testing the model...")
    _, mape_24h = model.backtest(
        Xy_filepath="data/gold/df.parquet",
        starting_ts=pd.Timestamp(
            datetime.now() - timedelta(hours=24), tz="Europe/Zurich"
        ),
        use_every_nth_ts=1,
    )
    logger.info("Back-testing done.")

    # Predict
    logger.info("Start forecasting the load for the next 24h...")
    model.train(Xy_filepath="data/gold/df.parquet", n_estimators=1_000)
    model.predict(
        in_df_filepath="data/gold/df.parquet",
        out_yhat_filepath="data/yhat.parquet",
    )
    logger.info("Forecasting done.")


@app.get("/update-forecast")
async def get_update_forecast(background_tasks: BackgroundTasks):
    logger.info(f"Received GET /update-forecast")

    load_dotenv()
    background_tasks.add_task(
        update_forecast, entsoe_api_key=os.getenv("ENTSOE_API_KEY")
    )
    return {"message": "Forecast updating task started..."}


@app.get("/latest-forecast")
async def get_latest_forecast():

    logger.info("Received GET /latest-forecast")

    # Load latest forecast
    yhat = pd.read_parquet("data/yhat.parquet")
    latest_forecasts = {
        "timestamps": yhat.index.tolist(),
        "predicted_24h_later_load": yhat["predicted_24h_later_load"].tolist(),
    }

    logger.info(
        f"Ready to send back: {len(latest_forecasts['timestamps'])} timestamps [{min(latest_forecasts['timestamps'])} -> {max(latest_forecasts['timestamps'])}]"
    )

    return latest_forecasts


@app.post("/entsoe-loads")
async def get_entsoe_loads(week_counter: WeekCounter):

    logger.info(
        f"Received POST /entsoe-loads - n_weeks_ago: {week_counter.n_weeks_ago}"
    )

    # Figure out till when the records should be sent, defaulting to a week ago
    cutoff_dt = datetime.now() - timedelta(weeks=week_counter.n_weeks_ago)
    cutoff_ts = pd.Timestamp(cutoff_dt, tz="Europe/Zurich")

    # Load past loads
    silver_df = pd.read_parquet("data/silver/df.parquet")

    # Only keep the data till
    silver_df = silver_df[silver_df.index >= cutoff_ts]

    entsoe_loads = {
        "timestamps": silver_df.index.tolist(),
        "24h_later_load": silver_df["24h_later_load"].fillna("NaN").tolist(),
        "24h_later_forecast": silver_df["24h_later_forecast"].fillna("NaN").tolist(),
    }

    logger.info(
        f"Ready to send back: {len(entsoe_loads['timestamps'])} timestamps [{min(entsoe_loads['timestamps'])} -> {max(entsoe_loads['timestamps'])}]"
    )

    return entsoe_loads
