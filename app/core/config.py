import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


# TODO rework
class Settings(BaseSettings):
    ENTSOE_API_KEY: str = os.getenv("ENTSOE_API_KEY")  # TODO implement error if missing
    BRONZE_DF_FILEPATH: Path = Path("data/bronze/df.pickle").resolve()
    SILVER_DF_FILEPATH: Path = Path("data/silver/df.pickle").resolve()
    GOLD_DF_FILEPATH: Path = Path("data/gold/df.pickle").resolve()
    WALKFORWARD_YHAT_FILEPATH: Path = Path("data/walkforward_yhat.pickle").resolve()
    YHAT_FILEPATH: Path = Path("data/yhat.pickle").resolve()
    OUR_MODEL_MAPE_FILEPATH: Path = Path("data/our_model_mape.joblib").resolve()
    ENTSOE_MAPE_FILEPATH: Path = Path("data/entsoe_mape.joblib").resolve()


settings = Settings()
