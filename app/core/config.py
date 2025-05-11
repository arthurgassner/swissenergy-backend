import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


# TODO rework
class Settings(BaseSettings):
    ENTSOE_API_KEY: str = os.getenv("ENTSOE_API_KEY")  # TODO implement error if missing
    BRONZE_DF_FILEPATH: str = "data/bronze/df.pickle"  # TODO use Path
    SILVER_DF_FILEPATH: str = "data/silver/df.pickle"
    GOLD_DF_FILEPATH: str = "data/gold/df.pickle"
    WALKFORWARD_YHAT_FILEPATH: str = "data/walkforward_yhat.pickle"
    YHAT_FILEPATH: str = "data/yhat.pickle"
    OUR_MODEL_MAPE_FILEPATH: str = "data/our_model_mape.joblib"
    ENTSOE_MAPE_FILEPATH: str = "data/entsoe_mape.joblib"


settings = Settings()
