import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    ROOT_FOLDERPATH: Path = Path(__file__).resolve().parent.parent.parent
    DATA_FOLDERPATH: Path = ROOT_FOLDERPATH / "data"
    BRONZE_DF_FILEPATH: Path = DATA_FOLDERPATH / "bronze/df.pickle"
    SILVER_DF_FILEPATH: Path = DATA_FOLDERPATH / "silver/df.pickle"
    GOLD_DF_FILEPATH: Path = DATA_FOLDERPATH / "gold/df.pickle"
    WALKFORWARD_YHAT_FILEPATH: Path = DATA_FOLDERPATH / "walkforward_yhat.pickle"
    YHAT_FILEPATH: Path = DATA_FOLDERPATH / "yhat.pickle"
    OUR_MODEL_MAPE_FILEPATH: Path = DATA_FOLDERPATH / "our_model_mape.joblib"
    ENTSOE_MAPE_FILEPATH: Path = DATA_FOLDERPATH / "entsoe_mape.joblib"
    LOGS_FILEPATH: Path = DATA_FOLDERPATH / "logs/.log"

    @property
    def ENTSOE_API_KEY(self) -> str:
        entsoe_api_key = os.getenv("ENTSOE_API_KEY")

        if entsoe_api_key is None:
            logger.error("Missing ENTSOE_API_KEY. Could not be found in the env variables.")
            raise ValueError

        return entsoe_api_key

    @property
    def MODEL_N_ESTIMATORS(self) -> int:
        model_n_estimators = os.getenv("MODEL_N_ESTIMATORS")

        if model_n_estimators is None:
            logger.error("Missing MODEL_N_ESTIMATORS. Could not be found in the env variables.")
            raise ValueError

        return int(model_n_estimators)


settings = Settings()
