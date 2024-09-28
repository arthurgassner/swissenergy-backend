from fastapi import BackgroundTasks, FastAPI
from dotenv import load_dotenv
import os

from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from .feature_extractor import FeatureExtractor

app = FastAPI(title="Data preparation")

def prep_data(entsoe_api_key: str):
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

@app.get("/prep-data")
async def get_prep_data(background_tasks: BackgroundTasks):
    load_dotenv()
    background_tasks.add_task(prep_data, entsoe_api_key=os.getenv('ENTSOE_API_KEY'))
    return {"message": "Data preparation started"} 