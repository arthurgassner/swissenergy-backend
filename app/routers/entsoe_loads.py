import pandas as pd
from fastapi import APIRouter
from loguru import logger

from app.core.config import settings
from app.schemas.EntsoeLoadsLatest import EntsoeLoadsLatestRequest

router = APIRouter()


@router.post("/entsoe-loads/latest")
async def post_entsoe_loads_latest(request: EntsoeLoadsLatestRequest):  # TODO -> EntsoeLoadsLatestResponse
    logger.info(f"Received POST /entsoe-loads/latest : {request}")

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
