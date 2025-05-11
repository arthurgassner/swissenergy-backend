import pandas as pd
from fastapi import APIRouter
from loguru import logger

from app.core.config import settings
from app.schemas.EntsoeLoadsLatest import (
    EntsoeLoadsLatestRequest,
    EntsoeLoadsLatestResponse,
)

router = APIRouter()


@router.post("/entsoe-loads/latest")
async def post_entsoe_loads_latest(request: EntsoeLoadsLatestRequest) -> EntsoeLoadsLatestResponse:
    logger.info(f"Received POST /entsoe-loads/latest : {request}")

    # Load past loads
    silver_df = pd.read_pickle(settings.SILVER_DF_FILEPATH)

    # Figure out till when the records should be sent
    end_ts = silver_df.index.max()
    cutoff_ts = end_ts - request.delta_time

    # Only keep the data till
    silver_df = silver_df[silver_df.index > cutoff_ts]

    response = EntsoeLoadsLatestResponse(
        timestamps=silver_df.index.tolist(),
        day_later_loads=silver_df["24h_later_load"].astype(float).fillna("NaN").tolist(),
        day_later_forecasts=silver_df["24h_later_forecast"].astype(float).fillna("NaN").tolist(),
    )

    if len(response.timestamps):
        logger.info(f"Ready to send back: {len(response.timestamps)} timestamps between {cutoff_ts} -> {end_ts}")
    else:
        logger.warning(
            f"Ready to send empty dict: {len(response.timestamps)} timestamps between {cutoff_ts} -> {end_ts}"
        )

    return response
