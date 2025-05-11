from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import settings
from app.routers.entsoe_loads import router as entsoe_loads_router
from app.routers.forecasts import router as forecasts_loads_router

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


@app.get("/")
async def get_root():
    logger.info(f"Received GET /")
    return {"message": "Welcome to the swissenergy-backend!"}


app.include_router(entsoe_loads_router)
app.include_router(forecasts_loads_router)
