from datetime import datetime
from typing import List

import pandas as pd
from pydantic import BaseModel


class EntsoeLoadsFetchLatestRequest(BaseModel):
    n_days: int = 0
    n_hours: int = 1

    @property
    def delta_time(self) -> pd.Timedelta:
        return pd.Timedelta(days=self.n_days, hours=self.n_hours)


class EntsoeLoadsFetchLatestResponse(BaseModel):
    timestamps: List[datetime]
    day_later_loads: List[float | str]
    day_later_forecasts: List[float | str]
