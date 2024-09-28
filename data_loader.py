from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
import pandas as pd
from datetime import datetime
from pathlib import Path

class DataLoader:
    def __init__(self, enstoe_api_key: str) -> None:
        self._entsoe_pandas_client = EntsoePandasClient(api_key=enstoe_api_key) # Get API key through website, after kindly asking the support

    def 
