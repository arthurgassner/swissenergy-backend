from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
import pandas as pd
from datetime import datetime
from pathlib import Path

class DataLoader:
    def __init__(self, entsoe_api_key: str) -> None:
        self._entsoe_pandas_client = EntsoePandasClient(api_key=entsoe_api_key) # Get API key through website, after kindly asking the support
        
    def update_df(self, out_df_filepath: str) -> None:
        # Load already-downloaded data
        current_df = pd.DataFrame()
        if Path(out_df_filepath).is_file():
            current_df = pd.read_parquet(out_df_filepath)

        # Figure out the timestamp of the latest-available row
        latest_available_ts = pd.Timestamp('20140101', tz='Europe/Zurich') # Very early ts
        if len(current_df): 
            latest_available_ts = current_df.index.max()

        # Fetch loads and forecasts
        end_ts = pd.Timestamp(datetime.now(), tz='Europe/Zurich') + pd.Timedelta(1, 'd')
        start_ts = latest_available_ts + pd.Timedelta(1, 'm') # start right after (i.e. 1min) the latest ts
        fetched_df = pd.DataFrame()
        try:
            fetched_df = self._entsoe_pandas_client.query_load_and_forecast(
                country_code='CH', 
                start=start_ts, 
                end=end_ts
            )
        except NoMatchingDataError:
            print(f"No data available between {start_ts} -> {end_ts}")
            
        # Append the newly-fetched data to the current data 
        current_df = pd.concat([current_df, fetched_df], axis=0)

        # Dump to output df
        current_df.to_parquet(out_df_filepath)