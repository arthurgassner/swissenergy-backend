import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from human_readable import precise_delta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Class responsible for downloading the data from the ENTSO-E servers."""

    def __init__(self, entsoe_api_key: str) -> None:
        """Create a new DataLoader instance.

        Args:
            entsoe_api_key (str): API key used to download data from the ENTSO-E servers.
                                  See https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
        """
        if entsoe_api_key is None:
            logger.error(f"entsoe_api_key cannot be None.")
            raise ValueError

        # Get API key through website, after kindly asking the support
        self._entsoe_pandas_client = EntsoePandasClient(api_key=entsoe_api_key)

    def _query_load_and_forecast(
        self,
        start_ts: pd.Timestamp,
        end_ts: Optional[pd.Timestamp] = None,
        max_retries: int = 10,
    ) -> pd.DataFrame:
        """Query the ENTSO-E API for the load and forecast data from `start_ts` to `end_ts`, breaking it down into yearly-queries.

        It seems that the ENTSO-E API tends to terminate the connection when asking for 10 years of data.
        Hence, the data is fetched year-by-year -- as it seems to lower the odds of aborted connections.

        Args:
            start_ts (pd.Timestamp): Starting ts (tz="Europe/Zurich") of the requested data
            end_ts (Optional[pd.Timestamp]): Ending ts (tz="Europe/Zurich") of the requested data, default to 24h away from now.
            max_retries (int): Max amount of retries, as the ENTSO-E API tends to abort the connection.

        Returns:
            pd.DataFrame: Fetched data.
                          - columns: ('Forcasted Load', 'Actual Load')
                          - dtypes: float64
                          - index: datetime64[ns, Europe/Zurich]
                          Empty dataframe if no data could be found
        """
        if end_ts is None:
            end_ts = pd.Timestamp(
                datetime.now() + timedelta(hours=24), tz="Europe/Zurich"
            )

        # Split up the query into yearly queries
        start_end_timestamps = []
        curr_start_ts = start_ts
        curr_end_ts = min(end_ts, curr_start_ts + timedelta(days=365))
        while curr_end_ts < end_ts:
            start_end_timestamps.append((curr_start_ts, curr_end_ts))
            curr_start_ts = curr_end_ts
            curr_end_ts = min(end_ts, curr_start_ts + timedelta(days=365))
        start_end_timestamps.append((curr_start_ts, end_ts))

        # Send each yearly-query to the ENTSO-E API
        fetched_dfs = []
        for curr_start_ts, curr_end_ts in start_end_timestamps:
            logging.info(
                f"Asking the ENTSO-E API for load/forecast data between {curr_start_ts} -> {curr_end_ts} ({precise_delta(curr_end_ts - curr_start_ts, minimum_unit="seconds")})"
            )
            n_retries = 0
            while n_retries < max_retries:
                try:
                    fetched_df = self._entsoe_pandas_client.query_load_and_forecast(
                        country_code="CH", start=curr_start_ts, end=curr_end_ts
                    )
                    break
                except NoMatchingDataError:
                    logger.warning(
                        f"No data available between {curr_start_ts} -> {curr_end_ts} ({precise_delta(curr_end_ts - curr_start_ts, minimum_unit="seconds")})"
                    )
                    fetched_df = pd.DataFrame(  # empty dataframe
                        columns=["Forecasted Load", "Actual Load"],
                        dtype=float,
                        index=pd.DatetimeIndex(
                            [], dtype="datetime64[ns, Europe/Zurich]"
                        ),
                    )
                    break
                except requests.ConnectionError as e:
                    n_retries += 1
                    if not n_retries < max_retries:
                        raise e
                    logger.warning(f"Thrown {e}. Retrying {n_retries}/{max_retries}...")
                time.sleep(1)  # Wait time between requests [s]
            fetched_dfs.append(fetched_df)

        return pd.concat(fetched_dfs)

    @staticmethod
    def enforce_data_quality(df: pd.DataFrame) -> None:
        """Enforce the data quality of df -- as we expect it to be when coming straight from the ENTSO-E API.

        If a poor data quality is detected, recover if possible, else throw a ValueError.

        Args:
            df (pd.DataFrame): Dataframe fresh from the ENTSO-E API.

        Raises:
            ValueError: If not isinstance(df.index, pd.DatetimeIndex)
            ValueError: If df.index.dtype != "datetime64[ns, Europe/Zurich]"
            ValueError: If len(df.columns) != 2
            ValueError: If df.columns != ["Forecasted Load", "Actual Load"]
            ValueError: If df.dtypes.to_list() != ['float64', 'float64']

        Returns:
            pd.DataFrame: Input dataframe with data quality enforced.
        """

        # Enforce the data quality of the index
        # errors
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(
                f"df.index should be an instance of pd.DatetimeIndex, but is: {type(df.index)}"
            )
            raise ValueError
        if df.index.dtype != "datetime64[ns, Europe/Zurich]":
            logger.error(
                f"df.index.dtype should be datetime64[ns, Europe/Zurich] but is: {df.index.dtype}"
            )
            raise ValueError

        # warnings
        if not df.index.is_unique:
            logger.warning(
                f"df.index should be unique, but has {(df.index.value_counts() > 1).sum()} duplicated index. Forcing index's uniqueness by aggregating duplicated index with median..."
            )
            df = df.groupby(df.index).median()
        if not df.index.is_monotonic_increasing:
            logger.warning(
                "df.index should be monotonic increasing, but isn't. Forcing index's monotonic increase by sorting the index..."
            )
            df = df.sort_index()

        # Enforce the data quality of the columns
        if len(df.columns) != 2:
            logger.error(f"df should only have 2 columns, but has {len(df.columns)}")
            raise ValueError
        if any([df.columns[0] != "Forecasted Load", df.columns[1] != "Actual Load"]):
            logger.error(
                f"df.columns should be ['Forecasted Load', 'Actual Load'], but is {df.columns}"
            )
            raise ValueError
        if (df.dtypes != "float64").any():
            logger.error(
                f"df.dtypes should be [dtype('float64'), dtype('float64')], but are {df.dtypes.to_list()}"
            )
            raise ValueError

        return df

    def fetch_df(self, out_df_filepath: str) -> None:
        """Fetch the forecast/load data from the ENTSO-E API, and dump it to disk.

        Args:
            out_df_filepath (str): Filepath where the dataframe (.pickle) should be stored.
        """

        # Fetch loads and forecasts
        fetched_df = self._query_load_and_forecast(
            start_ts=pd.Timestamp("2014-01-01 00:00", tz="Europe/Zurich")
        )

        # Dump to output df
        # Ensure the folderpath exists
        Path(out_df_filepath).parent.mkdir(parents=True, exist_ok=True)
        fetched_df.to_pickle(out_df_filepath)
