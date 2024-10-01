import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

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
            logger.error(f"Missing `entsoe_api_key`.")
            raise ValueError

        self._entsoe_pandas_client = EntsoePandasClient(
            api_key=entsoe_api_key
        )  # Get API key through website, after kindly asking the support

    @staticmethod
    def _get_latest_ts_with_actual_load(df: pd.DataFrame) -> pd.Timestamp:
        """Get the timestamp of the latest-available row with a non-NaN 'Actual Load'.

        Args:
            df (pd.DataFrame): Dataframe whose latest-available timestamp with non-NaN 'Actual Load' we want

        Returns:
            pd.Timestamp: Latest-available timestamp (tz="Europe/Zurich") with non-NaN 'Actual Load'.
                          pd.TimeStamp('20140101 00:00', tz="Europe/Zurich") if `df` is empty.
        """

        if (
            "Actual Load" in df.columns
            and (non_na_mask := ~df["Actual Load"].isna()).sum()
        ):
            return df[non_na_mask].index.max()

        return pd.Timestamp("20140101 00:00", tz="Europe/Zurich")

    def _query_load_and_forecast(self, start_ts: pd.Timestamp) -> pd.DataFrame:
        """Query the ENTSO-E API for the load and forecast data from `start_ts` to now+24h.

        Args:
            start_ts (pd.Timestamp): Starting ts (tz="Europe/Zurich") of the requested data

        Returns:
            pd.DataFrame: Fetched data.
                          - columns: ('Forcasted Load', 'Actual Load')
                          - dtypes: float64
                          - index: datetime64[ns, Europe/Zurich]
                          Empty dataframe if no data could be found
        """
        end_ts = pd.Timestamp(datetime.now() + timedelta(hours=24), tz="Europe/Zurich")

        try:
            fetched_df = self._entsoe_pandas_client.query_load_and_forecast(
                country_code="CH", start=start_ts, end=end_ts
            )
        except NoMatchingDataError:
            logger.warning(f"No data available between {start_ts} -> {end_ts}")
            fetched_df = pd.DataFrame(
                columns=["Forecasted Load", "Actual Load"],
                dtype=float,
                index=pd.DatetimeIndex([], dtype="datetime64[ns, Europe/Zurich]"),
            )  # empty dataframe

        return fetched_df

    def update_df(self, out_df_filepath: str) -> None:
        """Update the currently-on-disk dataframe (.parquet)
        by downloading -- through the ENTSO-E API -- the rows whose timestamps are after the latest on-disk timestamp.

        Args:
            out_df_filepath (str): Filepath where the dataframe (.parquet) should be stored.
        """
        # Load already-downloaded data
        current_df = pd.DataFrame()
        if Path(out_df_filepath).is_file():
            current_df = pd.read_parquet(out_df_filepath)

        # Figure out the timestamp of the latest-available row with a non-NaN 'Actual Load'
        latest_available_ts = DataLoader._get_latest_ts_with_actual_load(df=current_df)

        # Fetch loads and forecasts
        fetched_df = self._query_load_and_forecast(
            start_ts=latest_available_ts
            + pd.Timedelta(1, "m")  # right after (i.e. 1min) the latest ts
        )

        # Append the newly-fetched data to the current data
        current_df = pd.concat([current_df, fetched_df], axis=0)

        # TODO investigate why that is necessary
        current_df = current_df.drop_duplicates()

        # Dump to output df
        Path(out_df_filepath).parent.mkdir(  # Ensure the folderpath exists
            parents=True, exist_ok=True
        )
        current_df.to_parquet(out_df_filepath)
