import logging
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Class responsible for cleaning the data downloaded from the ENTSO-E servers."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _format(df: pd.DataFrame) -> pd.DataFrame:
        """Format `df` by
        - Setting back its index by 24h, so that the columns refer to "the values in 24h"
        - Renaming the columns to reflect this new format

        Args:
            df (pd.DataFrame): Dataframe to be formatted

        Returns:
            pd.DataFrame: Formatted dataframe
        """

        df = df.copy()

        # Setback index by 24h
        df.index -= pd.Timedelta(24, "h")

        # rename the columns to reflect the new index
        df = df.rename(
            columns={
                "Forecasted Load": "24h_later_forecast",
                "Actual Load": "24h_later_load",
            }
        )

        return df

    @staticmethod
    def _enforce_data_quality(df: pd.DataFrame) -> None:
        """Enforce the data quality of df.

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

        # Only keep rows below this threshold, as it seems the ENTSO-E sometimes logs extreme values
        upper_threshold = df["Actual Load"].quantile(q=0.999)
        mask = df["Actual Load"] < upper_threshold
        df = df[mask]

        return df

    @staticmethod
    def _force_1h_frequency(df: pd.DataFrame) -> pd.DataFrame:
        """Force a 1h-frequency, filling the gap with rows of NaN.

        Assumes that df.index.is_unique

        Args:
            df (pd.DataFrame): Dataframe with DatetimeIndex whose frequency should be 1h

        Returns:
            pd.DataFrame: df with a frequency of 1h, with rows of NaN where data was missing.
        """
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_unique

        return df.resample(rule=pd.Timedelta(1, "h")).min()

    @staticmethod
    def clean(df: pd.DataFrame, out_df_filepath: str) -> None:
        """Clean the dataframe df and dump the cleaned version to disk.

        Args:
            df (pd.DataFrame): Dirty dataframe
            out_df_filepath (str): Filesystem location where the cleaned dataframe will be dumped.
        """

        # Enfore data quality
        df = DataCleaner._enforce_data_quality(df)

        # Currently, the timestamp correponds to "in the next hour, this is the load"
        # whereas we want it to mean "the load 24h from this timestamp is"
        df = DataCleaner._format(df=df)

        # Enforce 1h frequency
        df = DataCleaner._force_1h_frequency(df=df)

        # Dump to output dataframe filepath
        Path(out_df_filepath).parent.mkdir(  # Ensure the folderpath exists
            parents=True, exist_ok=True
        )
        df.to_pickle(out_df_filepath)
