from pathlib import Path

import pandas as pd


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
    def clean(in_df_filepath: str, out_df_filepath: str) -> None:
        """Clean the dataframe (.parquet) and dump the cleaned version to disk.

        Args:
            in_df_filepath (str): Filesystem location of the dirty dataframe (.parquet)
            out_df_filepath (str): Filesystem location where the cleaned dataframe will be dumped.
        """
        # Load dataframe to be cleaned
        df = pd.read_parquet(in_df_filepath)

        # Currently, the timestamp correponds to "in the next hour, this is the load"
        # whereas we want it to mean "the load 24h from this timestamp is"
        df = DataCleaner._format(df=df)

        # Enforce 1h frequency
        df = DataCleaner._force_1h_frequency(df=df)

        # Dump to output dataframe filepath
        Path(out_df_filepath).parent.mkdir(  # Ensure the folderpath exists
            parents=True, exist_ok=True
        )
        df.to_parquet(out_df_filepath)
