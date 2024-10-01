from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


class FeatureExtractor:
    """Class responsible for extracted features out of the cleaned data."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _n_hours_ago_load(df: pd.DataFrame, n_hours: int) -> pd.Series:
        """For each timestamps in the index, compute the load n_hours ago

        Assume that each row's index is the current timestamp.
        That is, when we say "timedelta ago from now", we mean "timedelta ago from this timestamp".

        Args:
            df (pd.DataFrame): Dataframe containing the `24h_later_load`, whose index refers to "now" when saying "24h later".
            n_hours (int): How many hours ago is the load of interest ?

        Returns:
            pd.Series: Series whose index is the same as `df`, and whose values are the loads n_hours ago from their index.
        """

        assert "24h_later_load" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.freq == "h"

        return df["24h_later_load"].shift(24 + n_hours)

    @staticmethod
    def _rolling_window(df: pd.DataFrame, n_hours: int, stat: Callable) -> pd.Series:
        """For each timestamps in the index, compute the `stat` over the date comprised between that timestamp and `timedelta` ago.

        Args:
            df (pd.DataFrame): Dataframe containing the `24h_later_load`, whose index refers to "now" when saying "24h later".
            n_hours (int): Over how many hours should we compute the statistics
            stat (Callable): Statistic function

        Returns:
            pd.Series: Series whose index is the same as `df`, and whose values are the statistics computed over `n_hours` hours.
        """

        assert "24h_later_load" in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.freq == "h"

        # Compute the last-hour load for each row
        last_hour_loads = FeatureExtractor._n_hours_ago_load(df, n_hours=1)
        return last_hour_loads.rolling(window=n_hours).apply(stat)

    @staticmethod
    def extract_features(in_df_filepath: str, out_df_filepath: str) -> None:
        """Extract the features.

        Args:
            in_df_filepath (str): Filepath of the dataframe whose features must be extracted (.pickle)
            out_df_filepath (str): Filepath where to dump the extracted features (.pickle)
        """

        # Load data
        df = pd.read_pickle(in_df_filepath)[["24h_later_load"]]

        # Enrich the df with the datetime attributes
        df["year"] = df.index.year
        df["month"] = df.index.month
        df["day"] = df.index.day
        df["hour"] = df.index.hour
        df["weekday"] = df.index.weekday

        # Enrich each row with previous loads: 1h-ago, 2h-ago, 3h-ago, 24h-ago, 7days-ago
        df["1h_ago_load"] = FeatureExtractor._n_hours_ago_load(df, n_hours=1)
        df["2h_ago_load"] = FeatureExtractor._n_hours_ago_load(df, n_hours=2)
        df["3h_ago_load"] = FeatureExtractor._n_hours_ago_load(df, n_hours=3)
        df["24h_ago_load"] = FeatureExtractor._n_hours_ago_load(df, n_hours=24)
        df["7d_ago_load"] = FeatureExtractor._n_hours_ago_load(df, n_hours=7 * 24)

        # Enrich the df with statistics
        df["8h_min"] = FeatureExtractor._rolling_window(df, n_hours=8, stat=np.min)
        df["8h_max"] = FeatureExtractor._rolling_window(df, n_hours=8, stat=np.max)
        df["8h_median"] = FeatureExtractor._rolling_window(
            df, n_hours=8, stat=np.median
        )

        df["24h_min"] = FeatureExtractor._rolling_window(df, n_hours=24, stat=np.min)
        df["24h_max"] = FeatureExtractor._rolling_window(df, n_hours=24, stat=np.max)
        df["24h_median"] = FeatureExtractor._rolling_window(
            df, n_hours=24, stat=np.median
        )

        df["7d_min"] = FeatureExtractor._rolling_window(df, n_hours=7, stat=np.min)
        df["7d_max"] = FeatureExtractor._rolling_window(df, n_hours=7, stat=np.max)
        df["7d_median"] = FeatureExtractor._rolling_window(
            df, n_hours=7, stat=np.median
        )

        # Drop rows for which we could not compute features
        df = df.dropna(
            subset=[
                "year",
                "month",
                "day",
                "hour",
                "weekday",  # daily attributes
                "1h_ago_load",
                "2h_ago_load",
                "3h_ago_load",
                "24h_ago_load",
                "7d_ago_load",  # past loads
                "8h_min",
                "8h_max",
                "8h_median",
                "24h_min",
                "24h_max",
                "24h_median",
                "7d_min",
                "7d_max",
                "7d_median",  # statistics
            ]
        )

        # Dump to output df
        Path(out_df_filepath).parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the folderpath exists
        df.to_pickle(out_df_filepath)
