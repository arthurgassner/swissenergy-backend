import pandas as pd
import numpy as np
from typing import Callable

class FeatureExtractor:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _compute_timedelta_ago_load(df: pd.DataFrame, timedelta: pd.Timedelta) -> pd.Series:
        """For each timestamps in the index, compute the load timedelta ago 

        Assume that each row's index is the current timestamp.
        That is, when we say "timedelta ago from now", we mean "timedelta ago from this timestamp".

        df (pd.DataFrame): Dataframe containing the `24h_later_load`, whose index refers to now when saying "24h later".
        timedelta (pd.Timedelta): Time delta of interest, i.e. how long ago do we want the load ?
        """
        
        assert '24h_later_load' in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

        ts_to_24h_later_load = df['24h_later_load'].to_dict()
        return df.index.to_series().apply(lambda x: ts_to_24h_later_load.get(x - pd.Timedelta(24, 'h') - timedelta))
    
    @staticmethod
    def _compute_stat(df: pd.DataFrame, current_time: pd.Timestamp, timedelta: pd.Timedelta, stat: Callable) -> float:
        start_time = current_time -  pd.Timedelta(24, 'h') - timedelta
        end_time = current_time - pd.Timedelta(24, 'h') 
        
        relevant_data = df.loc[start_time:end_time, '24h_later_load']
    
        if len(relevant_data) == 0:
            return np.nan
    
        return stat(relevant_data.values)

    @staticmethod
    def _compute_stats(df: pd.DataFrame, timedelta: pd.Timedelta, stat: Callable) -> pd.Series:
        """For each timestamps in the index, compute the stat over the date comprised between now and timedelta ago. 

        Assume that each row's index is the current timestamp.
        That is, when we say "timedelta ago from now", we mean "timedelta ago from this timestamp".

        df (pd.DataFrame): Dataframe containing the `24h_later_load`, whose index refers to now when saying "24h later".
        timedelta (pd.Timedelta): Time delta of interest, i.e. how long ago do we want the statistics calculation to start ?
        stats (list[func]): Functions of the statistic to compute
        """

        assert '24h_later_load' in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

        return df.index.to_series().apply(lambda x: FeatureExtractor._compute_stat(df, x, timedelta, stat))
    
    @staticmethod
    def extract_features(in_df_filepath: str, out_df_filepath: str) -> None:
        # Load data
        df = pd.read_parquet(in_df_filepath, columns=['24h_later_load'])
        
        # Enrich the df with the datetime attributes
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['hour'] = df.index.hour
        df['weekday'] = df.index.weekday

        # Enrich each row with previous loads: 1h-ago, 2h-ago, 3h-ago, 24h-ago, 7days-ago
        df['1h_ago_load'] = FeatureExtractor._compute_timedelta_ago_load(df, timedelta=pd.Timedelta(1, 'h'))
        df['2h_ago_load'] = FeatureExtractor._compute_timedelta_ago_load(df, timedelta=pd.Timedelta(2, 'h'))
        df['3h_ago_load'] = FeatureExtractor._compute_timedelta_ago_load(df, timedelta=pd.Timedelta(3, 'h'))
        df['24h_ago_load'] = FeatureExtractor._compute_timedelta_ago_load(df, timedelta=pd.Timedelta(24, 'h'))
        df['7d_ago_load'] = FeatureExtractor._compute_timedelta_ago_load(df, timedelta=pd.Timedelta(7, 'd'))

        # Enrich the df with statistics
        df['8h_min'] = FeatureExtractor._compute_stats(df, pd.Timedelta(8, 'h'), np.min)
        df['8h_max'] = FeatureExtractor._compute_stats(df, pd.Timedelta(8, 'h'), np.max)
        df['8h_median'] = FeatureExtractor._compute_stats(df, pd.Timedelta(8, 'h'), np.median)

        df['24h_min'] = FeatureExtractor._compute_stats(df, pd.Timedelta(24, 'h'), np.min)
        df['24h_max'] = FeatureExtractor._compute_stats(df, pd.Timedelta(24, 'h'), np.max)
        df['24h_median'] = FeatureExtractor._compute_stats(df, pd.Timedelta(24, 'h'), np.median)

        df['7d_min'] = FeatureExtractor._compute_stats(df, pd.Timedelta(7, 'd'), np.min)
        df['7d_max'] = FeatureExtractor._compute_stats(df, pd.Timedelta(7, 'd'), np.max)
        df['7d_median'] = FeatureExtractor._compute_stats(df, pd.Timedelta(7, 'd'), np.median)

        # Drop rows for which we could not compute features
        df = df.dropna(subset=[
            'year', 'month', 'day', 'hour', 'weekday', # daily attributes
            '1h_ago_load', '2h_ago_load', '3h_ago_load', '24h_ago_load', '7d_ago_load', # past loads
            '8h_min', '8h_max', '8h_median', '24h_min', '24h_max', '24h_median', '7d_min', '7d_max', '7d_median', # statistics
        ])  
        
        # Dump to output df
        df.to_parquet(out_df_filepath)
