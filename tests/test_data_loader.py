import pandas as pd
import pytest

from model_server.data_loader import DataLoader


def test__get_latest_ts_with_actual_load__empty():
    """Empty dataframe should return 2014-01-01"""
    excepted_ts = pd.Timestamp("20140101", tz="Europe/Zurich")
    assert excepted_ts == DataLoader.get_latest_ts_with_actual_load(df=pd.DataFrame())
