import numpy as np
import pandas as pd
import pytest

from model_server.data_cleaner import DataCleaner


def test__format():
    """Formatting a dateframe sets back its index by 24 and renames its columns."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [7890.0, np.nan, np.nan],
            "Actual Load": [np.nan, 7890.0, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # When
    formatted_df = DataCleaner._format(df=df)

    # Then
    assert len(formatted_df.columns) == 2  # 2 columns
    assert (
        formatted_df.columns[0] == "24h_later_forecast"
        and formatted_df.columns[1] == "24h_later_load"
    )
    assert (formatted_df.dtypes == "float64").all()  # correct dtype
    assert isinstance(formatted_df.index, pd.DatetimeIndex)
    assert (
        formatted_df.index.dtype == "datetime64[ns, Europe/Zurich]"
    )  # correct timezone
    assert (
        df.index - formatted_df.index == pd.Timedelta(24, "h")
    ).all()  # -24h delay post-formatting
