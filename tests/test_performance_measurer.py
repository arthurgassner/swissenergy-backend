import numpy as np
import pandas as pd

from model_server.performance_measurer import PerformanceMeasurer


def test_mape__perfect_prediction():
    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [101.0, 202.0, 303.0],
            "Actual Load": [101.0, 202.0, 303.0],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240301 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240201 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when
    mape_df = PerformanceMeasurer.mape(
        "Actual Load",
        "Forecasted Load",
        data=df,
        cutoff_ts=pd.Timestamp("20240101 00:00", tz="Europe/Zurich"),
    )

    # then
    assert len(mape_df) == 3  # 3 predictions fell after the cutoff ts
    assert (mape_df["mape"] == 0.0).all()  # The predictions are always without error
