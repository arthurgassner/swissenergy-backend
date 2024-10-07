from datetime import timedelta

import numpy as np
import pandas as pd

from model_server.performance_measurer import PerformanceMeasurer


def test__mape__perfect_prediction():
    """Check that the MAPE of a perfect prediction is 0.0."""

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
    timedeltas = [timedelta(days=3 + i) for i in range(5)]
    mape_df = PerformanceMeasurer._mape(
        "Actual Load",
        "Forecasted Load",
        data=df,
        timedeltas=timedeltas,
    )

    # then
    assert len(mape_df) == len(timedeltas)  # 5 timedeltas where given
    assert (mape_df["mape"] == 0.0).all()  # The predictions are always without error
