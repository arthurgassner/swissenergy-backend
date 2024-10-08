import numpy as np
import pandas as pd
import pytest

from model_server.model import Model


def test__train_predict__missing_query_ts():
    """Check whether a ValueError is raised when the Xy.index is missing the query_ts."""

    # given
    model = Model(n_estimators=10)
    Xy = pd.DataFrame(
        {"feature1": [], "feature2": [], "24h_later_load": []},
        index=pd.DatetimeIndex([]),
    )
    query_ts = pd.Timestamp("2024-10-01 00:00", tz="Europe/Zurich")

    # when-then
    with pytest.raises(ValueError):
        model._train_predict(Xy, query_ts)


def test__train_predict__expected_prediction():
    """Check whether a model's prediction is of the expected shape."""

    # given
    model = Model(n_estimators=10)
    Xy = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 1],
            "feature2": [1, 2, 3, 1],
            "feature2": [1, 2, 3, 1],
            "24h_later_load": [1, 2, 3, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01 01:00", tz="Europe/Zurich"),
                pd.Timestamp("2024-01-01 02:00", tz="Europe/Zurich"),
                pd.Timestamp("2024-01-01 03:00", tz="Europe/Zurich"),
                pd.Timestamp("2024-01-01 04:00", tz="Europe/Zurich"),
            ]
        ),
    )
    query_ts = pd.Timestamp("2024-01-01 04:00", tz="Europe/Zurich")

    # when
    yhat = model._train_predict(Xy, query_ts)

    # then
    assert yhat == 2.0  # as the model overfit the training set
