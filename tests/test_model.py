import numpy as np
import pandas as pd
import pytest

from model_server.model import Model


def test__train_predict__missing_query_ts():
    """Check whether a ValueError is raised when the Xy.index is missing the query_ts."""

    # given
    model = Model(n_estimators=10)

    # when-then
    Xy = pd.DataFrame(
        {"feature1": [], "feature2": [], "24h_later_load": []},
        index=pd.DatetimeIndex([]),
    )
    query_ts = pd.Timestamp("2024-10-01 00:00", tz="Europe/Zurich")
    with pytest.raises(ValueError):
        model._train_predict(Xy, query_ts)
