import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


class PerformanceMeasurer:
    """Class responsible for measuring the performance of a time-series prediction model."""

    @staticmethod
    def mape(
        y_true_col: str, y_pred_col: str, data: pd.DataFrame, cutoff_ts: pd.Timestamp
    ) -> pd.Series:

        # Check the input is as we expect
        assert isinstance(data.index, pd.DatetimeIndex)
        assert data.index.is_monotonic_decreasing
        assert data.index.is_unique
        assert y_true_col in data.columns
        assert y_pred_col in data.columns

        data = data[data.index.to_series().apply(lambda x: x >= cutoff_ts)]

        starting_ts_to_mape = {}
        for curr_cutoff_ts in sorted(data.index.unique()):
            curr_mape = (
                mean_absolute_percentage_error(
                    y_true=data[y_true_col],
                    y_pred=data[y_pred_col],
                )
                * 100
            )
            starting_ts_to_mape[curr_cutoff_ts] = curr_mape

        return pd.DataFrame(
            {
                "mape": starting_ts_to_mape.values(),
            },
            index=pd.DatetimeIndex(starting_ts_to_mape.keys()),
        )
