from datetime import timedelta
from typing import List

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


class PerformanceMeasurer:
    """Class responsible for measuring the performance of a time-series prediction model."""

    @staticmethod
    def _mape(
        y_true_col: str,
        y_pred_col: str,
        data: pd.DataFrame,
        timedeltas: List[timedelta],
    ) -> pd.Series:
        """Measure the Mean Absolute Percentage Error (MAPE) between the ground-truth and a prediction,
        for each period between the latest ts in data.index and spanning timedelta.

        Args:
            y_true_col (str): Ground-truth's column name in `data`
            y_pred_col (str): Prediction's column name in `data`
            data (pd.DataFrame): Dataframe containing the ground-truth and prediction, with a pd.DatetimeIndex
            timedeltas (List[timedelta]): Timedelta from which we should compute the MAPE,
                                          starting at the latest ts in `data`.

        Returns:
            pd.Series: Series containing the MAPE values -- under 'mape' -- for each timedelta.
                       The index of row corresponds to the starting timestamp from which the MAPE was computed.
        """

        # Check the input is as we expect
        assert isinstance(data.index, pd.DatetimeIndex)
        assert data.index.is_monotonic_decreasing
        assert data.index.is_unique
        assert y_true_col in data.columns
        assert y_pred_col in data.columns

        starting_ts_to_mape = {}
        for timedelta in sorted(timedeltas):
            curr_starting_ts = data.index.max() - timedelta
            curr_data = data[
                data.index.to_series().apply(lambda x: x >= curr_starting_ts)
            ]
            curr_mape = (
                mean_absolute_percentage_error(
                    y_true=curr_data[y_true_col],
                    y_pred=curr_data[y_pred_col],
                )
                * 100
            )
            starting_ts_to_mape[curr_starting_ts] = curr_mape

        return pd.DataFrame(
            {
                "mape": starting_ts_to_mape.values(),
            },
            index=pd.DatetimeIndex(starting_ts_to_mape.keys()),
        )