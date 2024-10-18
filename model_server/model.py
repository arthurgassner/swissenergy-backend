from pathlib import Path
from typing import List, Optional

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm


class Model:
    """Class responsible for handling the training, inference and testing of a time-series prediction model."""

    def __init__(self, n_estimators: int) -> None:
        """Create a Model object, which encapsulates an LGBMRegressor with `n_estimators` estimators

        Args:
            n_estimators (int): Amount of estimators of the LGBMRegressor
        """
        # Create untrained-model and dump to disk
        self._model = lgb.LGBMRegressor(
            n_estimators=n_estimators, force_row_wise=True, verbose=-1
        )

    def _train_predict(self, Xy: pd.DataFrame, query_ts: pd.Timestamp) -> float:
        """Train a model on all the features whose index is BEFORE query_ts,
        and run an inference on the features EXACTLY AT query_ts.

        Args:
            Xy (pd.DataFrame): Dataframe containing the (features, target), where the target is '24h_later_load'
            query_ts (pd.Timestamp): Timestamp whose inference we are interested in

        Raises:
            ValueError: If the query_ts is missing from Xy's index,
                        i.e. the features for the timestamp of interest are missing.

        Returns:
            float: Predicted value for '24h_later_load'
        """

        assert isinstance(Xy.index, pd.DatetimeIndex)
        assert Xy.index.is_unique
        assert Xy.index.is_monotonic_increasing

        # Extract the serving Xy
        if not query_ts in Xy.index:
            raise ValueError(f"Query timestamp {query_ts} is missing from Xy's index.")
        X_serving = Xy.loc[[query_ts]].drop(columns=["24h_later_load"])

        # Prepare training data
        Xy = Xy.dropna(subset=("24h_later_load"))
        Xy = Xy[Xy.index < query_ts]  # Only train on data strictly before the ts
        X, y = Xy.drop(columns=["24h_later_load"]), Xy["24h_later_load"]

        # Train the model
        self._model.fit(X, y)

        # Predict
        return float(self._model.predict(X_serving)[0])

    def train_predict(
        self,
        Xy: pd.DataFrame,
        query_timestamps: List[pd.Timestamp],
        out_yhat_filepath: Optional[str] = None,
        already_computed_yhat_filepath: Optional[str] = None,
    ) -> pd.Series:
        """Train one model per query_ts in `query_timestamps`.
        Each model will only be training on the features in Xy available strictly BEFORE said query_ts.
        The features EXACTLY AT the query_ts will be used to predict the `24h_later_load`.

        Args:
            Xy (pd.DataFrame): Dataframe containing the (features, target), where the target is '24h_later_load'
            query_timestamps (List[pd.Timestamp]): Timestamps whose inference we are interested in
            out_yhat_filepath (str, optional): Where to save the predictions.
            already_computed_yhat_filepath (str, optional): Filepath of already-computed yhats.
                                                            If given, the timestamps whose yhat exists will not be recomputed.

        Returns:
            pd.Series: Dataframe with the predicted values under the column 'predicted_24h_later_load'.
                       The index corresponds to the query_timestamps.
        """

        already_computed_yhat = None
        already_computed_timestamps = set([])
        if already_computed_yhat_filepath:
            if Path(already_computed_yhat_filepath).is_file():
                already_computed_yhat = pd.read_pickle(already_computed_yhat_filepath)
                already_computed_timestamps = set(already_computed_yhat.index)

        predicted_values = []
        for query_ts in tqdm(query_timestamps):
            if query_ts in already_computed_timestamps:
                predicted_values.append(already_computed_yhat.loc[query_ts])
            else:
                predicted_values.append(self._train_predict(Xy, query_ts))

        yhat = pd.DataFrame(
            {"predicted_24h_later_load": predicted_values},
            index=pd.DatetimeIndex(query_timestamps),
        )

        if out_yhat_filepath:
            Path(out_yhat_filepath).parent.mkdir(parents=True, exist_ok=True)
            yhat.to_pickle(out_yhat_filepath)

        return yhat
