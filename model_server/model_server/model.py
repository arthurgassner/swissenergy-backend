import pandas as pd
from tqdm import tqdm
from typing import Tuple
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import joblib
from pathlib import Path


class Model:
    def __init__(self, model_filepath: str) -> None:
        self._model_filepath = Path(model_filepath)
        self._model_filepath.parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the folderpath exists

    def train(self, Xy_filepath: str, n_estimators: int) -> None:
        # Prepare training data
        Xy = pd.read_parquet(Xy_filepath)
        assert "24h_later_load" in Xy.columns
        Xy = Xy.dropna(
            subset=("24h_later_load")
        )  # Only train on data for which we have the target
        X, y = Xy.drop(columns=["24h_later_load"]), Xy["24h_later_load"]

        # Train model
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators, force_row_wise=True, verbose=0
        )
        model.fit(X, y)

        # Dump to disk
        joblib.dump(model, self._model_filepath)

    def backtest(
        self, Xy_filepath: str, starting_ts: pd.Timestamp, use_every_nth_ts: int = 1
    ) -> Tuple[pd.DataFrame, float]:
        """Backtest the model, by starting at the `starting_ts` timestamp.

        Each iteration during the foreward-walk, add `use_every_nth_ts` rows.
        """

        # Only load the model if it exists
        if not self._model_filepath.is_file():
            raise Exception(f"Missing model file at {self._model_filepath}")
        model = joblib.load(self._model_filepath)  # Load model

        # Prepare training data
        Xy = pd.read_parquet(Xy_filepath)
        assert "24h_later_load" in Xy.columns
        Xy = Xy.dropna(
            subset=("24h_later_load")
        )  # Only train on data for which we have the target

        cutoff_timestamps = Xy[Xy.index >= starting_ts].index.to_list()

        cutoff_ts_to_y = {}
        for cutoff_ts in tqdm(cutoff_timestamps[::use_every_nth_ts]):

            # Split train:val
            Xy_train = Xy[Xy.index < cutoff_ts]
            Xy_val = Xy[Xy.index == cutoff_ts]

            # Split X,y
            X_train, y_train = (
                Xy_train.drop(columns=["24h_later_load"]),
                Xy_train["24h_later_load"],
            )
            X_val, y_val = (
                Xy_val.drop(columns=["24h_later_load"]),
                Xy_val["24h_later_load"],
            )

            # Train model
            model.fit(X_train, y_train)

            # Compute prediction in 24h
            yhat_val = model.predict(X_val)

            cutoff_ts_to_y[cutoff_ts] = (yhat_val[0], y_val.iloc[0])

        results_df = pd.DataFrame(
            {
                "cutoff_ts": cutoff_ts_to_y.keys(),
                "predicted_24h_later_load": [e[0] for e in cutoff_ts_to_y.values()],
                "24h_later_load": [e[1] for e in cutoff_ts_to_y.values()],
            }
        )

        mape = (
            mean_absolute_percentage_error(
                results_df["24h_later_load"], results_df["predicted_24h_later_load"]
            )
            * 100
        )

        return results_df, mape

    def predict(self, in_df_filepath: str, out_yhat_filepath: str) -> pd.Series:
        # Only load the model if it exists
        if not self._model_filepath.is_file():
            raise Exception(f"Missing model file at {self._model_filepath}")
        model = joblib.load(self._model_filepath)  # Load model

        # Load data
        df = pd.read_parquet(in_df_filepath)

        # Figure out the timestamps of the next 24h for which we have features
        starting_ts = df.index.max() - pd.Timedelta(1, "d")
        timestamps = list(df.index[df.index > starting_ts])

        # Run predictions
        yhat = pd.DataFrame(
            {
                "predicted_24h_later_load": model.predict(
                    df.loc[timestamps].drop(columns=["24h_later_load"])
                )
            },
            index=timestamps,
        )

        # Dump to output df
        Path(out_yhat_filepath).parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure the folderpath exists
        yhat.to_parquet(out_yhat_filepath)

        return yhat
