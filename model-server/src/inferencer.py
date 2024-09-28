import pandas as pd
import joblib
import lightgbm as lgb

class Inferencer:
    def __init__(self, model_filepath: str) -> None:
        self._model = joblib.load(model_filepath)

    def predict(self, in_df_filepath: str) -> pd.Series:
        # Load data
        df = pd.read_parquet(in_df_filepath)

        # Figure out the timestamps of the next 24h for which we have features
        starting_ts = df.index.max() - pd.Timedelta(1, 'd')
        timestamps = list(df.index[df.index > starting_ts])

        # Run predictions
        yhat = pd.DataFrame({
            'predicted_24h_later_load': self._model.predict(df.loc[timestamps].drop(columns=['24h_later_load']))
        }, index=timestamps)

        return yhat
    
