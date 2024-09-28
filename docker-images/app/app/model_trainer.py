import pandas as pd
from tqdm import tqdm
from typing import Tuple
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd

class ModelTrainer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def backtest(Xy_filepath: str, model: lgb.LGBMRegressor, starting_ts: pd.Timestamp, use_every_nth_ts: int = 1) -> Tuple[pd.DataFrame, float]:
        """Backtest the model, by starting at the `starting_ts` timestamp.
        
        Each iteration during the foreward-walk, add `use_every_nth_ts` rows.
        """

        Xy = pd.read_parquet(Xy_filepath)

        assert '24h_later_load' in Xy.columns

        # Only train on data for which we have the target
        Xy = Xy.dropna(subset=('24h_later_load'))

        cutoff_timestamps = Xy[Xy.index >= starting_ts].index.to_list()
        
        cutoff_ts_to_y = {}
        for cutoff_ts in tqdm(cutoff_timestamps[::use_every_nth_ts]):    
            
            # Split train:val
            Xy_train = Xy[Xy.index < cutoff_ts]
            Xy_val = Xy[Xy.index == cutoff_ts]
            
            # Split X,y
            X_train, y_train = Xy_train.drop(columns=['24h_later_load']), Xy_train['24h_later_load']
            X_val, y_val = Xy_val.drop(columns=['24h_later_load']), Xy_val['24h_later_load']
        
            # Train model
            model.fit(X_train, y_train)
        
            # Compute prediction in 24h
            yhat_val = model.predict(X_val) 
        
            cutoff_ts_to_y[cutoff_ts] = (yhat_val[0], y_val.iloc[0])
            
        results_df = pd.DataFrame({
            'cutoff_ts': cutoff_ts_to_y.keys(), 
            'predicted_24h_later_load': [e[0] for e in cutoff_ts_to_y.values()], 
            '24h_later_load': [e[1] for e in cutoff_ts_to_y.values()]
        })

        mape = mean_absolute_percentage_error(results_df['24h_later_load'], results_df['predicted_24h_later_load']) * 100

        return results_df, mape