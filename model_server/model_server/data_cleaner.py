import pandas as pd
from pathlib import Path

class DataCleaner:
    def __init__(self) -> None:
        pass

    @staticmethod
    def clean(in_df_filepath: str, out_df_filepath: str) -> None:
        # Load dataframe to be cleaned
        df = pd.read_parquet(in_df_filepath)

        # Currently, the timestamp correponds to "in the next hour, this is the load"
        # whereas we want it to mean "the load 24h from this timestamp is"
        # Modify it so it fits
        df = df.set_index(df.index.to_series().apply(lambda x: x - pd.Timedelta(1, 'd'))) # Update the index
        df = df.rename(columns={ # rename the columns to reflect the new index
            'Forecasted Load': '24h_later_forecast',
            'Actual Load': '24h_later_load',
        })

        # Dump to output dataframe filepath
        Path(out_df_filepath).parent.mkdir(parents=True, exist_ok=True) # Ensure the folderpath exists 
        df.to_parquet(out_df_filepath)