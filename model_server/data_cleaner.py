from pathlib import Path

import pandas as pd


class DataCleaner:
    """Class responsible for cleaning the data downloaded from the ENTSO-E servers."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def clean(in_df_filepath: str, out_df_filepath: str) -> None:
        """Clean the dataframe (.parquet) and dump the cleaned version to disk.

        Args:
            in_df_filepath (str): Filesystem location of the dirty dataframe (.parquet)
            out_df_filepath (str): Filesystem location where the cleaned dataframe will be dumped.
        """
        # Load dataframe to be cleaned
        df = pd.read_parquet(in_df_filepath)

        # Currently, the timestamp correponds to "in the next hour, this is the load"
        # whereas we want it to mean "the load 24h from this timestamp is"
        # Modify it so it fits
        df = df.set_index(
            df.index.to_series().apply(lambda x: x - pd.Timedelta(1, "d"))
        )  # Update the index
        df = df.rename(
            columns={  # rename the columns to reflect the new index
                "Forecasted Load": "24h_later_forecast",
                "Actual Load": "24h_later_load",
            }
        )

        # Dump to output dataframe filepath
        Path(out_df_filepath).parent.mkdir(  # Ensure the folderpath exists
            parents=True, exist_ok=True
        )
        df.to_parquet(out_df_filepath)