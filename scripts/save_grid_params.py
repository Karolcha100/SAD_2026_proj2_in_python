import pandas as pd
import pathlib as pth
from datetime import datetime
from zoneinfo import ZoneInfo


def save_random_forest(
        df: pd.DataFrame,
        path_to_submissions: pth.Path = pth.Path("params-random-forest")
    ) -> None:
    """
    Save a DataFrame as a CSV random forest file with a grid search outputs.

    :param df: DataFrame to save.
    :param path_to_submissions: Directory where the submission file will be saved.
    """

    file_count = sum(1 for f in path_to_submissions.iterdir() if f.is_file())

    df.to_csv(path_to_submissions / f"random_forest_{file_count}.csv")