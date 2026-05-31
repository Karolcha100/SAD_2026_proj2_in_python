import pandas as pd
import pathlib as pth
from datetime import datetime
from zoneinfo import ZoneInfo


def save_submission_csv(
        df: pd.DataFrame,
        path_to_submissions: pth.Path = pth.Path("outputs-submissions/submissions")
    ) -> None:
    """
    Save a DataFrame as a CSV submission file with a Warsaw-time timestamp.

    :param df: DataFrame to save.
    :param path_to_submissions: Directory where the submission file will be saved.
    """
    warsaw_tz = ZoneInfo("Europe/Warsaw")
    submission_date = datetime.now(tz=warsaw_tz).strftime("%Y-%m-%d___%H:%M:%S")

    file_count = sum(1 for f in path_to_submissions.iterdir() if f.is_file())

    df.to_csv(path_to_submissions / f"{submission_date}___{file_count}.csv", index=False)


def save_model