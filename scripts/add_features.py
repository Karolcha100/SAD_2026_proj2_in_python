import numpy as np
import pandas as pd




def get_regr_features_cols() -> list:
    return [
        "temp", "atemp", "hum", "wind",
        "temp_sq", "hum_sq", "wind_sq",
        "temp_x_hum", "temp_x_wind", "atemp_x_hum",
        "month", "dayofweek", "weekend", "dayofyear",
        "sin_doy_1", "cos_doy_1",
        "sin_doy_2", "cos_doy_2",
        "sin_doy_3", "cos_doy_3",
        "wcond_2", "wcond_3",
    ]


def add_calendar_features_regr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-derived features from the date column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a parsed 'date' column of dtype datetime64.

    Returns
    -------
    pd.DataFrame
        DataFrame extended with calendar features.
    """
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear

    # Fourier terms to capture smooth annual seasonality
    for k in [1, 2, 3]:
        df[f"sin_doy_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / 365)
        df[f"cos_doy_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / 365)

    return df


def add_weather_features_regr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered weather interaction and nonlinear features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: temp, atemp, hum, wind, wcond.

    Returns
    -------
    pd.DataFrame
        DataFrame extended with weather features.
    """
    df = df.copy()
    df["temp_sq"] = df["temp"] ** 2
    df["hum_sq"] = df["hum"] ** 2
    df["wind_sq"] = df["wind"] ** 2
    df["temp_x_hum"] = df["temp"] * df["hum"]
    df["temp_x_wind"] = df["temp"] * df["wind"]
    df["atemp_x_hum"] = df["atemp"] * df["hum"]

    # One-hot encode wcond (drop first to avoid multicollinearity)
    wcond_dummies = pd.get_dummies(df["wcond"], prefix="wcond", drop_first=True)
    df = pd.concat([df, wcond_dummies], axis=1)

    return df


def get_lgb_features_cols() -> list[str]:
    return [
        "temp", "atemp", "hum", "wind", "wcond",
        "temp_sq", "hum_sq", "wind_sq",
        "temp_x_hum", "temp_x_wind", "atemp_x_hum", "temp_atemp_diff",
        "month", "dayofweek", "weekend", "dayofyear", "quarter",
        "sin_doy_1", "cos_doy_1",
        "sin_doy_2", "cos_doy_2",
        "sin_doy_3", "cos_doy_3",
    ]


def add_calendar_features_lgb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-derived features from the date column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a parsed 'date' column of dtype datetime64.

    Returns
    -------
    pd.DataFrame
        DataFrame extended with calendar features.
    """
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter

    for k in [1, 2, 3]:
        df[f"sin_doy_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / 365)
        df[f"cos_doy_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / 365)

    return df


def add_weather_features_lgb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered weather interaction and nonlinear features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: temp, atemp, hum, wind, wcond.

    Returns
    -------
    pd.DataFrame
        DataFrame extended with weather features.
    """
    df = df.copy()
    df["temp_sq"] = df["temp"] ** 2
    df["hum_sq"] = df["hum"] ** 2
    df["wind_sq"] = df["wind"] ** 2
    df["temp_x_hum"] = df["temp"] * df["hum"]
    df["temp_x_wind"] = df["temp"] * df["wind"]
    df["atemp_x_hum"] = df["atemp"] * df["hum"]
    df["temp_atemp_diff"] = df["atemp"] - df["temp"]

    # wcond as categorical (LightGBM handles it natively)
    df["wcond"] = df["wcond"].astype("category")

    return df






import pandas as pd
import numpy as np


def add_fourier_features(df: pd.DataFrame, n_harmonics: int = 3) -> pd.DataFrame:
    """Add Fourier seasonality terms for annual cycle.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``dayofyear`` column.
    n_harmonics : int
        Number of harmonics to generate.

    Returns
    -------
    pd.DataFrame
        DataFrame with added sin/cos Fourier columns.
    """
    for k in range(1, n_harmonics + 1):
        df[f"sin_doy_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / 365)
        df[f"cos_doy_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / 365)
    return df


def build_features(
        df: pd.DataFrame,
        to_drop_at_end: list[str]|None = None
    ) -> pd.DataFrame:
    """Build full feature matrix from raw bike sharing data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with columns: date, temp, atemp, hum, wind, wcond.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features, without target column.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter
    df["weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["doy_normalized"] = df["dayofyear"] / 365

    df["temp_sq"] = df["temp"] ** 2
    df["temp_cube"] = df["temp"] ** 3
    df["hum_sq"] = df["hum"] ** 2
    df["wind_sq"] = df["wind"] ** 2
    df["temp_atemp_diff"] = df["temp"] - df["atemp"]

    df["temp_x_hum"] = df["temp"] * df["hum"]
    df["temp_x_wind"] = df["temp"] * df["wind"]
    df["atemp_x_hum"] = df["atemp"] * df["hum"]
    df["hum_x_wind"] = df["hum"] * df["wind"]
    df["wcond_x_temp"] = df["wcond"] * df["temp"]

    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)

    wcond_dummies = pd.get_dummies(df["wcond"], prefix="wcond", drop_first=True)
    df = pd.concat([df, wcond_dummies], axis=1)

    df = add_fourier_features(df)

    drop_cols = ["date", "wcond", "dayofyear"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if to_drop_at_end is not None:
        df = df.drop(columns=to_drop_at_end, errors='ignore')
    return df


def build_features_without_dates(
        df: pd.DataFrame,
        to_drop_at_end: list[str]|None = None
    ) -> pd.DataFrame:
    """Build full feature matrix from raw bike sharing data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with columns: date, temp, atemp, hum, wind, wcond.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features, without target column.
    """
    df = df.drop(columns=["date"]).copy()

    df["temp_sq"] = df["temp"] ** 2
    df["temp_cube"] = df["temp"] ** 3
    df["hum_sq"] = df["hum"] ** 2
    df["wind_sq"] = df["wind"] ** 2
    df["temp_atemp_diff"] = df["temp"] - df["atemp"]

    df["temp_x_hum"] = df["temp"] * df["hum"]
    df["temp_x_wind"] = df["temp"] * df["wind"]
    df["atemp_x_hum"] = df["atemp"] * df["hum"]
    df["hum_x_wind"] = df["hum"] * df["wind"]
    df["wcond_x_temp"] = df["wcond"] * df["temp"]

    wcond_dummies = pd.get_dummies(df["wcond"], prefix="wcond", drop_first=True)
    df = pd.concat([df, wcond_dummies], axis=1)


    if to_drop_at_end is not None:
        df = df.drop(columns=to_drop_at_end, errors='ignore')
    return df