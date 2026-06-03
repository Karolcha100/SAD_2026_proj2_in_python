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