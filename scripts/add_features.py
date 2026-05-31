import numpy as np
import pandas as pd







def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
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


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
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