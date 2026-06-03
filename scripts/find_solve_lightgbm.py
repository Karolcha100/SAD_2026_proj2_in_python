import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error








def make_objective(
    X: pd.DataFrame,
    y: np.ndarray,
    kf: KFold,
) -> callable:
    """
    Create an Optuna objective function for LightGBM hyperparameter search.

    Parameters
    ----------
    X : pd.DataFrame
        Full training feature DataFrame.
    y : np.ndarray
        Target vector.
    kf : KFold
        Pre-initialised KFold splitter.

    Returns
    -------
    Callable
        Optuna objective that returns mean CV MSE.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "mse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

        fold_mse = []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)],
            )
            preds = model.predict(X_val)
            fold_mse.append(mean_squared_error(y_val, preds))

        return float(np.mean(fold_mse))

    return objective


def fit_final_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
) -> lgb.LGBMRegressor:
    """
    Fit a LightGBM model on the full training set with best hyperparameters.

    Parameters
    ----------
    X : pd.DataFrame
        Full training feature DataFrame.
    y : np.ndarray
        Full training target vector.
    params : dict
        Best hyperparameters from Optuna study.

    Returns
    -------
    lgb.LGBMRegressor
        Fitted model.
    """
    model = lgb.LGBMRegressor(
        objective="regression",
        verbosity=-1,
        **params,
    )
    model.fit(X, y)
    return model