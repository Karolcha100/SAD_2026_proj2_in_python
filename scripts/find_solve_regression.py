from itertools import product
from typing import Any

from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np


def cross_validate_regression(
    X: np.ndarray,
    y: np.ndarray,
    regressor_class: type[RegressorMixin],
    param_grid: dict[str, list[Any]],
    kf: KFold,
) -> dict:
    """
    Run k-fold cross-validation for any sklearn regressor over a parameter grid.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target vector (n_samples,).
    regressor_class : type[RegressorMixin]
        Sklearn-compatible regressor class (e.g. Ridge, ElasticNet).
    param_grid : dict[str, list[Any]]
        Mapping of parameter names to lists of values to evaluate.
        All combinations are tested (full Cartesian product).
        Example: {'alpha': [0.1, 1.0], 'l1_ratio': [0.3, 0.7]}
    kf : KFold
        Pre-initialised KFold splitter (ensures identical folds across models).

    Returns
    -------
    dict with keys:
        'param_combinations' : list[dict]  — all evaluated parameter combinations
        'fold_mse'           : np.ndarray  — shape (n_combinations, n_folds)
        'cv_mse'             : np.ndarray  — shape (n_combinations,) mean CV MSE
        'best_params'        : dict        — parameter combination with lowest CV MSE
        'best_cv_mse'        : float       — corresponding mean CV MSE
        'best_cv_rmse'       : float       — RMSE at best parameter combination
    """
    param_names = list(param_grid.keys())
    param_combinations = [
        dict(zip(param_names, values))
        for values in product(*param_grid.values())
    ]

    n_folds = kf.get_n_splits()
    fold_mse = np.zeros((len(param_combinations), n_folds))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"[Fold]: {fold_idx}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        for p_idx, params in enumerate(param_combinations):
            model = regressor_class(**params)
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_val_s)
            fold_mse[p_idx, fold_idx] = mean_squared_error(y_val, preds)

    cv_mse = fold_mse.mean(axis=1)
    best_idx = int(np.argmin(cv_mse))

    return {
        "param_combinations": param_combinations,
        "fold_mse": fold_mse,
        "cv_mse": cv_mse,
        "best_params": param_combinations[best_idx],
        "best_cv_mse": float(cv_mse[best_idx]),
        "best_cv_rmse": float(np.sqrt(cv_mse[best_idx])),
    }


def fit_final_regression(
    X: np.ndarray,
    y: np.ndarray,
    regressor_class: type[RegressorMixin],
    params: dict[str, Any],
) -> tuple[RegressorMixin, StandardScaler]:
    """
    Fit a regressor on the full training set.

    Parameters
    ----------
    X : np.ndarray
        Full training feature matrix.
    y : np.ndarray
        Full training target vector.
    regressor_class : type[RegressorMixin]
        Sklearn-compatible regressor class (e.g. Ridge, ElasticNet).
    params : dict[str, Any]
        Parameters passed to the regressor constructor (e.g. best_params from CV).

    Returns
    -------
    tuple of (fitted regressor, fitted StandardScaler)
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = regressor_class(**params)
    model.fit(X_s, y)
    return model, scaler