from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np








def cross_validate_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    kf: KFold,
) -> dict:
    """
    Run 5-fold cross-validation for Ridge regression over a grid of alpha values.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target vector (n_samples,).
    alphas : np.ndarray
        Array of regularisation strengths (lambda values) to evaluate.
    kf : KFold
        Pre-initialised KFold splitter (ensures identical folds across models).

    Returns
    -------
    dict with keys:
        'alphas'        : np.ndarray  — evaluated alpha values
        'fold_mse'      : np.ndarray  — shape (n_alphas, n_folds) per-fold MSE
        'cv_mse'        : np.ndarray  — shape (n_alphas,) mean CV MSE
        'best_alpha'    : float       — alpha with lowest mean CV MSE
        'best_cv_mse'   : float       — corresponding mean CV MSE
        'best_cv_rmse'  : float       — RMSE at best alpha
    """
    n_folds = kf.get_n_splits()
    fold_mse = np.zeros((len(alphas), n_folds))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"[Fold]: {fold_idx}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        for a_idx, alpha in enumerate(alphas):
            model = Ridge(alpha=alpha)
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_val_s)
            fold_mse[a_idx, fold_idx] = mean_squared_error(y_val, preds)

    cv_mse = fold_mse.mean(axis=1)
    best_idx = int(np.argmin(cv_mse))

    return {
        "alphas": alphas,
        "fold_mse": fold_mse,
        "cv_mse": cv_mse,
        "best_alpha": float(alphas[best_idx]),
        "best_cv_mse": float(cv_mse[best_idx]),
        "best_cv_rmse": float(np.sqrt(cv_mse[best_idx])),
    }



def fit_final_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> tuple[Ridge, StandardScaler]:
    """
    Fit a Ridge model on the full training set.

    Parameters
    ----------
    X : np.ndarray
        Full training feature matrix.
    y : np.ndarray
        Full training target vector.
    alpha : float
        Regularisation strength (best alpha from CV).

    Returns
    -------
    tuple of (fitted Ridge model, fitted StandardScaler)
    """
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=alpha)
    model.fit(X_s, y)
    return model, scaler