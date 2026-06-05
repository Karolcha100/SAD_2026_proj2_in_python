import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error



def select_n_components(
    X: np.ndarray,
    max_components: int = 10,
    random_state: int = 42,
) -> tuple[int, np.ndarray]:
    """
    Select the optimal number of GMM components using BIC.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix used to fit GMM.
    max_components : int
        Maximum number of components to evaluate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (best_n_components, bic_scores)
        best_n_components : int — number of components with lowest BIC.
        bic_scores : np.ndarray — BIC for each n in range(1, max_components+1).
    """
    bic_scores = []
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, random_state=random_state)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))

    best_n = int(np.argmin(bic_scores)) + 1
    return best_n, np.array(bic_scores)