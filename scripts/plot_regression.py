import itertools
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def plot_cv_results(
    param_grid: dict[str, list[Any]],
    cv_results: dict,
    regressor_name: str = "Regressor",
) -> None:
    """
    Plot cross-validation RMSE results from cross_validate_regression.

    For 1 parameter: 2D line plot (parameter vs CV RMSE).
    For 2 parameters: single 3D scatter plot (param1, param2, CV RMSE).
    For 3+ parameters: one 3D scatter plot per pair of parameters,
        marginalising over the remaining ones by taking the minimum CV RMSE.

    All 3D plots share a single azimuth slider for XY-plane rotation.

    Parameters
    ----------
    param_grid : dict[str, list[Any]]
        The same param_grid passed to cross_validate_regression.
    cv_results : dict
        Output dict from cross_validate_regression.
    regressor_name : str
        Display name used in plot titles.
    """
    param_names = list(param_grid.keys())
    combinations = cv_results["param_combinations"]
    cv_rmse = np.sqrt(cv_results["cv_mse"])
    best_params = cv_results["best_params"]
    best_rmse = cv_results["best_cv_rmse"]

    n_params = len(param_names)

    if n_params == 1:
        _plot_1d(param_names[0], combinations, cv_rmse, best_params, best_rmse, regressor_name)
        return

    if n_params == 2:
        axes_3d, fig = _build_figure([(param_names[0], param_names[1])], regressor_name)
    else:
        pairs = list(itertools.combinations(param_names, 2))
        axes_3d, fig = _build_figure(pairs, regressor_name)

    for ax, (p1, p2) in zip(axes_3d, (
        [(param_names[0], param_names[1])] if n_params == 2
        else list(itertools.combinations(param_names, 2))
    )):
        _plot_3d_on_ax(ax, p1, p2, combinations, cv_rmse, best_params, best_rmse)
        suffix = "" if n_params == 2 else "\n(min over remaining params)"
        ax.set_title(f"{p1} × {p2}{suffix}")

    _attach_azimuth_slider(fig, axes_3d)
    plt.show()


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _build_figure(
    pairs: list[tuple[str, str]],
    regressor_name: str,
    slider_height: float = 0.06,
) -> tuple[list, plt.Figure]:
    """
    Create a figure with one 3D subplot per pair and bottom space for the slider.

    Parameters
    ----------
    pairs : list of (str, str)
        Parameter name pairs — one subplot each.
    regressor_name : str
        Used in the figure suptitle.
    slider_height : float
        Fraction of figure height reserved for the azimuth slider.

    Returns
    -------
    tuple of (list of Axes3D, Figure)
    """
    n = len(pairs)
    n_cols = min(n, 2)
    n_rows = (n + 1) // n_cols
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows + 1))
    fig.suptitle(f"{regressor_name} — CV RMSE", fontsize=14)

    axes_3d = []
    for idx, _ in enumerate(pairs, start=1):
        ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
        axes_3d.append(ax)

    fig.subplots_adjust(bottom=slider_height + 0.05)
    return axes_3d, fig


def _attach_azimuth_slider(fig: plt.Figure, axes_3d: list) -> None:
    """
    Add a shared horizontal slider at the bottom of the figure that rotates
    all 3D subplots together by changing their azimuth angle.

    Parameters
    ----------
    fig : plt.Figure
        The figure to attach the slider to.
    axes_3d : list of Axes3D
        All 3D axes that will be rotated synchronously.
    """
    ax_slider = fig.add_axes([0.2, 0.01, 0.6, 0.03])
    slider = Slider(ax_slider, "Azimuth", 0, 360, valinit=225, valstep=1)

    def _on_change(val: float) -> None:
        for ax in axes_3d:
            ax.view_init(elev=ax.elev, azim=val)
        fig.canvas.draw_idle()

    slider.on_changed(_on_change)
    # Keep a reference so the slider is not garbage-collected
    fig._azimuth_slider = slider


# ---------------------------------------------------------------------------
# Internal plot helpers
# ---------------------------------------------------------------------------

def _plot_1d(
    param_name: str,
    combinations: list[dict],
    cv_rmse: np.ndarray,
    best_params: dict,
    best_rmse: float,
    regressor_name: str,
) -> None:
    """Render a 2D line plot for a single-parameter grid."""
    x_vals = [c[param_name] for c in combinations]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, cv_rmse, marker="o", linewidth=1.5, label="CV RMSE")
    ax.scatter(
        [best_params[param_name]], [best_rmse],
        color="red", zorder=5, s=100,
        label=f"Best ({param_name}={best_params[param_name]:.4g}, RMSE={best_rmse:.2f})",
    )
    ax.set_xlabel(param_name)
    ax.set_ylabel("CV RMSE")
    ax.set_title(f"{regressor_name} — CV RMSE vs {param_name}")
    ax.legend()
    plt.tight_layout()
    plt.show()


def _plot_3d_on_ax(
    ax,
    p1: str,
    p2: str,
    combinations: list[dict],
    cv_rmse: np.ndarray,
    best_params: dict,
    best_rmse: float,
) -> None:
    """
    Draw a marginal 3D scatter on an existing Axes3D.

    For each unique (p1, p2) pair the minimum CV RMSE over all remaining
    parameters is used (margin by minimum).

    Parameters
    ----------
    ax : Axes3D
        Target 3D axes.
    p1, p2 : str
        Names of the two parameters to place on X and Y axes.
    combinations : list[dict]
        All parameter combinations from cross_validate_regression.
    cv_rmse : np.ndarray
        CV RMSE for each combination (same order as combinations).
    best_params : dict
        Best parameter combination (highlighted in red).
    best_rmse : float
        CV RMSE of the best combination.
    """
    margin: dict[tuple, float] = {}
    for combo, rmse in zip(combinations, cv_rmse):
        key = (combo[p1], combo[p2])
        margin[key] = min(margin.get(key, np.inf), rmse)

    x_vals = np.array([k[0] for k in margin])
    y_vals = np.array([k[1] for k in margin])
    z_vals = np.array(list(margin.values()))

    sc = ax.scatter(x_vals, y_vals, z_vals, c=z_vals, cmap="viridis_r", s=60, alpha=0.85)
    plt.colorbar(sc, ax=ax, pad=0.1, label="CV RMSE")

    best_key = (best_params[p1], best_params[p2])
    if best_key in margin:
        ax.scatter(
            [best_key[0]], [best_key[1]], [best_rmse],
            color="red", s=120, zorder=5,
            label=f"Best RMSE={best_rmse:.2f}",
        )
        ax.legend(fontsize=8)

    ax.set_xlabel(p1)
    ax.set_ylabel(p2)
    ax.set_zlabel("CV RMSE")