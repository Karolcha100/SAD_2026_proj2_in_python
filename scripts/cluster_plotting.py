import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import numpy as np



def _plot_hist_while_columns(ax: Axes, col_name: str, data: np.ndarray, bins_n: int = 1_000) -> None:
    ax.hist(data, rwidth = 0.8, bins = bins_n, alpha = 1)
    ax.set_title(col_name)
    ax.set_xlabel(f"Value")
    ax.set_ylabel(f"Count")


def _plot_two_columns(
        ax: Axes, 
        col1_name: str, 
        col2_name: str, 
        col1_data: np.ndarray, 
        col2_data: np.ndarray, 
        alpha: float = 0.01,
        point_size: float = 1,
    ) -> None:
    ax.scatter(col2_data, col1_data, marker = ".", alpha=alpha, s = point_size)
    ax.set_ylabel(col1_name)
    ax.set_xlabel(col2_name)
    ax.set_title(f"{col1_name} vs {col2_name}")


def plot_all_columns(
        df: pd.DataFrame, 
        all_labels: pd.Series|None = None, 
        sel_labels: list[int]|None = None, 
        save_path: str|None = None, 
        alpha: float = 0.005,
        point_size: float = 1,
        bins_n: int = 1_000
    ) -> None:

    if all_labels is None and sel_labels is not None:
        raise ValueError(f"[plot_all_columns] {"all_labels"} should be provided when {"sel_labels"} are present!")

    fig, axs = plt.subplots(
        len(df.columns), len(df.columns), 
        figsize=(6*len(df.columns), 6*len(df.columns))
    )

    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i == j:
                if sel_labels is not None and all_labels is not None:
                    for sel_lab in sel_labels:
                        _plot_hist_while_columns(axs[i][j], col1, df.loc[all_labels == sel_lab][col1].to_numpy(), bins_n=bins_n)                  
                else:
                    _plot_hist_while_columns(axs[i][j], col1, df[col1].to_numpy(), bins_n=bins_n)
            else:
                if sel_labels is not None and all_labels is not None:
                    for sel_lab in sel_labels:
                        _plot_two_columns(
                            ax = axs[i][j],
                            col1_name=col1,
                            col2_name=col2,
                            col1_data=df.loc[all_labels == sel_lab][col1].to_numpy(),
                            col2_data=df.loc[all_labels == sel_lab][col2].to_numpy(),
                            alpha=alpha,
                            point_size=point_size
                        )
                else:
                    _plot_two_columns(
                    ax = axs[i][j],
                    col1_name=col1,
                    col2_name=col2,
                    col1_data=df[col1].to_numpy(),
                    col2_data=df[col2].to_numpy(),
                    alpha=alpha,
                    point_size=point_size,
                )


    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)






def plot_hist(df: pd.DataFrame, labels_selected: list[int]|None=None, save_path: str|None = None) -> None:
    selected_columns = [col for col in df.columns if col not in {"label"}]

    fig, axs = plt.subplots(len(selected_columns), figsize = (10, 2 * len(df.columns)), sharex=True)

    for i, col in enumerate(selected_columns):
        
        if labels_selected is not None:
            for lab_idx in labels_selected:
                axs[i].hist(df[df["label"] == lab_idx][col], rwidth = 0.8, bins = 100, alpha = 0.5, label = f"lab = {lab_idx}")
        else:
            axs[i].hist(df[col], rwidth = 0.8, bins = 1000, alpha = 1)
        axs[i].set_title(col)
        axs[i].legend(fontsize=10)
        axs[i].set_xlabel(f"ERK ratio value")
        axs[i].set_ylabel(f"Count")

    if labels_selected is not None:
        fig.suptitle(f"sel_labs = {labels_selected}")
    else:
        fig.suptitle(f"All Data")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)