"""
Functions for visualizing animal performance
"""

from matplotlib import pyplot as plt
from get_rat_data import *


def plot_animal_performance_by_stim(animal_id, violations_only=True, ax=None):
    if violations_only:
        fun = get_rat_viol_data
    else:
        fun = get_rat_data

    df = fun(animal_ids=animal_id)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    stim_table = (
        df.groupby(["s_a", "s_b"])
        .agg(perf_rate=("hit", "mean"), perf_count=("hit", "size"))
        .reset_index()
    )

    # plot each sa,sb pair with rate as color
    scatter = ax.scatter(
        stim_table.s_a,
        stim_table.s_b,
        c=stim_table.perf_rate,
        cmap="flare",
        vmin=0,
        vmax=1,
        marker=",",
        s=100,
    )

    ax.axline(xy1=(60, 60), slope=1, color="lightgray", linestyle="--")
    # Add a colorbar to the plot
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("hit rate")

    # add labels to each point
    for i, txt in enumerate(stim_table.perf_rate):
        ax.text(
            stim_table.s_a[i] - 2,
            stim_table.s_b[i] + 1.5,
            f"{round(txt, 2)} [{stim_table.perf_count[i]}]",
            fontsize=8,
        )

    return None
