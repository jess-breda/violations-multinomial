import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_kernel_validation(df, kernel_column):
    fig, ax = plt.subplots(figsize=(10, 3))

    # plot kernel
    plt.plot(df[kernel_column] / df[kernel_column].max(), label="Filter", marker="o")

    # Iterate through the prev_violation column and plot vertical lines when the value is 1
    for idx, value in df["violation"].items():
        if value == 1:
            plt.vlines(x=idx, ymin=0, ymax=1, color="r", label="Violation")

    # Plot black vertical lines for session boundaries
    last_idx_of_each_session = df["session"].duplicated(keep="last")
    for idx, is_duplicated in enumerate(last_idx_of_each_session):
        if not is_duplicated:
            plt.vlines(
                x=idx,
                ymin=0,
                ymax=2,
                color="k",
                linestyles="dashed",
                label="Session boundary",
            )

    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1, 1), loc=2)

    ax.set(
        title=f"Scaled exp filter with $\\tau$ = {kernel_column.split('_')[-1]}",
        xlabel="Trial",
        ylabel="Filter value",
    )

    return None
