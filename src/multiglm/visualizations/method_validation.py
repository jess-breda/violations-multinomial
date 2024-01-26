import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def plot_kernel_validation(df, kernel_column, reference_column="violation"):
    fig, ax = plt.subplots(figsize=(10, 3))

    # plot kernel
    plt.plot(
        df[kernel_column] / df[kernel_column].max(),
        label="Filter",
        marker="o",
        color="gray",
    )

    # Iterate through the prev_violation column and plot vertical lines when the value is 1
    for idx, value in df[reference_column].items():
        if value == 1:
            plt.vlines(x=idx, ymin=0, ymax=1, color="r", label=reference_column)

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


def plot_simulated_weights_binary(true_w, optimized_w, title="Recovered Weights"):
    """
    Plot true and stimulated weights for a binary logistic regression model.

    params
    ------
    true_w : np.ndarray, shape (D + 1, )
        true weight vector used to simulate data
    optimized_W : np.ndarray, shape (D + 1, )
        estimated weight vector from scipy.optimize.minimize
    title : str (default="Recovered Weights")
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    feature_and_bias_labels = np.arange(len(true_w))

    ax.plot(feature_and_bias_labels, true_w, label="true", color="gray", marker="o")
    ax.plot(
        feature_and_bias_labels,
        optimized_w,
        label="optimized",
        color="gray",
        marker="o",
        linestyle="dashed",
    )

    # Set the x-axis tick labels
    _ = ax.set_xticks(feature_and_bias_labels)
    ax.set(xlabel="Feature", ylabel="Weight", title=title)
    ax.legend()

    return None


### For binary/multi comparision


def are_dataframes_equal(df1, df2):
    """
    Function to make sure the 2 X_test dataframes are equal
    for the binary and multi class post filtering the multi-class
    data to only include L and R choices.
    """
    return df1.reset_index(drop=True).equals(df2.reset_index(drop=True))


# Helper function to map multi-class labels to binary labels
def map_multi_to_binary_labels(multi_labels):
    """
    Function to make sure y_test (binary) and Y_test (multi-class)
    are equal for post filtering the multi-class data to only
    include L and R choices, but keeping labels one-hot encoded.

    Maps [0, 1, 0] to 1 and [1, 0, 0] to 0
    """
    return np.argmax(multi_labels, axis=1)
