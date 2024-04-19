import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    ax: plt.Axes = None,
    cmap: str = "Greys",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        transition_matrix, cmap=cmap, annot=True, fmt=".2f", vmin=0, vmax=1, ax=ax
    )

    ax.set(xlabel="Next State", ylabel="Current State", title="Transitions")

    return None


def make_weight_df(params, plot_bias=False, name_col=None):
    # Extract the weights and biases
    weights = params.emissions.weights
    biases = params.emissions.biases

    # Convert the numpy array into a DataFrame
    num_states, num_features = weights.shape
    feature_names = [f"Feature {i+1}" for i in range(num_features)]

    # Prepare the data frame for weights
    df = pd.DataFrame(weights, columns=feature_names)
    df["State"] = [f"State {i+1}" for i in range(num_states)]

    # If plot_bias is True, add biases as an additional "feature"
    if plot_bias and biases is not None:
        df["Bias"] = biases
        feature_names = ["Bias"] + feature_names  # Adjust the order to put Bias first

    # Melt the DataFrame to long format
    df_long = df.melt(id_vars="State", var_name="Feature", value_name="Weight")

    if name_col:
        df_long["Name"] = name_col

    return df_long


def plot_binary_emission_weights(
    params, plot_bias=False, ax=None, legend=True, **kwargs
):
    """
    Plot weights for each feature across different states.

    Params
    -------
    params : HMMParameterSet (Num States, Num Features)
        The parameters of the model with params.emissions.weights
        containing the weights for each feature across states.
    plot_bias : bool (default=True)
        Whether to include the bias terms in the plot.
    ax : matplotlib.Axes (default=None)
        ax to plot to
    """

    df = make_weight_df(params, plot_bias)

    # Create the plot
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, color="k", lw=2)
    sns.pointplot(
        data=df, x="Feature", y="Weight", hue="State", palette="husl", **kwargs
    )

    ax.set(xlabel="", ylabel="Weight", title="Emission Weights")

    if legend:
        pass
    else:
        ax.legend().remove()

    return None


def plot_binary_hmm_params(params, plot_bias=False, ax=None, title=""):

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    fig.suptitle(title, fontsize=28)

    plot_transition_matrix(params.transitions.transition_matrix, ax=ax[0])
    plot_binary_emission_weights(params, plot_bias, ax=ax[1])

    plt.tight_layout()


def print_binary_hmm_params(params):
    print(
        f"Model has {len(params.initial.probs)} states with initial probabilities:\n  {params.initial.probs}"
    )
    print(f"\nTransition_matrix:\n {params.transitions.transition_matrix}")

    print(
        f"\nEmissions have {params.emissions.weights.shape[1]} features with weights:\n {params.emissions.weights}"
    )
    print(f"\nEmissions have biases:\n {params.emissions.biases}")
