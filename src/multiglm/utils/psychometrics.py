""" 
Functions and plotting specific to psychometric analysis

Written by Jess Breda, 2023-12-27
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.pyplot as plt


## Helpers
def sigmoid_for_curve_fit(x, alpha, beta):
    return 1 / (1 + np.exp(-(alpha + beta * x)))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


##  For fitting psych to data
def compute_psych_df(df, by_session=False):
    """
    Function to compute a DataFrame used for
    psychometric analysis where x is the difference
    between stimuli and y is the probability of
    going right. p right is calculated from choice
    variable where 1 : right, 0 : left.

    params
    ------
    df : pd.DataFrame
        DataFrame with trials as row index for all animals
        with 's_a', 's_b', 'animal_id', 'session' and 'choice'
        columns
    by_session : bool (default=False)
        whether or not to compute performance for each
        animal and session or just for each animal

    returns
    -------
    psych_df : pd.DataFrame
        dataframe with columns `animal_id`, `delta_stim`
        and `p_right`. if by_session will also have a
        `session`
    """

    # create delta stim column
    df = df.copy()
    df["delta_stim"] = df.s_a - df.s_b

    if by_session:
        group = ["animal_id", "session", "delta_stim"]
    else:
        group = ["animal_id", "delta_stim"]

    psych_df = df.groupby(group).choice.mean().reset_index()

    return psych_df.rename(columns={"choice": "p_right"})


def fit_psych_(df):
    # TODO might need to check for session column
    x = df.delta_stim
    y = df.p_right

    params, covariance = curve_fit(sigmoid_for_curve_fit, x, y)
    x_vals = np.linspace(-10, 10, 20)
    psych = sigmoid_for_curve_fit(x_vals, *params)

    return x_vals, psych


def plot_fitted_psych(x_vals, psych, ax=None, title="", **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    sns.lineplot(x=x_vals, y=psych, ax=ax, **kwargs)

    _ = ax.set(
        xlabel="$s_a - s_b$",
        ylabel="P(right)",
        title=title,
        ylim=(0, 1),
        xlim=(-10, 10),
    )

    return None


def plot_raw_psych_data(df, ax=None, title="", **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    # TODO determine how to do this when sessions are included
    sns.scatterplot(df.groupby("delta_stim").p_right.mean(), **kwargs)
    _ = ax.set(
        xlabel="$s_a - s_b$",
        ylabel="P(right)",
        ylim=(0, 1),
        title=title,
        xlim=(-10, 10),
    )


## For simulation from weights


def generate_x_data(min_val=-2, max_val=2, num_points=10):
    """
    Function to generate x data pairs for a grid of points
    (sa, sb) within a range. Used for simulating data with
    varition in deltas between two values.

    params
    ------
    min_val: float
        minimum value for both sa and sb
    max_val: float
        maximum value for both sa and sb
    num_points: int
        number of points to generate in the range
        note grid is num_points x num_points

    returns
    -------
    data: pd.DataFrame
        dataframe with columns sa and sb and rows
        for each pair of values

    """
    # Generate linearly spaced values within the range
    sa_values = np.linspace(min_val, max_val, num_points)
    sb_values = np.linspace(min_val, max_val, num_points)

    # Create a grid of all possible (sa, sb) pairs
    sa_grid, sb_grid = np.meshgrid(sa_values, sb_values)

    # Flatten the grids to create a list of pairs
    sa_flat = sa_grid.flatten()
    sb_flat = sb_grid.flatten()

    # Create a DataFrame from the pairs
    data = pd.DataFrame({"s_a": sa_flat, "s_b": sb_flat})

    return data


def simulate_psych(delta_weights_df, X_enhanced, filter_val):
    """
    function to simulate psychometric x and y data from weights
    given x values and a filter value

    delta_weights_df : pd.Dataframe
        long form df with columns "feature", "weight" for a single animal
    X_enhanced: pd.Dataframe with columns "s_a", "s_b"
        a grid of sa,sbs with varying deltas
    filter_val : int
        the value of the filter to be implemented
    """
    # Extract weights
    w_sa = delta_weights_df.loc[delta_weights_df["feature"] == "s_a", "weight"].values[
        0
    ]
    w_sb = delta_weights_df.loc[delta_weights_df["feature"] == "s_b", "weight"].values[
        0
    ]
    w_sa_x = delta_weights_df.loc[
        delta_weights_df["feature"] == "prev_violation_exp_x_s_a", "weight"
    ].values[0]
    w_sb_x = delta_weights_df.loc[
        delta_weights_df["feature"] == "prev_violation_exp_x_s_b", "weight"
    ].values[0]
    bias = delta_weights_df.loc[delta_weights_df["feature"] == "bias", "weight"].values[
        0
    ]
    # Filter control
    f_x = filter_val

    # Vectorized computation of z
    z = (
        ((w_sa + (w_sa_x * f_x)) * X_enhanced["s_a"])
        + ((w_sb + (w_sb_x * f_x)) * X_enhanced["s_b"])
        + bias
    )

    # Apply sigmoid function to z
    ys = sigmoid(z)

    # Delta x computation (if needed)
    deltax = X_enhanced["s_a"] - X_enhanced["s_b"]

    return deltax, ys


# Function to fit the psychometric curve
def fit_psych(deltax, ys):
    params, covariance = curve_fit(sigmoid_for_curve_fit, deltax, ys, p0=[0, 1])
    return params, covariance
