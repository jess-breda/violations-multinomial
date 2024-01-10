import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pathlib
import sys

[
    sys.path.append(str(folder))
    for folder in pathlib.Path("../src/").iterdir()
    if folder.is_dir()
]
from fitting_utils import get_taus_df
from get_rat_data import get_rat_viol_data
from design_matrix_generator import DesignMatrixGenerator


def generate_design_mat_with_filter_viol():
    """
    function to make a design matrix for all animals
    with the prev_violation column filtered by the
    animal's prev_violation_tau that was fit to the
    animal's data
    """

    X_animals = []
    taus_df = get_taus_df()
    animals = taus_df.animal_id.to_list()

    for animal_id in animals:
        # load df
        df = (
            get_rat_viol_data(animal_ids=animal_id)
            .query("training_stage_cat >= 3")
            .copy()
        )

        # get tau
        prev_violation_tau = taus_df.query("animal_id == @animal_id")[
            f"prev_violation_tau"
        ].values[0]

        # create design matrix with filt prev violation
        dmg = DesignMatrixGenerator()
        X = dmg.generate_base_matrix(df, return_labels=False, include_stage=True)
        X = dmg.exp_filter_column(X, tau=prev_violation_tau, column="prev_violation")
        X["animal_id"] = [animal_id] * len(X)
        X["prev_violation_tau"] = [prev_violation_tau] * len(X)
        X_animals.append(X)

    prev_violation_df = pd.concat(X_animals, ignore_index=True)

    return prev_violation_df


def plot_histogram_facet_grid(
    df,
    var,
    binwidth=1,
    color=None,
    **kwargs,
):
    """
    function to plot a histogram facet grid
    """
    g = sns.FacetGrid(df, col="animal_id", col_wrap=4, **kwargs)
    g.map(sns.histplot, var, binwidth=binwidth, color=color)

    return g


def create_trials_df_w_viol_iti(min_stage=3, add_prev_viol_tau=True):
    """
    function to load trials data for all animals
    in min_stage and above and add a column with
    violation iti data

    params
    ------
    min_stage : int (default=3)
        minimum training stage to include in trials df

    returns
    -------
    viol_itis_df : pd.DataFrame
        trials df with violation iti column added
        where viol iti is not null only on violation
        trials to indicate how many trials ago the
        last violation occurred

        e.g. on hit trials, viol iti is a nan, if on
        a violation trial that followed a previous
        violation trial, viol iti is 0.
    """

    # load raw trials data
    raw_df = get_rat_viol_data().query("training_stage_cat >= @min_stage").copy()
    viol_iti_df = add_violation_iti_column(raw_df)

    if add_prev_viol_tau:
        # load tau data
        taus_df = get_taus_df()

        tau_mapping = taus_df.set_index("animal_id").to_dict()["prev_violation_tau"]
        viol_iti_df["prev_violation_tau"] = viol_iti_df["animal_id"].map(tau_mapping)

    return viol_iti_df


def add_violation_iti_column(df):
    """
    Function to add a violation_iti column to a trials DataFrame
    given a violation column. The violation_iti column indicates
    for a violation trial how many trials ago the last violation
    occurred.

    This means, on non-violation trials, the iti is nan.
    However, if the current trial is a violation trial
    and the previous trial was also a violation trial, then the
    violation_iti value for the current trial will be 0.
    Counter resets a start of the session.

    params
    ------
    df : pd.DataFrame
        trials DataFrame with a `violation`, `session` and
        `animal_id` `trial` column

    returns
    -------
    df : pd.DataFrame
        trials DataFrame with a `violation_iti` column
        indicating how many trials ago the last violation
        occurred
    """

    # Check that the required columns are present
    required_columns = ["animal_id", "session", "trial", "violation"]
    if not all([col in df.columns for col in required_columns]):
        raise ValueError(
            f"Required columns {required_columns} not present in DataFrame."
        )

    # Function to apply to each animal-session group
    def calculate_intervals(group):
        # Find trials with violations
        violation_trials = group["trial"][group["violation"] == 1]
        # Calculate intervals and shift to align with the next violation trial
        intervals = violation_trials.diff() - 1
        # Assign the calculated intervals back to the group
        group.loc[violation_trials.index, "violation_iti"] = intervals
        return group

    # Initialize the violation_iti column
    df["violation_iti"] = np.nan

    # Apply the function to each animal-session group. Only adds iti values
    # to the violation trials
    viol_iti_df = df.groupby(["animal_id", "session"]).apply(calculate_intervals)
    return viol_iti_df.reset_index(drop=True)


def create_frac_consecutive_viols_df(viol_iti_df):
    """
    Function to take the trials form viol_iti_df and for each animal,
    calculate the fractions of violation trials where violation_iti is zero.
    This is a metric of continuous or repeating violations (as opposed to
    non-repeating violations)

    params
    ------
    viol_iti_df : pandas.DataFrame
        DataFrame with violation_iti column created by create_trials_df_w_viol_iti()
        with animal_id, session, trial as row index

    returns
    -------
    frac_viol_iti_zero_df : pandas.DataFrame
        DataFrame with animal_id and frac_violation_iti_zero columns with
        animal_id as row index
    """
    ## calculating the fraction of trials where violation_iti is zero for
    ## each animal, then adding it to the summary_df
    total_violation_trials = (
        viol_iti_df[viol_iti_df["violation"] == 1].groupby("animal_id").size()
    )

    # Count the number of trials where violation_iti equals 0 per animal
    violation_iti_zero_trials = (
        viol_iti_df[viol_iti_df["violation_iti"] == 0].groupby("animal_id").size()
    )

    # Calculate the percentage
    create_frac_consecutive_viols_df = (
        violation_iti_zero_trials / total_violation_trials
    ).reset_index(name="frac_consecutive_viols")

    return create_frac_consecutive_viols_df


def calculate_consecutive_violations(viol_iti_df):
    def count_max_consecutive_trues(group):
        """
        Apply function to group. Will find a streak of violations
        and count the number of violations in the streak. It will
        then store the count at the end of the streak when a non-
        violation trial occurs.

        Group is typically animal_id, session
        """
        streak_counter = 0
        max_streaks = []

        for iti_zero in group["iti_is_zero"]:
            if iti_zero:
                streak_counter += 1
                max_streaks.append(
                    np.nan
                )  # We don't store the count until the streak ends
            else:
                if streak_counter > 0:
                    max_streaks.append(
                        streak_counter
                    )  # Store the count at the end of the streak
                    streak_counter = 0  # Reset the counter
                else:
                    max_streaks.append(np.nan)  # No streak to count

        # Fill in the last streak count if the data ends with a streak
        if streak_counter > 0:
            max_streaks[-1] = streak_counter

        group["n_consecutive_viols"] = max_streaks
        return group

    viol_iti_df["iti_is_zero"] = viol_iti_df["violation_iti"] == 0

    viol_iti_df = viol_iti_df.groupby(["animal_id", "session"]).apply(
        count_max_consecutive_trues
    )

    return viol_iti_df.copy().reset_index(drop=True)


## Plots ##


def plot_prev_viol_median(data, **kwargs):
    """
    function to plot a vertical line at the median
    of the prev_violation_exp column

    used with g.map_dataframe()
    """
    median = data["prev_violation_exp"].median()
    plt.axvline(median, color="k", linestyle="--")
    plt.text(
        median + 0.1,
        plt.gca().get_ylim()[1] * 0.8,
        f"{np.round(median, 2)}",
        color="k",
    )


def plot_prev_viol_tau_palette(figsize=(5, 1)):
    """
    handy plot util to plot a legend of the prev_violation_tau
    for presentation visuals
    """
    taus_df = get_taus_df()
    palette = sns.color_palette("husl", n_colors=taus_df.prev_violation_tau.nunique())

    # Create a list of patches for the legend
    legend_patches = [
        mpatches.Patch(color=palette[i], label=tau)
        for i, tau in enumerate(sorted(taus_df.prev_violation_tau.unique()))
    ]

    # Plot the legend separately
    plt.figure(figsize=figsize)
    plt.legend(
        handles=legend_patches,
        title="Prev Violation Tau",
        bbox_to_anchor=(1, 1),
        loc="center",
    )
    plt.axis("off")
    plt.show()

    return None
