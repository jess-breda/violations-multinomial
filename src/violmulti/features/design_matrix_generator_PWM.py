"""
Child class of DesignMatrixGenerator for creating design matrices
specific to the PWM dataset.
"""

import pandas as pd
from pandas import Series
import numpy as np
from violmulti.features.design_matrix_generator import *
from typing import List, Tuple
import warnings


## CLASS
class DesignMatrixGeneratorPWM(DesignMatrixGenerator):
    def __init__(
        self,
        df: pd.DataFrame,
        config: dict,
        verbose: bool = False,
    ):
        super().__init__(df, config, verbose)
        if len(self.df["animal_id"].unique()) > 1:
            print("DMG: Note more than 1 animal in dataframe!")

    def assert_single_animal(self):
        """
        Assert only one animal in data frame- and easy thing to miss
        when creating design matrices since the column is dropped.
        """
        if len(self.df["animal_id"].unique()) == 1:
            print("DMG: Note more than 1 animal in dataframe!")

    def create(self) -> Tuple[pd.DataFrame, np.ndarray]:
        X, y = super().create()

        return X, y


## METHODS


def prev_avg_stim(df, mask_prev_violation):
    """
    Function to create previous stimulus avg loudness feature
    as done by Roy et al. 2021 in Psytrack paper.
    """

    avg_stim = combine_two_cols(df.s_a, df.s_b, operation="mean")

    avg_stim_standardized = standardize(avg_stim)

    prev_avg_stim = shift_n_trials_up(avg_stim_standardized, df.session, shift_size=1)

    if mask_prev_violation:
        prev_avg_stim = mask_prev_event(prev_avg_stim, df.violation, df.session)

    return prev_avg_stim


def prev_correct_side(df):
    """
    Function to create a previous correct side feature to
    approximate win-stay-lose-switch behavior. Mapping:
        left (0)   ->  -1
        right (1)  ->  1
        violations ->  masked to 0 (animal does not know correct side)
    """

    correct = remap_values(df.correct_side, mapping={0: -1})

    prev_correct = shift_n_trials_up(correct, df.session, shift_size=1)

    prev_correct = mask_prev_event(prev_correct, df.violation, df.session)

    return prev_correct.astype(int)


def prev_choice(df):
    """
    Function to create a previous choice feature to approximate
    side bias behavior. Mapping:
        left (0)       ->  -1
        right (1)      ->   1
        violations (2) ->   0 (has it's own prev_viol regressor)
    """

    choice = remap_values(df.choice, mapping={0: -1, 2: 0})

    prev_choice = shift_n_trials_up(choice, df.session, shift_size=1)

    return prev_choice.astype(int)


def prev_violation(df):
    """
    Function to create a previous violation feature
    Mapping:
        left or right (0, 1) -> 0
        violation        (2) -> 1
    """

    choice = remap_values(df.choice, mapping={1: 0, 2: 1})

    prev_violation = shift_n_trials_up(choice, df.session, shift_size=1)

    return prev_violation.astype(int)


def prev_trial_not_started(df):
    """
    Function to create a previous trial not started
    Mapping:
        trial not started (>0) -> 1
        started (0) -> 0
    """

    return binarize(df.n_prev_trial_not_started, comparison=operator.gt, value=0)


def filtered_prev_viol(df, tau):
    """
    Function to create a filtered previous violation feature
    given a specified time constant (tau) for the exponential filter.
    """

    prev_viol = prev_violation(df)
    prev_viol.name = "prev_viol"  # needed for ExpFilter class

    filtered_prev_viol = exp_filter_column(prev_viol, df.session, tau)

    return filtered_prev_viol


def filtered_prev_disengaged(df, tau, binarize_pre_filt):
    """
    Function to create a filtered previous disengaged feature which
    is the summation of the prev trial not started and prev violation
    binary regressors. This can then be filtered with overlap (ie max of)
    2 or a binarized to 0 and 1 (ie no indication of overlap)
    """

    combined_cols = combine_two_cols(
        prev_trial_not_started(df), prev_violation(df), operator.add
    )

    if binarize_pre_filt:
        combined_cols = binarize(combine_two_cols, operator.gt, 0)

    filt_prev_disengaged = exp_filter_column(combined_cols, df.session, tau)

    return filt_prev_disengaged


def stim_filt_viol_intrx(df, stim_name, tau):
    """
    Function to create an interaction feature between a
    stimulus term ("s_a", or "s_b") and a filtered previous
    violation term.
    """

    filt_viol_col = filtered_prev_viol(df, tau)

    stim_col = standardize(df[stim_name])

    intrx_col = combine_two_cols(stim_col, filt_viol_col, operator.mul)

    return intrx_col


def get_animals_tau(df, var_name):

    taus_df = pd.read_csv(
        "/Users/jessbreda/Desktop/github/violations-multinomial/data/processed/tau_sweeps/taus_df.csv"
    )

    # design matrix generator checks tests for single animal requirement in df
    animal_id = df.animal_id.iloc[0]

    tau = taus_df.query("animal_id == @animal_id")[f"{var_name}_tau"].values[0]

    return tau


def multi_choice_labels():
    """
    quick function to return labels dictionary that
    maps choice values to multi values such that
    L (0) : 0, R (1) : 1, and Viol (2) : 2.
    """
    return {"column_name": "choice", "mapping": {0: 0, 1: 1, 2: 2}}


def binary_choice_labels():
    """
    quick function to return labels dictionary that
    maps choice values to binary values such that
    L (0) : 0, R (1) : 1, and Viol (2) : np.nan.
    DesignMatrixGenerator drops nan rows by default.
    """

    return {"column_name": "choice", "mapping": {0: 0, 1: 1, 2: np.nan}}


### SSM Helper Functions
def partition_data_by_session_for_EM(
    X: pd.DataFrame, y: np.ndarray[int]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Prepare jagged lists of feature arrays and label arrays for model fitting.

    Parameters
    ----------
    X : (n_trials, m_features + 1), where +1 is the "session" column.
        The input DataFrame containing trial data.
        It must have a column named "session".
        Each row represents a trial with associated features.
    y : (n_trials,)
        The input array containing labels for each trial.

    Returns:
    -------
    Tuple :  A tuple containing two jagged lists:
        - jagged_X_list: (n_trials_in_session, m_features)
            A list of arrays where each array corresponds to a session.
        - jagged_y_list: (n_trials_in_session, 1)
            A list of arrays where each array corresponds to a session.
            The second dimension 1 is necessary for the SSM model.1
        The number of trials per session can vary.
    """
    jagged_X_list = prepare_X_for_EM(X)
    jagged_y_list = prepare_y_for_EM(X, y)

    print(f"DMG: {len(jagged_X_list)} sessions found and reshaped for EM.")

    return jagged_X_list, jagged_y_list


def prepare_X_for_EM(df: pd.DataFrame) -> List[np.ndarray]:
    """
    Prepare a jagged list of arrays from the DataFrame for model fitting.

    Parameters:
    df : (n_trials, m_features + 2), where +2 is the "session" and "bias" columns
        The input DataFrame containing trial data.
        It must have a column named "session".
        Each row represents a trial with associated features.

    Returns:
    jagged_X_list: (n_sessions, (n_trials_in_session, m_features))
        A jagged list of arrays where each array corresponds to a session.
        The number of trials per session can vary.
    """
    assert "session" in df.columns, "The DataFrame must have a column named 'session'."

    # Infer feature columns
    feature_cols = [col for col in df.columns if col != "session"]

    # Group by the session column
    grouped = df.groupby("session")

    # Create the jagged array
    jagged_X_list = [group[feature_cols].to_numpy() for _, group in grouped]

    return jagged_X_list


def prepare_y_for_EM(df: pd.DataFrame, y: np.ndarray) -> List[np.ndarray]:
    """
    Prepare a jagged list of label arrays from the DataFrame and label array.

    Parameters:
    df : (n_trials, m_features + 2), where + 2 is the "session" and "bias" columns
        The input DataFrame containing trial data, where the
        session column is used for grouping.

    y : (n_trials,)
        The input array containing labels for each trial.

    Returns:
    jagged_y_list : (n_sessions, (n_trials_in_session, 1 ))
        A jagged list of label arrays where each array corresponds to a session.
        The number of trials per session can vary.
    """
    # Tests
    if len(y) != len(df):
        raise ValueError(
            "The length of the label array y must match the number of rows in the DataFrame"
        )
    assert "session" in df.columns, "The DataFrame must have a column named 'session'."

    # Group by the session column
    grouped = df.reset_index().groupby("session")

    # Create the jagged list of labels
    jagged_y_list = [y[group.index].reshape(-1, 1) for _, group in grouped]

    return jagged_y_list
