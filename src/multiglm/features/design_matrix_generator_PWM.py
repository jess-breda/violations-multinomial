"""
Child class of DesignMatrixGenerator for creating design matrices
specific to the PWM dataset.
"""

import pandas as pd
from pandas import Series
import numpy as np
from multiglm.features.design_matrix_generator import *


## CLASS
class DesignMatrixGeneratorPWM(DesignMatrixGenerator):
    def __init__(self, df, config, verbose=False):
        super().__init__(df, config, verbose)
        # self.X["choice"] = df.choice  # FOR DEBUG INIT
        self.run_init_tests()

    def run_init_tests(self):

        assert (
            len(self.df["animal_id"].unique()) == 1
        ), "More than 1 animal in dataframe!"

    def create(self):
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


def get_animals_tau(df, var_name):

    taus_df = pd.read_csv(
        "/Users/jessbreda/Desktop/github/animal-learning/data/processed/tau_sweeps/taus_df.csv"
    )

    # design matrix generator checks tests for single animal requirement in df
    animal_id = df.animal_id.iloc[0]

    tau = taus_df.query("animal_id == @animal_id")[f"{var_name}_tau"].values[0]

    return tau


# binary and multi label maps?
# interaction terms
