"""
Parent class for generating design matrices for different models.
Written by Jess Breda, 2023-10-23
Overhauled on 2024-03-01
"""

import pandas as pd
from pandas import Series
import numpy as np
import operator
from multiglm.features.exp_filter import ExpFilter


## CLASS
class DesignMatrixGenerator:
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.X = pd.DataFrame()
        self.split_data_and_label_configs()

    def split_data_and_label_configs(self):
        """
        Function to split the config into data and label configurations
        since they follow different patters for how they are used.

        Data config take the form:
            "design_matrix_column_name_1" : function(s) to make
            "design_matrix_column_name_2" : function(s) to make


        Label config take the form:
            "labels" : {
                column_name : "column_name",
                mapping     : { old_value : new_value, ...}
                dropna      : True/False (can use this to drop violations in binary)
            }
        """

        self.config_labels = {}
        self.config_data = {}

        if "labels" not in self.config.keys():
            print("No labels found in config, only creating data matrix.")

            self.config_data = self.config

        else:
            for key, value in self.config.items():
                if value == "labels":
                    self.config_labels = value

                else:
                    self.config_data[key] = value

    def create(self):

        for key, func in self.config_data.items():

            self.X[key] = func(self.df)

        return self.X


## METHODS


def get_session_start_mask(sessions: Series, shift_size: int) -> Series:
    """
    method for creating a mask to zero out the first N trials
    of a session where N is defined by the shift size
    """

    # find session boundaries
    was_first_trial_in_session = sessions.diff().ne(0)

    # initialize mask to be updated in the loop (cleaner naming)
    was_first_n_trials_in_session = was_first_trial_in_session.copy()

    # propagate first trial in session True forward for N trials
    # where N is defined by the shift size and |= logical or
    # note this has been tested as does not cause issue if the number
    # of trials in a session is < shift_size
    for N in range(1, shift_size):
        was_first_n_trials_in_session |= was_first_trial_in_session.shift(N).fillna(
            False
        )

    return ~was_first_n_trials_in_session  # mask first trials to 0


def get_prev_event_mask(
    event_bool: Series,
    sessions: Series,
) -> Series:
    """
    method for creating a mask to zero out a trial value if it
    followed a trial with an event (e.g. violation)
    """

    was_prev_event = shift_n_trials_up(event_bool, sessions, shift_size=1).astype(bool)

    return ~was_prev_event


def mask_prev_event(col: Series, event_bool: Series, sessions: Series) -> Series:
    """
    method for masking a column to 0 if a specified event (e.g. violation)
    happened on the previous trial.

    *EX* of this is when creating a previous correct regressor to track
    win-stay-lose-shift behavior. If the previous trial was a violation,
    the previous correct side is unknown to the animal so the value
    should be masked to 0.

    mask_prev_event(df["prev_correct"], df["violation"], df["session"])
    """

    mask = get_prev_event_mask(event_bool, sessions)

    return col * mask


def add_bias_column(len: int) -> Series:
    """
    method for creating a bias column of 1s
    """

    return np.ones(len, dtype="int")


def copy(col: Series) -> Series:
    """
    method for copying a column
    """

    return col.copy()


def standardize(col: Series) -> Series:
    """
    method for standardizing a column to have
    mean 0 and std 1
    """

    return (col - col.mean()) / col.std()


def scale_by_max(col: Series) -> Series:
    """
    method for scaling a column by it's maximum
    value such that all the values are less than
    or equal to 1
    """
    return col / col.max()


def binarize(col: Series, comparison: operator, value: float) -> Series:
    """
    method for converting a column to a binary 0/1 int
    given comparison logic and value

    possible comparison options from operator class:
        - eq : == equal
        - ne : != not equal
        - gt : > greater than
        - lt : < less than
        - ge : >= grater than or equal to
        - lt : <= less than or equal to
        - and_ : bit wise AND
        - or_ : bit wise OR
        - xor : bitwise XOR
    """
    return comparison(col, value).astype(int)


def remap_values(col: Series, mapping: dict) -> Series:
    """
    method for mapping from old to new column values
    governed by a dictionary {old : new, ...}
    """
    return col.replace(mapping)


def shift_n_trials_up(col: Series, sessions: Series, shift_size: int = 1) -> Series:
    """
    method for shifting a column up N trials where N is defined by shift size
    and masking the first N trials of a session to 0 to avoid accidentally
    pulling data from the previous session
    """
    prev_data = col.shift(shift_size).fillna(0)

    # whenever shifting forward, you need to mask the the trials to 0 that
    # occurred at the beginning of the session. For example, if shift size is
    # 1, the first trial of session N should not have information from sess. N-1
    return mask_session_boundaries(prev_data, sessions, shift_size)


def mask_session_boundaries(col: Series, sessions: Series, shift_size: int) -> Series:
    """
    method for masking the first N trials of a session to 0 to avoid accidentally
    pulling data from the previous session
    """

    mask = get_session_start_mask(sessions, shift_size)

    return col * mask


def combine_two_cols(col1: Series, col2: Series, operation: operator) -> Series:
    """
    method for combining two columns using operator library

    possible methods from operator class:
        - .add
        - .truediv
        - .floordiv
        - .mul
        - .matmul
        - .sub
        - .pow

    can also use a custom mean function
    """

    if operation == "mean":
        return (col1 + col2) / 2
    else:
        return operation(col1, col2)


def exp_filter_column(col: Series, sessions: Series, tau: float) -> Series:
    """
    method for applying an exponential filter to a column on a session
    by session basis with a given tau
    """

    filtered_df = ExpFilter(
        tau=tau, column=col.name, verbose=False
    ).apply_filter_to_dataframe(pd.DataFrame({f"{col.name}": col, "session": sessions}))

    return filtered_df[f"{col.name}_exp"]
