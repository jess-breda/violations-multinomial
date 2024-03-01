"""
Parent class for generating design matrices for different models.
Written by Jess Breda, 2023-10-23
Overhauled on 2024-03-01
"""

import pandas as pd
import numpy as np
import operator
from multiglm.features.exp_filter import ExpFilter


class DesignMatrixGenerator:
    def __init__(self, df, config, verbose=False):
        self.df = df
        self.config = config
        self.verbose = verbose
        self.animal_id = df.animal_id.iloc[0]
        self.temp_X = pd.DataFrame()  # columns
        self.X = pd.DataFrame()
        self.X["choice"] = df.choice  # FOR DEBUG INIT

        self.run_init_tests()

    def run_init_tests(self):

        assert (
            len(self.df["animal_id"].unique()) == 1
        ), "More than 1 animal in dataframe!"

    def create_mask(df, mask_violations):
        """
        masks used for zeroing out data given events. This is generally
        used when representing previous trial history. For example, a
        the first trial in the session has no previous trial information
        so this can be used to set to 0.
        """

        session_boundaries_mask = df["session"].diff() == 0

        if not mask_violations:
            return session_boundaries_mask
        else:
            was_previous_violation = (
                df["violation"].shift() * session_boundaries_mask
            ).fillna(0)

            # counterintuitive, but want to set previous violations
            # to 0 to be able to mask with
            prev_violation_mask = was_previous_violation == 0

            return session_boundaries_mask * prev_violation_mask

    def create(self):

        for key, func in self.config.items():

            if key == "final_cols":
                continue

            self.temp_X[key] = func(self.df)

        self.X[self.config["final_cols"]] = self.temp_X[self.config["final_cols"]]

    @staticmethod
    def add_bias_column(df):

        return np.ones(len(df), dtype="int")

    @staticmethod
    def copy(df, col_name):

        return df[col_name]

    @staticmethod
    def normalize_column(df, col_name):

        col_data = df[col_name]

        return (col_data - col_data.mean()) / col_data.std()

    @staticmethod
    def prev_trial_avg(df, col_names, mask_violations=True, normalize=True):
        """
        On current trial t, take the previous trials average of
        features in cols list.

        params
        ------
        df : pd.Dataframe
            raw trial dataframe for an animal
        cols : list
            list of columns in df to take average of.
            *EX*: ["s_a", "s_b"]
        mask_violations : bool
            if the previous trial was a violation, should the value
            for the previous average be masked to or kept as is.
            *EX*: if computing previous stimulus average, one
            *may mask prev violations since the both stimuli may not
            *have played.
        normalize : bool, default = True
            if avg column should use .normalize_column() method for
            mean 0 and std 1
        """

        cols_data = df[col_names].copy()

        # Shift to previous trial & average
        cols_data["prev_avg"] = cols_data.shift().mean(axis=1).fillna(0)

        if normalize:
            cols_data["prev_avg"] = DesignMatrixGenerator.normalize_column(
                cols_data, "prev_avg"
            )

        # Apply masks- see docstring above for context
        mask = DesignMatrixGenerator.create_mask(df, mask_violations=mask_violations)

        return cols_data["prev_avg"] * mask

    @staticmethod
    def combine_two_cols(df, col_names, operation):
        """
        method for combining two columns using operator libary

        possible methods from operator class:
            - .add
            - .divide
            - .mult
            - .subtract
        """

        assert len(col_names) == 2, IndexError("Method only for two columns!")

        return operation(df[col_names[0]], col_names[1])

    @staticmethod
    def prev_trial_value(
        df,
        col_name,
        method=None,
        mask_violations=True,
        **kwargs,
    ):
        """
        method for shifting the values of a column up one trial such
        that on trial t, represents the values from trial t-1

        params
        ------
        df :
        col :
        method :
        mask_violations : bool, default = True

        mapping : dict, default = None

        kwargs :
            if method is binary: comparison operator and value to compare to
            if method is map: dict map to use

        """

        prev_col_data = df[col_name].shift().fillna(0)

        if method is not None:
            prev_col_data = DesignMatrixGenerator.apply_custom_method(
                prev_col_data, method, **kwargs
            )
        else:
            print("Raw values used for previous history")

        mask = DesignMatrixGenerator.create_mask(df, mask_violations=mask_violations)

        return prev_col_data * mask

    @staticmethod
    def apply_custom_method(col_data, method, **kwargs):
        print(method)
        if method == "scale_by_max":
            out_col = DesignMatrixGenerator.scale_by_max(col_data)
        elif method == "binarize":
            out_col = DesignMatrixGenerator.binarize(col_data, **kwargs)
        elif method == "map_values":
            out_col = DesignMatrixGenerator.implement_map(col_data, **kwargs)
        else:
            raise KeyError(f"{method} is an unknown method!")

        return out_col

    @staticmethod
    def scale_by_max(col_data):
        """
        method for scaling a column by it's maximum
        value such that all the values are less than 1
        """
        return col_data / col_data.max()

    @staticmethod
    def binarize(col_data, comparison, value):
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
        return comparison(col_data, value).astype(int)

    @staticmethod
    def implement_map(col_data, mapping):
        """
        method for mapping from old to new column values
        """
        return col_data.replace(mapping)


# METHODS TODO
# exp_filter_single_tau(df, col, tau=None, make_binary_history) if none, use LUT
# combination methods -> add, etc
# labels


# note some of these can be combined and might need to drop taus

# config could have a labels, (bool, type) that it skips in create and
# then uses to return
# label : {"name" : "choice", type = "binary/multi"}

# ==== scraps below
