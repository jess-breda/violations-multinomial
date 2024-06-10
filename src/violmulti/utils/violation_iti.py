""" 
Class for computing violation ITI (inter-trial-interval) 
for a given dataset. Additional methods allow for validation
of the computed ITI function with simulated data as well as
plotting. 

Note: a single object can be run on multiple datasets using
the computer_and_add_viol_iti_column method.

Written by: Jess Breda 2024-01-31
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ViolationITI:
    def __init__(self):
        pass

    def compute_and_add_viol_iti_column(self, df):
        self.run_checks(df)

        df["violation_iti"] = np.nan
        df_with_iti = (
            df.groupby(["animal_id", "session"])
            .apply(self.calculate_violation_intervals)
            .reset_index(drop=True)
        )

        return df_with_iti

    def plot_data_over_trials(self, df, start_idx=None, end_idx=None):
        if start_idx:
            plot_data = df.iloc[start_idx:end_idx]
        else:
            plot_data = df.copy()

        n_sessions = plot_data.session.nunique()

        fig, ax = plt.subplots(
            n_sessions, 1, figsize=(10, (n_sessions * 3)), constrained_layout=True
        )

        for i, (session, group) in enumerate(plot_data.groupby("session")):
            cur_ax = ax[i] if n_sessions > 1 else ax
            cur_ax.plot(
                group.trial,
                group.violation,
                "o",
                label="Viol Hist",
            )
            cur_ax.plot(
                group.trial,
                group.violation_iti,
                "x",
                color="red",
                label="Viol ITI",
            )

            cur_ax.set(
                xlabel="Trial",
                ylabel="was viol / viol ITI",
                title=f"Session: {session}",
            )
        cur_ax.legend()

    @staticmethod
    def simulate_data(n_sessions=3, n_trials=25, random_state=None):
        data = []

        for session in range(1, n_sessions + 1):
            for trial in range(1, n_trials + 1):
                # Randomly decide if a violation occurred (for simplicity, say 20% chance)
                np.random.seed(random_state)
                violation = np.random.choice(
                    [0, 1],
                    p=[0.8, 0.2],
                )
                data.append(["example_animal", session, trial, violation])

        # Create DataFrame
        columns = ["animal_id", "session", "trial", "violation"]
        simulated_df = pd.DataFrame(data, columns=columns)

        return simulated_df

    @staticmethod
    def run_checks(df):
        required_columns = ["animal_id", "trial", "session", "violation"]
        assert all(
            column in df.columns for column in required_columns
        ), "Required columns not present!"

    @staticmethod
    def calculate_violation_intervals(group):
        """
        function to be used with a pandas "apply" when
        grouping by session (can also group by animal_id
        if running multiple animals at a time)
        """

        # Get trial numbers of trials with a violation
        violation_trials = group["trial"][group["violation"] == 1]

        # Calculate number of trials between trials with a violation
        # shift by 1 to align trials properly
        intervals = violation_trials.diff() - 1

        # Assign the calculated intervals back to the group,
        # All violation trials (except the first of a group) get a value,
        # every other trial type gets an nan
        group.loc[violation_trials.index, "violation_iti"] = intervals

        return group
