""" 
Class for cleaning the new 2024 dataset to correct for column names,
data types and timeout trials.

Sessions are defined as a single day of data collection. Trials 
are all the trials in the session that were NOT timeout trials. 
This means that the trial count in this cleaned df *WILL* differ
from the trial count of the raw df/data that was scraped from 
Brody lab data.mat files.

Timeout trials are renamed as "trial_not_started". These trials are
dropped from the data, but the contiguous history of them remains. 
E.g. for a hit trial following 3 timeout trials in a row, the 
"n_prev_trial_not_started" would be 3. 

Written by Jess Breda 2024-02-01
"""

import numpy as np
import pandas as pd


def clean_datasets(animal_ids, column_rename_map, save_out=True):
    """
    Wrapper function that creates cleaned data frames for each animal
    in animal ids using the DatasetCleaner, saves them and then
    concatenates them into a single data frame.

    animal_ids : list of animal ids to align and visualize
    column_rename_map : dictionary of old column names to new column names
                        likely from the src/multiglm/data/__init__.py file
    save_out : bool, whether or not to save out cleaned df for an animal

    """
    all_animal_dfs = []
    for animal_id in animal_ids:
        try:
            clean = DatasetCleaner(animal_id, column_rename_map, save_out=save_out)
            all_animal_dfs.append(clean.run())
        except Exception as e:
            print(f"Error cleaning {animal_id}: {e}")

    all_animals = pd.concat(all_animal_dfs, ignore_index=True)

    if save_out:  # already in gitignore due to large size
        all_animals.to_csv("../data/cleaned/all_animals_cleaned.csv", index=False)
    return all_animals


class DatasetCleaner:
    def __init__(
        self,
        animal_id,
        column_rename_map,
        load_path="/Volumes/brody/jbreda/PWM_data_scrape/",
        save_out=False,
        save_path="../data/cleaned/by_animal/",
    ):
        self.animal_id = animal_id
        self.column_remap = column_rename_map
        self.load_path = load_path
        self.save_out = save_out
        self.save_path = save_path

    def run(self):
        print(f"** RUNNING {self.animal_id} **")
        self.raw_df = self.load_animal_df()

        # these functions are meant to run in order and
        # modify the raw_df in place
        self.rename_columns()
        self.map_correct_side_and_choice()
        self.make_session_column()
        self.add_old_session_column()
        self.drop_and_account_for_trial_non_starts()

        self.cleaned_df = self.raw_df.copy()

        if self.save_out:
            self.cleaned_df.to_csv(self.save_path + f"{self.animal_id}_cleaned.csv")
        return self.cleaned_df

    def load_animal_df(self):
        return pd.read_csv(self.load_path + f"{self.animal_id}_trials_data.csv")

    def rename_columns(self):
        if not hasattr(self, "raw_df"):
            self.raw_df = self.load_animal_df()

        self.raw_df.rename(columns=self.column_remap, inplace=True)

        return None

    def map_correct_side_and_choice(self):
        self.raw_df["correct_side"] = self.raw_df.correct_side.map(
            {"RIGHT": 1, "LEFT": 0}
        )

        self.raw_df["choice"] = self.raw_df.apply(self.determine_animal_choice, axis=1)

        return None

    def make_session_column(self):
        # convert to date object first
        self.raw_df["session_date"] = pd.to_datetime(
            self.raw_df.session_date, format="%y%m%d"
        )

        # defining a session as all trials from a single day
        self.raw_df["session"] = (
            self.raw_df["session_date"].rank(method="dense").astype(int)
        )

        return None

    def add_old_session_column(self):
        """
        Note- this isn't a perfect alignment between the old and new-
        you can see the quality of alignment in figures/dataset_alignment
        However, it is better than not aligning and will be useful for
        trying to approximate how to truncate the new dataset to replicate
        results with the old dataset when violation stop being tracked at
        session 200.
        """
        alignment_df = pd.read_csv(
            f"../data/processed/dataset_alignment/{self.animal_id}_alignment_df.csv"
        )

        delta_for_align_new_to_old = (
            alignment_df.query("source == 'old'").align_session.values[0]
            - alignment_df.query("source == 'new'").align_session.values[0]
        )

        self.raw_df["session_relative_to_old"] = (
            self.raw_df["session"] + delta_for_align_new_to_old
        )

        return None

    def drop_and_account_for_trial_non_starts(self):
        self.raw_df = (
            self.raw_df.groupby("session")
            .apply(self.calc_n_prev_trial_not_started)
            .reset_index(drop=True)
        )

        return None

    def add_trial_column(self):
        self.raw_df = (
            self.raw_df.groupby("session")
            .apply(self.calc_trial_counts)
            .reset_index(drop=True)
        )

        return None

    @staticmethod
    def determine_animal_choice(row):
        if row.hit == 0:
            return 0
        elif row.hit == 1:
            return 1
        elif row.violation == 1:
            return 2
        elif row.trial_not_started == 1:
            return 3  # will be dropped anyway
        else:
            return -1  # this should happen, make the error clear

    @staticmethod
    def calc_n_prev_trial_not_started(session_group):
        """
        For non-timeout/trial not started trials, determine how many
        consecutive previous trials were timeouts/not started (if any!)

        see 2024_01_23_dev_new_dataset_cleaning.ipynb for validation
        """
        # Convert dtype
        session_group.trial_not_started = session_group.trial_not_started.astype(bool)

        # Calculate the cumulative sum of 'trial_not_started',
        # resetting when a trial is started to only count consecutive non-starts
        session_group["trial_not_started_cumsum"] = session_group[
            "trial_not_started"
        ].cumsum() - session_group["trial_not_started"].cumsum().where(
            ~session_group["trial_not_started"]
        ).ffill().fillna(
            0
        )

        # Shift the cumulative values down to align with the next trial
        # in order to create  "prev history" variable
        session_group["n_prev_trial_not_started"] = session_group[
            "trial_not_started_cumsum"
        ].shift(fill_value=0)

        # Remove the trials where 'trial_not_started', so only the
        # history of them remains on valid trials
        filtered_df = session_group.query("trial_not_started != True").copy()
        # filtered_df = session_group.copy()

        # Drop the temporary cumulative sum column
        filtered_df.drop(["trial_not_started_cumsum"], axis=1, inplace=True)

        return filtered_df

    @staticmethod
    def calc_trial_counts(session_group):
        """
        add trial counter now that timeout/trial not started trials
        are no longer in the data
        """
        session_group["trial"] = np.arange(1, len(session_group) + 1)

        return session_group
