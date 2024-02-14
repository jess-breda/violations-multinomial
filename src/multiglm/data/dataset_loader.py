""" 
Class for the flexible loading of various datasets to be used in
experiments (and called by the experiment.py class family).

This is intentionally written to be extendable- simply add a new
data_type into determine_load_function and corresponding load_function
to execute.

Potential improvements:
-----------------------
Currently this loads the whole new dataset
and then filters based on animal_id or session number. It may
be more efficient to implement this filtering on the csv during 
load, rather than on the data frame after load.

Example usage:
--------------
loader = DatasetLoader(animal_ids = ["W078", "W079"], data_type="old_viols")
data = loader.load_data()

Written By Jess breda 2024-02-06
"""

import pandas as pd
from multiglm.data import ANIMAL_IDS


class DatasetLoader:
    def __init__(self, animal_ids=None, data_type="new_trained"):
        if animal_ids is None:
            animal_ids = ANIMAL_IDS  # load all the animals
        self.animal_ids = animal_ids
        self.data_type = data_type
        self.determine_load_function()

    def determine_load_function(self):
        """
        Given the data type requested by the user, determine the
        appropriate function to load the data.

        Options
        -------
            - new_trained: Only trained (i.e. latest stage, full performance)
                data from the new dataset see `crated_trained_threshold_df.ipynb`
                for more info
            - new_all: all data from the new dataset
            - new_match_old_viols: New dataset truncated to session 200 to
                closely resemble the old dataset where violations stopped
                being tracked at session 200
            - old_viols: the old, public dataset until sessions stopped being
                tracked at session 200
        Returns
        -------
            function: The function to load the data

        """
        if self.data_type == "new_trained":
            self.load_function = self.load_new_trained
        elif self.data_type == "new_all":
            self.load_function = self.load_new_all
        elif self.data_type == "new_match_old_viols":
            self.load_function = self.load_new_match_old_viols
        elif self.data_type == "old_viols":
            self.load_function = self.load_old_viols
        else:
            raise ValueError(f"Invalid data type requested: {self.data_type}")

        return self.load_function

    def load_data(self):
        print("Loading data for animal ids: ", self.animal_ids)
        return self.load_function()

    def load_new_trained(self):
        data = pd.read_csv("../data/processed/all_animals_trained_threshold.csv")
        data = data.query("animal_id in @self.animal_ids").copy()
        return data

    def load_new_all(self):
        data = pd.read_csv("../data/cleaned/all_animals_cleaned.csv")
        data = data.query("animal_id in @self.animal_ids").copy()
        return data

    def load_new_match_old_viols(self):
        data = self.load_new_all()
        data = data.query("session_relative_to_old < 200").copy()
        return data

    def load_old_viols(self):
        data = pd.read_csv("../data/cleaned/old_dataset/old_violation_data.csv")
        data = data[data["animal_id"].isin(self.animal_ids)]
        return data
