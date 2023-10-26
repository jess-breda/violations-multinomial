"""
Parent class for all experiments. This class contains
init and storing information. Eventually may be updated
to include running information

Written by Jess Breda 2023-10-23
"""


import pathlib
import sys
import pickle

try:
    [
        sys.path.append(str(folder))
        for folder in pathlib.Path("../src/").iterdir()
        if folder.is_dir()
    ]
except:
    [
        sys.path.append(str(folder))
        for folder in pathlib.Path("../../src/").iterdir()
        if folder.is_dir()
    ]

from get_rat_data import get_rat_viol_data
from fitting_utils import get_taus_df


class Experiment:
    def __init__(self, params):
        self.animals = params["animals"]
        self.sigmas = params["sigmas"]
        self.df = get_rat_viol_data(animal_ids=self.animals)
        self.taus = get_taus_df()
        self.random_state = params.get("random_state", 23)
        self.test_size = params.get("test_size", 0.2)
        self.null_models = []
        self.model_config = params["model_config"]

        if self.animals is None:
            self.animals = self.df.animal_id.unique()
        self.n_animals = len(self.animals)

    @staticmethod
    def store(data, df):
        """
        Function to store the fit information for a single
        animal and model sweep. This creates a single row of the
        self.fit_models data frame.

        params
        ------
        data : dict
            dictonary with keys corresponding to the columns
            of df and values corresponding to the values
            of a single fit row
        df : pd.DataFrame
            dataframe of fit where each row corresponds to
            a single animal with fitting parameters
        """
        # assure df and data have same columns & keys, respectively
        # assert df.columns == data.keys(), "fit data & df columns don't match!"

        # append to df
        next_index = len(df)
        for key, value in data.items():
            df.loc[next_index, key] = value
        return None

    def save(self, file_name, file_path="../data/results/"):
        """
        Function to save the experiment object as a pickle file
        """
        with open(file_path + file_name, "wb") as f:
            pickle.dump(self, f)
