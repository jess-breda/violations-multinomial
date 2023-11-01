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
from train_test_splitter import TrainTestSplitter
from null_model import NullModel


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

    def generate_design_matrix_for_animal(self, animal_df, violation_tau, model_name):
        """
        Function to generate the design matrix and labels
        with the given animal data and tau and the model configs
        determining which design matrix generator to use

        params
        ------
        animal_df : pd.DataFrame
            dataframe of animal data for a single animal to generate
            design matrix from
        violation_tau : float
            tau value for the animal for prev_violation history

        returns
        -------
        X : pd.DataFrame (N, D + 2, bias & session)
            design matrix for the animal
        y : np.ndarray (N,C) if multi, (N,1) if binary model type
            labels for the anima

        TODO clean up tau naming etc. for example in interactions design matrix
        TODO the column for exp filter is flexible but now passing in a tau
        TODO specific to violations. consider changing to tau_args that is a dict
        TODO in the form {"filter_column": tau_value} and dmg handles this internally
        TODO this will require get_taus_df to be able to unpack multiple columns
        """
        # Determine class for design matrix generator e.g. DesignMatrixGeneratorInteractions
        design_matrix_generator_class = self.model_config[model_name][
            "design_matrix_generator"
        ]

        # Determine arguments for design matrix generator e.g. {filter_column": "prev_violation"}
        design_matrix_args = self.model_config[model_name].get(
            "design_matrix_generator_args", {}
        )

        # Create design matrix generator object e.g. DesignMatrixGeneratorInteractions(model_type="multi")
        design_matrix_generator = design_matrix_generator_class(
            model_type=self.model_config[model_name]["model_type"]
        )

        # Generate design matrix (X) and labels (y) given args & model_type
        X, y = design_matrix_generator.generate_design_matrix(
            df=animal_df, tau=violation_tau, **design_matrix_args
        )

        return X, y

    def get_animal_train_test_sessions(self, animal_df):
        """
        Function to get the train and test sessions for a single animal
        and return the object with the sessions stored as attributes

        params
        ------
        animal_df : pd.DataFrame
            dataframe with column `session` for a single animal. note this
            is usually not the design matrix, but instead df used to generate
            the design matrices

        returns
        -------
        tts : TrainTestSplitter object
            object with train and test sessions stored as attributes
            that can be used to apply the split to a design matrix
        """
        # Initialize TrainTestSplitter object & get sessions for split
        tts = TrainTestSplitter(
            test_size=self.test_size, random_state=self.random_state
        )
        tts.get_sessions_for_split(animal_df)

        return tts

    def compute_null_model(self, animal_df, test_sessions):
        """
        Function to compute the null model for a single animal
        and store the information in self.null_models

        params
        ------
        animal_df : pd.DataFrame
            dataframe with columns `choice` and `session` for a single
            animal. Note this is not the design matrix!
        test_sessions : list
            list of sessions to use for testing, likely computed by
            the TrainTestSplitter object

        returns
        -------
        None
        """

        null_model = NullModel(test_sessions=test_sessions, mode=self.null_mode)
        self.null_models.append(null_model.compute_and_store(animal_df))

        return None

    def fit_and_evaluate_model(
        self, X_train, X_test, Y_train, Y_test, sigma, model_name
    ):
        """
        Function to fit and evaluate a model given the training
        and testing data and the sigma (L2 regularization) value

        returns
        -------
        W_fit : np.ndarray (D, C) if multi, (D,1) if binary model type
            weights for the model
        nll : float
            negative log likelihood of the model
        """

        # Initialize model e.g. MultiLogisticRegression(sigma=sigma)
        model = self.model_config[model_name]["model_class"](sigma=sigma)

        # fit & eval model
        W_fit = model.fit(X_train, Y_train)
        nll = model.eval(X_test, Y_test)

        return W_fit, nll

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

    def save(
        self,
        file_name,
        file_path="/Users/jessbreda/Desktop/github/animal-learning/data/results/",
    ):
        """
        Function to save the experiment object as a pickle file
        """
        with open(file_path + file_name, "wb") as f:
            pickle.dump(self, f)
