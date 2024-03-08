"""

Parent class for all experiments to build their .run() method off of. 

To see a list of possible params, see params.py

Has flexible methods for:
- inits from params
- loading dataset
- generates design matrix
- train test splits
- compute null model
- generating filter params from params
- fit and eval models (binary or multi)
- storing model fits
- saving full experiment info


Written by Jess Breda 2023-10-23
Edited by Jess Breda 2024-02-06 for flexible data loading
"""

import gzip
import pickle

from multiglm.utils.fitting_utils import get_taus_df
from multiglm.utils.train_test_splitter import TrainTestSplitter
from multiglm.models.null_model import NullModel
from multiglm.data.dataset_loader import DatasetLoader
from multiglm.data import ANIMAL_IDS
from multiglm.features.design_matrix_generator_PWM import DesignMatrixGeneratorPWM


class Experiment:
    def __init__(self, params):
        self.params = params

        # set up animals
        self.animals = params["animals"]
        if self.animals is None:
            self.animals = ANIMAL_IDS
        self.n_animals = len(self.animals)

        # set up data
        self.df = self.load_dataset()
        self.taus_df = get_taus_df()

        # set up train/test
        self.random_state = params.get("random_state", 47)
        self.test_size = params.get("test_size", 0.2)

        # set up model config (a large-sub dictionary of params)
        # see src/experiment/init_params.py for an example
        self.model_config = params["model_config"]
        self.sigmas = params["sigmas"]

        # init space for null model
        self.null_models = []

    def load_dataset(self):
        """
        Function to return dataset of given data_type with specified
        animal ids that will be passed into design matrix generator
        """
        self.data_type = self.params.get("data_type", "new_trained")
        print(f"Loading dataset type : {self.data_type}")
        loader = DatasetLoader(animal_ids=self.animals, data_type=self.data_type)

        return loader.load_data()

    def generate_design_matrix_for_animal(self, animal_df, model_name, verbose=False):
        """
        Function to generate the design matrix and labels
        with the given animal data and tau and the model configs
        determining which design matrix generator to use

        params
        ------
        animal_df : pd.DataFrame
            dataframe of animal data for a single animal to generate
            design matrix from

        model_name : str
            model_config primary key to determine which model and subsequent
            dmg configuration to use

        returns
        -------
        X : pd.DataFrame (N, D + 2, bias & session)
            design matrix for the animal
        y : np.ndarray (N,C) if multi, (N,1) if binary model type
            labels for the animal
        """

        dmg_config = self.model_config[model_name]["dmg_config"]
        dmg = DesignMatrixGeneratorPWM(animal_df, dmg_config, verbose)

        X, y = dmg.create()

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

    def create_filter_params(self, animal_id, model_name):
        """
        Create filter params dict for a given animal given the
        filter_implementation dict and the tau df.

        If the value is 1, it is being swept and we need to look it up
        in the tau_df. If the value is -1 or 0, it is not being swept
        and we can leave these values as is. The DesignMatrixGenerator
        will handle the filtering in this case (see returns for more info).

        params
        ------
        animal_id : str
            animal id to create filter params for

        returns
        -------
        filter_params : dict
            dictionary with the column names, filter_values for any variable
            that is in filter_implementation
            if the value is 1, the filter_value is looked up in the tau_df
            if the value is -1 or 0, the filter_value is left as is
            to be handled by the DesignMatrixGenerator where 0 means leave
            the column as is, don't filter and -1 means drop the column.
        """

        filter_params = {}
        filter_implementation = self.model_config[model_name].get(
            "filter_implementation", {}
        )

        if not filter_implementation or len(filter_implementation) == 0:
            return filter_params  # empty dict
        else:
            for key, value in filter_implementation.items():
                # If value is 1, we need to look up the tau & set that as the filter value
                if value == 1:
                    # Query the self.taus with the condition and extract the value
                    filter_params[key] = self.taus_df.query("animal_id == @animal_id")[
                        f"{key}_tau"
                    ].values[0]
                # If the value is 0 or -1, no filtering is being applied and
                # the DesignMatrixGenerator will handle this
                elif value == -1 or value == 0:
                    filter_params[key] = value

            return filter_params

    def fit_and_evaluate_model(
        self,
        X_train,
        X_test,
        Y_train,
        Y_test,
        sigma,
        model_name,
        lr_only,
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
        nll = model.eval(X_test, Y_test, lr_only)

        if self.eval_train:
            train_nll = model.eval(X_train, Y_train, lr_only)
            return W_fit, nll, train_nll
        else:
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
        compress=True,
    ):
        """
        Function to save the experiment object as a pickle file
        """

        if compress:
            with gzip.open(file_path + file_name + ".gz", "wb") as f:
                pickle.dump(self, f)

        else:
            with open(file_path + file_name, "wb") as f:
                pickle.dump(self, f)


def load_experiment(
    save_name,
    save_path="/Users/jessbreda/Desktop/github/animal-learning/data/results/",
    compressed=True,
):
    """
    function to load experiment object from pickle file
    """

    if compressed:
        with gzip.open(save_path + save_name + ".gz", "rb") as f:
            experiment = pickle.load(f)

    else:
        with open(save_path + save_name, "rb") as f:
            experiment = pickle.load(f)

    return experiment
