"""
Child Experiment class that specifically runs a sigma
sweep over a stable (provided) design matrix for each
animal. Model can be binary or multi

Written by Jess Breda
"""

import pathlib
import sys
import pandas as pd
from experiment import Experiment

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


class ExperimentSigmaSweep(Experiment):
    """
    Model that runs a sigma sweep for a given
    set of animals, sigmas and parameters
    """

    def __init__(self, params):
        super().__init__(params)
        vars = [
            "animal_id",
            "model_name",
            "nll",
            "train_nll",
            "sigma",
            "features",
            "weights",
            "n_train_trials",
            "n_test_trials",
        ]

        # only one model being tested here so we assume the name is the
        # first key in the model_config dict
        self.model_name = next(iter(self.model_config.keys()))

        if params["tau_columns"] is not None:
            tau_columns = [f"{col_name}_tau" for col_name in params["tau_columns"]]
        else:
            tau_columns = []
        self.fit_models = pd.DataFrame(columns=vars + tau_columns)
        self.eval_train = params.get("eval_train", False)

    def run(self):
        print("minimum training stage is ", self.min_training_stage)
        for animal_id in self.animals:
            animal_df = self.df.query(
                "animal_id == @animal_id and training_stage >= @self.min_training_stage"
            )

            print(f"\n >>>> evaluating animal {animal_id} <<<<")
            self.run_single_animal(animal_id, animal_df)

    def run_single_animal(self, animal_id, animal_df):
        """
        Run an experiment given a fixed design matrix for
        a single animal that sweeps over sigmas
        """
        # get filter params (if any) & build design matrix
        filter_params = super().create_filter_params(animal_id, self.model_name)
        X, Y = super().generate_design_matrix_for_animal(
            animal_df, filter_params, self.model_name
        )

        # train test split
        tts = super().get_animal_train_test_sessions(animal_df)
        X_train, X_test, Y_train, Y_test = tts.apply_session_split(X, Y)

        for sigma in self.sigmas:
            print(f"\n ***** evaluating model {self.model_name} w/ sigma {sigma} *****")
            W_fit, test_nll, train_nll = super().fit_and_evaluate_model(
                X_train,
                X_test,
                Y_train,
                Y_test,
                sigma,
                self.model_name,
                lr_only=False,
            )

            # Store
            data = {
                "animal_id": animal_id,
                "model_name": self.model_name,
                "nll": test_nll,
                "train_nll": train_nll,
                "sigma": sigma,
                "features": X_test.columns,
                "weights": W_fit,
                "n_train_trials": len(X_train),
                "n_test_trials": len(X_test),
                **{f"{key}_tau": value for key, value in filter_params.items()},
            }
            super().store(data, self.fit_models)
