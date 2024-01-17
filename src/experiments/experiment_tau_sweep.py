"""
Child Experiment class that specifically runs a tau
sweep for a chosen variable. Model can be binary or
multi. 

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


class ExperimentTauSweep(Experiment):
    """
    Experiment class for running an experiment to
    differing features (or model types) for a set
    set of animals, taus and parameters
    """

    def __init__(self, params):
        super().__init__(params)

        # tau_sweep is a dict with the keys for all column(s)
        # that are being filtered. the values are either T/F
        # with the single true indicating which column is the sweep experiment
        self.tau_sweep = params["tau_sweep"]
        tau_columns = [f"{key}_tau" for key in self.tau_sweep.keys()]
        self.sweep_column = next(key for key, value in self.tau_sweep.items() if value)
        self.taus = params["taus"]  # actual values to sweep over

        vars = [
            "animal_id",
            "model_name",
            "model_type",
            "nll",
            "train_nll",
            "sigma",
            "features",
            "weights",
            "n_train_trials",
            "n_test_trials",
        ]
        tau_columns = [f"{key}_tau" for key in self.tau_sweep.keys()]
        self.fit_models = pd.DataFrame(columns=vars + tau_columns)
        self.eval_train = params.get("eval_train", False)

        # only one model being tested here so we assume the name is the
        # first key in the model_config dict
        self.model_name = next(iter(self.model_config.keys()))

    def create_filter_params(self, animal_id):
        """
        Create filter params dict for a given animal given the
        tau_sweep dict. This will create a dict with the
        column names, taus (looked up from tau_df) for any variable
        that is in tau_sweep with a value of False

        !! NOTE- this child method overrides the parent method
        !! this is because tau sweeps have unique filter params format
        !! where some columns are being swept and others already have
        !! a tau value
        """

        filter_params = {}

        # Iterate over each item in the tau_sweep dictionary
        for key, value in self.tau_sweep.items():
            # If the value is false, it is not being swept and we
            # need to look it up in the tau_df
            if not value:
                # Query the self.taus with the condition and extract the value
                filter_params[key] = self.taus_df.query("animal_id == @animal_id")[
                    f"{key}_tau"
                ].values[0]

        return filter_params

    def run(self):
        """
        Run experiment for all animals, sigmas, and models
        """
        print("minimum training stage is ", self.min_training_stage)
        for animal_id in self.animals:
            animal_df = self.df.query(
                "animal_id == @animal_id and training_stage >= @self.min_training_stage"
            )

            # get the exp filter params for the animal, columns
            filter_params = self.create_filter_params(animal_id)

            print(
                f"\n >>>> evaluating animal {animal_id} sweeping taus of {self.sweep_column} <<<<"
            )

            self.run_single_animal(animal_id, animal_df, filter_params)

    def run_single_animal(self, animal_id, animal_df, filter_params):
        tts = super().get_animal_train_test_sessions(animal_df)

        # Iterate over taus, create DMs, split, iterate over sigmas, fit
        for tau in self.taus:
            # Update the filter params with the tau & use this for
            # design matrix generation
            filter_params[self.sweep_column] = tau

            X, Y = super().generate_design_matrix_for_animal(
                animal_df, filter_params, self.model_name
            )

            X_train, X_test, Y_train, Y_test = tts.apply_session_split(X, Y)

            for sigma in self.sigmas:
                print(f"\n ***** evaluating tau {tau}, sigma {sigma} *****")

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
                    "model_type": self.model_config[self.model_name]["model_type"],
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
