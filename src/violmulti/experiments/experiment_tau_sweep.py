"""
Child Experiment class that specifically runs a tau
sweep for a chosen variable. Model can be binary or
multi. 

Written by Jess Breda
"""

import pandas as pd
from violmulti.experiments.experiment import Experiment


class ExperimentTauSweep(Experiment):
    """
    Experiment class for running an experiment to
    differing features (or model types) for a set
    set of animals, taus and parameters
    """

    def __init__(self, params):
        params["tau_sweep"] = True
        super().__init__(params)
        super().unpack_config_for_single_model()  # gets model name and type

        # get params for the tau sweep for documentation purposes
        self.get_tau_sweep_params(params)

    def get_tau_sweep_params(self, params):
        """
        Helper function to unpack nested params dict specific
        to the tau_sweep subdict of the dmg config dict
        """
        self.tau_params = params["model_config"][f"{self.model_name}"]["dmg_config"][
            "tau_sweep"
        ]

        self.taus = self.tau_params["taus"]

        self.tau_sweep_column = self.tau_params["col_name"]

    def iterate_sweep_idx(self, idx):
        """
        function to manually iterate the dmg_config tau sweep

        note- this update is also tracked by self.tau_params but cannot be implemented
        in self.tau_param
        """
        self.params["model_config"][f"{self.model_name}"]["dmg_config"]["tau_sweep"][
            "current_idx"
        ] = idx

    def run(self):
        """
        Run experiment for all animals, sigmas, and models
        """
        for animal_id in self.animals:
            animal_df = self.df.query("animal_id == @animal_id")

            # # get the exp filter params for the animal, columns
            # filter_params = self.create_filter_params(animal_id)

            print(
                f"\n >>>> evaluating animal {animal_id} sweeping taus of {self.taus} <<<<"
            )

            self.run_single_animal(animal_id, animal_df)

    def run_single_animal(self, animal_id, animal_df):
        tts = super().get_animal_train_test_sessions(animal_df)

        # Iterate over taus, create DMs, split, iterate over sigmas, fit
        for idx, tau in enumerate(self.taus):

            # updates dmg_config["tau_sweep"]["current_index"] on compile for proper
            # design matrix generation over various taus
            self.iterate_sweep_idx(idx)

            X, Y = super().generate_design_matrix_for_animal(animal_df, self.model_name)

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
                    "model_type": self.model_type,
                    "nll": test_nll,
                    "train_nll": train_nll,
                    "sigma": sigma,
                    "features": X_test.columns,
                    "weights": W_fit,
                    "n_train_trials": len(X_train),
                    "n_test_trials": len(X_test),
                    "tau": tau,
                }
                super().store(data, self.fit_models)
