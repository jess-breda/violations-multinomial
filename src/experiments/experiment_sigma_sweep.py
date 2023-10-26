"""
Child Experiment class that specifically runs a sigma
sweep over a stable (provided) design matrix for each
animal. Model can be binary or multi

Written by Jess Breda
"""

import pathlib
import sys
import pandas as pd
import numpy as np
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

from train_test_splitter import TrainTestSplitter
from design_matrix_generator_interactions import DesignMatrixGeneratorInteractions


class ExperimentSigmaSweep(Experiment):
    """
    Model that runs a sigma sweep for a given
    set of animals, sigmas and parameters
    """

    def __init__(self, params):
        super().__init__(params)
        self.fit_models = pd.DataFrame(
            columns=[
                "animal_id",
                "model_name",
                "nll",
                "sigma",
                "tau",
                "features",
                "weights",
                "n_train_trials",
                "n_test_trials",
            ]
        )

    def run(self, save_name=None):
        # TODO- training stage is hard coded- consider changing
        for animal_id in self.animals:
            print(f"\n\n !!!!! evaluating animal {animal_id} !!!!!\n\n")
            animal_df = self.df.query("animal_id == @animal_id and training_stage > 2")
            tau = self.taus.query("animal_id == @animal_id")["tau"].values[0]
            self.run_single_animal(animal_id, animal_df, tau)

    def run_single_animal(self, animal_id, animal_df, tau):
        # Make design matrix given model configs
        design_matrix_generator_class = globals()[
            self.model_config["design_matrix_generator"]
        ]
        design_matrix_args = self.model_config.get("design_matrix_generator_args", {})
        design_matrix_generator = design_matrix_generator_class(
            model_type=self.model_config["model_type"]
        )

        X, Y = design_matrix_generator.generate_design_matrix(
            df=animal_df,
            tau=tau,
            **design_matrix_args,
        )

        # Split into train and test.
        tts = TrainTestSplitter(self.test_size, self.random_state)
        tts.get_sessions_for_split(X)
        X_train, X_test, Y_train, Y_test = tts.apply_session_split(X, Y)

        # Fit models for each sigma
        for sigma in self.sigmas:
            model = self.model_config["model_class"](sigma=sigma)
            W_fit = model.fit(X_train, Y_train)
            nll = model.eval(X_test, Y_test)

            # Store
            data = {
                "animal_id": animal_id,
                "model_name": self.model_config["model_type"],
                "nll": nll,
                "sigma": sigma,
                "tau": tau,
                "features": X_test.columns,
                "weights": W_fit,
                "n_train_trials": len(X_train),
                "n_test_trials": len(X_test),
            }
            self.store(data, self.fit_models)

    def store(self, data, df):
        return super().store(data, df)
