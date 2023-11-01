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


class ExperimentCompareModels(Experiment):
    """
    Experiment class for comparing models with
    differing features (or model types) for a set
    set of animals, sigmas and parameters
    """

    def __init__(self, params):
        super().__init__(params)
        self.null_mode = params["null_mode"]
        self.null_models = []
        self.fit_models = pd.DataFrame(
            columns=[
                "animal_id",
                "model_name",
                "model_type",
                "nll",
                "sigma",
                "tau",
                "features",
                "weights",
                "n_train_trials",
                "n_test_trials",
            ]
        )

    def run(self):
        """
        Run experiment for all animals, sigmas, and models
        TODO- training stage is hard coded- consider changing
        """

        for animal_id in self.animals:
            animal_df = self.df.query("animal_id == @animal_id and training_stage > 2")

            tau = self.taus.query("animal_id == @animal_id")["tau"].values[0]

            print(f"\n >>>> evaluating animal {animal_id} <<<<")
            self.run_single_animal(animal_id, animal_df, tau)

        self.null_models = pd.concat(self.null_models, ignore_index=True)

    def run_single_animal(self, animal_id, animal_df, tau):
        """
        Run experiment for a single animal that sweeps over models and sigmas
        """
        # Train test split & Null Model- only compute once for each animal
        # and keep stable across models, sigmas
        tts = super().get_animal_train_test_sessions(animal_df)
        super().compute_null_model(animal_df, tts.test_sessions)

        # Fit and evaluate models, sigmas
        for model_name, config in self.model_config.items():
            # Generate design matrix & split
            X, Y = super().generate_design_matrix_for_animal(animal_df, tau, model_name)
            X_train, X_test, Y_train, Y_test = tts.apply_session_split(X, Y)

            for sigma in self.sigmas:
                print(f"\n ***** evaluating model {model_name} w/ sigma {sigma} *****")
                W_fit, nll = super().fit_and_evaluate_model(
                    X_train, X_test, Y_train, Y_test, sigma, model_name
                )

                # Store
                data = {
                    "animal_id": animal_id,
                    "model_name": model_name,
                    "model_type": config["model_type"],
                    "nll": nll,
                    "sigma": sigma,
                    "tau": tau,
                    "features": X_test.columns,
                    "weights": W_fit,
                    "n_train_trials": len(X_train),
                    "n_test_trials": len(X_test),
                }
                super().store(data, self.fit_models)
