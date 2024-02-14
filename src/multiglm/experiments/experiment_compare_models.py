import pathlib
import sys
import pandas as pd
from multiglm.experiments.experiment import Experiment


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
        if params["tau_columns"]:
            tau_columns = [f"{col_name}_tau" for col_name in params["tau_columns"]]
            self.fit_models = pd.DataFrame(columns=vars + tau_columns)
        else:
            self.fit_models = pd.DataFrame(columns=vars)
        self.eval_train = params.get("eval_train", False)

    def run(self, min_training_stage=3):
        """
        Run experiment for all animals, sigmas, and models
        """
        print("minimum training stage is ", self.min_training_stage)
        for animal_id in self.animals:
            animal_df = self.df.query(
                "animal_id == @animal_id and training_stage >= @self.min_training_stage"
            )

            print(f"\n >>>> evaluating animal {animal_id} <<<<")
            self.run_single_animal(animal_id, animal_df)

        self.null_models = pd.concat(self.null_models, ignore_index=True)

    def run_single_animal(self, animal_id, animal_df):
        """
        Run experiment for a single animal that sweeps over models and sigmas
        """
        # Train test split & Null Model- only compute once for each animal
        # and keep stable across models, sigmas
        tts = super().get_animal_train_test_sessions(animal_df)
        super().compute_null_model(animal_df, tts.test_sessions)

        # Fit and evaluate models, sigmas
        for model_name, config in self.model_config.items():
            # filter params are unique to each model name
            filter_params = super().create_filter_params(animal_id, model_name)

            # Generate design matrix & split
            X, Y = super().generate_design_matrix_for_animal(
                animal_df, filter_params, model_name
            )

            # if lr_only, then test set is just L/R trials and cost is computed only
            # on these trials
            lr_only_eval = self.model_config[model_name].get("lr_only_eval", False)
            X_train, X_test, Y_train, Y_test = tts.apply_session_split(
                X, Y, lr_only_eval
            )

            for sigma in self.sigmas:
                print(f"\n ***** evaluating model {model_name} w/ sigma {sigma} *****")
                W_fit, test_nll, train_nll = super().fit_and_evaluate_model(
                    X_train,
                    X_test,
                    Y_train,
                    Y_test,
                    sigma,
                    model_name,
                    lr_only_eval,
                )

                # Store
                data = {
                    "animal_id": animal_id,
                    "model_name": model_name,
                    "model_type": config["model_type"],
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
