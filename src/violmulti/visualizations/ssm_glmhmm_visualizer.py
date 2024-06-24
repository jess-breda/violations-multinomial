import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import violmulti.utils.save_load as save_load
from typing import Union, Tuple

from violmulti.models.ssm_glm_hmm import SSMGLMHMM


class SSMVisualizer:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.model_path = None
        self.data_path = None

    def load_model(
        self,
        animal_id: str,
        n_states: int,
        model_name: str,
        n_fold: int,
        n_init: int,
    ) -> SSMGLMHMM:

        model = save_load.load_model_from_pickle(
            animal_id, n_states, model_name, n_fold, n_init, self.model_path
        )

        return model

    def load_data(
        self, animal_id: str, model_name: str, n_fold: int
    ) -> Tuple[pd.DataFrame, np.ndarray]:

        X, y = save_load.load_data_and_labels_from_parquet(
            animal_id, model_name, n_fold, self.data_path
        )

        return X, y

    def plot_transition_matrix():
        pass

    def plot_binary_weights_by_state():
        pass

    def plot_log_like_by_iter():
        pass
        # TODO have this deal with multiple iterations? or write a wrapper
        # in the mega fit to do it

    def plot_posterior_probabilities_over_trials():
        pass

    def plot_posterior_probabilities_summary():
        pass
        # ie collapsing to the y axis in the over trials plot

    def plot_state_occupancies():
        pass

    def plot_n_state_switches_per_session():
        pass

    def plot_dwell_time_by_state():
        pass

    def plot_hit_rate_by_state():
        pass

    def plot_binary_psychometric_by_state():
        pass

    def plot_psychometric():
        pass


class SSMVisualizerMegaFit(SSMVisualizer):
    def __init__(self, experiment_name):
        super().__init__(experiment_name)
        self.n_iterations = None

    def load_all_models(self):
        pass

    def determine_best_fit_model(self):
        pass

        self.best_model = None  # model object

    def plot_log_like_by_iter_by_init():
        pass

    def plot_megafit_summary():
        # mega summary plot
        pass


# plot utils (put in own util)
