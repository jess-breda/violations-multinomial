"""
Model classes for running SSM-based model experiments. 
Experiments can be run on the cluster or locally, and 
all results will be saved to experiment directory on cup.

Written by Jess Breda 2024-06-24
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

import violmulti.utils.save_load as save_load
from violmulti.data.dataset_loader import DatasetLoader
from violmulti.features.design_matrix_generator_PWM import *
from violmulti.models.ssm_glm_hmm import SSMGLMHMM


class SSMExperimentRunner:

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.experiment_dir = self.get_experiment_dir()
        self.config = self.load_format_and_unpack_config()
        self.experiment_runner = self.determine_experiment_runner()
        # self.raw_df = self.load_raw_data()

    def get_experiment_dir(self) -> Path:
        """
        Determine the path to the main data directory on
        Cup where results, raw and processed data are
        stored (from local mac machine or cluster)
        """
        self.cup_data_path = save_load.determine_path_to_cup_data()
        return Path(
            self.cup_data_path / "results" / f"experiment_{self.experiment_name}"
        )

    def _load_config(self) -> dict:

        return save_load.load_config_from_yaml(
            config_path=self.experiment_dir / "config.yaml"
        )

    def load_format_and_unpack_config(self) -> dict:
        """
        Set up the config dictionary to be used in the experiment.
        This means unpacking any necessary attributes that are frequently
        used (e.g. animal_ids, model_config, dmg_config, etc.). As
        well as formatting the design matrix generator config since
        it utilizes lambda functions and these need to be instantiated
        from the str yaml format.

        See the utils.create_experiment_folder.py for the standard
        config format and options.

        # TODO- unclear how this will work with model comparison
        # TODO if we have multiple DMGs?

        """
        # Load
        config = self._load_config()

        # Unpack
        self.animal_ids = config["animal_ids"]
        self.model_config = config["model_config"]

        # Format (lambda functions)
        self.dmg_config = save_load.convert_dmg_config_functions(config["dmg_config"])

        return config

    def determine_experiment_runner(self):
        """
        Given the experiment type from the config,
        determine which experiment runner to use.
        """

        runner = MAP_EXPERIMENT_TO_RUNNER.get(self.config["experiment_type"])

        if runner is None:
            raise ValueError(
                f"Unknown experiment type: {self.config['experiment_type']}"
            )
        return runner

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load the raw data to be passed into the DesignMatrixGenerator
        given configuration. All animals loaded into a single pandas
        dataframe with "animal_id" column that can be used for
        later grouping. See DatasetLoader for more details.

        Note data_tpe is almost always "new_trained" and this means
        use the newest version of the dataset and only trained data.
        """

        return DatasetLoader(
            animal_ids=self.config["animal_ids"],
            data_type=self.config["data_type"],
            relative_data_path=str(self.cup_data_path),
        ).load_data()


class MegaFitExperimentRunner(SSMExperimentRunner):

    def __init__(self, experiment_name):
        super().__init__(experiment_name)


MAP_EXPERIMENT_TO_RUNNER = {
    "mega_fit": MegaFitExperimentRunner,
}
