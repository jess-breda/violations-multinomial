import ssm
import logging
import pickle
from pathlib import Path
import numpy as np


# TODO
# 1. Add type hints to all methods
# 2. Add docstrings to all methods
# 3. Add weight/state initialization logic
# need to determine what the inputs are for this and how
# much is inferred (ie will need the name of where to pull from)
# will also need a noise additive function
# 4. Determine seed setting in original class and write own if none


class SSMGLMHMM(ssm.HMM):
    """
    Child class of ssm.HMM that adds additional functionality for
    fitting glm-hmm models specific to my (Jess Breda's) use case
    of fitting models to binary and trinomial trial-by-trial choice
    data.

    """

    def __init__(
        self,
        model_config: dict,
        model_name: str = "glmhmm",
        n_fold: int = 1,
        results_path=None,
    ):

        self.model_config = model_config
        self.unpack_model_config()
        self.set_up_priors()

        self.n_fold = n_fold
        self.model_name = model_name
        self.results_path = results_path

        super().__init__(
            K=self.K,
            D=1,  # never have more than 1 output dimension
            M=self.M,
            observations="input_driven_obs",
            observation_kwargs=self.observation_kwargs,
            transitions=self.transitions,
            transition_kwargs=self.transition_kwargs,
        )

        # TODO logic here for initializing weights and transitions
        # TODO if model config var exists

    def unpack_model_config(self):
        """
        Method to unpack the model config dictionary into class
        attributes. Some are exact duplicates from ssm.HMM, others
        are custom to this class.
        """
        self.K = self.model_config["n_states"]
        self.M = self.model_config["n_features"]
        self.C = self.model_config["n_categories"]
        self.transitions = self.model_config.get("transitions", "standard")
        self.n_iters = self.model_config.get("n_iters", 200)
        self.prior_sigma = self.model_config.get("prior_sigma", None)
        self.prior_alpha = self.model_config.get("prior_alpha", 0)
        self.prior_kappa = self.model_config.get("prior_kappa", 0)
        self.masks = self.model_config.get("masks", None)
        self.tolerance = self.model_config.get("tolerance", 1e-4)
        self.seed = self.model_config.get("seed", 0)
        logging.info(f"Unpacked model config: {self.model_config}")

    def set_up_priors(self):
        # Set up the kwargs for the model- can't pass in 0 values and
        # have them be ignored, so need to do this manually
        if self.transitions == "sticky":
            self.transition_kwargs = dict(self.prior_alpha, self.prior_kappa)
        elif self.transitions == "standard":
            self.transition_kwargs = None
        else:
            raise ValueError("Invalid transition type for SSM GLM-HMM.")
        logging.info(f"Transition kwargs set: {self.transition_kwargs}")

        if self.prior_sigma is None:
            self.observation_kwargs = dict(C=self.C)
        else:
            self.observation_kwargs = dict(C=self.C, prior_sigma=self.prior_sigma)
        logging.info(f"Observation kwargs set: {self.observation_kwargs}")

    def initialize_weights(self):
        """
        Initialize the weights of the model. Placeholder for actual implementation.
        """
        np.random.seed(self.seed)
        pass

    def initialize_transitions(self):
        """
        Initialize the transitions of the model. Placeholder for actual implementation.
        """
        # Initialize transitions logic here

    def fit(self, X, y):
        self.X = X
        self.y = y

        self.log_probs = super().fit(
            datas=self.y,
            inputs=self.X,
            masks=self.masks,
            method="em",
            num_iters=self.n_iters,
            tolerance=self.tolerance,
        )

        return self.log_probs

    def compute_stats_of_interest(self):
        """
        Statistics of interest that are easiest to calculate
        after fitting the model while the data is still in memory.

        1. log likelihood
        2. posterior state probs (in list of list by session)
        """

        self.log_like = self.log_likelihood(self.y, self.X)
        self.posterior_state_probs = self.get_posterior_state_probs()

    def get_posterior_state_probs(self):
        posterior_probs = []
        for sesssion_choices, sesssion_inputs in zip(self.y, self.X):

            # expected_states returns
            # [posterior_state_probs, posterior_joint_probs, normalizer]
            # so we only need the first element of the returned list
            posterior_probs.append(
                self.expected_states(data=sesssion_choices, input=sesssion_inputs)[0]
            )
        return posterior_probs

    def save(self):

        self.compute_stats_of_interest()

        if self.results_path is None:
            self.results_path = Path.cwd() / "model_results"

            if not self.results_path.exists():
                self.results_path.mkdir()
        else:
            self.results_path = Path(self.results_path)
            self.results_path.mkdir(
                parents=True, exist_ok=True
            )  # Ensure the directory exists

        with open(
            f"{self.results_path}/{self.model_name}_model_fold_{self.n_fold}.pkl", "wb"
        ) as f:
            pickle.dump(self, f)
