from violmulti.data.dataset_loader import DatasetLoader
from violmulti.features.design_matrix_generator_PWM import (
    DesignMatrixGeneratorPWM,
    prepare_data_for_ssm,
)
from violmulti.visualizations.ssm_visualizer import *
import ssm
import matplotlib.pyplot as plt


class RunBinaryGLMHMM:

    def __init__(self, params):
        self.animal_ids = params["animal_ids"]
        self.seed = params.get("seed", 0)
        self.raw_df = self.load_trials_data()
        self.dmg_config = params["dmg_config"]
        self.model_config = params["model_config"]
        self.unpack_model_config()
        self.save_path = params.get("save_path", None)

    def load_trials_data(self):
        df = DatasetLoader(
            animal_ids=self.animal_ids,
            data_type="new_trained",
            relative_data_path="../data/",
        ).load_data()
        return df

    def unpack_model_config(self):
        self.n_states = self.model_config["n_states"]
        self.n_features = self.model_config["n_features"]
        self.n_categories = self.model_config["n_categories"]
        self.transitions = self.model_config.get("transitions", "standard")
        self.n_iters = self.model_config.get("n_iters", 200)
        self.prior_sigma = self.model_config.get("prior_sigma", None)
        self.prior_alpha = self.model_config.get("prior_alpha", 0)
        self.prior_kappa = self.model_config.get("prior_kappa", 0)

        if self.transitions != "standard":
            self.transition_kwargs = dict(self.prior_alpha, self.prior_kappa)
        else:
            self.transition_kwargs = None

        if self.prior_sigma is None:
            self.observation_kwargs = dict(C=self.n_categories)
        else:
            self.observation_kwargs = dict(
                C=self.n_categories, prior_sigma=self.prior_sigma
            )

    def run(self):

        for animal_id, animal_df in self.raw_df.groupby("animal_id"):
            print(f"Running GLM-HMM for animal {animal_id}")
            Xs, ys = self.generate_design_matrix(animal_df.reset_index())
            log_probs, model = self.fit_model()
            self.visualize(
                log_probs,
                model,
                Xs,
                ys,
                animal_id,
                self.feature_names,
                seed=self.seed,
                save_path=self.save_path,
            )
            print("-----------\n")

    def generate_design_matrix(self, animal_df):
        dmg = DesignMatrixGeneratorPWM(animal_df, self.dmg_config, verbose=True)
        X, y = dmg.create()  # N total trials long
        self.feature_names = [col for col in X.columns if col != "session"]
        self.Xs, self.ys = prepare_data_for_ssm(X, y)  # Jagged arrays N sessions long

        return self.Xs, self.ys

    def fit_model(self):

        # Initialize
        # TODO - add in weight initialization
        glmhmm = ssm.HMM(
            K=self.n_states,
            D=1,
            M=self.n_features,
            observations="input_driven_obs",
            observation_kwargs=self.observation_kwargs,
            transitions=self.transitions,
            transition_kwargs=self.transition_kwargs,
        )

        # Fit
        log_probs = glmhmm.fit(
            datas=self.ys,
            inputs=self.Xs,
            masks=None,
            method="em",
            num_iters=self.n_iters,
            tolerance=10**-4,
        )

        return log_probs, glmhmm

    @staticmethod
    def visualize(
        log_probs, model, Xs, ys, animal_id, feature_names, seed=0, save_path=None
    ):
        layout = """
            ABBB
            CDDD
            EFFF
            GHHH
        """

        n_states = model.K

        fig = plt.figure(constrained_layout=True, figsize=(23, 16))

        ax_dict = fig.subplot_mosaic(layout)  # ax to plot to
        plt.suptitle(f"{animal_id} GLM-HMM Summary Plot", fontweight="semibold")

        plot_transition_matrix(model.transitions.params, ax=ax_dict["A"])

        weights = model.observations.params
        plot_bernoulli_weights_by_state(
            weights, ax=ax_dict["C"], feature_names=feature_names
        )
        plot_log_probs_over_iters(log_probs, ax=ax_dict["E"], color="black")

        state_occupancies = get_state_occupancies(model, ys, inputs=Xs)
        plot_state_occupancies(state_occupancies, ax=ax_dict["G"])

        posterior_state_probs = get_posterior_state_probs(model, ys, inputs=Xs)

        # get 4 random sessions given between 0 and len(ys)
        np.random.seed(seed)
        random_sessions = np.random.choice(range(len(ys)), 4)
        plots = ["B", "D", "F", "H"]
        for session, plot in zip(random_sessions, plots):
            plot_state_posterior(
                posterior_state_probs[session],
                ax=ax_dict[plot],
                title=f"Session {session}",
            )

        if save_path:
            save_name = f"{animal_id}_BGLM_HMM_{n_states}_states_summary.png"
            plt.savefig(save_path + "/" + save_name, bbox_inches="tight")
            print(f"VIZ: Summary figure saved")
            plt.close("all")
