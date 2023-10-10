import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class ModelVisualizerSweeps:
    def __init__(self, experiment):
        self.model_fits = experiment.model_fits

    ## Plotting Functions ##

    def plot_nll_over_sigmas_by_animal(self, df=None):
        """
        Plot the test NLL for each sigma value for each animal
        in the experiment. Minimum is marked in red.

        params
        ------
        df: pd.DataFrame (default=None)
            dataframe containing the results of the experiment
            grouped by sigma. if None, will calculate it.
        """

        if df is None:
            df = self.find_best_fit(group="sigma")

        n_animals = df.animal_id.nunique()
        fig, ax = plt.subplots(
            n_animals, 1, figsize=(15, 5 * n_animals), sharex=True, sharey=True
        )

        df["is_min"] = df.groupby("animal_id")["nll"].transform(lambda x: x == x.min())

        if n_animals == 1:
            ax = [ax]

        for idx, (animal_id, sub_df) in enumerate(df.groupby("animal_id")):
            plt.xticks(rotation=90)

            current_ax = ax[idx] if n_animals > 1 else ax[0]

            sns.scatterplot(
                x="sigma",
                y="nll",
                data=sub_df,
                ax=current_ax,
                hue="is_min",
                palette=["grey", "red"],
            )

            # aesthetics
            plt.xticks(rotation=90)
            sns.despine()
            current_ax.legend().remove()
            current_ax.set(
                ylabel="Test NLL",
                title=f"Animal {animal_id}",
            )
            # if on the last plot, add the x-axis label
            if idx == n_animals - 1:
                current_ax.set(xlabel="Sigma")
            else:
                current_ax.set(xlabel="")

        return None

    def plot_weights_by_animal(self, df=None):
        if df is None:
            df = self.find_best_fit(group="animal_id")

        n_animals = df.animal_id.nunique()
        fig, ax = plt.subplots(
            n_animals, 1, figsize=(10, 5 * n_animals), sharex=True, sharey=True
        )

        for i, animal in enumerate(df.animal_id.unique()):
            animal_df = df.query("animal_id == @animal")
            self.plot_weights(
                animal_df.features.values[0],
                animal_df.weights.values[0],
                ax=ax[i],
            )
            ax[i].set_title(f"Animal {animal}, sigma = {animal_df.sigma.values[0]}")

    def plot_weights_across_animals(self, hue=None, df=None, ax=None):
        # Step 1: Find the best fit DataFrame for each animal
        if df is None:
            df = self.find_best_fit(group="animal_id")

        # Step 2 & 3: Prepare the data for seaborn
        plot_data = []
        for _, row in df.iterrows():
            features = row["features"]
            weights = row["weights"]
            animal_id = row["animal_id"]

            for feature, weight in zip(features, weights):
                plot_data.append(
                    {"Feature": feature, "Weight": weight, "Animal": animal_id}
                )

        df_plot = pd.DataFrame(plot_data)

        # Plot using seaborn
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x="Feature",
            y="Weight",
            hue=hue,
            data=df_plot,
            color="cornflowerblue",
            ax=ax,
        )
        ax.axhline(0, color="black")

        ax.set(
            xlabel="Feature",
            ylabel="Weight Value",
            title="Weight by Feature Across Animals",
        )
        plt.xticks(rotation=45)

    @staticmethod
    def plot_weights_binary(X, w, ax=None, title=None):
        """
        Plots the weights for each column in the design matrix X.

        Parameters:
        X (numpy.ndarray): The design matrix with shape (m, n)
        w (numpy.ndarray): The weight vector with shape (n,)

        """

        # if X is a df, grab the columns
        if isinstance(X, pd.DataFrame):
            X = X.columns

        if len(X) != len(w):
            raise ValueError("The number of columns in X must match the length of w.")
        # plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(X, w)
        ax.axhline(0, color="black")

        # aesthetics
        plt.xticks(rotation=45)
        ax.set(
            xlabel="Feature",
            ylabel="Weight Value",
            title="Weight by Feature" if None else title,
        )

    ## Wranlging Functions ##

    def find_best_fit(self, group="model_name"):
        """
        Find the best fit for a given group. For example,
        if search over sigma and taus and group is "tau",
        this function will return the best sigma fit for
        each animal, tau.

        If you just want the best fit, group by "model_name"
        or "animal_id"

        params
        ------
        group: str (default: "tau")
            group to find best fit for


        returns
        -------
        best_fit_df: pd.DataFrame
            dataframe containing the best fit for each
            animal, group
        """

        best_fit_dfs = []

        for animal_id, sub_df in self.model_fits.groupby(["animal_id"]):
            best_idx = sub_df.groupby(group)["nll"].idxmin()
            best_fit_df = sub_df.loc[best_idx]
            best_fit_dfs.append(best_fit_df)

        return pd.concat(best_fit_dfs, ignore_index=True)

    # Plot weights for the best fit model by animal

    # def query_min_nll(self, df=None):
    #     """
    #     Query the row with the minimum NLL for a given animal_id and model_name
    #     """
    #     query = self.model_fits.query(
    #         "animal_id == @animal_id and model_name == @model_name"
    #     ).sort_values(by="nll", ascending=True)

    #     return query.iloc[0]


class ModelVisualizerCompare:
    def __init__(self, experiment):
        self.experiment = experiment
        self.fit_models = experiment.fit_models
        self.null_models = experiment.null_models

        self._init_config_dtypes_()

    def _init_config_dtypes_(self):
        cat_columns = ["model_name", "animal_id"]
        self.fit_models[cat_columns] = self.fit_models[cat_columns].astype("category")
        int_columns = ["n_test_trials", "n_train_trials"]
        self.fit_models[int_columns] = self.fit_models[int_columns].astype(int)
        float_columns = ["nll", "sigma", "tau"]
        self.fit_models[float_columns] = self.fit_models[float_columns].astype(float)

        return None

    ## Plotting functions
    def plot_nll_over_sigmas_by_animal(self, model_name="binary", df=None):
        """
        Plot the test NLL for each sigma value for each animal
        in the experiment. Minimum is marked in red.
        NOTE: this only works for a single model_name!

        params
        ------
        df: pd.DataFrame (default=None)
            dataframe containing the results of the experiment
            grouped by sigma. if None, will calculate it.
        """

        if df is None:
            df = self.fit_models.query(f"model_name == '{model_name}'").copy()

        n_animals = df.animal_id.nunique()
        fig, ax = plt.subplots(
            n_animals, 1, figsize=(15, 5 * n_animals), sharex=True, sharey=True
        )

        df["is_min"] = df.groupby("animal_id")["nll"].transform(lambda x: x == x.min())

        if n_animals == 1:
            ax = [ax]

        for idx, (animal_id, sub_df) in enumerate(df.groupby("animal_id")):
            plt.xticks(rotation=90)

            current_ax = ax[idx] if n_animals > 1 else ax[0]

            sns.scatterplot(
                x="sigma",
                y="nll",
                data=sub_df,
                ax=current_ax,
                hue="is_min",
                palette=["grey", "red"],
            )

            # aesthetics
            plt.xticks(rotation=90)
            sns.despine()
            current_ax.legend().remove()
            current_ax.set(
                ylabel="Test NLL",
                title=f"Animal {animal_id}",
            )
            # if on the last plot, add the x-axis label
            if idx == n_animals - 1:
                current_ax.set(xlabel="Sigma")
            else:
                current_ax.set(xlabel="")

        return None

    def plot_model_comparison(
        self, type="point", hue=None, ax=None, ylim=None, **kwargs
    ):
        """
        Plot the model comparison for each animal
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        if not hasattr(self, "bits_per_trial_df"):
            self.compute_bits_per_trial_df()

        if type == "point":
            sns.pointplot(
                data=self.bits_per_trial_df.query("model_name != 'null'"),
                x="model_name",
                y="bits_per_trial",
                hue=hue,
                ax=ax,
                **kwargs,
            )
        elif type == "bar":
            sns.barplot(
                data=self.bits_per_trial_df.query("model_name != 'null'"),
                x="model_name",
                y="bits_per_trial",
                hue=hue,
                ax=ax,
                **kwargs,
            )

        if hue == "animal_id":
            ax.legend().remove()
        if hue == "tau":
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.xlabel("Model")
        plt.ylabel("Bits/Trial")

        if ylim is not None:
            ax.set(ylim=ylim)

        return None

    ## Wrangling functions
    def find_best_fit(self, group="model_name"):
        """
        Find the best fit for a given group. For example,
        if search over sigma and taus and group is "tau",
        this function will return the best sigma fit for
        each animal, tau.

        If you just want the best fit, group by "model_name"
        or "animal_id"

        params
        ------
        group: str (default: "model_name")
            group to find best fit for


        returns
        -------
        best_fit_df: pd.DataFrame
            dataframe containing the best fit for each
            animal, group
        """

        best_fit_dfs = []

        for animal_id, sub_df in self.fit_models.groupby(["animal_id"]):
            best_idx = sub_df.groupby(group)["nll"].idxmin()
            best_fit_df = sub_df.loc[best_idx]
            best_fit_dfs.append(best_fit_df)

        return pd.concat(best_fit_dfs, ignore_index=True)

    def merge_null_and_fit_models_dfs(self):
        # for each model, find best fit given sigma
        best_fit_df = self.find_best_fit(group=["model_name"]).copy()
        null_df = self.null_models.copy()

        # Select relevant columns & set inidices for merge
        best_fit_df_selected = best_fit_df.set_index(["animal_id", "model_name"])[
            ["nll", "sigma", "tau", "n_train_trials", "n_test_trials"]
        ]
        null_df_selected = null_df.set_index(["animal_id", "model_name"])[
            ["nll", "n_train_trials", "n_test_trials"]
        ]

        merged_df = pd.concat([best_fit_df_selected, null_df_selected])
        merged_df.sort_values(["animal_id", "model_name"], inplace=True)
        merged_df.reset_index(inplace=True)
        merged_df["log_like"] = merged_df["nll"] * -1

        self.all_models_df = merged_df.copy()

    def compute_bits_per_trial_df(self):
        """
        Compute the bits per trial for each model for each animal
        in the experiment.

        Requires having a merged dataframe with null model
        and fit model(s) containing nll and n_test trials.
        """
        if not hasattr(self, "all_models_df"):
            self.merge_null_and_fit_models_dfs()

        bits_per_trial_df = (
            self.all_models_df.groupby("animal_id")
            .apply(self._calculate_bits_per_trial)
            .reset_index(drop=True)
        )

        self.bits_per_trial_df = bits_per_trial_df
        return self.bits_per_trial_df

    def _calculate_bits_per_trial(self, group):
        """
        Calculate the number of bits per trial relative for each model
        relative to the null model (L_0) for each animal (group)

        bit/trial = (log model - log null model) / (n_test_trials * log(2))

        params
        ------
        group : pd.DataFrame
            groupby object with columns `animal_id`, `model_name`,
            `n_test_trials` and `ll` (log-likelihood)
            grouped by animal_id
        """
        bits_per_trial = []

        n_test_trials = group["n_test_trials"].iloc[
            0
        ]  # assumes each animal has same test size

        null_ll = group.query("model_name == 'null'")["log_like"].iloc[0]

        for _, row in group.iterrows():
            if row["model_name"] == "null":
                bits_per_trial.append((null_ll) / (n_test_trials * np.log(2)))
            else:
                bits_per_trial.append(
                    (row["log_like"] - null_ll) / (n_test_trials * np.log(2))
                )

        group["bits_per_trial"] = bits_per_trial
        return group
