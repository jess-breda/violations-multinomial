import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class ModelVisualizer:
    def __init__(self, experiment):
        self.experiment = experiment
        self.fit_models = experiment.fit_models
        self._init_config_dtypes_()

    def _init_config_dtypes_(self):
        """
        Helper function to assign the correct dtypes to the
        fit_models dataframe for plotting
        """
        cat_columns = ["model_name", "animal_id"]
        self.fit_models[cat_columns] = self.fit_models[cat_columns].astype("category")

        int_columns = ["n_test_trials", "n_train_trials"]
        self.fit_models[int_columns] = self.fit_models[int_columns].astype(int)

        float_columns = ["nll", "sigma", "tau"]
        self.fit_models[float_columns] = self.fit_models[float_columns].astype(float)

        return None

    # DF HELPERS
    def find_best_fit(self, group="animal_id"):
        """
        Find the best fit for a given group. For example,
        if search over sigma and taus and group is "tau",
        this function will return the best sigma fit for
        each animal, tau.

        If you just want the best fit over all parameters,
        or you only swept over one parameter, group by animal_id.

        params
        ------
        group: str (default: "tau")
            group to find best fit for


        returns
        -------
        best_fit_df: pd.DataFrame
            dataframe containing the best fit for each
            animal, group (or just each animal if group is "animal_id")
        """

        if group == "animal_id":
            best_idx = self.fit_models.groupby("animal_id").nll.idxmin()
            return self.fit_models.loc[best_idx].copy().reset_index(drop=True)

        else:
            best_fit_dfs = []
            for _, sub_df in self.fit_models.groupby(["animal_id"]):
                best_idx = sub_df.groupby(group)["nll"].idxmin()
                best_fit_df = sub_df.loc[best_idx]
                best_fit_dfs.append(best_fit_df)
            return pd.concat(best_fit_dfs, ignore_index=True)

    def unpack_features_and_weights(self, df=None):
        """
        Unpacks the "feature" and "weight" columns of a row
        to crete a long-form dataframe where each row corresponds
        to a animal_id, feature, class weight.

        params
        ------
        df : pd.DataFrame
            A pandas DataFrame with "feature" and "weight"
            columns as arrays. Note the weight column can be
            a 1d (binary model) or 2d (mutliclass model)

        returns
        -------
        melted_df : pd.DataFrame
            A long-form pandas DataFrame where each row corresponds
            to a animal_id, feature, class weight.
        """
        if df is None:
            df = self.find_best_fit(group="animal_id")

        # creates df where each row corresponds to a animal_feature
        unpacked_df = pd.concat(
            df.apply(self.unpack_row, axis=1).tolist(), ignore_index=True
        )

        melted_df = pd.melt(
            unpacked_df,
            id_vars=["animal_id", "sigma", "tau", "model_name", "nll", "feature"],
            var_name="weight_class",
            value_name="weight",
        )

        return melted_df

    @staticmethod
    def unpack_row(row):
        """
        Unpacks the "feature" and "weight" columns of a row
        to crete a long-form dataframe where each row corresponds
        to a animal_id, feature. Note this function is flexible
        to the number of classes

        params
        ------
        row : pd.Series
            A row of a pandas DataFrame with "feature" and "weight"
            columns as arrays.
        """

        n_classes = row["weights"].shape[1]
        temp_df = pd.DataFrame(
            {
                "animal_id": row["animal_id"],
                "sigma": row["sigma"],
                "tau": row["tau"],
                "model_name": row["model_name"],
                "nll": row["nll"],
                "feature": row["features"],
            }
        )

        if n_classes == 3:
            class_names = ["L", "R", "V"]
        else:
            class_names = [f"weight_class_{i+1}" for i in range(n_classes)]

        for i in range(n_classes):
            temp_df[class_names[i]] = row["weights"][:, i]

        return temp_df

    # PLOTS

    ## SIGMAS
    def plot_nll_over_sigmas_by_animal(self, group="sigma", df=None, **kwargs):
        """
        Plot the test NLL for each sigma value for each animal
        in the experiment. Minimum is marked in red.

        params
        ------
        group: str (default: "sigma")
            group to find best fit for, default will collapse over
            all other parameters to animal_id, sigma
            if doing model_comparison group = ["sigma", "model_name"]
        df: pd.DataFrame (default=None)
            dataframe containing the results of the experiment
            grouped by sigma. if None, will calculate it.
        kwargs: dict
            additional keyword arguments to pass to seaborn.lineplot()

        """

        if df is None:
            df = self.find_best_fit(group=group)

        n_animals = df.animal_id.nunique()
        fig, ax = plt.subplots(
            n_animals, 1, figsize=(15, 5 * n_animals), sharex=True, sharey=False
        )

        df["is_min"] = df.groupby("animal_id")["nll"].transform(lambda x: x == x.min())

        if n_animals == 1:
            ax = [ax]

        for idx, (animal_id, sub_df) in enumerate(df.groupby("animal_id")):
            plt.xticks(rotation=90)

            current_ax = ax[idx] if n_animals > 1 else ax[0]

            sns.lineplot(
                x="sigma",
                y="nll",
                data=sub_df,
                ax=current_ax,
                marker="o",
                **kwargs,
            )

            current_ax.axvline(
                sub_df[sub_df.is_min].sigma.values, color="red", linestyle="--"
            )

            # aesthetics
            plt.xticks(rotation=90)
            sns.despine()
            current_ax.set(
                ylabel="Test NLL",
                title=f"Animal {animal_id}",
            )
            # if on the last plot, add the x-axis label
            if idx == n_animals - 1:
                current_ax.set(xlabel="Sigma")
            else:
                current_ax.set(xlabel="")
                current_ax.legend().remove()

        return None

    def plot_best_sigma_counts(self, df=None, ax=None):
        """
        Plot count across the best sigmas for each animal in
        an experiment

        params
        ------
        df: pd.DataFrame (default=None)
            dataframe containing the results of the experiment
            grouped by animal_id. if None, will calculate it.
        """

        if df is None:
            df = self.find_best_fit(group="animal_id")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        sns.countplot(data=df, x="sigma", color="gray", ax=ax)
        _ = ax.set(xlabel="Best sigma", ylabel="Number of animals")

        return None

    def plot_best_sigma_by_animal(self, df=None, ax=None):
        """
        Plot animal id by best sigma

        params
        ------
        df: pd.DataFrame (default=None)
            dataframe containing the results of the experiment
            grouped by animal_id. if None, will calculate it.
        """

        if df is None:
            df = self.find_best_fit(group="animal_id")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        sns.pointplot(
            data=df, x="animal_id", y="sigma", ax=ax, join=False, hue="animal_id"
        )

        _ = ax.set_xticks(
            ax.get_xticks(), ax.get_xticklabels(), rotation=90, ha="right"
        )
        _ = ax.set(ylabel="Best sigma", xlabel="")
        ax.get_legend().set_visible(False)

        return None

    def plot_sigma_summary(self, df=None, title=None):
        """
        Wrapper function around plot_best_sigma_by_animal()
        and plot_best_sigma_counts()
        """

        if df is None:
            df = self.find_best_fit(group="animal_id")

        fig, ax = plt.subplots(1, 2, figsize=(16, 5))
        self.plot_best_sigma_by_animal(df=df, ax=ax[0])
        self.plot_best_sigma_counts(df=df, ax=ax[1])

        if title is None:
            plt.suptitle("Fit Sigma Summary")
        else:
            plt.suptitle(title)

        return None

    ## WEIGHTS
    @staticmethod
    def plot_weights(df, ax, title="", **kwargs):
        """
        Workhorse function for plotting weights across features
        and classes.

        params
        ------
        df : pd.DataFrame
            dataframe with columns "feature", "weight_class", "weight"
            row indexed by animal id, feature and weight class
            created by unpack_features_and_weights()
        ax : matplotlib axis
            axis to plot on.
        title : str (default="")
            title of plot
        kwargs : dict
            additional keyword arguments to pass to seaborn.barplot()
        """

        sns.barplot(
            x="feature", y="weight", hue="weight_class", data=df, ax=ax, **kwargs
        )
        ax.axhline(y=0, color="black")

        _ = ax.set_xticks(
            ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right"
        )
        _ = ax.set(xlabel="", ylabel="Weight", title=title)

        return None

    def plot_weights_by_animal(self, df=None, **kwargs):
        """
        Wrapper around plot_weights to plot the weights for each
        animal in the experiment.

        params
        ------
        df : pd.DataFrame (default=None)
            dataframe with columns "feature", "weight_class", "weight"
            row indexed by animal id, feature and weight class
            created by unpack_features_and_weights()
        kwargs : dict
            additional keyword arguments to pass to seaborn.barplot()
        """
        if df is None:
            df = self.unpack_features_and_weights()

        n_animals = df.animal_id.nunique()
        fig, ax = plt.subplots(
            n_animals, 1, figsize=(12, 6 * n_animals), sharex=True, sharey=True
        )

        for i, (animal_id, df_animal) in enumerate(df.groupby("animal_id")):
            self.plot_weights(
                df_animal, ax=ax[i], title=f"Animal {animal_id}", **kwargs
            )

    def plot_weights_summary(self, df=None, ax=None, animal_id=None, **kwargs):
        """
        Wrapper around plot_weights to plot weights for all
        animals in the experiment with variance indicated on plot.

        params
        ------
        df : pd.DataFrame (default=None)
            dataframe with columns "feature", "weight_class", "weight"
            row indexed by animal id, feature and weight class
            created by unpack_features_and_weights()
        animal_id : str (default=None)
            animal id to plot weights for. if None, will plot
            weights for all animals
        ax : matplotlib axis (default=None)
            axis to plot on. if None, plot_weights() will create a
            new figure
        kwargs : dict
            additional keyword arguments to pass to seaborn.barplot()
        """

        if df is None:
            df = self.unpack_features_and_weights()

        if animal_id is not None:
            df = df.query("animal_id == @animal_id")
            title = f"Animal {animal_id}"
        else:
            title = "All animals"

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        self.plot_weights(df, ax, title=title, **kwargs)

        return None


class ModelVisualizerCompare(ModelVisualizer):
    """
    Model class with additional methods for visualizing
    the results of a model comparison experiment.
    """

    def __init__(self, experiment):
        super().__init__(experiment)
        self.null_models = experiment.null_models

    # DF HELPERS
    def concat_null_and_fit_models_dfs(self):
        """
        Function to concat the fit models and null models data frames
        on animal_id and model_name columns. Creates a new attribute
        all_models_df.

        """
        # for each model, find best fit given sigma
        best_fit_df = super().find_best_fit(group=["model_name"]).copy()
        null_df = self.null_models.copy()

        # Select relevant columns & set inidices for merge
        best_fit_df_selected = best_fit_df.set_index(["animal_id", "model_name"])[
            ["nll", "sigma", "tau", "model_type", "n_train_trials", "n_test_trials"]
        ]
        null_df_selected = null_df.set_index(["animal_id", "model_name"])[
            ["nll", "model_type", "n_train_trials", "n_test_trials"]
        ]

        # concat, order and reset index
        merged_df = pd.concat([best_fit_df_selected, null_df_selected])
        merged_df.sort_values(["animal_id", "model_name"], inplace=True)
        merged_df.reset_index(inplace=True)
        merged_df["log_like"] = merged_df["nll"] * -1  # makes bits/trial easier

        self.all_models_df = merged_df.copy()

        return None

    def compute_bits_per_trial_df(self):
        """
        Compute the bits per trial for each animal, model
        in the experiment. Creates a new attribute bits_per_trial_df
        and returns it.
        """
        if not hasattr(self, "all_models_df"):
            self.concat_null_and_fit_models_dfs()

        bits_per_trial_df = (
            self.all_models_df.groupby("animal_id")
            .apply(self._calculate_bits_per_trial)
            .reset_index(drop=True)
        )

        self.bits_per_trial_df = bits_per_trial_df

        return self.bits_per_trial_df

    def _calculate_bits_per_trial(self, group):
        """
        Calculate the number of bits per trial for each model in the
        animal group relative to the null model (L_0).

        bit/trial = (log model - log null model) / (n_test_trials * log(2))

        params
        ------
        group : pd.DataFrame
            groupby object with columns `animal_id`, `model_name`,
            `n_test_trials` and `ll` (log-likelihood)
            grouped by animal_id

        returns
        -------
        group : pd.DataFrame
            groupby object with new column `bits_per_trial` added
            to each row

        Note: if the model type is null, the bits per trial will be the
        raw bits per trial. If the model type is not null, bits per trial
        represents the delta bits per trial relative to the null model.

        E.g. if the null model has 0.5 bits per trial and model X has
        0.6 bits per trial, model X has 0.5 + 0.6 = 1.1 bits per trial
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

    # PLOTS
    def plot_model_comparison(
        self, type="point", hue=None, ax=None, ylim=None, **kwargs
    ):
        """
        Plot the model comparison (delta bits/trial for the null
        model) for each model in the experiment.

        params
        ------
        type : str
            type of plot to use. "point" or "bar"
        hue : str, default = None
            column name to use for color encoding
        ax : matplotlib.axes.Axes, default = None
            axes to plot on
        ylim : tuple, default = None
            y-axis limits
        kwargs : dict
            keyword arguments to pass to seaborn.pointplot or
            seaborn.barplot
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

    def plot_nll_over_sigmas_by_animal_by_model(self):
        """
        Wrapper function to plot the negative log-likelihood across sigmas
        with hue as model type. One plot per animal.
        """
        super().plot_nll_over_sigmas_by_animal(
            group=["sigma", "model_name"], hue="model_name", palette="Greys"
        )
        return None
