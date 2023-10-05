import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ModelVisualizer:
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
        pass

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

        # no16 = self.model_fits.query("sigma != 16")

        best_fit_dfs = []

        # TODO might break if tau doesn't exist in experiment
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
