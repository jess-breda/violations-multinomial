import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

import sys
import pathlib

[
    sys.path.append(str(folder))
    for folder in pathlib.Path("../src/").iterdir()
    if folder.is_dir()
]
from get_rat_data import get_rat_viol_data
from generate_design_matrix import DesignMatrixGenerator
from multiclass_logistic_regression import MultiClassLogisticRegression


class SigmaTauSearchExperiment:
    def __init__(self, params):
        self.animals = params["animals"]
        self.sigmas = params["sigmas"]
        self.taus = params["taus"]
        self.model_names = self.generate_model_names(params)
        self.random_state = params["random_state"]
        self.test_size = params["test_size"]
        self.df = get_rat_viol_data(animal_ids=self.animals)
        self.stored_fits = []

        if self.animals is None:
            self.animals = self.df.animal_id.unique()

    def run(self):
        for animal in self.animals:
            print(f"\n\n !!!!! evaluating animal {animal} !!!!!\n\n")
            if self.df.animal_id.nunique() > 1:
                # Load in data for specific animal
                animal_df = self.df.query("animal_id == @animal and training_stage > 2")
            else:
                animal_df = self.df.query("training_stage > 2")

            # Create a DesignMatrixGenerator object & get train/test sessions for animal
            dmg = DesignMatrixGenerator(verbose=False)
            dmg.get_train_test_sessions(
                animal_df, test_size=self.test_size, random_state=self.random_state
            )

            # Iterate over sigma/tau combinations
            for sigma in self.sigmas:
                for idx, tau in enumerate(self.taus):
                    # Generate design matrix & create train/test splits
                    X, Y = dmg.generate_design_matrix(
                        self.df,
                        tau=tau,
                        filter_column="prev_violation",
                        return_labels=True,
                    )
                    (
                        X_train,
                        X_test,
                        Y_train,
                        Y_test,
                    ) = dmg.apply_session_train_test_split(X, Y)

                    # Fit model & evaluate
                    model = MultiClassLogisticRegression(sigma=sigma)
                    W_fit = model.fit(X_train, Y_train)
                    nll = model.eval(X_test, Y_test)

                    # Store results
                    self.store(
                        animal, self.model_names[idx], nll, sigma, tau, X_test, W_fit
                    )

        self.results = pd.concat(self.stored_fits, ignore_index=True)
        return self.results

    def store(self, animal, model_name, nll, sigma, tau, X, W_fit):
        # Create a DataFrame for this iteration
        iter_df = pd.DataFrame(
            {
                "animal_id": [animal],
                "model_name": [model_name],
                "nll": [nll],
                "sigma": [sigma],
                "tau": [tau],
                "features": [list(X.columns)],
                "weights": [list(W_fit)],  # Convert numpy array to list
            }
        )
        # Append to the list of stored fits
        self.stored_fits.append(iter_df)

    @staticmethod
    def generate_model_names(params):
        taus = params["taus"]
        names = []

        for tau in taus:
            if tau is None:
                names.append("base")
            else:
                name = "t_" + str(tau)
                names.append(name)

        return names

    ## WRANGLING ##

    def find_best_fit(self, group="tau"):
        # if group is tau, will find the best sigma for
        # each tau tested.
        # if group is sigma, will find the best tau for
        # each sigma tested.

        best_fit_dfs = []

        for animal_id, sub_df in self.results.groupby(["animal_id"]):
            best_idx = sub_df.groupby(group)["nll"].idxmin()
            best_fit_df = sub_df.loc[best_idx][
                ["animal_id", "model_name", "sigma", "tau", "nll"]
            ]
            best_fit_dfs.append(best_fit_df)

        return pd.concat(best_fit_dfs, ignore_index=True)

    def create_pivot_for_tau_heatmap(self, df=None):
        if df is None:
            df = self.find_best_fit(group="tau")

        pivot = df.pivot(index="animal_id", columns="model_name", values="nll")

        # reorder the df to match the order of the model names
        # e.g. so that t_10 doesn't come before t_5

        columns_with_numbers = [
            col for col in pivot.columns if "_" in col and col.split("_")[-1].isdigit()
        ]
        columns_without_numbers = [
            col for col in pivot.columns if col not in columns_with_numbers
        ]

        # Sort both lists
        columns_with_numbers.sort(key=lambda x: int(x.split("_")[-1]))
        columns_without_numbers.sort()

        # Combine the lists
        sorted_columns = columns_without_numbers + columns_with_numbers

        # Reorder the DataFrame based on the sorted column names
        pivot = pivot[sorted_columns]

        return pivot

    def query_min_nll(self, animal_id, model_name):
        """
        Query the row with the minimum NLL for a given animal_id and model_name
        """
        query = self.results.query(
            "animal_id == @animal_id and model_name == @model_name"
        ).sort_values(by="nll", ascending=True)

        return query.iloc[0]

    ## PLOTTING ##

    def plot_nll_over_taus(self, df=None):
        if df is None:
            df = self.find_best_fit(group="tau")

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
                x="tau",
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
                current_ax.set(xlabel="Tau")
            else:
                current_ax.set(xlabel="")

        return None

    def plot_tau_heatmap(self, df=None, vmin=None, vmax=None, cmap="Blues"):
        if df is None:
            df = self.create_pivot_for_tau_heatmap()

        fig, ax = plt.subplots(figsize=(30, 30))
        sns.heatmap(df, annot=True, cmap=cmap, fmt=".2e", ax=ax, vmin=vmin, vmax=vmax)

        plt.title("NLL by Animal ID and Model")
        plt.xlabel("Model")
        plt.ylabel("Animal ID")

        yticks = ax.get_yticks()
        xticks = ax.get_xticks()

        for i, (index, row) in enumerate(df.iterrows()):
            min_val_col = row.idxmin()
            min_val_idx = df.columns.tolist().index(min_val_col)

            y_coord = yticks[i] - 0.5
            x_coord = xticks[min_val_idx] - 0.5

            rect = patches.Rectangle(
                (x_coord, y_coord),
                1,
                1,
                linewidth=3,
                edgecolor="White",
                facecolor="none",
            )

            ax.add_patch(rect)

        return None

    def plot_best_sigma_tau_by_animal(self, jitter=0.025, df=None):
        # TODO could add hue as a param to color by trained/not trained"

        if df is None:
            df = self.find_best_fit(group=["animal_id"])

        fig, ax = plt.subplots(figsize=(10, 8))

        # jitter points so animals don't overlap
        df["tau"] = df["tau"] + np.random.uniform(-jitter, jitter, len(df))
        df["sigma"] = df["sigma"] + np.random.uniform(-jitter, jitter, len(df))

        ax.grid()
        sns.scatterplot(data=df, x="tau", y="sigma", hue="animal_id", ax=ax)

        # aesthetics
        ymin = self.results.sigma.min() - 0.5
        ymax = self.results.sigma.max() + 0.5
        xmin = self.results.tau.min() - 0.5
        xmax = self.results.tau.max() + 0.5

        ax.set(
            xlim=(xmin, xmax),
            ylim=(ymin, ymax),
            xlabel="Tau",
            ylabel="Sigma",
            title="Best Fit Parameters",
        )

        return None

    def plot_class_weights(self, animal_id, model_name):
        """
        Wrapper function to plot class weights for a given animal and model
        """
        row = self.query_min_nll(animal_id, model_name)
        self._plot_class_weights(
            row["features"], np.array(row["weights"]), title=f"{animal_id} {model_name}"
        )

    def _plot_class_weights(self, feature_names, W_fit, title=""):
        """
        Internal function to plot the weights for each feature and class as bar charts.
        """
        D, C = W_fit.shape
        classes = ["L", "R", "V"]

        weight_data = [
            {"Weight": W_fit[d, c], "Feature": feature_names[d], "Class": classes[c]}
            for c in range(C)
            for d in range(D)
        ]
        df_weights = pd.DataFrame(weight_data)

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.axhline(y=0, color="black")
        sns.barplot(x="Feature", y="Weight", hue="Class", data=df_weights, ax=ax)
        plt.xticks(rotation=45)
        plt.legend(loc="upper left")
        plt.title(title)
