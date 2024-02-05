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
from multiclass_logistic_regression import MultiClassLogisticRegression


class SigmaTauSearchExperimentReward:
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
                animal_df = self.df.query("animal_id == @animal and training_stage > 3")
            else:
                animal_df = self.df.query("training_stage > 3")

            # Create a DesignMatrixGenerator object & get train/test sessions for animal
            dmg = DesignMatrixGeneratorReward(verbose=False)
            dmg.get_train_test_sessions(
                animal_df, test_size=self.test_size, random_state=self.random_state
            )

            # Iterate over sigma/tau combinations
            for sigma in self.sigmas:
                for idx, tau in enumerate(self.taus):
                    # Generate design matrix & create train/test splits
                    X, Y = dmg.generate_design_matrix(
                        animal_df,
                        tau=tau,
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
            n_animals, 1, figsize=(15, 5 * n_animals), sharex=True, sharey=False
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

    def plot_nll_over_sigmas(self, df=None):  # self
        if df is None:
            df = self.find_best_fit(group="sigma")

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


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


class DesignMatrixGeneratorReward:
    def __init__(self, verbose=True):
        self.verbose = verbose

    @staticmethod
    def normalize_column(col):
        return (col - col.mean()) / col.std()

    def generate_design_matrix(
        self, df, tau, return_labels=True, drop_session_column=False
    ):
        """
        Function to generate "base" design matrix given a dataframe
        with violations tracked. In this case "base" means using the
        same regressors as Nick Roy did in Psytrack, but adjusted to
        take into account 3 choice options (L,R, Violation).

        N = number of trials
        D = number of features
        C = number of classes, in this case 3 (L, R, Violation)

        params
        ------
        df : pd.DataFrame
            dataframe with columns `s_a` `s_b` `session`, `violation`
            `correct_side` and `choice`, likely generated by
            get_rat_viol_data()
        tau : TODO
        filter_column : TODO
            if tau is not None, which column to filter/drop
        return_label : bool (default = True)
            whether to return one-hot encoded choice labels
        drop_session_column : bool (default = False)
            whether to drop 'session' column. should be set to
            false if doing session based train/test split
            following design matrix generation

        returns
        -------
        X : pd.DataFrame, shape (N, D + 1)
            design matrix with regressors for s_a, s_b,
            prev sound avg, correct side and choice info,
            normalized to standard normal with bias column added
        Y : np.ndarray, shape (N, C), where C = 3 if return_labels=True
            one-hot encoded choice labels for each trial as left,
            right or violation
        """
        # Initialize
        X = pd.DataFrame()
        stim_cols = ["s_a", "s_b"]
        X["session"] = df.session

        # Masks- if first trial in a session and/or previous trial
        # was a violation, "prev" variables get set to 0
        session_boundaries_mask = df["session"].diff() == 0
        X["prev_violation"] = (
            df["violation"].shift() * session_boundaries_mask
        ).fillna(0)
        prev_violation_mask = X["prev_violation"] == 0

        # # Violation Exp Filter
        # ! TODO tau might need to be animal specific!!!
        self.exp_filter = ExpFilter(
            tau=4, verbose=self.verbose, column="prev_violation"
        )
        self.exp_filter.apply_filter_to_dataframe(X)
        X.drop(columns=["prev_violation"], inplace=True)

        # Stimuli (s_a, s_b) get normalized
        for col in stim_cols:
            X[stim_cols] = self.normalize_column(df[stim_cols])

        # Average previous stimulus (s_a, s_b) loudness
        X["prev_sound_avg"] = df[stim_cols].shift().mean(axis=1)
        X["prev_sound_avg"] = self.normalize_column(X["prev_sound_avg"])
        X["prev_sound_avg"] *= session_boundaries_mask * prev_violation_mask

        # Prev correct side (L, R) (0, 1) -> (-1, 1),
        X["prev_correct"] = (
            df.correct_side.replace({0: -1}).astype(int).shift()
            * session_boundaries_mask
            * prev_violation_mask
        )

        # prev choice regressors (L, R, V) (0, 1, Nan) -> (-1, 1, 0),
        X["prev_choice"] = (
            df.choice.replace({0: -1}).fillna(0).astype(int).shift()
            * session_boundaries_mask
        )

        # Prev rewarded
        X["prev_rewarded"] = (df.hit.shift() * session_boundaries_mask).fillna(0)

        X.fillna(0, inplace=True)  # remove nan from shift()
        X.insert(0, "bias", 1)  # add bias column

        # Apply exponential filter if tau is not None
        if tau is not None:
            self.exp_filter = ExpFilter(
                tau=tau, verbose=self.verbose, column="prev_rewarded"
            )
            self.exp_filter.apply_filter_to_dataframe(X)
            X.drop(columns=["prev_rewarded"], inplace=True)

        if drop_session_column:
            X.drop(columns=["session"], inplace=True)

        if return_labels:
            Y = self.one_hot_encode_labels(df)
            return X, Y
        else:
            return X

    @staticmethod
    def one_hot_encode_labels(df):
        """
        Function to one-hot encode choice labels for each trial as
        left, right or violation (C = 3)

        params
        ------
        df : pd.DataFrame
            dataframe with columns `choice` likely generated by
            get_rat_viol_data()

        returns
        -------
        Y : np.ndarray, shape (N, C), where C = 3
            one-hot encoded choice labels for each trial as left,
            right or violation
        """

        Y = pd.get_dummies(df["choice"], "choice", dummy_na=True).to_numpy(copy=True)
        return Y

    def get_train_test_sessions(self, df, test_size, random_state=45):
        """
        This function will return a list of sessions to use for training
        and testing respectively. To apply, see function
        see apply_session_train_test_split()


        Parameters:
        -----------
        df : pd.DataFrame
            dataframe with `sessions` column
        test_size : float
            Proportion of data to use for test set
        """
        unique_sessions = df["session"].unique()
        train_sessions, test_sessions = train_test_split(
            unique_sessions, test_size=test_size, random_state=random_state
        )

        self.train_sessions = train_sessions
        self.test_sessions = test_sessions

    def apply_session_train_test_split(self, X, Y):
        """
        train_sessions = np.array
            sessions for the
        test_values = np.array indices for the test values

        """
        # TODO add in a check to see if self._train_session
        # TODO exists, and if not, call get_train_test_sessions

        # Filter rows based on session values for X
        X_train = X[X["session"].isin(self.train_sessions)].copy()
        X_test = X[X["session"].isin(self.test_sessions)].copy()

        # Filter rows based on session values for Y
        # Assuming the index of Y corresponds to that of X
        Y_train = Y[X["session"].isin(self.train_sessions).values]
        Y_test = Y[X["session"].isin(self.test_sessions).values]

        X_train.drop(columns=["session"], inplace=True)
        X_test.drop(columns=["session"], inplace=True)

        return X_train, X_test, Y_train, Y_test


class ExpFilter:
    def __init__(self, tau, column="violation", len_factor=5, verbose=True):
        self.tau = tau
        self.column = column
        self.len_factor = len_factor
        self.verbose = verbose

    def create_kernel(self):
        """
        create an exp decay kernal with time constant tau and
        kernel length = len factor * tau
        """

        return np.array(
            [np.exp(-i / self.tau) for i in range(self.len_factor * self.tau)]
        )

    def plot_kernel(self):
        kernel = self.create_kernel()
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.plot(kernel)
        plt.title(f"Exponential filter kernel | Tau: {self.tau}")
        plt.show()

    def apply_filter_to_session(self, session_df):
        """
        apply kernel to individual sessions for independent
        filtering of column history
        """
        kernel = self.create_kernel()

        # Convolve the kernel with selected column
        convolution_result = np.convolve(session_df[self.column], kernel, mode="full")[
            : len(session_df)
        ]

        session_df[f"{self.column}_exp_{self.tau}"] = convolution_result

        return session_df

    def apply_filter_to_dataframe(self, source_df, output_df=None):
        """
        Function to apply exp kernel to a column given and
        entire dataframe on a session-by-session basis
        """
        if self.tau == 0:
            return

        if output_df is None:
            output_df = source_df

        for session_id, session_data in source_df.groupby("session"):
            filtered_session = self.apply_filter_to_session(session_data.copy())
            output_df.loc[
                output_df["session"] == session_id, f"{self.column}_exp_{self.tau}"
            ] = filtered_session[f"{self.column}_exp_{self.tau}"]

            if self.verbose:
                print(
                    f"Exp filter added for session {session_id} | Column: {self.column}, Tau: {self.tau}"
                )

        # scale column by max to bound between 0 and 1
        output_df[f"{self.column}_exp_{self.tau}"] /= output_df[
            f"{self.column}_exp_{self.tau}"
        ].max()
