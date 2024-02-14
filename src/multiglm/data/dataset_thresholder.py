import pandas as pd
import numpy as np
import seaborn as sns


class PerformanceThresholder:
    def __init__(self) -> None:
        self.data_dict = {"hit": {}, "violation": {}}

    def compute_rolling_mean(self, df, column, window_size):
        """
        Function to compute the rolling mean of a column for each animal_id
        across sessions
        """
        grouped_over_sessions = (
            df.groupby(["animal_id", "session", "training_stage"])[column]
            .mean()
            .reset_index()
        )
        rolling_mean = (
            grouped_over_sessions.groupby("animal_id")
            .apply(
                lambda x: x[column].rolling(window=window_size, min_periods=1).mean()
            )
            .reset_index()
        )

        rolling_mean.rename(columns={column: f"rolling_mean_{column}"}, inplace=True)
        rolling_mean["window_size"] = window_size

        return pd.merge(
            grouped_over_sessions,
            rolling_mean.drop(columns=["animal_id"]),  # already in grouped df
            left_index=True,
            right_on="level_1",  # index of grouped df
            how="left",
        ).drop(columns=["level_1"])

    def compute_rolling_means_iter_window(self, df, column, window_sizes):
        """
        Function to compute the rolling mean of a column for each animal_id
        across sessions for multiple window sizes
        """
        rolling_mean_over_windows = pd.concat(
            [self.compute_rolling_mean(df, column, w) for w in window_sizes],
            ignore_index=True,
        )

        self.data_dict[column]["rolling_mean_windows_df"] = rolling_mean_over_windows

        return rolling_mean_over_windows

    def calculate_threshold_crossings(
        self, df, column, window_size, threshold, training_stage_threshold=3
    ):
        # Filter the rolling mean df for a given window size and training stage
        filtered_df = df.query(
            "window_size == @window_size and training_stage > @training_stage_threshold"
        ).copy()

        crossing_df = self.calculate_low_to_high_crossings(
            filtered_df, column, threshold
        )

        crossing_summary_df = self.compute_and_append_crossing_stats(
            filtered_df, crossing_df, column, window_size
        )

        return crossing_summary_df

    def compute_crossings_iter_thresholds(self, df, window_size, column, thresholds):
        crossings_over_thresholds = pd.concat(
            [
                self.calculate_threshold_crossings(df, column, window_size, t)
                for t in thresholds
            ],
            ignore_index=True,
        )

        self.data_dict[column]["crossing_thresholds_df"] = crossings_over_thresholds

        return crossings_over_thresholds

    @staticmethod
    def calculate_low_to_high_crossings(df, column, threshold):
        """
        Function to locate and count low -> high crossings for a column across
        a threshold for each animal_id
        """
        # Calculate threshold crossings
        column_to_check = f"rolling_mean_{column}"
        df["crossed_threshold"] = (df[column_to_check].shift() < threshold) & (
            df[column_to_check] >= threshold
        )

        # Count threshold crossings
        crossings_count = (
            df.groupby("animal_id")["crossed_threshold"].sum().reset_index()
        )

        crossings_count["threshold"] = threshold

        return crossings_count

    @staticmethod
    def compute_and_append_crossing_stats(
        rolling_mean_df, crossing_df, column, window_size
    ):
        """
        Function to compute the min, median and max crossing sessions for each
        animal_id and append to the crossing_df
        """
        grouped = rolling_mean_df.groupby("animal_id")

        crossing_df["min_cross_sess"] = grouped.apply(
            lambda x: x.loc[x["crossed_threshold"]].session.min()
        ).values
        crossing_df["med_cross_sess"] = grouped.apply(
            lambda x: x.loc[x["crossed_threshold"]].session.median()
        ).values
        crossing_df["max_cross_sess"] = grouped.apply(
            lambda x: x.loc[x["crossed_threshold"]].session.max()
        ).values

        # Assign additional info
        crossing_df["window_size"] = window_size
        crossing_df["type"] = column
        crossing_df.rename(
            columns={"crossed_threshold": "crossed_threshold_count"}, inplace=True
        )

        return crossing_df


def line_plot_facet_grid(data, hue, x, y):
    g = sns.FacetGrid(data=data, col="animal_id", col_wrap=4, hue=hue, height=4)
    g.map(sns.lineplot, x, y)

    return g
