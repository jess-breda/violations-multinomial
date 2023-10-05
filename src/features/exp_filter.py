"Class for applying exponential filter to a column of a dataframe"

import numpy as np
import matplotlib.pyplot as plt


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
        plt.xlabel("Trial")
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
