import numpy as np
import matplotlib.pyplot as plt


class ExpFilter:
    def __init__(self, tau, verbose=True, len_factor=5):
        """
        An exponential filter that can be applied to a column
        of a dataframe. The filter is applied to each session
        independently and then scaled across all sessions.

        params
        ------
        tau : int
            time constant for exponential decay
        verbose : bool, (default=True)
            whether to print out progress
        len_factor : int, (default=5)
            factor to multiply tau by to get length of kernel. 5
            allows for full decay of kernel
        """
        self.tau = tau
        self.verbose = verbose
        self.len_factor = len_factor

    def create_kernel(self):
        """
        create an exp decay kernel with time constant tau and
        kernel length = len factor * tau
        """

        return np.array(
            [np.exp(-i / self.tau) for i in range(self.len_factor * round(self.tau))]
        )

    def plot_kernel(self):
        """
        plot the kernel created by create_kernel given the
        current tau value and len_factor
        """
        kernel = self.create_kernel()
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.plot(kernel)
        plt.title(f"Exponential filter kernel | Tau: {self.tau}")
        plt.xlabel("Trial")
        plt.show()

    def apply_filter_to_session(self, column_name, session_df):
        """
        Apply kernel to individual sessions for independent
        filtering of column history. This is done by convolving
        the kernel with the column of interest and then
        truncating the result to the length of the session.

        params
        ------
        column_name : str
            name of column in session_df to apply filter to
        session_df : pd.DataFrame
            dataframe containing data for a single session

        returns
        -------
        session_df : pd.DataFrame
            dataframe with new column containing filtered
            column of interest indicated by column_exp_tau
        """
        kernel = self.create_kernel()

        # Convolve the kernel with selected column
        convolution_result = np.convolve(session_df[column_name], kernel, mode="full")[
            : len(session_df)
        ]

        session_df[f"{column_name}_exp"] = convolution_result

        return session_df

    def apply_filter_to_dataframe(self, column_name, source_df, output_df=None):
        """
        Apply filter to all sessions in a data frame for a given
        column name. This is done by applying the filter to each
        session individually with apply_filter_to_session() and then
        max scaling across all sessions

        params
        ------
        column_name : str
            column to apply filter to in source_df to create
            column_name_exp in output_df
        source_df : pd.DataFrame
            dataframe containing data for all sessions and column
            to filter
        output_df : pd.DataFrame, default=None
            dataframe to add new column to. If None, the column
            will be added to a copy of source_df.

        returns
        -------
        output_df : pd.DataFrame
            dataframe with new column containing filtered
            column of interest indicated by column_exp_tau
        """
        if not column_name in source_df.columns:
            raise ValueError(f"{column_name} column not found in X!")

        if self.tau == 0:
            return

        if output_df is None:
            output_df = source_df.copy()

        for session_id, session_data in source_df.groupby("session"):
            filtered_session = self.apply_filter_to_session(
                column_name, session_data.copy()
            )
            output_df.loc[output_df["session"] == session_id, f"{column_name}_exp"] = (
                filtered_session[f"{column_name}_exp"]
            )

            if self.verbose:
                print(
                    f"Exp filter added for session {session_id} | Column: {column_name}, Tau: {self.tau}"
                )

        # scale column by max to bound between 0 and 1
        output_df[f"{column_name}_exp"] /= output_df[f"{column_name}_exp"].max()

        return output_df
