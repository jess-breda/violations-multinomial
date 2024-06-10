"""
Class to split data into train and test sets based on session
values. 
Written by Jess Breda, 2023-10-23
"""

from sklearn.model_selection import train_test_split
import numpy as np


class TrainTestSplitter:
    def __init__(self, test_size=0.2, random_state=None):
        """
        Initialize the TrainTestSplitter class.

        params
        ------

        test_size : float (default=0.2)
            proportion of the data to include in the test set.
        random_state : int (default=None)
            random seed for reproducibility.
        """
        self.test_size = test_size
        self.random_state = random_state

    def get_sessions_for_split(self, df):
        """
        This function will compute a list of sessions to use for training
        and testing respectively and store them as attributes of the class.

        params:
        -------
        df : pd.DataFrame
            dataframe with `sessions` column
        test_size : float
            Proportion of data to use for test set

        computes:
        --------
        train_sessions : list
            list of sessions to use for training
        test_sessions : list
            list of sessions to use for testing
        """
        unique_sessions = df["session"].unique()
        self.train_sessions, self.test_sessions = train_test_split(
            unique_sessions, test_size=self.test_size, random_state=self.random_state
        )

    def apply_session_split(self, X, Y, lr_only=False):
        """
        Function to apply session train/test split computed by
        get_sessions_for_split() to design matrix and labels.

        params
        ------
        X : pd.DataFrame, shape (N, D + 2)
            design matrix with bias column and session column
        Y : np.ndarray, shape (N, C) or (N, )
            one-hot encoded choice labels for mutli class (l, r, v) or
            binary class (l, r) respectively
        lr_only : bool (default=False)
            whether to filter out violation trials from the test set
            for the multi-class case. this is used when running
            model comparison between binary and multi.

        returns
        -------
        X_train : pd.DataFrame, shape (N_train, D + 1)
            design matrix for training set
        X_test : pd.DataFrame, shape (N_test, D + 1)
            design matrix for test set
        Y_train : np.ndarray, shape (N_train, C) or (N_train, )
            one-hot encoded  or binary encoded choice labels
            for training set
        Y_test : np.ndarray, shape (N_test, K) or (N_test, )
            on-hot encoded or binary encoded choice labels for
            test set. K = 2 if drop_violations=True, K = 3 otherwise
        """
        ## Checks
        if not "session" in X.columns:
            raise ValueError("session column not found in X, can't split!")

        if not hasattr(self, "train_sessions"):
            raise ValueError("train_sessions and test_sessions not defined!")

        # Filter rows based on session values for X
        X_train = X[X["session"].isin(self.train_sessions)].copy()
        X_test = X[X["session"].isin(self.test_sessions)].copy()

        # Filter rows based on session values for Y
        # Assuming the index of Y corresponds to that of X
        Y_train = Y[X["session"].isin(self.train_sessions).values]
        Y_test = Y[X["session"].isin(self.test_sessions).values]

        X_train.drop(columns=["session"], inplace=True)
        X_test.drop(columns=["session"], inplace=True)

        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        # Additional code to filter out violations if flag is set
        if lr_only:
            self.filter_violations_from_test_set()
            return (
                self.X_train,
                self.filtered_X_test,
                self.Y_train,
                self.filtered_Y_test,
            )

        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def filter_violations_from_test_set(self):
        """
        Filters out the violation trials from Y_test and X_test. For
        the multi-class case to allow for comparison with the binary
        case on only L & R trials.

        Assumes that the violation is encoded as [0, 0, 1] in Y_test.
        """
        violation_filter = np.all(self.Y_test == np.array([0, 0, 1]), axis=1)
        non_violation_idx = np.where(~violation_filter)[0]

        self.filtered_Y_test = self.Y_test[non_violation_idx]
        self.filtered_X_test = self.X_test.iloc[non_violation_idx]

        return None
