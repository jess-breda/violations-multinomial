from sklearn.model_selection import train_test_split
import pandas as pd


def get_train_test_sessions(df, test_size=0.2, random_state=45):
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

    return train_sessions, test_sessions


def get_taus():
    """Load df with tau values for each animal
    that were found in macro sweep with randomstate = 47"""
    taus_df = pd.read_csv(
        "/Users/jessbreda/Desktop/github/animal-learning/data/results/prev_violation_tau.csv"
    )
    return taus_df
