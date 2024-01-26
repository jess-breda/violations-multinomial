from sklearn.model_selection import train_test_split
import pandas as pd


def get_prev_violation_taus_df():
    """Load df with prev_violation tau values for each animal
    that were found in macro sweep with randomstate = 47"""
    taus_df = pd.read_csv(
        "/Users/jessbreda/Desktop/github/animal-learning/data/results/prev_violation_tau.csv"
    )
    return taus_df


def get_taus_df(min_training_stage=3):
    """
    Load df with tau values for each animal and variable
    with tau sweep performed with randomstate = 47
    """
    if min_training_stage == 4:
        taus_df = pd.read_csv(
            "/Users/jessbreda/Desktop/github/animal-learning/data/results/tau_df_stg_4.csv"
        )
    elif min_training_stage == 3:
        taus_df = pd.read_csv(
            "/Users/jessbreda/Desktop/github/animal-learning/data/results/tau_df.csv"
        )
    else:
        raise ValueError("min_training_stage must be 3 or 4!")
    return taus_df


def create_violation_interaction_pairs(cols):
    """
    Quick function for creating interaction pairs for
    a subset of columns in a design matrix against the filtered
    previous violation history column.

    params
    ------
    cols : list of str
        columns interact with prev_violation_exp column.
        e.g. ["s_a", "s_b"]

    returns
    -------
    interaction_pairs : list of tuples
        each tuple contains the names of two columns to interact
        when generating the design matrix
    """
    interaction_pairs = [(f"prev_violation_exp", col) for col in cols]
    return interaction_pairs
