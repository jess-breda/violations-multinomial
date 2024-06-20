from pathlib import Path
from typing import Union, Tuple
import pickle
import pandas as pd
import numpy as np

from violmulti.models.ssm_glm_hmm import SSMGLMHMM


def save_model_to_pickle(
    model_object: SSMGLMHMM,
    animal_id: str = "",
    model_name: str = "glmhmm",
    n_fold: int = 0,
    model_path=None,
) -> None:
    """
    Save the model object to a pickle file.

    Default naming convention is:
    animal_{animal_id}_{model_object.K}_states_model_{model_name}_fold_{n_fold}.pkl

    Parameters
    ----------
    model_object : SSMGLMHMM
        The model object to be saved.
    animal_id : str (default="")
        The animal id of the data that model was fit to. Default is an empty string.
    model_name : str (default="glmhmm")
        The name of the model to be saved. Typically the key used to
        access the model_config dictionary (if multiple models are run
        on the same data).
    n_fold : int (default=0)
        The fold number of the model. Default is 0, assuming no cross-validation.
    model_path : str or Path object (default=None)
        The path to save the model object. Default is None, which saves the
        model object to a folder called "models" in the current working
        directory.
    """

    # if no path is provided, save to a folder called "model_results"
    # in the current working directory, if path is provided, just convert to Path object
    model_path = create_required_directory(model_path, "models")

    with open(
        f"{model_path}/animal_{animal_id}_{model_object.K}_states_model_{model_name}_fold_{n_fold}.pkl",
        "wb",
    ) as f:
        pickle.dump(model_object, f)


def load_model_from_pickle(
    animal_id: str,
    n_states: int,
    model_name: str,
    n_fold: int,
    model_path: Union[str, Path],
) -> SSMGLMHMM:
    """
    Load model object from pickle file.

    Default naming convention is:
    animal_{animal_id}_{n_states}_states_model_{model_name}_fold_{n_fold}.pkl

    Parameters
    ----------
    animal_id : str
        The animal id of the data that model was fit to. Default is an empty string.
    n_states : int
        The number of states in the model.
    model_name : str
        The name of the model to be saved. Typically the key used to
        access the model_config dictionary (if multiple models are run
        on the same data).
    n_fold : int
        The fold number of the model, (0 if no cross-validation was done)
    model_path : str or Path object
        The path where the model is located.

    Returns
    -------
    model : SSMLGLMHMM
        The model object loaded from the pickle file.
    """

    with open(
        f"{model_path}/animal_{animal_id}_{n_states}_states_model_{model_name}_fold_{n_fold}.pkl",
        "rb",
    ) as f:
        model = pickle.load(f)

    return model


def save_data_and_labels_to_parquet(
    X: pd.DataFrame,
    y: np.ndarray,
    animal_id: str = "",
    model_name: str = "",
    n_fold: int = 0,
    data_path: Union[str, Path] = None,
) -> None:
    """
    Function to save a DataFrame X and labels y to individual
    compressed parquet files.

    Default naming convention is:
    animal_{animal_id}_model_{model_name}_fold_{n_fold}_X.parquet
    animal_{animal_id}_model_{model_name}_fold_{n_fold}_y.parquet

    Parameters
    ----------
    X : pd.DataFrame (n_trials, m_features + 2) where +2 is the "session" and "bias" columns
        The DataFrame to be saved
    y : np.ndarray (n_trials, )
        The labels to be saved (usually representing animals choice)
    animal_id : str (default="")
        The animal id of the data used to fit the model (if only one was fit).
    model_name : str (default="")
        The name of the model that fit the data. Typically the key used to
        access the model_config dictionary (if multiple models are run
        on the same data).
    n_fold : int (default=0)
        The fold number of the model. Default is 0, assuming no cross-validation.
    data_path : str or Path object (default=None)
        The path to save the data and labels. Default is None, which saves the
        data and labels to a folder called "data" in the current working directory.

    """

    # if no path is provided, save to a folder called "data"
    # in the current working directory, if path is provided, just convert to Path object
    data_path = create_required_directory(data_path, "data")

    df_path = (
        data_path / f"animal_{animal_id}_model_{model_name}_fold_{n_fold}_X.parquet"
    )
    labels_path = (
        data_path / f"animal_{animal_id}_model_{model_name}_fold_{n_fold}_y.parquet"
    )

    # Save DataFrame X
    X.to_parquet(df_path, engine="pyarrow", compression="gzip")

    # Save labels y
    # note! convert numpy array y to a pandas DF to be able use parquet
    # since it doesn't work with pd.Series
    labels_df = pd.DataFrame(y, columns=["label"])
    labels_df.to_parquet(labels_path, engine="pyarrow", compression="gzip")

    print(f"DataFrame saved to {df_path}")
    print(f"Labels saved to {labels_path}")


def load_data_and_labels_from_parquet(
    animal_id: str, model_name: str, n_fold: int, data_path: Union[str, Path]
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Function to load a DataFrame X and labels y from individual
    compressed parquet files.

    Default naming convention is:
    animal_{animal_id}_model_{model_name}_fold_{n_fold}_X.parquet
    animal_{animal_id}_model_{model_name}_fold_{n_fold}_y.parquet

    Parameters
    ----------
    animal_id : str
        The animal id of the data that model was fit to (can be empty string).
    model_name : str
        The name of the model that fit the data. Typically the key used to
        access the model_config dictionary (if multiple models are run
        on the same data).
    n_fold : int
        The fold number of the model. Default is 0, assuming no cross-validation.
    data_path : str or Path object
        The path where the data and labels parquets are located.

    Returns
    -------
    X : pd.DataFrame (n_trials, m_features + 2) where +2 is the "session" and "bias" columns
        The DataFrame to be loaded
    y : np.ndarray (n_trials, )
        The labels to be loaded (usually representing animals choice)
    """

    df_path = (
        f"{data_path}/animal_{animal_id}_model_{model_name}_fold_{n_fold}_X.parquet"
    )
    labels_path = (
        f"{data_path}/animal_{animal_id}_model_{model_name}_fold_{n_fold}_y.parquet"
    )

    # Load DataFrame X
    X = pd.read_parquet(df_path, engine="pyarrow")

    # Load labels y, convert back from df to numpy array
    labels_df = pd.read_parquet(labels_path, engine="pyarrow")
    y = labels_df["label"].to_numpy()

    print(f"DataFrame loaded from {df_path}")
    print(f"Labels loaded from {labels_path}")

    return X, y


def create_required_directory(path, folder_name):
    """
    Check to see if path_to_check exists. If it does not, create the directory
    with the folder_name.
    """
    if path is None:
        # Set the default path in the current working directory with the folder_name
        created_path = Path.cwd() / folder_name
        # Create the directory if it does not exist
        created_path.mkdir(exist_ok=True)
    else:
        # Convert string path to Path object and create directory
        created_path = Path(path)
        created_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    print(f"Directory ensured at: {created_path}")

    return created_path
