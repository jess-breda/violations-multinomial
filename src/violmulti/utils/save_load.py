"""
Functions to save and load different experiment components
such as models, data, and configuration files. Additional
utility functions to create directories and determine paths
to Cup storage locations

Written by Jess Breda 2024-06-24
"""

from pathlib import Path
from typing import Union, Tuple, Dict, Any
import pickle
import yaml
import pandas as pd
import numpy as np

from violmulti.models.ssm_glm_hmm import SSMGLMHMM
from violmulti.features.design_matrix_generator_PWM import *  # for deserialize_function_or_call

## == Model Utility Functions == ##


def save_model_to_pickle(
    model_object: SSMGLMHMM,
    animal_id: str = "",
    model_name: str = "glmhmm",
    n_fold: int = 0,
    n_init: int = 0,
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
    n_init : int (default=0)
        The initialization number of the model (if multiple are used)
    model_path : str or Path object (default=None)
        The path to save the model object. Default is None, which saves the
        model object to a folder called "models" in the current working
        directory.
    """

    # if no path is provided, save to a folder called "model_results"
    # in the current working directory, if path is provided, just convert to Path object
    model_path = create_required_directory(model_path, "models")

    with open(
        f"{model_path}/animal_{animal_id}_{model_object.K}_states_model_{model_name}_fold_{n_fold}_init_{n_init}.pkl",
        "wb",
    ) as f:
        pickle.dump(model_object, f)


def load_model_from_pickle(
    animal_id: str,
    n_states: int,
    model_name: str,
    n_fold: int,
    n_init: int,
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
    n_init : int
        The initialization number of the model (if multiple are used)
    model_path : str or Path object
        The path where the model is located.

    Returns
    -------
    model : SSMLGLMHMM
        The model object loaded from the pickle file.
    """

    with open(
        f"{model_path}/animal_{animal_id}_{n_states}_states_model_{model_name}_fold_{n_fold}_init_{n_init}.pkl",
        "rb",
    ) as f:
        model = pickle.load(f)

    return model


## == Data Utility Functions == ##


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


## == Config Utility Functions == ##


def save_config_to_yaml(config: Dict[str, Any], file_path: str) -> None:
    """
    Saves an experiment configuration to a YAML file.

    Parameters
    ----------
    config : dict
        The configuration dictionary to save to a YAML file.
    file_path : str
        The path to the YAML file to save the configuration to.
    """
    with open(file_path, "w") as file:
        yaml.safe_dump(config, file)


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Loads an experiment configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        The path to the YAML file containing the configuration.

    Returns
    -------
    config : dict
        The configuration dictionary loaded from the YAML file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def deserialize_function_or_call(func_str: str) -> Any:
    """
    Deserializes a string representing a lambda function
    or a function call.

    Parameters
    ----------
    func_str : str
        The string representation of the lambda function
        or function call.

    Returns
    -------
    func : Any
        The deserialized function or the result of the
        function call.

    Raises
    ------
    ValueError
        If the function string format is unknown or if
        there is a syntax error.

    Examples
    --------
    deserialize_function_or_call("lambda df: standardize(df.s_a)") -> lambda df: standardize(df.s_a)
    deserialize_function_or_call("binary_choice_labels()" -> binary_choice_labels()
    """
    try:
        if "lambda" in func_str:
            return eval(func_str)
        elif "()" in func_str:  # Simple check to see if it's a function call
            return eval(func_str)
        else:
            raise ValueError(f"Unknown function format: {func_str}")
    except (SyntaxError, NameError) as e:
        raise ValueError(f"Error evaluating function string: {func_str} - {e}")


def convert_dmg_config_functions(config: Dict[str, str]) -> Dict[str, Any]:
    """
    Converts strings in the configuration to actual functions or the results
    of function calls. This is specific to the DesignMatrixGenerator configuration
    where the functions are serialized as strings that indicate the operations
    needed to create each column of the Design Matrix and Labels.

    Parameters
    ----------
    config : dict
        The dmg configuration dictionary with string representations of functions.

    Returns
    -------
    deserialized_config : dict
        The dmg configuration dictionary with deserialized functions.
    """
    deserialized_config = {}
    for key, func_str in config.items():
        deserialized_config[key] = deserialize_function_or_call(func_str)
    return deserialized_config


## == General Utility Functions == ##
def create_required_directory(path: Union[Path, str], folder_name: str) -> Path:
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


def determine_path_to_cup_data() -> Path:
    """
    Quick function to determine the path for data loading and storage based on
    whether the code is being run on Spock (on the cluster) or locally.

    Returns
    -------
    data_base_path : Path
        The base path for data storage containing the raw, cleaned, processed, and results
        subfolders. The "results" subfolder is where experiment directories will be created.
    """
    cwd = Path.cwd()
    # spock path starts with /mnt
    # local path stats with /Users/jessbreda
    # if cwd.parts[1] == "mnt":
    #     on_spock = True
    # else:
    #     on_spock = False

    if cwd.parts[1] == "Users":
        on_spock = False
    else:
        on_spock = True

    prefix = "/jukebox" if on_spock else "/Volumes"
    return Path(
        f"{prefix}/brody/jbreda/behavioral_analysis/violations_multinomial/data"
    )
