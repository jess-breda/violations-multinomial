import yaml
from typing import Any, Dict
from violmulti.features.design_matrix_generator_PWM import *  # for deserialize_function_or_call


def save_config_to_yaml(config: Dict[str, Any], file_path: str) -> None:
    """
    Saves a configuration to a YAML file.

    Parameters
    ----------
    config : dict
        The configuration dictionary to save to a YAML file.
    file_path : str
        The path to the YAML file to save the configuration to.
    """
    with open(file_path, "w") as file:
        yaml.dump(config, file)


def load_config_from_yaml(file_path: str) -> Dict[str, Any]:
    """
    Loads a configuration from a YAML file.

    Parameters
    ----------
    file_path : str
        The path to the YAML file containing the configuration.

    Returns
    -------
    config : dict
        The configuration dictionary loaded from the YAML file.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def deserialize_function_or_call(func_str: str) -> Any:
    """
    Deserializes a string representing a lambda function or a function call.

    Parameters
    ----------
    func_str : str
        The string representation of the lambda function or function call.

    Returns
    -------
    func : Any
        The deserialized function or the result of the function call.

    Raises
    ------
    ValueError
        If the function string format is unknown or if there is a syntax error.

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
