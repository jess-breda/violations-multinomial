import argparse
from pathlib import Path
import yaml
import sys


def create_experiment_directory(
    experiment_name: str,
    config_type: str,
    on_spock: bool = False,
) -> str:

    # Create the experiment directory and subdirectories in data/results/experiment_name
    data_path = experiment_path_manager(on_spock)
    experiment_dir = Path(data_path / "results" / f"experiment_{experiment_name}")
    subdirectories = ["models", "logs", "figures", "data"]
    for subdir in subdirectories:
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Create config file
    config_data = get_init_config_data(config_type)
    config_data["relative_data_path"] = str(data_path)
    config_path = experiment_dir / "config.yaml"

    # Write the config file
    try:
        with open(config_path, "w") as config_file:
            yaml.safe_dump(config_data, config_file)
        return str(experiment_dir)
    except IOError as e:
        print(f"Failed to write config file: {e}")


def experiment_path_manager(on_spock: bool) -> Path:
    """
    Quick function to determine the path for data loading and storage based on
    whether the code is being run on Spock (on the cluster) or locally.

    Returns
    -------
    data_base_path : Path
        The base path for data storage containing the raw, cleaned, processed, and results
        subfolders. The "results" subfolder is where experiment directories will be created.
    """

    prefix = "/jukebox" if on_spock else "/Volumes"
    return Path(
        f"{prefix}/brody/jbreda/behavioral_analysis/violations_multinomial/data"
    )


def get_init_config_data(config_type: str):
    """
    Returns initialized configuration dict based on the type specified.
    """
    if config_type == "standard":
        return {
            "_experiment_description": "string to enter here",
            "animal_ids": [None],
            "data_type": "new_trained",
            "experiment_type": None,
            "dmg_config": None,
            "model_config": None,
        }
    else:
        # Handle other types or raise an error
        raise ValueError(f"Unsupported config type: {config_type}")


def main():
    parser = argparse.ArgumentParser(description="Create a new experiment directory.")
    parser.add_argument("experiment_name", type=str, help="Name of the experiment.")
    parser.add_argument(
        "config_type", type=str, help="Type of configuration for the experiment."
    )
    parser.add_argument(
        "--on_spock",
        action="store_true",
        help="Flag to set if running on Spock (cluster).",
    )
    args = parser.parse_args()

    # Call the directory creation function
    exp_dir = create_experiment_directory(
        args.experiment_name, args.config_type, args.on_spock
    )
    print(exp_dir)  # This will be captured by the shell script
    sys.stderr.write(f"Experiment {args.experiment_name} created at: {exp_dir}\n")


if __name__ == "__main__":
    main()
