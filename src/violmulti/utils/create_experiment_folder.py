import argparse
from pathlib import Path
import yaml
import sys
import violmulti.utils.save_load as save_load


def create_experiment_directory(
    experiment_name: str,
    config_type: str,
) -> str:

    # Create the experiment directory and subdirectories in data/results/experiment_name
    cup_data_path = save_load.determine_path_to_cup_data()
    experiment_dir = Path(cup_data_path / "results" / f"experiment_{experiment_name}")
    subdirectories = ["models", "logs", "figures", "data"]
    for subdir in subdirectories:
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Create config file
    config_data = get_init_config_data(config_type)
    config_data["relative_data_path"] = str(cup_data_path)
    config_path = experiment_dir / "config.yaml"

    # Write the config file
    try:
        save_load.save_config_to_yaml(config_data, config_path)
        return str(experiment_dir)
    except IOError as e:
        print(f"Failed to write config file: {e}")


def get_init_config_data(config_type: str):
    """
    Returns initialized configuration dict based on the type specified.
    """
    if config_type == "standard":
        return {
            "_experiment_description": "string to enter here",
            "animal_ids": [None],
            "data_type": "new_trained",
            "experiment_type": "standard",  # could take from keys in runner class!
            "dmg_config": None,
            "model_config": {
                "n_inits": 1,
                "n_iters": 100,
                "n_states": None,
                "n_features": None,
                "n_categories": None,  # add other things!
            },
        }
    elif config_type == "mega_fit":
        return {
            "_experiment_description": "string to enter here",
            "animal_ids": None,  # all animals
            "data_type": "new_trained",
            "experiment_type": "mega_fit",  # could take from keys in runner class!
            "dmg_config": {
                "s_a_stand": "lambda df: standardize(df.s_a)",
                "s_b_stand": "lambda df: standardize(df.s_b)",
                "bias": "lambda df: (add_bias_column(df))",
                "session": "lambda df: copy(df.animal_id_session)",  # special session column
                "labels": "binary_choice_labels()",
            },
            "model_config": {
                "n_inits": 20,
                "n_iters": 100,
                "n_states": None,
                "n_features": None,
                "n_categories": None,  # add other things!
            },
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
    args = parser.parse_args()

    # Call the directory creation function
    exp_dir = create_experiment_directory(
        args.experiment_name,
        args.config_type,
    )
    print(exp_dir)  # This will be captured by the shell script
    sys.stderr.write(f"Experiment {args.experiment_name} created at: {exp_dir}\n")


if __name__ == "__main__":
    main()
