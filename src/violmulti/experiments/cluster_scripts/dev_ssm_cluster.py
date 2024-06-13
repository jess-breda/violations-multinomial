import ssm
from violmulti.data.dataset_loader import DatasetLoader


import ssm
from violmulti.data.dataset_loader import DatasetLoader
from violmulti.features.design_matrix_generator_PWM import *
from violmulti.utils.config_utils import *
import sys
import os


def main(experiment_name):

    # make the config path
    config_path = f"./{experiment_name}/config.yaml"
    config_path = os.path.abspath(config_path)
    config = load_config_from_yaml(config_path)

    animal_ids = config["animal_ids"]
    relative_data_path = config["relative_data_path"]
    data_type = config["data_type"]

    dmg_config = convert_dmg_config_functions(config["dmg_config"])

    df = DatasetLoader(
        animal_ids=animal_ids,
        data_type=data_type,
        relative_data_path=relative_data_path,
    ).load_data()

    dmg = DesignMatrixGeneratorPWM(df.reset_index(), dmg_config, verbose=True)
    X, y = dmg.create()

    print("X shape: ", X.shape)
    print("y shape: ", y.shape)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # print("Usage: python script_name.py path_to_config.json")
        # config_path = "./example_experiment/config.json"
        sys.exit(1)
    else:

        experiment_name = sys.argv[1]

    main(experiment_name)
