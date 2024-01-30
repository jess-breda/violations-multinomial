import pathlib
import platform
import pandas as pd

DATA_PATH = pathlib.Path("X:\jbreda\learning_project\data")


def determine_data_path(mode="local"):
    """
    Quick function to determine the path to the data directory
    given the operating system

    params
    ------
    mode: str (default: "local")
        "local" or "cup" depending on where the data is stored
        note "cup" requires a VPN to princeton network
    """
    if platform.system() == "Windows":
        data_path = pathlib.Path("X:\jbreda\learning_project\data")
    else:
        if mode == "cup":
            data_path = pathlib.Path("/Volumes/brody/jbreda/learning_project/data")
        elif mode == "local":
            data_path = pathlib.Path(
                "/Users/jessbreda/Desktop/github/animal-learning/data"
            )
    return data_path


def get_rat_viol_data(animal_ids=None, mode="local"):
    """
    Function to load the rat violation dataframe

    params
    ------
    animal_ids: str or list (default: None)
        animal ids to return data for, will return
        all data if None
    mode: str (default: "local")
        "local" or "cup" depending on where the data is stored
        note "cup" requires a VPN to princeton network

    returns
    -------
    rat_df: pd.DataFrame
        Dataframe containing data for specified animal(s)
        from main PWM dataset where violations were tracked with
        trials as row index. See
        `notebooks/create_violation_dataset.ipynb` for more info.
    """
    data_path = determine_data_path(mode=mode)
    file_name = "processed/violation_data.csv"

    rat_df = pd.read_csv(data_path / file_name)

    if animal_ids:
        print(f"returning truncated viol data for {animal_ids}")
        rat_df = rat_df.query("animal_id == @animal_ids").copy()
        rat_df.reset_index(drop=True, inplace=True)
    else:
        print(f"returning viol dataset for all animals")

    return rat_df


def get_full_rat_data(animal_ids=None, mode="local"):
    """
    Function to load all rat data from the main PWM dataset
    without truncate at end of violations being tracked.

    This is similar to the raw "rat_behavior.csv() with some
    additional formating:

        subject_id -> animal_id
        violation column added
        n_trials column added
        training_stage_cat column added
        delay column is rounded to 2 decimal places

    params
    ------
    animal_ids: str or list (default: None)
        animal ids to return data for, will return
        all data if None
    mode: str (default: "local")
        "local" or "cup" depending on where the data is stored
        note "cup" requires a VPN to princeton network

    returns
    -------
    rat_df pd.DataFrame
        Dataframe containing data for specified animal(s)
        from main PWM dataset for all trials
    """

    data_path = determine_data_path(mode=mode)
    file_name = "processed/all_data.csv"

    rat_df = pd.read_csv(data_path / file_name)

    if animal_ids:
        print(f"returning full data for {animal_ids}")
        rat_df = rat_df.query("animal_id == @animal_ids").copy()
        rat_df.reset_index(drop=True, inplace=True)
    else:
        print(f"returning full dataset for all animals")

    return rat_df
