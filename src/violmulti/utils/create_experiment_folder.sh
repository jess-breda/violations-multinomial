#!/bin/bash

# Example usage:
# ./create_experiment_folder.sh experiment_name standard 
    # - will create a "experiment_name" folder with standard config running
    # - from local cuputer accessing cup
# ./create_experiment_folder.sh experiment_name standard --on_spock --cd
    # - will create a "experiment_name" folder with standard config running
    # - from spock accessing cup
    # - will change directory to the newly created folder

# Flag to check if we should change directory after script execution
change_dir=false

# Prepare arguments for the Python script
python_args=()
for arg in "$@"; do
    case "$arg" in
        --cd)
            change_dir=true
            ;;
        --on_spock)
            on_spock=true
            ;;
        *)
            python_args+=("$arg")
            ;;
    esac
done

# Load module and activate environment if --on_spock flag is present
if [ "$on_spock" = true ]; then
    module load anacondapy/2024.02
    conda activate viol-multi
    echo "Anaconda loaded and viol-multi environment activated"
fi

# Run the Python script without the --cd argument and capture the new directory path
NEW_DIR=$(python create_experiment_folder.py "${python_args[@]}")

# Check if we should change the directory
if [ "$change_dir" = true ]; then
    echo "Changing directory to $NEW_DIR"
    cd "$NEW_DIR"
    # Open a new interactive shell if the directory change is intended to persist
    exec bash
fi