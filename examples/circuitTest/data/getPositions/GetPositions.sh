#!/usr/bin/env bash
set -euo pipefail

NEURONS_PER_FILE=1000
FILES_PER_FOLDER=50

# Number of folders you want to pre-create
NUM_FOLDERS=1

for ((i=0; i<NUM_FOLDERS; i++)); do
    folder="positions/$i"
    mkdir -p "$folder"
done

python ../../../scripts/run_get_positions.py \
    "../simulation/simulation_config.json" \
    "positions" \
    "$NEURONS_PER_FILE" \
    "$FILES_PER_FOLDER"

