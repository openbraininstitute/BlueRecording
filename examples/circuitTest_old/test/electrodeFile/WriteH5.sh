#!/usr/bin/env bash

# SPDX-License-Identifier: GPL-3.0-or-later

python ../../../scripts/run_initialize_h5.py 'electrodes.csv' '../../data/simulation/simulation_config.json' 'coeffs.h5' 

# Parameters
NEURONS_PER_FILE=1000
FILES_PER_FOLDER=50

# Run the Python script locally
python ../../../scripts/run_write_weights.py \
    "../../data/simulation/simulation_config.json" \
    "../../data/getPositions/positions" \
    "coeffs.h5" \
    "$NEURONS_PER_FILE" \
    "$FILES_PER_FOLDER"