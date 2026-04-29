#!/usr/bin/env bash
# download_examples_data.sh — Download example datasets for BlueRecording.
#
# Downloads atlas, network, and FEM field data from Zenodo.
# Idempotent: skips any dataset that is already present.
#
# Usage: ./download_examples_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

# -------------------------
# Detect OS and ensure unzip is available
# -------------------------
OS="$(uname -s)"

if ! command -v unzip &>/dev/null; then
    if [[ "$OS" == "Linux" ]]; then
        echo "=== Installing unzip (required for data extraction) ==="
        SUDO=""
        if [[ $(id -u) -ne 0 ]] && command -v sudo &>/dev/null; then
            SUDO="sudo"
        fi
        $SUDO apt update
        $SUDO apt install -y unzip
    else
        echo "Error: unzip is not installed and could not be installed automatically."
        exit 1
    fi
fi

# -------------------------
# Atlas dataset
# -------------------------
ATLAS_DIR="examples/data/atlas"

if [ -d "$ATLAS_DIR" ] && [ "$(ls -A "$ATLAS_DIR")" ]; then
    echo "=== Skipping atlas download — $ATLAS_DIR already exists and is not empty ==="
else
    echo "=== Downloading atlas dataset ==="
    mkdir -p examples/data
    curl -L -o examples/data/atlas.zip \
        "https://zenodo.org/record/10927050/files/atlas.zip?download=1"

    echo "=== Unpacking atlas dataset ==="
    unzip -q examples/data/atlas.zip -d examples/data

    echo "=== Cleaning up ==="
    rm examples/data/atlas.zip

    echo "=== Atlas dataset ready at $ATLAS_DIR ==="
fi

# -------------------------
# Networks dataset
# -------------------------
CONFIG_DIR="examples/sscx_100_cells/configuration"
NETWORK_DIR="$CONFIG_DIR/networks"

if [ -d "$NETWORK_DIR" ] && [ "$(ls -A "$NETWORK_DIR")" ]; then
    echo "=== Skipping networks download — $NETWORK_DIR already exists and is not empty ==="
else
    echo "=== Downloading networks dataset ==="

    mkdir -p "$CONFIG_DIR"

    curl -L -o networks.zip \
        "https://zenodo.org/record/10927050/files/networks.zip?download=1"

    echo "=== Unpacking networks dataset ==="
    unzip -q networks.zip -d "$CONFIG_DIR"

    echo "=== Cleaning up ==="
    rm networks.zip

    echo "=== Networks dataset ready at $NETWORK_DIR ==="
fi

# -------------------------
# Single-cell L5 TPC FEM field files
# -------------------------
L5_TPC_DIR="examples/single_cell_l5_tpc"
L5_TPC_FILE1="$L5_TPC_DIR/Infinite_VeryFar_HighRes.h5"
L5_TPC_FILE2="$L5_TPC_DIR/Infinite_Close_HighRes_SmallSphere.h5"

if [ -f "$L5_TPC_FILE1" ] && [ -f "$L5_TPC_FILE2" ]; then
    echo "=== Skipping single_cell_l5_tpc FEM field download — files already exist ==="
else
    echo "=== Downloading single_cell_l5_tpc FEM field files ==="
    mkdir -p "$L5_TPC_DIR"

    if [ ! -f "$L5_TPC_FILE1" ]; then
        curl -L -o "$L5_TPC_FILE1" \
            "https://zenodo.org/record/10927050/files/Infinite_VeryFar_HighRes.h5?download=1"
    fi

    if [ ! -f "$L5_TPC_FILE2" ]; then
        curl -L -o "$L5_TPC_FILE2" \
            "https://zenodo.org/record/10927050/files/Infinite_Close_HighRes_SmallSphere.h5?download=1"
    fi

    echo "=== single_cell_l5_tpc FEM field files ready ==="
fi

echo ""
echo "=== All example data downloaded ==="
