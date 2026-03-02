#!/usr/bin/env bash
# setup.sh — Full MPI-enabled Python environment setup

INSTALL_MODE="normal"
SKIP_SYSTEM=0

# -------------------------
# Parse arguments (with --help)
# -------------------------
for arg in "$@"; do
    case $arg in
        --dev) INSTALL_MODE="dev" ;;
        --no-system) SKIP_SYSTEM=1 ;;
        --atlas) DOWNLOAD_ATLAS=1 ;;
        --help|-h)
            echo "Usage: source setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev         Install development version (includes -e pip install)"
            echo "  --no-system   Skip system package installation"
            echo "  --atlas       Download and unpack atlas dataset"
            echo "  --help, -h    Show this help message"
            return 0 2>/dev/null || exit 0
            ;;
    esac
done

echo "=== Install mode: $INSTALL_MODE ==="
echo "=== Skip system installation: $SKIP_SYSTEM ==="

# -------------------------
# Detect OS
# -------------------------
OS="$(uname -s)"
echo "=== Detecting platform: $OS ==="

if [[ $SKIP_SYSTEM -eq 0 ]]; then
    if [[ "$OS" == "Darwin" ]]; then
        echo "macOS detected"
        if ! command -v brew &> /dev/null; then
            echo "Homebrew is required but not installed."
            exit 1
        fi
        brew install openmpi hdf5-mpi python

    elif [[ "$OS" == "Linux" ]]; then
        echo "Linux detected"
        sudo apt update
        sudo apt install -y \
            openmpi-bin \
            libopenmpi-dev \
            libhdf5-openmpi-dev \
            python3 \
            python3-dev \
            python3-pip \
            python3-venv
    else
        echo "Unsupported OS: $OS"
        exit 1
    fi
fi

# -------------------------
# Virtual environment
# -------------------------
echo "=== Checking for existing virtual environment ==="
if [ -d "venv" ]; then
    echo "Virtual environment exists — activating."
    source venv/bin/activate
else
    echo "=== Creating virtual environment ==="
    python3 -m venv venv
    source venv/bin/activate

    echo "=== Configuring MPI build environment ==="
    export CC=$(which mpicc)
    export CXX=$(which mpicxx)
    export MPICC=$(which mpicc)

    export HDF5_MPI=ON
    if [[ "$OS" == "Linux" ]]; then
        export HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi
    else
        export HDF5_DIR=$(dirname "$(dirname "$(which h5cc)")")
    fi

    echo "=== Installing Python dependencies (MPI-enabled) ==="
    pip install --upgrade pip setuptools wheel cython numpy
    pip install --no-binary=mpi4py mpi4py
    pip install --no-cache-dir --no-binary=h5py h5py --no-build-isolation
fi

# -------------------------
# Install project
# -------------------------
echo "=== Installing project ==="
if [[ "$INSTALL_MODE" == "dev" ]]; then
    pip install -e ".[all]"
else
    pip install .
fi

echo "=== Setup complete ==="

# -------------------------
# Download atlas data if requested via --atlas
# -------------------------
if [[ "$DOWNLOAD_ATLAS" == "1" ]]; then
    ATLAS_DIR="examples/data/atlas"

    if [ -d "$ATLAS_DIR" ] && [ "$(ls -A "$ATLAS_DIR")" ]; then
        echo "=== Skipping atlas download — $ATLAS_DIR already exists and is not empty ==="
    else
        echo "=== Downloading atlas dataset (requested via --atlas) ==="
        mkdir -p examples/data
        curl -L -o examples/data/atlas.zip "https://zenodo.org/record/10927050/files/atlas.zip?download=1"

        echo "=== Unpacking atlas dataset ==="
        unzip -q examples/data/atlas.zip -d examples/data

        echo "=== Cleaning up ==="
        rm examples/data/atlas.zip

        echo "=== Atlas dataset ready at $ATLAS_DIR ==="
    fi
else
    echo "=== Skipping atlas download — --atlas not given ==="
fi