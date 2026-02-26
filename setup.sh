#!/usr/bin/env bash
# setup.sh — Full MPI-enabled Python environment setup

INSTALL_MODE="normal"

# -------------------------
# Parse arguments
# -------------------------
if [[ "$1" == "--dev" ]]; then
    INSTALL_MODE="dev"
fi

echo "=== Install mode: $INSTALL_MODE ==="

# -------------------------
# Detect OS
# -------------------------
echo "=== Detecting platform ==="
OS="$(uname -s)"

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
    export HDF5_DIR=$(dirname "$(dirname "$(which h5cc)")")

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