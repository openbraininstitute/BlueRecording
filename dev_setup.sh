#!/usr/bin/env bash
# dev_setup.sh — Full development environment setup for BlueRecording
#
# Builds everything from source: libsonatareport, NEURON, neurodamus-models
# with reporting enabled. Editable install with test + notebook dependencies.
#
# Usage: ./dev_setup.sh [OPTIONS]
#
# This script MUST be executed (not sourced). It generates an env.sh file
# that you source afterward to activate the environment in your shell.
#
# For platform use (without this script):
#   pip install bluerecording[neuron]   — weights only, neuron from pip
#   pip install bluerecording           — simulations, neuron already from source
# See pyproject.toml for notes on dependencies not declared there
# (e.g. neurodamus-models).

# -------------------------
# Pinned versions (edit here to update)
# -------------------------
NEURON_COMMIT="9.0.1"
NEURODAMUS_COMMIT="4.2.1"

SKIP_SYSTEM=0
CLEAN_INSTALL=0

# -------------------------
# Parse arguments
# -------------------------
for arg in "$@"; do
    case $arg in
        --no-system) SKIP_SYSTEM=1 ;;
        --clean-install) CLEAN_INSTALL=1 ;;
        --help|-h)
            echo "Usage: ./dev_setup.sh [OPTIONS]"
            echo ""
            echo "Sets up a full development environment with NEURON built from"
            echo "source (with libsonatareport), neurodamus-models, and an editable"
            echo "install of bluerecording with test and notebook dependencies."
            echo ""
            echo "After completion, run 'source env.sh' to activate the environment."
            echo ""
            echo "Options:"
            echo "  --no-system      Skip system package installation (brew/apt)"
            echo "  --clean-install  Remove venv, cloned repos, and build artifacts (keeps data)"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $arg"
            echo ""
            echo "Run './dev_setup.sh --help' for usage."
            exit 1
            ;;
    esac
done

# -------------------------
# Clean install
# -------------------------
if [[ $CLEAN_INSTALL -eq 1 ]]; then
    echo "This will remove:"
    echo "  - venv/"
    echo "  - nrn/"
    echo "  - libsonatareport/"
    echo "  - neurodamus-models/"
    echo "  - bluerecording.egg-info/"
    echo "  - build/"
    echo ""
    echo "Downloaded data will NOT be removed."
    echo ""
    printf "Are you sure? [y/N] "
    read -r confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -rf venv nrn libsonatareport neurodamus-models bluerecording.egg-info build
        echo "=== Clean complete ==="
    else
        echo "=== Clean cancelled ==="
        exit 0
    fi
fi

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
        brew install openmpi hdf5-mpi python bison

    elif [[ "$OS" == "Linux" ]]; then
        echo "Linux detected"
        SUDO=""
        if [[ $(id -u) -ne 0 ]] && command -v sudo &>/dev/null; then
            SUDO="sudo"
        fi
        $SUDO apt update
        $SUDO apt install -y \
            openmpi-bin \
            libopenmpi-dev \
            libhdf5-openmpi-dev \
            python3 \
            python3-dev \
            python3-pip \
            python3-venv \
            bison
    else
        echo "Unsupported OS: $OS"
        exit 1
    fi
fi

# -------------------------
# Set globals
# -------------------------
export NEURODAMUS_NEOCORTEX_ROOT="$(pwd)/neurodamus-models/build/install"
export HOC_LIBRARY_PATH="$NEURODAMUS_NEOCORTEX_ROOT/share/neurodamus_neocortex/hoc"
export PATH=$NEURODAMUS_NEOCORTEX_ROOT/bin:$PATH

export SONATAREPORT_DIR="$(pwd)/libsonatareport/build/install"
export PATH=$(pwd)/nrn/build/install/bin:$PATH
export PYTHONPATH=$(pwd)/nrn/build/install/lib/python:${PYTHONPATH:-}

if [[ "$OS" == "Darwin" ]]; then
  export CORENEURONLIB="$NEURODAMUS_NEOCORTEX_ROOT/lib/libcorenrnmech.dylib"
  export NRNMECH_LIB_PATH="$NEURODAMUS_NEOCORTEX_ROOT/lib/libnrnmech.dylib"
  # override system bison
  export PATH="/opt/homebrew/opt/bison/bin:$PATH"
elif [[ "$OS" == "Linux" ]]; then
  export CORENEURONLIB="$NEURODAMUS_NEOCORTEX_ROOT/lib/libcorenrnmech.so"
  export NRNMECH_LIB_PATH="$NEURODAMUS_NEOCORTEX_ROOT/lib/libnrnmech.so"
else
  echo "Unsupported platform: OS=$OS" >&2
  exit 1
fi

# -------------------------
# Virtual environment
# -------------------------

echo "=== Checking for virtual environment ==="
if [ -d "venv" ]; then
    echo "Activating existing ./venv"
    source venv/bin/activate
else
    echo "=== Creating virtual environment ==="
    uv venv --python ">=3.10,<3.14" venv
    source venv/bin/activate
fi

uv pip install --upgrade pip setuptools wheel cython numpy

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

echo "=== Installing base dependencies ==="

uv pip install mpi4py
uv pip install --no-cache --no-binary h5py h5py --no-build-isolation

# =========================================================================
# Build components
# =========================================================================

# -------------------------
# libsonatareport — needed by NEURON for reporting
# -------------------------
if [ ! -d "libsonatareport" ]; then
    git clone https://github.com/openbraininstitute/libsonatareport.git --recursive --depth=1
    cmake -B libsonatareport/build -S libsonatareport \
        -DCMAKE_INSTALL_PREFIX=$SONATAREPORT_DIR -DCMAKE_BUILD_TYPE=Release -DSONATA_REPORT_ENABLE_SUBMODULES=ON -DSONATA_REPORT_ENABLE_MPI=ON -GNinja

    cmake --build libsonatareport/build
    cmake --install libsonatareport/build
fi

# -------------------------
# NEURON — built from source with reporting support
# -------------------------
if uv pip show neuron &>/dev/null; then
    echo "Error: NEURON is already installed from pip in this environment."
    echo "A pip-installed NEURON conflicts with a source build."
    echo "Run 'uv pip uninstall neuron' first, or use --clean-install."
    exit 1
fi
if [ ! -d "nrn" ]; then
    echo "=== Building NEURON from source ==="
    git clone --branch=master https://github.com/neuronsimulator/nrn.git
    cd nrn && git checkout $NEURON_COMMIT && cd ..
    uv pip install --upgrade pip -r nrn/nrn_requirements.txt

    NRN_C_COMPILER=gcc
    NRN_CXX_COMPILER=g++

    cmake -B nrn/build -S nrn -G Ninja \
        -DPYTHON_EXECUTABLE=$(which python) \
        -DCMAKE_INSTALL_PREFIX=$(pwd)/nrn/build/install \
        -DNRN_ENABLE_MPI=ON \
        -DNRN_ENABLE_INTERVIEWS=OFF \
        -DNRN_ENABLE_CORENEURON=ON \
        -DCMAKE_C_COMPILER=$NRN_C_COMPILER \
        -DCMAKE_CXX_COMPILER=$NRN_CXX_COMPILER \
        -DCORENRN_ENABLE_REPORTING=ON \
        -DCMAKE_PREFIX_PATH=$SONATAREPORT_DIR

    cmake --build nrn/build --parallel
    cmake --build nrn/build --target install
fi

# -------------------------
# neurodamus
# -------------------------
uv pip install git+https://github.com/openbraininstitute/neurodamus.git@${NEURODAMUS_COMMIT}

# -------------------------
# neurodamus-models — with reporting (linked against libsonatareport)
# -------------------------
if [ ! -d "neurodamus-models" ]; then
    git clone --depth=1 https://github.com/openbraininstitute/neurodamus-models.git
    NEURODAMUS_PYTHON=$(python -c "import neurodamus; from pathlib import Path; print(Path(neurodamus.__file__).parent / 'data')")

    cmake -B neurodamus-models/build -S neurodamus-models/ \
        -DPython_EXECUTABLE=$(which python) \
        -DCMAKE_INSTALL_PREFIX=$NEURODAMUS_NEOCORTEX_ROOT \
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
        -DNEURODAMUS_CORE_DIR=${NEURODAMUS_PYTHON} \
        -DNEURODAMUS_MECHANISMS=neocortex \
        -DNEURODAMUS_NCX_V5=ON \
        -DCMAKE_PREFIX_PATH=$SONATAREPORT_DIR

    cmake --build neurodamus-models/build
    cmake --install neurodamus-models/build
fi

# -------------------------
# Install project (editable, with test + notebook deps)
# -------------------------
echo "=== Installing project ==="
uv pip install -e ".[test,notebooks,lint]"

echo "=== Setup complete ==="

# -------------------------
# Write env.sh for environment activation
# -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/env.sh"

cat > "$ENV_FILE" << EOF
# Auto-generated by dev_setup.sh — source this to activate the environment.
# Usage: source env.sh
source "$SCRIPT_DIR/venv/bin/activate" 2>/dev/null || true
export NEURODAMUS_NEOCORTEX_ROOT="$NEURODAMUS_NEOCORTEX_ROOT"
export HOC_LIBRARY_PATH="$HOC_LIBRARY_PATH"
export SONATAREPORT_DIR="$SONATAREPORT_DIR"
export CORENEURONLIB="$CORENEURONLIB"
export NRNMECH_LIB_PATH="$NRNMECH_LIB_PATH"
export PATH="$NEURODAMUS_NEOCORTEX_ROOT/bin:$(pwd)/nrn/build/install/bin:\$PATH"
export PYTHONPATH="$(pwd)/nrn/build/install/lib/python:\${PYTHONPATH:-}"
EOF

echo ""
echo "=== Environment file written to env.sh ==="
echo "To activate the environment, run:"
echo ""
echo "    source env.sh"
echo ""
echo "To download example datasets (atlas, networks, FEM fields), run:"
echo ""
echo "    ./download_examples_data.sh"
echo ""
