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
        --data) DOWNLOAD_DATA=1 ;;
        --help|-h)
            echo "Usage: source setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev         Install development version (includes -e pip install)"
            echo "  --no-system   Skip system package installation"
            echo "  --data        Download and unpack datasets"
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
export SONATAREPORT_DIR="$(pwd)/libsonatareport/build/install"
export NEURODAMUS_NEOCORTEX_ROOT="$(pwd)/neurodamus-models/build/install"
export HOC_LIBRARY_PATH="$NEURODAMUS_NEOCORTEX_ROOT/share/neurodamus_neocortex/hoc"
export PATH=$(pwd)/nrn/build/install/bin:$PATH
export PYTHONPATH=$(pwd)/nrn/build/install/lib/python:$PYTHONPATH
export PATH=$NEURODAMUS_NEOCORTEX_ROOT/bin:$PATH

if [[ "$OS" == "Darwin" ]]; then
  export CORENEURONLIB="$NEURODAMUS_NEOCORTEX_ROOT/lib/libcorenrnmech.dylib"
  export NRNMECH_LIB_PATH="$NEURODAMUS_NEOCORTEX_ROOT/lib/libnrnmech.dylib"
  # override system bison
  export PATH="/opt/homebrew/opt/bison/bin:$PATH"
elif [[ "$OS" == "Linux" ]]; then
  export CORENEURONLIB="$NEURODAMUS_NEOCORTEX_ROOT/lib/libcorenrnmech.so"
  export NRNMECH_LIB_PATH="$NEURODAMUS_NEOCORTEX_ROOT/lib/libnrnmech.so"
else
  echo "Unsupported platform: PLATFORM=$PLATFORM OS=$OS" >&2
  exit 1
fi

# -------------------------
# Virtual environment
# -------------------------

VENV_EXISTS=0   # default: not found
if [ -d "venv" ]; then
    VENV_EXISTS=1
fi

echo "=== Checking for existing virtual environment ==="
if [ $VENV_EXISTS -eq 1 ]; then
    echo "Virtual environment exists — activating."
    source venv/bin/activate
else
    echo "=== Creating virtual environment ==="
    python3 -m venv venv
    source venv/bin/activate

    pip install --upgrade pip setuptools wheel cython numpy
fi

# -------------------------
# Install libsonatareport
# -------------------------
if [ ! -d "libsonatareport" ]; then
    git clone https://github.com/openbraininstitute/libsonatareport.git --recursive --depth=1
    cmake -B libsonatareport/build -S libsonatareport \
    -DCMAKE_INSTALL_PREFIX=$SONATAREPORT_DIR -DCMAKE_BUILD_TYPE=Release -DSONATA_REPORT_ENABLE_SUBMODULES=ON -DSONATA_REPORT_ENABLE_MPI=ON -GNinja

    cmake --build libsonatareport/build
    cmake --install libsonatareport/build
fi

# -------------------------
# Install neuron with libsonatareport
# -------------------------
if [ ! -d "nrn" ]; then
    git clone --recursive https://github.com/neuronsimulator/nrn.git
    python -m pip install --upgrade pip -r nrn/nrn_requirements.txt
    cmake -B nrn/build -S nrn -G Ninja \
        -DPYTHON_EXECUTABLE=$(which python) \
        -DCMAKE_INSTALL_PREFIX=$(pwd)/nrn/build/install \
        -DNRN_ENABLE_MPI=ON \
        -DNRN_ENABLE_INTERVIEWS=OFF \
        -DNRN_ENABLE_CORENEURON=ON \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCORENRN_ENABLE_REPORTING=ON \
        -DCMAKE_PREFIX_PATH=$SONATAREPORT_DIR -GNinja
    cmake --build nrn/build --parallel
    cmake --build nrn/build --target install
fi

# -------------------------
# Install h5py, mpi4pi and neurodamus
# -------------------------
if [ $VENV_EXISTS -eq 0 ]; then
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
    
    pip install --no-binary=mpi4py mpi4py
    pip install --no-cache-dir --no-binary=h5py h5py --no-build-isolation
    pip install neurodamus
fi

# -------------------------
# Install neurodamus-models
# -------------------------
if [ ! -d "neurodamus-models" ]; then
  git clone https://github.com/openbraininstitute/neurodamus-models.git

  DATADIR=$(python -c "import neurodamus; from pathlib import Path; print(Path(neurodamus.__file__).parent / 'data')")

  cmake -B neurodamus-models/build -S neurodamus-models/ \
      -DPython_EXECUTABLE=$(which python) \
      -DCMAKE_INSTALL_PREFIX=$NEURODAMUS_NEOCORTEX_ROOT \
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      -DCMAKE_PREFIX_PATH=$SONATAREPORT_DIR \
      -DNEURODAMUS_CORE_DIR=${DATADIR} \
      -DNEURODAMUS_MECHANISMS=neocortex \
      -DNEURODAMUS_NCX_V5=ON

  cmake --build neurodamus-models/build
  cmake --install neurodamus-models/build
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
# Download atlas data if requested via --data
# -------------------------
if [[ "$DOWNLOAD_DATA" == "1" ]]; then
    ATLAS_DIR="examples/data/atlas"

    if [ -d "$ATLAS_DIR" ] && [ "$(ls -A "$ATLAS_DIR")" ]; then
        echo "=== Skipping atlas download — $ATLAS_DIR already exists and is not empty ==="
    else
        echo "=== Downloading atlas dataset (requested via --atlas) ==="
        mkdir -p examples/data
        curl -L -o examples/data/atlas.zip \
            "https://zenodo.org/record/10927050/files/atlas.zip?download=1"

        echo "=== Unpacking atlas dataset ==="
        unzip -q examples/data/atlas.zip -d examples/data

        echo "=== Cleaning up ==="
        rm examples/data/atlas.zip

        echo "=== Atlas dataset ready at $ATLAS_DIR ==="
    fi
else
    echo "=== Skipping atlas download — --atlas not given ==="
fi


# -------------------------
# Download networks data if requested via --data
# -------------------------
if [[ "$DOWNLOAD_DATA" == "1" ]]; then
    CONFIG_DIR="examples/circuitTest/data/simulation/configuration"
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
fi

# -------------------------
# Run compare-to-reference-solutions
# -------------------------
if [[ "$DOWNLOAD_DATA" == "1" ]]; then
    neurodamus examples/compare-to-reference-solutions/data/simulation/simulation_config.json
fi