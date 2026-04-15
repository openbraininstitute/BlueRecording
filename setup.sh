#!/usr/bin/env bash
# setup.sh — Full MPI-enabled Python environment setup

# -------------------------
# Pinned versions (edit here to update)
# -------------------------
NEURON_COMMIT="9.0.1"

if command -v deactivate &> /dev/null; then
    deactivate
fi

INSTALL_MODE="normal"
SKIP_SYSTEM=0
DOWNLOAD_DATA=0
NO_CACHE=0
QUICK=0

# -------------------------
# Parse arguments (with --help)
# -------------------------
for arg in "$@"; do
    case $arg in
        --dev) INSTALL_MODE="dev" ;;
        --no-system) SKIP_SYSTEM=1 ;;
        --data) DOWNLOAD_DATA=1 ;;
        --no-cache) NO_CACHE=1 ;;
        --quick) QUICK=1 ;;
        --help|-h)
            echo "Usage: source setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev         Install development version (includes -e pip install)"
            echo "  --no-system   Skip system package installation"
            echo "  --data        Download and unpack datasets"
            echo "  --no-cache    Remove venv, cloned repos, and build artifacts (keeps data)"
            echo "  --quick       Skip NEURON, libsonatareport, neurodamus, and neurodamus-models builds"
            echo "  --help, -h    Show this help message"
            return 0 2>/dev/null || exit 0
            ;;
        *)
            echo "Error: Unknown option: $arg"
            echo ""
            echo "Run 'source setup.sh --help' for usage."
            return 1 2>/dev/null || exit 1
            ;;
    esac
done

# -------------------------
# No-cache mode
# -------------------------
if [[ $NO_CACHE -eq 1 ]]; then
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
        return 0 2>/dev/null || exit 0
    fi
fi

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
        brew install openmpi hdf5-mpi python bison

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

echo "=== Checking for existing virtual environment ==="
if [ -d "venv" ]; then
    echo "Virtual environment exists — activating."
    source venv/bin/activate
else
    echo "=== Creating virtual environment ==="
    python3 -m venv venv
    source venv/bin/activate

    pip install --upgrade pip setuptools wheel cython numpy

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
fi

# -------------------------
# Install libsonatareport, NEURON, neurodamus, neurodamus-models
# (skipped with --quick)
# -------------------------
if [[ $QUICK -eq 0 ]]; then

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
    # Install NEURON from source (with libsonatareport support)
    # -------------------------
    if [ ! -d "nrn" ]; then
        echo "=== Building NEURON from source ==="
        git clone --branch=master https://github.com/neuronsimulator/nrn.git
        cd nrn && git checkout $NEURON_COMMIT && cd ..
        pip install --upgrade pip -r nrn/nrn_requirements.txt

        if [[ "$OS" == "Darwin" ]]; then
            NRN_C_COMPILER=gcc
            NRN_CXX_COMPILER=g++
        else
            NRN_C_COMPILER=gcc
            NRN_CXX_COMPILER=g++
        fi

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
    # Install neurodamus
    # -------------------------
    pip install git+https://github.com/openbraininstitute/neurodamus.git@main

    # -------------------------
    # Install neurodamus-models
    # -------------------------
    if [ ! -d "neurodamus-models" ]; then
        git clone https://github.com/openbraininstitute/neurodamus-models.git
        NEURODAMUS_PYTHON=$(python -c "import neurodamus; from pathlib import Path; print(Path(neurodamus.__file__).parent / 'data')")

        cmake -B neurodamus-models/build -S neurodamus-models/ \
            -DPython_EXECUTABLE=$(which python) \
            -DCMAKE_INSTALL_PREFIX=$NEURODAMUS_NEOCORTEX_ROOT \
            -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
            -DCMAKE_PREFIX_PATH=$SONATAREPORT_DIR \
            -DNEURODAMUS_CORE_DIR=${NEURODAMUS_PYTHON} \
            -DNEURODAMUS_MECHANISMS=neocortex \
            -DNEURODAMUS_NCX_V5=ON

        cmake --build neurodamus-models/build
        cmake --install neurodamus-models/build
    fi

fi # end --quick guard


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


    # -------------------------
    # Download networks data if requested via --data
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
    # Download single_cell_l5_tpc FEM field files if requested via --data
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
else
    echo "=== Skipping data download and generation — --data not given ==="
fi