# SPDX-License-Identifier: GPL-3.0-or-later
from importlib.metadata import version

__version__ = version("bluerecording")


def _check_dependencies():
    """Verify runtime dependencies that need special attention.

    - h5py is declared in pyproject.toml but must be the MPI-enabled build.
      The default pip wheel lacks MPI support.
    - neuron is an optional extra ([neuron]). Required at runtime but kept
      optional because it may be built from source.
    """
    import h5py

    if not h5py.get_config().mpi:
        raise ImportError(
            "h5py is installed but was built without MPI support.\n"
            "bluerecording requires parallel HDF5 I/O.\n"
            "Fix with:\n"
            "  pip uninstall h5py\n"
            "  HDF5_MPI=ON pip install --no-cache-dir --no-binary=h5py h5py "
            "--no-build-isolation\n"
            "Or use './dev_setup.sh' followed by 'source env.sh' "
            "which handles this automatically."
        )

    try:
        # noqa: F401 — mpi4py must be imported before neuron to ensure
        # MPI_Init has been called. When NEURON is statically linked against
        # MPI (NRN_ENABLE_MPI_DYNAMIC=OFF), it expects MPI to already be
        # initialized by the time it loads.
        from mpi4py import MPI  # noqa: F401  # isort: skip
        import neuron  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "bluerecording requires NEURON.\n"
            "Install with: pip install bluerecording[neuron]\n"
            "Or use './dev_setup.sh' followed by 'source env.sh' "
            "to build from source."
        ) from err


_check_dependencies()
