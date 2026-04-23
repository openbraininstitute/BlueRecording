# SPDX-License-Identifier: GPL-3.0-or-later
from importlib.metadata import version

__version__ = version("bluerecording")


def _check_dependencies():
    """Verify runtime dependencies that need special attention.

    h5py is declared in pyproject.toml but must be the MPI-enabled build.
    The default pip wheel lacks MPI support.
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
            "Or use 'source setup.sh' which handles this automatically."
        )


_check_dependencies()