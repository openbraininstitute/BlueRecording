# SPDX-License-Identifier: GPL-3.0-or-later
from importlib.metadata import version

__version__ = version("bluerecording")


def _check_dependencies():
    """Verify that runtime dependencies not declared in pyproject.toml are available.

    bluerecording requires mpi4py, h5py (with MPI support), NEURON, and neurodamus.
    These are not listed in pyproject.toml because they need special build flags
    or must be built from source depending on the install mode. See setup.sh and
    the README for installation instructions.
    """
    try:
        from mpi4py import MPI  # noqa: F401
    except ImportError:
        raise ImportError(
            "bluerecording requires mpi4py.\n"
            "It is not listed in pyproject.toml because it must match your system's MPI.\n"
            "Install it with: pip install mpi4py\n"
            "Or use 'source setup.sh' which handles this automatically."
        )

    try:
        import h5py
    except ImportError:
        raise ImportError(
            "bluerecording requires h5py built with MPI support for parallel HDF5 I/O.\n"
            "It is not listed in pyproject.toml because the pip wheel lacks MPI support.\n"
            "Install it with: HDF5_MPI=ON pip install --no-cache-dir --no-binary=h5py h5py "
            "--no-build-isolation\n"
            "Or use 'source setup.sh' which handles this automatically."
        )

    if not h5py.get_config().mpi:
        raise ImportError(
            "h5py is installed but was built without MPI support.\n"
            "This typically happens when h5py is installed from a prebuilt pip wheel.\n"
            "It needs to be recompiled against your system's MPI-enabled HDF5 library.\n"
            "Fix with:\n"
            "  pip uninstall h5py\n"
            "  HDF5_MPI=ON pip install --no-cache-dir --no-binary=h5py h5py "
            "--no-build-isolation\n"
            "Or use 'source setup.sh' which handles this automatically."
        )

    try:
        import neuron  # noqa: F401
    except ImportError:
        raise ImportError(
            "bluerecording requires NEURON (neuron).\n"
            "It is not listed in pyproject.toml because in some install modes it must be\n"
            "built from source (e.g. with libsonatareport support for SONATA reporting).\n"
            "Install it with: pip install neuron\n"
            "Or use 'source setup.sh' which handles this automatically."
        )

    try:
        import neurodamus  # noqa: F401
    except ImportError:
        raise ImportError(
            "bluerecording requires neurodamus.\n"
            "It is not listed in pyproject.toml because setup.sh installs it from\n"
            "a Git branch/commit (see the 'neurodamus' section in setup.sh).\n"
            "Install it with: pip install neurodamus\n"
            "Or use 'source setup.sh' which handles this automatically."
        )


_check_dependencies()