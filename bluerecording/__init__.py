# SPDX-License-Identifier: GPL-3.0-or-later
from importlib.metadata import version

__version__ = version("bluerecording")

from mpi4py import MPI
import h5py

if not h5py.get_config().mpi:
    raise RuntimeError("h5py lacks MPI support. Reinstall it with MPI support.")

if MPI.COMM_WORLD.size < 1:
    raise RuntimeError("MPI not initialized correctly.")