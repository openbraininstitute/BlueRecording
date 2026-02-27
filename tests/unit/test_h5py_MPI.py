from mpi4py import MPI
import pytest
import h5py
import numpy as np
from pathlib import Path

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@pytest.mark.mpi(ranks=2)
def test_mpi_ranks():
    assert size == 2


@pytest.mark.mpi(ranks=2)
def test_h5py_mpio(tmp_path):
    assert size == 2

    # Only rank 0 determines the file path
    if rank == 0:
        file_path = tmp_path / "parallel.h5"
    else:
        file_path = None

    # Broadcast path so all ranks use IDENTICAL path
    file_path = comm.bcast(file_path, root=0)

    n_elems_per_rank = 10
    n_elems = size * n_elems_per_rank
    local_data = np.arange(rank * 10, (rank + 1) * 10, dtype=np.int32)

    # Rank 0 creates dataset
    if rank == 0:
        with h5py.File(str(file_path), "w") as f:
            f.create_dataset(
                "data",
                shape=(n_elems,),
                dtype="i",
                data=np.full(n_elems, -1, dtype=np.int32),
            )

    comm.Barrier()

    # Collective open for parallel write
    with h5py.File(str(file_path), "r+", driver="mpio", comm=comm) as f:
        f["data"][rank*n_elems_per_rank:(rank+1)*n_elems_per_rank] = local_data

    comm.Barrier()

    # Collective read
    with h5py.File(str(file_path), "r", driver="mpio", comm=comm) as f:
        expected = np.arange(n_elems, dtype=np.int32)
        data = f["data"][:]
        assert np.array_equal(data, expected)