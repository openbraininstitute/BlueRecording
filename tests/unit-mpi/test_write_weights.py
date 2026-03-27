# SPDX-License-Identifier: GPL-3.0-or-later
from mpi4py import MPI
import numpy as np
import h5py
import pytest

from bluerecording.circuit import init_circuit
from bluerecording import positions
from bluerecording.writeH5 import writeH5File
from bluerecording.writeH5_prelim import initializeH5File

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.skip_in_ci
@pytest.mark.mpi(ranks=2)
def test_circuit_write_weights_mpi(tmp_path):
    """Test that write_weights with 2 MPI ranks produces the same result as the reference."""
    assert size == 2

    path_to_simconfig = "examples/circuitTest/data/simulation_config.json"
    electrode_csv = "examples/circuitTest/test/electrodeFile/electrodes.csv"
    ref_path = "examples/circuitTest/weights_ref.h5"

    # Broadcast tmp_path from rank 0
    output_dir = comm.bcast(tmp_path, root=0)
    output_path = str(output_dir / "weights.h5")

    node_manager, ids, cols, population, population_name = init_circuit(path_to_simconfig)
    positions_df, cols = positions.get_positions(
        node_manager, ids, cols, population,
        path_to_simconfig=path_to_simconfig,
    )
    initializeH5File(cols, population_name, output_path, electrode_csv)
    writeH5File(positions_df, cols, population_name, output_path)

    comm.Barrier()

    # Only rank 0 does the comparison
    if rank == 0:
        with h5py.File(ref_path, "r") as ref, h5py.File(output_path, "r") as new:
            ref_ids = ref[population_name + "/node_ids"][:]
            new_ids = new[population_name + "/node_ids"][:]
            np.testing.assert_array_equal(ref_ids, new_ids)

            ref_offsets = ref[population_name + "/offsets"][:]
            new_offsets = new[population_name + "/offsets"][:]
            np.testing.assert_array_equal(ref_offsets, new_offsets)

            dset = "electrodes/" + population_name + "/scaling_factors"
            ref_sf = ref[dset][:]
            new_sf = new[dset][:]
            np.testing.assert_allclose(ref_sf, new_sf, rtol=1e-6, atol=1e-9)
