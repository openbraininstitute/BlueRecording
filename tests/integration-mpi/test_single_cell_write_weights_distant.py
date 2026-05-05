# SPDX-License-Identifier: GPL-3.0-or-later
import h5py
import numpy as np
import pytest
from mpi4py import MPI

from bluerecording import positions
from bluerecording.circuit import init_circuit
from bluerecording.weights import initialize_h5_file, write_h5_file

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.skip_in_ci
@pytest.mark.mpi(ranks=2)
def test_single_cell_write_weights_distant_mpi(tmp_path):
    """Test write_weights for single_cell_l5_tpc with 2 MPI ranks (distant electrodes)."""
    assert size == 2

    path_to_simconfig = "examples/single_cell_l5_tpc/simulation_config_near.json"
    electrode_csv = "examples/single_cell_l5_tpc/distant_electrodes.csv"
    ref_path = "examples/single_cell_l5_tpc/reference/weights_distant_ref.h5"
    field_path = "examples/single_cell_l5_tpc/Infinite_VeryFar_HighRes.h5"

    output_dir = comm.bcast(tmp_path, root=0)
    output_path = str(output_dir / "weights.h5")

    node_manager, ids, cols, population, population_name, morphologies_dir = init_circuit(path_to_simconfig)
    positions_df, cols, _ = positions.get_positions(
        node_manager,
        ids,
        cols,
        population,
        morphologies_dir=morphologies_dir,
    )
    initialize_h5_file(cols, population_name, output_path, electrode_csv)
    write_h5_file(positions_df, cols, population_name, output_path, path_to_fields=[field_path, field_path])

    comm.Barrier()

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
