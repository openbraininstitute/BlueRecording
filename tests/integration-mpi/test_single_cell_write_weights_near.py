# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
from mpi4py import MPI

from bluerecording import positions
from bluerecording.circuit import init_circuit
from bluerecording.compare import compare_weights
from bluerecording.weights import Electrode, get_weights, save_weights

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.skip_in_ci
@pytest.mark.mpi(ranks=2)
def test_single_cell_write_weights_near_mpi(tmp_path):
    """Test write_weights for single_cell_l5_tpc with 2 MPI ranks (near electrodes)."""
    assert size == 2

    path_to_simconfig = "examples/single_cell_l5_tpc/simulation_config_near.json"
    electrode_csv = "examples/single_cell_l5_tpc/near_electrodes.csv"
    ref_path = "examples/single_cell_l5_tpc/reference/weights_near_ref.h5"
    field_path = "examples/single_cell_l5_tpc/Infinite_Close_HighRes_SmallSphere.h5"

    output_dir = comm.bcast(tmp_path, root=0)
    output_path = str(output_dir / "weights.h5")

    node_manager, ids, cols, population, population_name, morphologies_dir = init_circuit(path_to_simconfig)
    pos_df, cols, _ = positions.get_positions(node_manager, ids, cols, population, morphologies_dir=morphologies_dir)
    electrodes = Electrode.from_csv(electrode_csv)
    weights = get_weights(pos_df, cols, electrodes=electrodes, path_to_fields=[field_path, field_path])
    save_weights(weights, cols, population_name, output_path, electrodes=electrodes)

    comm.Barrier()

    if rank == 0:
        match, report = compare_weights(ref_path, output_path)
        assert match, report
