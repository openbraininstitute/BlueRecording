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
def test_sscx_100_cells_write_weights_mpi(tmp_path):
    """Test that write_weights with 2 MPI ranks produces the same result as the reference."""
    assert size == 2

    path_to_simconfig = "examples/sscx_100_cells/simulation_config.json"
    electrode_file = "examples/sscx_100_cells/electrodes.json"
    ref_path = "examples/sscx_100_cells/reference/weights_ref.h5"

    output_dir = comm.bcast(tmp_path, root=0)
    output_path = str(output_dir / "weights.h5")

    cells, cols, population, population_name, morphologies_dir = init_circuit(path_to_simconfig)
    pos_df, cols, _ = positions.get_positions(cells, cols, population, morphologies_dir=morphologies_dir)
    electrodes = Electrode.from_json(electrode_file)
    weights = get_weights(pos_df, cols, electrodes=electrodes)
    save_weights(weights, cols, population_name, output_path, electrodes=electrodes)

    comm.Barrier()

    if rank == 0:
        match, report = compare_weights(ref_path, output_path)
        assert match, report
