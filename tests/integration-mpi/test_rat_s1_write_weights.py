# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
from mpi4py import MPI

from bluerecording import positions
from bluerecording.circuit import init_circuit
from bluerecording.compare import compare_weights
from bluerecording.weights import Electrode, get_weights, save_weights
from tests.conftest import EXAMPLE_RAT_S1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.mpi(ranks=2)
def test_rat_s1_write_weights_mpi(tmp_path):
    """Test write_weights for rat_s1_forelimb_l56_10cells with 2 MPI ranks."""
    assert size == 2

    output_dir = comm.bcast(tmp_path, root=0)
    output_path = str(output_dir / "weights.h5")

    circuit_config = str(EXAMPLE_RAT_S1 / "circuit_config.json")
    csv = str(EXAMPLE_RAT_S1 / "electrodes.csv")

    node_manager, ids, cols, population, population_name, morphologies_dir, gid_offset = init_circuit(circuit_config)
    pos_df, cols, _ = positions.get_positions(node_manager, ids, cols, population, morphologies_dir=morphologies_dir)
    electrodes = Electrode.from_csv(csv)
    weights = get_weights(pos_df, cols, electrodes=electrodes)
    save_weights(weights, cols, population_name, output_path, electrodes=electrodes, gid_offset=gid_offset)

    comm.Barrier()

    if rank == 0:
        ref = str(EXAMPLE_RAT_S1 / "reference" / "weights_ref.h5")
        match, report = compare_weights(ref, output_path)
        assert match, report
