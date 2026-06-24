# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration test for compute_and_save_weights with multiple tasks."""
import pytest
from mpi4py import MPI

from bluerecording.compare import compare_weights
from bluerecording.weights import ComputeWeightsTask, compute_and_save_weights
from tests.conftest import EXAMPLE_RAT_S1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.mpi(ranks=2)
def test_compute_and_save_weights_multi_task(tmp_path):
    """compute_and_save_weights produces correct results for 2 tasks.

    Runs with the same electrode file twice (different output paths) and
    verifies both outputs match the single-task reference.
    """
    assert size == 2

    output_dir = comm.bcast(tmp_path, root=0)
    output_1 = str(output_dir / "weights_1.h5")
    output_2 = str(output_dir / "weights_2.h5")

    circuit_config = str(EXAMPLE_RAT_S1 / "circuit_config.json")
    csv = str(EXAMPLE_RAT_S1 / "electrodes.csv")

    tasks = [
        ComputeWeightsTask(electrodes=csv, output=output_1),
        ComputeWeightsTask(electrodes=csv, output=output_2),
    ]

    compute_and_save_weights(circuit_config, tasks)

    comm.Barrier()

    if rank == 0:
        ref = str(EXAMPLE_RAT_S1 / "reference" / "weights_ref.h5")
        for output_path in [output_1, output_2]:
            match, report = compare_weights(ref, output_path)
            assert match, f"{output_path}: {report}"
