# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration test for compute_and_save_weights with multiple tasks."""
import h5py
import numpy as np
import pytest
from mpi4py import MPI

from bluerecording import positions
from bluerecording.circuit import init_circuit
from bluerecording.weights import (
    ComputeWeightsTask,
    Electrode,
    compute_and_save_weights,
    get_weights,
    save_weights,
)
from tests.conftest import EXAMPLE_RAT_S1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def _get_node_scaling_factors(h5, population_name, node_id):
    """Extract scaling_factors rows for a given node_id using offsets."""
    node_ids = h5[f"{population_name}/node_ids"][:]
    offsets = h5[f"{population_name}/offsets"][:]
    dset = f"electrodes/{population_name}/scaling_factors"

    idx = np.where(node_ids == node_id)[0]
    assert len(idx) == 1, f"node_id {node_id} not found or duplicated"
    i = idx[0]
    start = offsets[i]
    end = offsets[i + 1] if i + 1 < len(offsets) else h5[dset].shape[0]
    return h5[dset][start:end, :]


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
        # Both outputs should exist and contain the same data
        ref = str(EXAMPLE_RAT_S1 / "reference" / "weights_ref.h5")
        with h5py.File(ref, "r") as r:
            ref_pop = [k for k in r.keys() if k != "electrodes"][0]
            ref_ids = r[f"{ref_pop}/node_ids"][:]

        for output_path in [output_1, output_2]:
            with h5py.File(ref, "r") as r, h5py.File(output_path, "r") as n:
                new_ids = n[f"{ref_pop}/node_ids"][:]
                np.testing.assert_array_equal(np.sort(ref_ids), np.sort(new_ids))

                for node_id in ref_ids:
                    ref_sf = _get_node_scaling_factors(r, ref_pop, node_id)
                    new_sf = _get_node_scaling_factors(n, ref_pop, node_id)
                    np.testing.assert_allclose(
                        ref_sf,
                        new_sf,
                        rtol=1e-6,
                        atol=1e-9,
                        err_msg=f"scaling_factors mismatch for node_id {node_id} in {output_path}",
                    )
