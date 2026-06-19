# SPDX-License-Identifier: GPL-3.0-or-later
import h5py
import numpy as np
import pytest
from mpi4py import MPI

from bluerecording import positions
from bluerecording.circuit import init_circuit
from bluerecording.weights import Electrode, get_weights, save_weights
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
def test_rat_s1_write_weights_mpi(tmp_path):
    """Test write_weights for rat_s1_forelimb_l56_10cells with 2 MPI ranks."""
    assert size == 2

    output_dir = comm.bcast(tmp_path, root=0)
    output_path = str(output_dir / "weights.h5")

    circuit_config = str(EXAMPLE_RAT_S1 / "circuit_config.json")
    csv = str(EXAMPLE_RAT_S1 / "electrodes.csv")

    node_manager, ids, cols, population, population_name, morphologies_dir = init_circuit(circuit_config)
    pos_df, cols, _ = positions.get_positions(node_manager, ids, cols, population, morphologies_dir=morphologies_dir)
    electrodes = Electrode.from_csv(csv)
    weights = get_weights(pos_df, cols, electrodes=electrodes)
    save_weights(weights, cols, population_name, output_path, electrodes=electrodes)

    comm.Barrier()

    if rank == 0:
        ref = str(EXAMPLE_RAT_S1 / "reference" / "weights_ref.h5")
        with h5py.File(ref, "r") as r, h5py.File(output_path, "r") as n:
            ref_ids = r[f"{population_name}/node_ids"][:]
            new_ids = n[f"{population_name}/node_ids"][:]

            # Both files must contain the same set of node_ids (order may differ)
            np.testing.assert_array_equal(np.sort(ref_ids), np.sort(new_ids))

            # Compare per-node scaling_factors data (order-independent)
            for node_id in ref_ids:
                ref_sf = _get_node_scaling_factors(r, population_name, node_id)
                new_sf = _get_node_scaling_factors(n, population_name, node_id)
                np.testing.assert_allclose(
                    ref_sf, new_sf, rtol=1e-6, atol=1e-9,
                    err_msg=f"scaling_factors mismatch for node_id {node_id}",
                )
