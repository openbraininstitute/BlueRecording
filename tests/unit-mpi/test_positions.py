# SPDX-License-Identifier: GPL-3.0-or-later
from mpi4py import MPI
import numpy as np
import pandas as pd
import pytest

from bluerecording import positions
from bluerecording.circuit import init_circuit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.skip_in_ci
@pytest.mark.mpi(ranks=2)
def test_circuit_get_positions_mpi(tmp_path):
    """Test that get_positions produces correct results when split across 2 MPI ranks.

    Each rank writes its own positions{rank}.pkl.  We concatenate them and compare
    against the single-rank reference file.
    """
    assert size == 2

    # Broadcast tmp_path from rank 0 so all ranks write to the same directory
    output_dir = comm.bcast(tmp_path, root=0)

    path_to_simconfig = "examples/sscx_100_cells/simulation_config.json"

    node_manager, ids, cols, population, _, morphologies_dir = init_circuit(path_to_simconfig)
    positions_df, _, _ = positions.get_positions(
        node_manager, ids, cols, population,
        morphologies_dir=morphologies_dir,
    )
    positions.save_positions(positions_df, str(output_dir))

    comm.Barrier()

    # Only rank 0 does the comparison
    if rank == 0:
        ref_path = "examples/sscx_100_cells/reference/positions0_ref.pkl"
        df_ref = pd.read_pickle(ref_path)

        # Concatenate outputs from both ranks
        dfs = []
        for r in range(size):
            pkl_path = output_dir / f"positions{r}.pkl"
            dfs.append(pd.read_pickle(str(pkl_path)))
        df_new = pd.concat(dfs, axis=1)

        assert df_ref.shape == df_new.shape, (
            f"Shape mismatch: ref {df_ref.shape} vs new {df_new.shape}"
        )

        # Reorder df_new so that gid groups appear in the same order as the
        # reference.  Within each gid the column order is already correct
        # (sections in simulation order), so we only need to reorder gid blocks.
        ref_gid_order = df_ref.columns.get_level_values("id").unique()
        reordered = pd.concat([df_new.xs(gid, level="id", axis=1, drop_level=False)
                               for gid in ref_gid_order], axis=1)

        pd.testing.assert_frame_equal(
            df_ref,
            reordered,
            check_exact=False,
            rtol=5e-4,
            atol=0.1,
        )
