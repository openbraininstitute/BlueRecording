# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

from bluerecording import positions
from bluerecording.circuit import init_circuit

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.skip_in_ci
@pytest.mark.mpi(ranks=2)
def test_single_cell_get_positions_mpi(tmp_path):
    """Test get_positions for single_cell_l5_tpc split across 2 MPI ranks."""
    assert size == 2

    output_dir = comm.bcast(tmp_path, root=0)

    path_to_simconfig = "examples/single_cell_l5_tpc/simulation_config_near.json"

    node_manager, ids, cols, population, _, morphologies_dir = init_circuit(path_to_simconfig)
    positions_df, _, _ = positions.get_positions(
        node_manager,
        ids,
        cols,
        population,
        morphologies_dir=morphologies_dir,
    )
    positions.save_positions(positions_df, str(output_dir))

    comm.Barrier()

    if rank == 0:
        ref_path = "examples/single_cell_l5_tpc/reference/positions0_ref.pkl"
        df_ref = pd.read_pickle(ref_path)

        dfs = []
        for r in range(size):
            pkl_path = output_dir / f"positions{r}.pkl"
            dfs.append(pd.read_pickle(str(pkl_path)))
        df_new = pd.concat(dfs, axis=1)

        assert df_ref.shape == df_new.shape, f"Shape mismatch: ref {df_ref.shape} vs new {df_new.shape}"

        ref_gid_order = df_ref.columns.get_level_values("id").unique()
        col_indices = []
        for gid in ref_gid_order:
            col_indices.extend(np.where(df_new.columns.get_level_values("id") == gid)[0])
        reordered = df_new.iloc[:, col_indices]
        reordered.columns = df_ref.columns

        pd.testing.assert_frame_equal(
            df_ref,
            reordered,
            check_exact=False,
            rtol=5e-4,
            atol=0.1,
        )
