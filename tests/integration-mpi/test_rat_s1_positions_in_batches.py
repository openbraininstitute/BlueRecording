# SPDX-License-Identifier: GPL-3.0-or-later
"""Its own file, so that the cells are built in a fresh mpirun: NEURON can't be initialized twice in a process."""

import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI

from bluerecording import positions
from tests.conftest import EXAMPLE_RAT_S1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.mpi(ranks=2)
def test_rat_s1_positions_in_batches_mpi(tmp_path):
    """Positions built a few cells at a time on 2 MPI ranks: same cells per rank as one build, and the reference."""
    assert size == 2

    output_dir = comm.bcast(tmp_path, root=0)
    circuit_config = str(EXAMPLE_RAT_S1 / "circuit_config.json")

    positions_df, cols, _ = positions.compute_positions(circuit_config, cells_per_batch=4)
    np.testing.assert_array_equal(np.unique(cols[:, 0]), np.arange(10)[rank::size])
    positions.save_positions(positions_df, str(output_dir))

    comm.Barrier()

    if rank == 0:
        df_ref = pd.read_pickle(str(EXAMPLE_RAT_S1 / "reference" / "positions0_ref.pkl"))
        df_new = pd.concat([pd.read_pickle(str(output_dir / f"positions{r}.pkl")) for r in range(size)], axis=1)
        assert df_ref.shape == df_new.shape

        new_ids = df_new.columns.get_level_values("id")
        col_indices = [i for gid in df_ref.columns.get_level_values("id").unique() for i in np.where(new_ids == gid)[0]]
        reordered = df_new.iloc[:, col_indices]
        reordered.columns = df_ref.columns
        pd.testing.assert_frame_equal(df_ref, reordered, check_exact=False)
