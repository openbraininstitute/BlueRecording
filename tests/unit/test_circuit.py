# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pandas as pd
import pytest

from bluerecording.circuit import _round_robin_batches
from bluerecording.positions import _get_new_index, _merge_batches


@pytest.mark.parametrize(
    ("n_cells", "cells_per_batch", "n_ranks"),
    [(10, 4, 4), (10, 3, 1), (23, 8, 3), (5, 1, 2), (5, None, 2), (5, 50, 2)],
)
def test_round_robin_batches(n_cells, cells_per_batch, n_ranks):
    """Batches stay within cells_per_batch (or one round) and give each rank the cells of a single build."""
    gids = np.arange(100, 100 + n_cells, dtype=np.uint32)
    batches = _round_robin_batches(gids, cells_per_batch, n_ranks)

    assert all(len(batch) <= max(cells_per_batch or n_cells, n_ranks) for batch in batches)
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), gids)
    for rank in range(n_ranks):
        # neurodamus deals each batch's cells to the ranks round-robin
        built = np.sort(np.concatenate([batch[rank::n_ranks] for batch in batches]))
        np.testing.assert_array_equal(built, gids[rank::n_ranks])


def test_merge_batches():
    """Positions of interleaved batches are merged back into node ID order."""
    rows = [(gid, sec) for gid in range(5) for sec in (0, 1, 1, 2)]
    cols = np.array(rows, dtype=np.uint64)
    index = _get_new_index(cols)
    positions_df = pd.DataFrame(np.arange(3 * len(index), dtype=float).reshape(3, -1), columns=index)
    neurite_types = np.arange(len(cols), dtype=np.int32)

    def batch(gids):
        in_batch = np.isin(cols[:, 0], gids)
        points = np.isin(positions_df.columns.get_level_values(0), gids)
        return positions_df.loc[:, points], cols[in_batch], neurite_types[in_batch]

    empty = batch([])
    merged_df, merged_cols, merged_types = _merge_batches([batch([0, 2, 4]), empty, batch([1, 3])])
    pd.testing.assert_frame_equal(merged_df, positions_df)
    np.testing.assert_array_equal(merged_cols, cols)
    np.testing.assert_array_equal(merged_types, neurite_types)

    assert _merge_batches([empty])[1].shape == (0, 2)
