# SPDX-License-Identifier: GPL-3.0-or-later
import pandas as pd
import pytest

from bluerecording import positions
from bluerecording.circuit import init_circuit


@pytest.mark.skip_in_ci
def test_single_cell_get_positions(tmp_path):
    """Test get_positions for single_cell_l5_tpc."""
    simconfig = "examples/single_cell_l5_tpc/simulation_config_near.json"
    ref_path = "examples/single_cell_l5_tpc/reference/positions0_ref.pkl"

    nm, ids, cols, pop, _, morphologies_dir = init_circuit(simconfig)
    pos_df, _, _ = positions.get_positions(nm, ids, cols, pop, morphologies_dir=morphologies_dir)
    positions.save_positions(pos_df, tmp_path)

    df_ref = pd.read_pickle(ref_path)
    df_new = pd.read_pickle(str(tmp_path / "positions0.pkl"))

    assert df_ref.index.equals(df_new.index)
    assert df_ref.columns.equals(df_new.columns)
    pd.testing.assert_frame_equal(df_ref, df_new, check_exact=False)


@pytest.mark.skip_in_ci
def test_sscx_100_cells_get_positions(tmp_path):
    """Test get_positions for sscx_100_cells."""
    simconfig = "examples/sscx_100_cells/simulation_config.json"
    ref_path = "examples/sscx_100_cells/reference/positions0_ref.pkl"

    nm, ids, cols, pop, _, morphologies_dir = init_circuit(simconfig)
    pos_df, _, _ = positions.get_positions(nm, ids, cols, pop, morphologies_dir=morphologies_dir)
    positions.save_positions(pos_df, tmp_path)

    df_ref = pd.read_pickle(ref_path)
    df_new = pd.read_pickle(str(tmp_path / "positions0.pkl"))

    assert df_ref.index.equals(df_new.index)
    assert df_ref.columns.equals(df_new.columns)
    pd.testing.assert_frame_equal(df_ref, df_new, check_exact=False)


def test_rat_s1_get_positions(tmp_path):
    """Test get_positions using a circuit_config.json (no simulation config).

    Exercises the resolve_simulation_config path that generates a temporary simulation
    config on the fly from a circuit config.
    """
    from tests.conftest import EXAMPLE_RAT_S1

    circuit_config = str(EXAMPLE_RAT_S1 / "circuit_config.json")
    ref_path = str(EXAMPLE_RAT_S1 / "reference" / "positions0_ref.pkl")

    nm, ids, cols, pop, _, morphologies_dir = init_circuit(circuit_config)
    pos_df, _, _ = positions.get_positions(
        nm,
        ids,
        cols,
        pop,
        morphologies_dir=morphologies_dir,
    )

    positions.save_positions(pos_df, tmp_path)

    df_ref = pd.read_pickle(ref_path)
    df_new = pd.read_pickle(str(tmp_path / "positions0.pkl"))

    assert df_ref.index.equals(df_new.index)
    assert df_ref.columns.equals(df_new.columns)
    pd.testing.assert_frame_equal(df_ref, df_new, check_exact=False)
