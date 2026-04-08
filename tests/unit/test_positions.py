# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pandas as pd
import pytest

from bluerecording import positions
from bluerecording.circuit import init_circuit

from tests.helpers import (
    SOMA_POS, make_report_data,
    make_morphology, make_morphology_short, make_morphology_far_axon,
)


# ---------------------------------------------------------------------------
# Helper to extract section info from report data
# ---------------------------------------------------------------------------

def _get_sections(data, gid):
    cols = np.array(list(data.columns))
    return np.unique(cols[np.where(cols[:, 0] == gid), 1:].flatten())


# ---------------------------------------------------------------------------
# Morphology structure tests
# ---------------------------------------------------------------------------

def test_mutable_morph(tmp_path):
    morph = make_morphology(tmp_path / "morph.h5")
    m = positions.MutableMorph(morph)
    assert m.indices == [[0, 1, 2, 3], [4, 5], [6, 7, 8]]


def test_get_axon_points(tmp_path):
    morph = positions.MutableMorph(make_morphology(tmp_path / "morph.h5"))
    points, lengths = positions.get_axon_points(morph, SOMA_POS)
    np.testing.assert_almost_equal(lengths, [0, 1, 2, 3, 1073], decimal=2)
    np.testing.assert_almost_equal(points, [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 1073]], decimal=2)


def test_get_axon_points_extrapolate(tmp_path):
    morph = positions.MutableMorph(make_morphology_short(tmp_path / "morph.h5"))
    points, lengths = positions.get_axon_points(morph, SOMA_POS)
    np.testing.assert_almost_equal(lengths, [0, 1, 2, 3, 4, 1060], decimal=2)
    np.testing.assert_almost_equal(points, [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 1060]], decimal=2)


def test_get_new_idx():
    data = make_report_data()
    expected_columns = [
        [1]*23 + [2]*7,
        [0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 10, 10, 10, 10, 10, 10,
         0, 1, 1, 1, 1, 1, 1],
    ]
    expected_idx = list(zip(*expected_columns))
    expected_mi = pd.MultiIndex.from_tuples(expected_idx, names=['id', 'section'])
    result = positions.getNewIndex(data.columns)
    pd.testing.assert_index_equal(result, expected_mi)


def test_interpolate_dendrite(tmp_path):
    data = make_report_data()
    morph = positions.MutableMorph(make_morphology(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[3]
    num_compartments = np.shape(data[1][sec_name])[-1]
    sec_pts = np.array(morph.sections[sec_name - 1].points)
    seg_pos = positions.interp_points(sec_pts, num_compartments)
    np.testing.assert_almost_equal(seg_pos, [[0, 0, 0], [33.33, 0, 0], [66.66, 0, 0], [100, 0, 0]], decimal=2)


def test_interpolate_ais(tmp_path):
    data = make_report_data()
    morph = positions.MutableMorph(make_morphology(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[1]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions.get_axon_points(morph, SOMA_POS)
    seg_pos = positions.interp_points_axon(axon_pts, running_lens, sec_name, num_compartments, SOMA_POS)
    np.testing.assert_almost_equal(seg_pos, [[0, 0, 0], [0, 0, 6], [0, 0, 12], [0, 0, 18], [0, 0, 24], [0, 0, 30]], decimal=2)


def test_interpolate_ais_far_axon(tmp_path):
    """Edge case: only the soma is < 30 um from soma."""
    data = make_report_data()
    morph = positions.MutableMorph(make_morphology_far_axon(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[1]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions.get_axon_points(morph, SOMA_POS)
    seg_pos = positions.interp_points_axon(axon_pts, running_lens, sec_name, num_compartments, SOMA_POS)
    np.testing.assert_almost_equal(seg_pos, [[0, 0, 0], [0, 0, 6], [0, 0, 12], [0, 0, 18], [0, 0, 24], [0, 0, 30]], decimal=2)


def test_interpolate_ais_short(tmp_path):
    """No point > 30 um from soma."""
    data = make_report_data()
    morph = positions.MutableMorph(make_morphology_short(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[1]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions.get_axon_points(morph, SOMA_POS)
    seg_pos = positions.interp_points_axon(axon_pts, running_lens, sec_name, num_compartments, SOMA_POS)
    np.testing.assert_almost_equal(seg_pos, [[0, 0, 0], [0, 0, 6], [0, 0, 12], [0, 0, 18], [0, 0, 24], [0, 0, 30]], decimal=2)


def test_interpolate_ais_2(tmp_path):
    """No point between 30-60 um, but one farther than 60 um."""
    data = make_report_data()
    morph = positions.MutableMorph(make_morphology(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[2]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions.get_axon_points(morph, SOMA_POS)
    seg_pos = positions.interp_points_axon(axon_pts, running_lens, sec_name, num_compartments, SOMA_POS)
    np.testing.assert_almost_equal(seg_pos, [[0, 0, 30], [0, 0, 36], [0, 0, 42], [0, 0, 48], [0, 0, 54], [0, 0, 60]], decimal=2)


def test_interpolate_ais_2_short(tmp_path):
    """No points > 30 um from soma."""
    data = make_report_data()
    morph = positions.MutableMorph(make_morphology_short(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[2]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions.get_axon_points(morph, SOMA_POS)
    seg_pos = positions.interp_points_axon(axon_pts, running_lens, sec_name, num_compartments, SOMA_POS)
    np.testing.assert_almost_equal(seg_pos, [[0, 0, 30], [0, 0, 36], [0, 0, 42], [0, 0, 48], [0, 0, 54], [0, 0, 60]], decimal=2)


def test_interpolate_myelin(tmp_path):
    data = make_report_data()
    morph = positions.MutableMorph(make_morphology(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[-1]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions.get_axon_points(morph, SOMA_POS)
    seg_pos = positions.interp_points_axon(axon_pts, running_lens, sec_name, num_compartments, SOMA_POS)
    np.testing.assert_almost_equal(seg_pos, [[0, 0, 60], [0, 0, 260], [0, 0, 460], [0, 0, 660], [0, 0, 860], [0, 0, 1060]], decimal=2)


def test_interpolate_myelin_short(tmp_path):
    data = make_report_data()
    morph = positions.MutableMorph(make_morphology_short(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[-1]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions.get_axon_points(morph, SOMA_POS)
    seg_pos = positions.interp_points_axon(axon_pts, running_lens, sec_name, num_compartments, SOMA_POS)
    np.testing.assert_almost_equal(seg_pos, [[0, 0, 60], [0, 0, 260], [0, 0, 460], [0, 0, 660], [0, 0, 860], [0, 0, 1060]], decimal=2)


# ---------------------------------------------------------------------------
# Integration tests (require data)
# ---------------------------------------------------------------------------

@pytest.mark.skip_in_ci
def test_single_cell_get_positions(tmp_path):
    """Test get_positions for single_cell_l5_tpc."""
    simconfig = "examples/single_cell_l5_tpc/simulation_config_near.json"
    ref_path = "examples/single_cell_l5_tpc/reference/positions0_ref.pkl"

    nm, ids, cols, pop, _ = init_circuit(simconfig)
    pos_df, _ = positions.get_positions(nm, ids, cols, pop, path_to_simconfig=simconfig)
    positions.save_positions(pos_df, tmp_path)

    df_ref = pd.read_pickle(ref_path)
    df_new = pd.read_pickle(str(tmp_path / "positions0.pkl"))

    assert df_ref.index.equals(df_new.index)
    assert df_ref.columns.equals(df_new.columns)
    pd.testing.assert_frame_equal(df_ref, df_new, check_exact=False)


@pytest.mark.skip_in_ci
@pytest.mark.slow
def test_circuit_get_positions(tmp_path):
    """Test get_positions for sscx_100_cells."""
    simconfig = "examples/sscx_100_cells/simulation_config.json"
    ref_path = "examples/sscx_100_cells/reference/positions0_ref.pkl"

    nm, ids, cols, pop, _ = init_circuit(simconfig)
    pos_df, _ = positions.get_positions(nm, ids, cols, pop, path_to_simconfig=simconfig)
    positions.save_positions(pos_df, tmp_path)

    df_ref = pd.read_pickle(ref_path)
    df_new = pd.read_pickle(str(tmp_path / "positions0.pkl"))

    assert df_ref.index.equals(df_new.index)
    assert df_ref.columns.equals(df_new.columns)
    pd.testing.assert_frame_equal(df_ref, df_new, check_exact=False, rtol=5e-4, atol=0.1)
