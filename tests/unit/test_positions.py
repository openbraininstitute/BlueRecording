# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pandas as pd

from bluerecording import positions
from tests.helpers import (
    SOMA_POS,
    make_morphology,
    make_morphology_far_axon,
    make_morphology_short,
    make_morphology_two_axon_branches,
    make_report_data,
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


def test_positioned_morphology(tmp_path):
    morph = make_morphology(tmp_path / "morph.h5")
    m = positions._PositionedMorphology(morph)
    np.testing.assert_array_equal(m.offsets, [0, 4, 6, 9])
    assert m.num_sections == 3


def test_save_positions_creates_missing_directory(tmp_path):
    """save_positions creates parent directories if they don't exist."""
    output_dir = tmp_path / "nested" / "subdir"
    assert not output_dir.exists()
    df = pd.DataFrame({"a": [1, 2, 3]})
    positions.save_positions(df, output_dir)
    assert (output_dir / "positions0.pkl").exists()


def test_get_axon_points(tmp_path):
    morph = positions._PositionedMorphology(make_morphology(tmp_path / "morph.h5"))
    points, lengths = positions._get_axon_points(morph, SOMA_POS)
    np.testing.assert_almost_equal(lengths, [0, 1, 2, 3, 1073], decimal=2)
    np.testing.assert_almost_equal(points, [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 1073]], decimal=2)


def test_get_axon_points_extrapolate(tmp_path):
    morph = positions._PositionedMorphology(make_morphology_short(tmp_path / "morph.h5"))
    points, lengths = positions._get_axon_points(morph, SOMA_POS)
    np.testing.assert_almost_equal(lengths, [0, 1, 2, 3, 4, 1060], decimal=2)
    np.testing.assert_almost_equal(
        points, [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 1060]], decimal=2
    )


def test_get_axon_points_picks_longest_branch(tmp_path):
    """Regression: longest branch must be selected when extrapolation is needed.

    With two short axonal branches (both < 1060 µm), the algorithm must pick
    the longer one (branch A, 100 µm) for extrapolation, not the last one
    visited (branch B, 10 µm).
    """
    morph = positions._PositionedMorphology(make_morphology_two_axon_branches(tmp_path / "morph.h5"))
    points, lengths = positions._get_axon_points(morph, SOMA_POS)

    # Longest branch: soma(0,0,0) → root tip(0,0,5) → branch A tip(0,0,100)
    # then extrapolated to 1060 µm along z.
    # With the bug we'd get branch B: soma → root tip → (0,0,10) → extrapolated
    np.testing.assert_almost_equal(points, [[0, 0, 0], [0, 0, 5], [0, 0, 100], [0, 0, 1060]], decimal=2)
    np.testing.assert_almost_equal(lengths, [0, 5, 100, 1060], decimal=2)


def test_get_new_idx():
    data = make_report_data()
    expected_columns = [
        [1] * 23 + [2] * 7,
        [0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 10, 10, 10, 10, 10, 10, 0, 1, 1, 1, 1, 1, 1],
    ]
    expected_idx = list(zip(*expected_columns, strict=False))
    expected_mi = pd.MultiIndex.from_tuples(expected_idx, names=["id", "section"])
    result = positions._get_new_index(data.columns)
    pd.testing.assert_index_equal(result, expected_mi)


def test_interpolate_dendrite(tmp_path):
    data = make_report_data()
    morph = positions._PositionedMorphology(make_morphology(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[3]
    num_compartments = np.shape(data[1][sec_name])[-1]
    sec_pts = np.array(morph.sections[sec_name - 1].points)
    seg_pos = positions._interp_points(sec_pts, num_compartments)
    np.testing.assert_almost_equal(seg_pos, [[0, 0, 0], [33.33, 0, 0], [66.66, 0, 0], [100, 0, 0]], decimal=2)


def test_interpolate_ais(tmp_path):
    data = make_report_data()
    morph = positions._PositionedMorphology(make_morphology(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[1]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions._get_axon_points(morph, SOMA_POS)
    seg_pos = positions._interp_points_axon(axon_pts, running_lens, sec_name, num_compartments)
    np.testing.assert_almost_equal(
        seg_pos, [[0, 0, 0], [0, 0, 6], [0, 0, 12], [0, 0, 18], [0, 0, 24], [0, 0, 30]], decimal=2
    )


def test_interpolate_ais_far_axon(tmp_path):
    """Edge case: only the soma is < 30 um from soma."""
    data = make_report_data()
    morph = positions._PositionedMorphology(make_morphology_far_axon(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[1]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions._get_axon_points(morph, SOMA_POS)
    seg_pos = positions._interp_points_axon(axon_pts, running_lens, sec_name, num_compartments)
    np.testing.assert_almost_equal(
        seg_pos, [[0, 0, 0], [0, 0, 6], [0, 0, 12], [0, 0, 18], [0, 0, 24], [0, 0, 30]], decimal=2
    )


def test_interpolate_ais_short(tmp_path):
    """No point > 30 um from soma."""
    data = make_report_data()
    morph = positions._PositionedMorphology(make_morphology_short(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[1]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions._get_axon_points(morph, SOMA_POS)
    seg_pos = positions._interp_points_axon(axon_pts, running_lens, sec_name, num_compartments)
    np.testing.assert_almost_equal(
        seg_pos, [[0, 0, 0], [0, 0, 6], [0, 0, 12], [0, 0, 18], [0, 0, 24], [0, 0, 30]], decimal=2
    )


def test_interpolate_ais_2(tmp_path):
    """No point between 30-60 um, but one farther than 60 um."""
    data = make_report_data()
    morph = positions._PositionedMorphology(make_morphology(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[2]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions._get_axon_points(morph, SOMA_POS)
    seg_pos = positions._interp_points_axon(axon_pts, running_lens, sec_name, num_compartments)
    np.testing.assert_almost_equal(
        seg_pos, [[0, 0, 30], [0, 0, 36], [0, 0, 42], [0, 0, 48], [0, 0, 54], [0, 0, 60]], decimal=2
    )


def test_interpolate_ais_2_short(tmp_path):
    """No points > 30 um from soma."""
    data = make_report_data()
    morph = positions._PositionedMorphology(make_morphology_short(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[2]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions._get_axon_points(morph, SOMA_POS)
    seg_pos = positions._interp_points_axon(axon_pts, running_lens, sec_name, num_compartments)
    np.testing.assert_almost_equal(
        seg_pos, [[0, 0, 30], [0, 0, 36], [0, 0, 42], [0, 0, 48], [0, 0, 54], [0, 0, 60]], decimal=2
    )


def test_interpolate_myelin(tmp_path):
    data = make_report_data()
    morph = positions._PositionedMorphology(make_morphology(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[-1]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions._get_axon_points(morph, SOMA_POS)
    seg_pos = positions._interp_points_axon(axon_pts, running_lens, sec_name, num_compartments)
    np.testing.assert_almost_equal(
        seg_pos, [[0, 0, 60], [0, 0, 260], [0, 0, 460], [0, 0, 660], [0, 0, 860], [0, 0, 1060]], decimal=2
    )


def test_interpolate_myelin_short(tmp_path):
    data = make_report_data()
    morph = positions._PositionedMorphology(make_morphology_short(tmp_path / "morph.h5"))
    sections = _get_sections(data, 1)
    sec_name = sections[-1]
    num_compartments = np.shape(data[1][sec_name])[-1]
    axon_pts, running_lens = positions._get_axon_points(morph, SOMA_POS)
    seg_pos = positions._interp_points_axon(axon_pts, running_lens, sec_name, num_compartments)
    np.testing.assert_almost_equal(
        seg_pos, [[0, 0, 60], [0, 0, 260], [0, 0, 460], [0, 0, 660], [0, 0, 860], [0, 0, 1060]], decimal=2
    )
