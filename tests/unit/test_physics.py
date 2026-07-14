# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pandas as pd

from bluerecording.physics import (
    SegmentGeometry,
    _distances_in_planar_coords,
    _get_array_spacing,
    _get_thickness,
    _line_source_coeffs,
    get_coeffs_dipole_reciprocity,
    get_coeffs_line_source,
    get_coeffs_objective_csd_disk,
    get_coeffs_objective_csd_plane,
    get_coeffs_objective_csd_sphere,
    get_coeffs_point_source,
    get_coeffs_reciprocity,
)
from bluerecording.weights import _get_segment_midpts
from tests.helpers import (
    GIDS,
    create_e_field,
    create_potential_field,
    make_two_section_data,
    make_two_section_positions,
)


def _scalar_line_coeff(start_pos, end_pos, electrode_pos, sigma):
    """Compute a single line-source coefficient via _line_source_coeffs (test helper)."""
    start = np.array(start_pos, dtype=np.float64).reshape(1, 3)
    end = np.array(end_pos, dtype=np.float64).reshape(1, 3)
    epos = np.array(electrode_pos, dtype=np.float64).reshape(1, 3)
    seg_lengths = np.linalg.norm(end - start, axis=1)
    sigma_arr = np.array([sigma])
    return _line_source_coeffs(start, end, seg_lengths, epos, sigma_arr).item()


# ---------------------------------------------------------------------------
# Line-source coefficient tests
# ---------------------------------------------------------------------------


def test_get_coeffs_line_source():
    positions = make_two_section_positions()
    data = make_two_section_data()
    electrode_pos = np.array([10, 10, 10])
    sigma = 1

    geom = SegmentGeometry.from_positions(positions)
    coeffs = get_coeffs_line_source(geom, data.columns, electrode_pos, sigma)

    soma_dist = np.sqrt(3 * 10**2)  # µm
    expected_soma = 1e-3 / (4 * np.pi * sigma * soma_dist)
    expected_line = _scalar_line_coeff(np.array([0, 0, 0]), np.array([0, 0, 1]), electrode_pos, sigma)
    expected = pd.DataFrame(data=np.hstack((expected_soma, expected_line))[np.newaxis, :], columns=data.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def test_line_source():
    seg = [np.array([0, 0, 0]), np.array([1, 0, 0])]
    epos = np.array([2, 0, 1])
    sigma = 1
    ds = 1  # µm
    h, r, l = 1, 1, 2  # µm
    expected = 1e-3 / (4 * np.pi * sigma * ds) * np.log(np.abs((np.sqrt(h**2 + r**2) - h) / (np.sqrt(l**2 + r**2) - l)))
    result = _scalar_line_coeff(seg[0], seg[1], epos, sigma)
    np.testing.assert_almost_equal(result, expected)


def test_line_source_2():
    seg = [np.array([0, 0, 0]), np.array([1, 0, 0])]
    epos = np.array([-2, 0, 1])
    sigma = 1
    ds = 1  # µm
    h, r, l = -3, 1, -2  # µm
    expected = 1e-3 / (4 * np.pi * sigma * ds) * np.log(np.abs((np.sqrt(h**2 + r**2) - h) / (np.sqrt(l**2 + r**2) - l)))
    result = _scalar_line_coeff(seg[0], seg[1], epos, sigma)
    np.testing.assert_almost_equal(result, expected)


def test_line_source_3():
    seg = [np.array([0, 0, 0]), np.array([1, 0, 0])]
    epos = np.array([0.5, 0, 1])
    sigma = 1
    ds = 1  # µm
    h, r, l = -0.5, 1, 0.5  # µm
    expected = 1e-3 / (4 * np.pi * sigma * ds) * np.log(np.abs((np.sqrt(h**2 + r**2) - h) / (np.sqrt(l**2 + r**2) - l)))
    result = _scalar_line_coeff(seg[0], seg[1], epos, sigma)
    np.testing.assert_almost_equal(result, expected)


def test_coefficients_are_in_V_per_nA():
    """Verify output units are V/nA as required by the SONATA electrodes spec.

    For a point source at distance r (µm) with sigma (S/m):
        coeff = 1e-3 / (4π σ r)  [V/nA]

    With σ=1 S/m and r=1 µm: coeff = 1e-3/(4π) ≈ 7.96e-5 V/nA.
    """
    positions = pd.DataFrame(
        data=np.array([[0.0, 0.0, 0.0]]).T,
        columns=pd.MultiIndex.from_tuples([(1, 0)], names=["id", "section"]),
    )
    electrode_pos = np.array([1.0, 0.0, 0.0])  # 1 µm away
    sigma = 1.0  # S/m

    result = get_coeffs_point_source(positions, electrode_pos, sigma)
    expected_V_per_nA = 1e-3 / (4 * np.pi * 1.0 * 1.0)  # ≈ 7.96e-5

    np.testing.assert_allclose(result.values.item(), expected_V_per_nA, rtol=1e-10)


# ---------------------------------------------------------------------------
# Point-source coefficient tests
# ---------------------------------------------------------------------------


def test_get_coeffs_point_source():
    positions = make_two_section_positions()
    electrode_pos = np.array([10, 10, 10])
    sigma = 1
    midpts = _get_segment_midpts(positions, GIDS)
    coeffs = get_coeffs_point_source(midpts, electrode_pos, sigma)

    soma_dist = np.sqrt(3 * 10**2)  # µm
    expected_soma = 1e-3 / (4 * np.pi * sigma * soma_dist)
    seg_dist = np.sqrt(10**2 + 10**2 + (10 - 0.5) ** 2)  # µm
    expected_seg = 1e-3 / (4 * np.pi * sigma * seg_dist)
    expected = pd.DataFrame(data=np.hstack((expected_soma, expected_seg))[np.newaxis, :], columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def test_point_source_multi_electrode_varying_sigma():
    """Verify multi-electrode point source with per-electrode sigma matches manual computation."""
    rng = np.random.default_rng(99)

    # 3 segments at known positions
    columns = [[1, 1, 2], [0, 1, 0]]
    idx = list(zip(*columns, strict=False))
    mi = pd.MultiIndex.from_tuples(idx, names=["id", "section"])
    segment_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 3.0, 1.0],
            [10.0, -2.0, 4.0],
        ]
    ).T
    midpts = pd.DataFrame(data=segment_positions, columns=mi)

    # 4 electrodes at random positions, each with a different sigma
    electrode_positions = rng.uniform(-50, 50, size=(4, 3))
    sigmas = np.array([0.2, 0.3, 0.5, 1.0])

    result = get_coeffs_point_source(midpts, electrode_positions, sigmas)

    # Manually compute expected values
    positions_um = midpts.values.T  # (3, 3)
    for i in range(4):
        for j in range(3):
            dist = np.linalg.norm(positions_um[j] - electrode_positions[i])  # µm
            expected = 1e-3 / (4 * np.pi * sigmas[i] * dist)
            np.testing.assert_allclose(
                result.iloc[i, j],
                expected,
                rtol=1e-12,
                err_msg=f"Electrode {i}, segment {j}: sigma={sigmas[i]}",
            )

    # Also verify single-electrode calls match multi-electrode rows
    for i in range(4):
        single = get_coeffs_point_source(midpts, electrode_positions[i], sigmas[i])
        np.testing.assert_allclose(
            result.iloc[i].values,
            single.values.flatten(),
            rtol=1e-12,
            err_msg=f"Single vs multi mismatch for electrode {i}",
        )


# ---------------------------------------------------------------------------
# Reciprocity coefficient tests
# ---------------------------------------------------------------------------


def test_get_coeffs_reciprocity(tmp_path):
    positions = make_two_section_positions()
    field_path = create_potential_field(tmp_path / "potential.h5")
    midpts = _get_segment_midpts(positions, GIDS)
    potentials = get_coeffs_reciprocity(midpts, field_path)

    columns = [[1, 1], [0, 1]]
    mi = pd.MultiIndex.from_tuples(list(zip(*columns, strict=False)), names=["id", "section"])
    expected = pd.DataFrame(data=np.array([0, 0.5e-6])[np.newaxis, :], columns=mi)
    pd.testing.assert_frame_equal(potentials, expected)


def test_get_coeffs_dipole_reciprocity(tmp_path):
    positions = make_two_section_positions()
    field_path = create_e_field(tmp_path / "efield.h5")
    midpts = _get_segment_midpts(positions, GIDS)
    potentials = get_coeffs_dipole_reciprocity(midpts, field_path)

    columns = [[1, 1], [0, 1]]
    mi = pd.MultiIndex.from_tuples(list(zip(*columns, strict=False)), names=["id", "section"])
    expected = pd.DataFrame(data=-1 * np.array([0.5e-6, 0])[np.newaxis, :] ** 2, columns=mi)
    pd.testing.assert_frame_equal(potentials, expected)


# ---------------------------------------------------------------------------
# Objective CSD tests
# ---------------------------------------------------------------------------


def test_objective_csd_sphere():
    positions = make_two_section_positions()
    all_epos = np.array([[0, 0, 0], [2, 0, 0]])
    midpts = _get_segment_midpts(positions, GIDS)

    coeffs = get_coeffs_objective_csd_sphere(midpts, all_epos[0], all_epos)
    expected = pd.DataFrame(data=np.array([[1, 1]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = get_coeffs_objective_csd_sphere(midpts, all_epos[0], all_epos, radius=0.1)
    expected = pd.DataFrame(data=np.array([[1, 0]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def test_objective_csd_disk():
    positions = make_two_section_positions()
    all_epos = np.array([[0, 0, 0], [1, 0, 0]])
    midpts = _get_segment_midpts(positions, GIDS)

    coeffs = get_coeffs_objective_csd_disk(midpts, all_epos[0], all_epos)
    expected = pd.DataFrame(data=np.array([[1, 1]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = get_coeffs_objective_csd_disk(midpts, all_epos[1], all_epos)
    expected = pd.DataFrame(data=np.array([[0, 0]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = get_coeffs_objective_csd_disk(midpts, all_epos[0], all_epos, radius=0.1)
    expected = pd.DataFrame(data=np.array([[1, 0]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = get_coeffs_objective_csd_disk(midpts, all_epos[0], all_epos, diskThickness=10)
    expected = pd.DataFrame(data=np.array([[1, 1]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def test_objective_csd_plane():
    positions = make_two_section_positions()
    all_epos = np.array([[0, 0, 0], [1, 0, 0]])
    midpts = _get_segment_midpts(positions, GIDS)

    coeffs = get_coeffs_objective_csd_plane(midpts, all_epos[0], all_epos)
    expected = pd.DataFrame(data=np.array([[1, 1]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = get_coeffs_objective_csd_plane(midpts, all_epos[1], all_epos)
    expected = pd.DataFrame(data=np.array([[0, 0]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


# ---------------------------------------------------------------------------
# Array geometry helpers
# ---------------------------------------------------------------------------


def test_array_spacing():
    all_epos = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    main_axis, spacing = _get_array_spacing(all_epos)
    np.testing.assert_equal(main_axis, np.array([0, 0, 1])[:, np.newaxis])
    np.testing.assert_equal(spacing, np.array([1, 1]))


def test_array_thickness():
    assert _get_thickness(np.array([1, 1])) == 0.5


def test_planar_coords():
    positions = make_two_section_positions()
    all_epos = np.array([[0, 0, 0], [1, 0, 0]])
    midpts = _get_segment_midpts(positions, GIDS)
    main_axis, _ = _get_array_spacing(all_epos)

    axial, radial = _distances_in_planar_coords(midpts, all_epos[0], main_axis)
    np.testing.assert_equal(axial, np.array([0, 0])[:, np.newaxis])
    np.testing.assert_equal(radial, np.array([0, 0.5]))

    axial, radial = _distances_in_planar_coords(midpts, all_epos[1], main_axis)
    np.testing.assert_equal(axial, np.array([1, 1])[:, np.newaxis])
    np.testing.assert_equal(radial, np.array([0, 0.5]))


# ---------------------------------------------------------------------------
# Precompute segment geometry tests
# ---------------------------------------------------------------------------


def test_precompute_segment_geometry_basic():
    """Test _precompute_segment_geometry with soma + one line-source segment."""
    positions = make_two_section_positions()

    result = SegmentGeometry.from_positions(positions)

    # Should identify 1 soma and 1 line-source segment
    assert result.is_soma.shape == (2,)
    assert result.is_soma[0] is np.True_  # first segment is soma
    assert result.is_soma[1] is np.False_  # second is line-source

    # Soma position
    assert result.soma_positions.shape == (1, 3)
    np.testing.assert_array_equal(result.soma_positions[0], [0.0, 0.0, 0.0])

    # Line-source segment: start=(0,0,0), end=(0,0,1) in µm
    assert result.start_pos.shape == (1, 3)
    assert result.end_pos.shape == (1, 3)
    np.testing.assert_array_equal(result.start_pos[0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.end_pos[0], [0.0, 0.0, 1.0])

    # Length should be 1 µm
    assert result.seg_lengths.shape == (1,)
    np.testing.assert_almost_equal(result.seg_lengths[0], 1.0)

    # Direction should be along z-axis
    assert result.seg_dirs.shape == (1, 3)
    np.testing.assert_array_almost_equal(result.seg_dirs[0], [0.0, 0.0, 1.0])


def test_precompute_segment_geometry_multi_neuron():
    """Test _precompute_segment_geometry with multiple neurons and sections."""
    columns = [
        [1, 1, 1, 1, 2, 2, 2],
        [0, 1, 1, 1, 0, 2, 2],
    ]
    idx = list(zip(*columns, strict=False))
    mi = pd.MultiIndex.from_tuples(idx, names=["id", "section"])
    values = np.array(
        [
            [0.0, 0.0, 0.0],  # neuron 1 soma
            [1.0, 0.0, 0.0],  # section 1 start
            [2.0, 0.0, 0.0],  # section 1 mid (end of seg1, start of seg2)
            [3.0, 0.0, 0.0],  # section 1 end
            [10.0, 0.0, 0.0],  # neuron 2 soma
            [10.0, 1.0, 0.0],  # section 2 start
            [10.0, 2.0, 0.0],  # section 2 end
        ]
    ).T
    positions = pd.DataFrame(data=values, columns=mi)

    result = SegmentGeometry.from_positions(positions)

    # Expected: 2 somas, 3 line-source segments
    assert np.sum(result.is_soma) == 2
    assert np.sum(~result.is_soma) == 3

    assert result.soma_positions.shape == (2, 3)
    np.testing.assert_array_equal(result.soma_positions[0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.soma_positions[1], [10.0, 0.0, 0.0])

    assert result.start_pos.shape == (3, 3)
    assert result.end_pos.shape == (3, 3)

    # All segments are 1 µm long
    np.testing.assert_array_almost_equal(result.seg_lengths, [1.0, 1.0, 1.0])


def test_precompute_segment_geometry_no_soma():
    """Test with no soma segments (all line-source)."""
    columns = [[1, 1], [1, 1]]
    idx = list(zip(*columns, strict=False))
    mi = pd.MultiIndex.from_tuples(idx, names=["id", "section"])
    values = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).T
    positions = pd.DataFrame(data=values, columns=mi)

    result = SegmentGeometry.from_positions(positions)

    assert np.sum(result.is_soma) == 0
    assert result.soma_positions.shape == (0, 3)
    assert result.start_pos.shape == (1, 3)
    assert result.end_pos.shape == (1, 3)
    np.testing.assert_almost_equal(result.seg_lengths[0], 1.0)


def test_precompute_segment_geometry_only_soma():
    """Test with only soma segments (no line-source)."""
    columns = [[1], [0]]
    idx = list(zip(*columns, strict=False))
    mi = pd.MultiIndex.from_tuples(idx, names=["id", "section"])
    values = np.array([[5.0, 3.0, 1.0]]).T
    positions = pd.DataFrame(data=values, columns=mi)

    result = SegmentGeometry.from_positions(positions)

    assert np.sum(result.is_soma) == 1
    assert np.sum(~result.is_soma) == 0
    assert result.soma_positions.shape == (1, 3)
    np.testing.assert_array_equal(result.soma_positions[0], [5.0, 3.0, 1.0])
    assert result.start_pos.shape == (0, 3)
    assert result.seg_lengths.shape == (0,)


# ---------------------------------------------------------------------------
# Vectorized vs scalar consistency
# ---------------------------------------------------------------------------


def test_vectorized_matches_scalar():
    """Verify vectorized implementation matches scalar for non-trivial input."""
    rng = np.random.default_rng(42)

    # Build positions DataFrame with ~10 segments across 2-3 neurons:
    # neuron 1 (gid=1): soma + section 1 with 4 boundary points (3 segments)
    # neuron 2 (gid=2): soma + section 1 with 3 boundary points (2 segments)
    #                         + section 2 with 4 boundary points (3 segments)
    # Total: 2 soma + 8 line-source segments = 10 output coefficients

    # Generate random positions (deterministic seed)
    n1_soma = rng.uniform(-50, 50, size=3)
    n1_s1_p0 = n1_soma + rng.uniform(5, 20, size=3)
    n1_s1_p1 = n1_s1_p0 + rng.uniform(5, 20, size=3)
    n1_s1_p2 = n1_s1_p1 + rng.uniform(5, 20, size=3)
    n1_s1_p3 = n1_s1_p2 + rng.uniform(5, 20, size=3)

    n2_soma = rng.uniform(-50, 50, size=3)
    n2_s1_p0 = n2_soma + rng.uniform(5, 20, size=3)
    n2_s1_p1 = n2_s1_p0 + rng.uniform(5, 20, size=3)
    n2_s1_p2 = n2_s1_p1 + rng.uniform(5, 20, size=3)
    n2_s2_p0 = n2_soma + rng.uniform(-20, -5, size=3)
    n2_s2_p1 = n2_s2_p0 + rng.uniform(5, 20, size=3)
    n2_s2_p2 = n2_s2_p1 + rng.uniform(5, 20, size=3)
    n2_s2_p3 = n2_s2_p2 + rng.uniform(5, 20, size=3)

    columns_tuples = [
        (1, 0),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),
    ]
    mi = pd.MultiIndex.from_tuples(columns_tuples, names=["id", "section"])

    values = np.column_stack(
        [
            n1_soma,
            n1_s1_p0,
            n1_s1_p1,
            n1_s1_p2,
            n1_s1_p3,
            n2_soma,
            n2_s1_p0,
            n2_s1_p1,
            n2_s1_p2,
            n2_s2_p0,
            n2_s2_p1,
            n2_s2_p2,
            n2_s2_p3,
        ]
    )
    positions = pd.DataFrame(data=values, columns=mi)

    output_columns_tuples = [
        (1, 0),
        (1, 1),
        (1, 1),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 1),
        (2, 2),
        (2, 2),
        (2, 2),
    ]
    output_mi = pd.MultiIndex.from_tuples(output_columns_tuples, names=["id", "section"])

    electrode_positions = rng.uniform(-100, 100, size=(5, 3))
    sigma = 0.3

    # For each electrode, compute scalar result by iterating segments manually
    for epos in electrode_positions:
        scalar_coeffs = []
        i = 0
        col_section_ids = np.array([c[-1] for c in positions.columns])
        n_cols = len(positions.columns)

        while i < n_cols:
            section_id = col_section_ids[i]
            if section_id == 0:
                # Soma: point source
                soma_pos = positions.iloc[:, i].values
                dist = np.linalg.norm(soma_pos - epos)  # µm
                scalar_coeffs.append(1e-3 / (4 * np.pi * sigma * dist))
                i += 1
            elif i + 1 < n_cols and col_section_ids[i] == col_section_ids[i + 1]:
                # Line-source segment: start at i, end at i+1
                start = positions.iloc[:, i].values
                end = positions.iloc[:, i + 1].values
                scalar_coeffs.append(_scalar_line_coeff(start, end, epos, sigma))
                i += 1
            else:
                # Last boundary point of a section (no next pair) — skip
                i += 1

        # Vectorized computation
        geom = SegmentGeometry.from_positions(positions)
        vec_result = get_coeffs_line_source(geom, output_mi, epos, sigma)

        np.testing.assert_allclose(
            vec_result.values.flatten(),
            np.array(scalar_coeffs),
            rtol=1e-10,
            err_msg=f"Mismatch for electrode at {epos}",
        )

    # Also test multi-electrode call
    geom = SegmentGeometry.from_positions(positions)
    batch_result = get_coeffs_line_source(geom, output_mi, electrode_positions, sigma, verbose=False)
    for i, epos in enumerate(electrode_positions):
        single_result = get_coeffs_line_source(geom, output_mi, epos, sigma)
        np.testing.assert_allclose(
            batch_result.iloc[i].values,
            single_result.values.flatten(),
            rtol=1e-10,
            err_msg=f"Batch mismatch for electrode {i} at {epos}",
        )
