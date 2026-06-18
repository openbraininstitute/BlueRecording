# SPDX-License-Identifier: GPL-3.0-or-later
import h5py
import numpy as np
import pandas as pd
import pytest

from bluerecording.physics import (
    _distances_in_planar_coords,
    _get_array_spacing,
    _get_line_coeffs,
    _get_thickness,
    get_coeffs_dipole_reciprocity,
    get_coeffs_line_source,
    get_coeffs_line_source_batch,
    get_coeffs_objective_csd_disk,
    get_coeffs_objective_csd_plane,
    get_coeffs_objective_csd_sphere,
    get_coeffs_point_source,
    get_coeffs_reciprocity,
)
from bluerecording.physics import (
    precompute_segment_geometry as _precompute_segment_geometry,
)
from bluerecording.weights import (
    Electrode,
    ElectrodeType,
    ObjectiveCSDParams,
    _add_data,
    _get_objective_csd_array,
    _get_offsets,
    _get_segment_midpts,
    _sort_electrode_names,
    _write_electrode_metadata_to_h5,
)
from tests.helpers import (
    GIDS,
    POPULATION_NAME,
    create_e_field,
    create_potential_field,
    create_weights_file,
    make_electrodes,
    make_electrodes_objective,
    make_electrodes_objective_array,
    make_report_data,
    make_report_data_backwards,
    make_sec_counts,
    make_two_section_data,
    make_two_section_positions,
)

# ---------------------------------------------------------------------------
# Weight computation tests
# ---------------------------------------------------------------------------


def test_get_segment_midpts():
    positions = make_two_section_positions()
    columns = [[1, 1], [0, 1]]
    idx = list(zip(*columns, strict=False))
    mi = pd.MultiIndex.from_tuples(idx, names=["id", "section"])
    expected = pd.DataFrame(data=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]).T, columns=mi)
    expected.index = range(len(expected))

    result = _get_segment_midpts(positions, GIDS)
    pd.testing.assert_frame_equal(result, expected)


def test_write_neuron(tmp_path):
    path = create_weights_file(tmp_path / "weights.h5")
    with h5py.File(path, "r") as f:
        np.testing.assert_equal(f[f"electrodes/{POPULATION_NAME}/scaling_factors"][:], np.ones((25, 2)))
        np.testing.assert_equal(f[f"{POPULATION_NAME}/offsets"][:], np.array([0, 19, 25]))


def test_write_neuron_creates_missing_directory(tmp_path):
    """save_weights creates parent directories if they don't exist."""
    from bluerecording.weights import save_weights

    path = tmp_path / "nested" / "subdir" / "weights.h5"
    assert not path.parent.exists()

    cols = np.array([[1, 0], [1, 1]], dtype=np.int64)
    electrodes = make_electrodes()
    weights = pd.DataFrame(data=np.ones((1, 2)), columns=pd.MultiIndex.from_arrays(cols.T, names=["id", "section"]))

    save_weights(weights, cols, POPULATION_NAME, str(path), electrodes)
    assert path.exists()


def test_add_coeffs(tmp_path):
    path = create_weights_file(tmp_path / "weights.h5")
    data = make_report_data()
    with h5py.File(path, "r+") as h5:
        test_data = pd.DataFrame(data=np.arange(25)[np.newaxis, :], columns=data.columns)
        _add_data(h5, GIDS, test_data, POPULATION_NAME)
        expected = np.array([np.arange(25), np.ones(25)]).T
        np.testing.assert_equal(h5[f"electrodes/{POPULATION_NAME}/scaling_factors"][:], expected)


def test_add_coeffs_backwards(tmp_path):
    path = create_weights_file(tmp_path / "weights.h5")
    data_bw = make_report_data_backwards()
    with h5py.File(path, "r+") as h5:
        test_data = pd.DataFrame(data=np.arange(25)[np.newaxis, :], columns=data_bw.columns)
        _add_data(h5, GIDS, test_data, POPULATION_NAME)
        expected = np.array([np.hstack((np.arange(6, 25), np.arange(6))), np.ones(25)]).T
        np.testing.assert_equal(h5[f"electrodes/{POPULATION_NAME}/scaling_factors"][:], expected)


def testget_coeffs_line_source():
    positions = make_two_section_positions()
    data = make_two_section_data()
    electrode_pos = np.array([10, 10, 10])
    sigma = 1

    coeffs = get_coeffs_line_source(positions, data.columns, electrode_pos, sigma)

    soma_dist = np.sqrt(3 * 10**2) * 1e-6
    expected_soma = 1 / (4 * np.pi * sigma * soma_dist) * 1e-9
    expected_line = _get_line_coeffs(np.array([0, 0, 0]), np.array([0, 0, 1]), electrode_pos, sigma)
    expected = pd.DataFrame(data=np.hstack((expected_soma, expected_line))[np.newaxis, :], columns=data.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def test_line_source():
    seg = [np.array([0, 0, 0]), np.array([1, 0, 0])]
    epos = np.array([2, 0, 1])
    sigma = 1
    ds = 1e-6
    h, r, l = 1e-6, 1e-6, 2e-6
    expected = 1 / (4 * np.pi * sigma * ds) * np.log(np.abs((np.sqrt(h**2 + r**2) - h) / (np.sqrt(l**2 + r**2) - l)))
    result = _get_line_coeffs(seg[0], seg[1], epos, sigma)
    np.testing.assert_almost_equal(result, expected * 1e-9)


def test_line_source_2():
    seg = [np.array([0, 0, 0]), np.array([1, 0, 0])]
    epos = np.array([-2, 0, 1])
    sigma = 1
    ds = 1e-6
    h, r, l = -3e-6, 1e-6, -2e-6
    expected = 1 / (4 * np.pi * sigma * ds) * np.log(np.abs((np.sqrt(h**2 + r**2) - h) / (np.sqrt(l**2 + r**2) - l)))
    result = _get_line_coeffs(seg[0], seg[1], epos, sigma)
    np.testing.assert_almost_equal(result, expected * 1e-9)


def test_line_source_3():
    seg = [np.array([0, 0, 0]), np.array([1, 0, 0])]
    epos = np.array([0.5, 0, 1])
    sigma = 1
    ds = 1e-6
    h, r, l = -0.5e-6, 1e-6, 0.5e-6
    expected = 1 / (4 * np.pi * sigma * ds) * np.log(np.abs((np.sqrt(h**2 + r**2) - h) / (np.sqrt(l**2 + r**2) - l)))
    result = _get_line_coeffs(seg[0], seg[1], epos, sigma)
    np.testing.assert_almost_equal(result, expected * 1e-9)


def testget_coeffs_point_source():
    positions = make_two_section_positions()
    electrode_pos = np.array([10, 10, 10])
    sigma = 1
    midpts = _get_segment_midpts(positions, GIDS)
    coeffs = get_coeffs_point_source(midpts, electrode_pos, sigma)

    soma_dist = np.sqrt(3 * 10**2) * 1e-6
    expected_soma = 1 / (4 * np.pi * sigma * soma_dist) * 1e-9
    seg_dist = np.sqrt(10**2 + 10**2 + (10 - 0.5) ** 2) * 1e-6
    expected_seg = 1 / (4 * np.pi * sigma * seg_dist) * 1e-9
    expected = pd.DataFrame(data=np.hstack((expected_soma, expected_seg))[np.newaxis, :], columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def testget_coeffs_reciprocity(tmp_path):
    positions = make_two_section_positions()
    field_path = create_potential_field(tmp_path / "potential.h5")
    midpts = _get_segment_midpts(positions, GIDS)
    potentials = get_coeffs_reciprocity(midpts, field_path)

    columns = [[1, 1], [0, 1]]
    mi = pd.MultiIndex.from_tuples(list(zip(*columns, strict=False)), names=["id", "section"])
    expected = pd.DataFrame(data=np.array([0, 0.5e-6])[np.newaxis, :], columns=mi)
    pd.testing.assert_frame_equal(potentials, expected)


def testget_coeffs_dipole_reciprocity(tmp_path):
    positions = make_two_section_positions()
    field_path = create_e_field(tmp_path / "efield.h5")
    midpts = _get_segment_midpts(positions, GIDS)
    center = midpts.mean(axis=1)
    potentials = get_coeffs_dipole_reciprocity(midpts, field_path, center)

    columns = [[1, 1], [0, 1]]
    mi = pd.MultiIndex.from_tuples(list(zip(*columns, strict=False)), names=["id", "section"])
    expected = pd.DataFrame(data=-1 * np.array([0.5e-6, 0])[np.newaxis, :] ** 2, columns=mi)
    pd.testing.assert_frame_equal(potentials, expected)


def test_sort_electrode_names():
    keys = [0, 1, 10, "S1nonbarrel_neurons", 3]
    result = _sort_electrode_names(keys, "S1nonbarrel_neurons")
    assert np.array_equal(result, [0, 1, 3, 10])


def test_electrode_type():
    valid = [
        "PointSource",
        "LineSource",
        "Reciprocity",
        "DipoleReciprocity",
        "ObjectiveCSD_Sphere",
        "ObjectiveCSD_Disk",
    ]
    for t in valid:
        ElectrodeType(t)
    with pytest.raises(ValueError):
        ElectrodeType("sadasd")


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


def test_get_objective_csd_array(tmp_path):
    electrodes = make_electrodes_objective_array()
    # Build the ordered list matching sorted electrode names
    electrodes_ordered = sorted(electrodes, key=lambda e: e.name)

    idx, count = _get_objective_csd_array(ElectrodeType.OBJECTIVE_CSD_DISK, None, 0, electrodes_ordered, 0)
    assert idx == [2, 3, 4, 5]
    assert count == 0

    idx, count = _get_objective_csd_array(ElectrodeType.OBJECTIVE_CSD_DISK, ["2:3", "4:5"], 0, electrodes_ordered, 4)
    np.testing.assert_equal(idx, np.arange(4, 5))
    assert count == 1


# ---------------------------------------------------------------------------
# H5 initialization tests
# ---------------------------------------------------------------------------


def test_make_electrode_dict():
    expected = make_electrodes()
    result = Electrode.from_csv("tests/data/electrode.csv")
    e = result[0]
    np.testing.assert_equal(e.position, expected[0].position)
    assert e.type == expected[0].type
    assert e.region == expected[0].region
    assert e.layer == expected[0].layer


def test_make_electrode_dict_objective_csd():
    result = Electrode.from_csv("tests/data/electrode_objective.csv")
    by_name = {e.name: e for e in result}

    assert by_name["sphere"].type == ObjectiveCSDParams(
        electrode_type=ElectrodeType.OBJECTIVE_CSD_SPHERE, radius=15.0, thickness=None
    )
    assert by_name["disk"].type == ObjectiveCSDParams(
        electrode_type=ElectrodeType.OBJECTIVE_CSD_DISK, radius=500.0, thickness=25.0
    )
    assert by_name["plane"].type == ObjectiveCSDParams(
        electrode_type=ElectrodeType.OBJECTIVE_CSD_PLANE, radius=None, thickness=30.0
    )
    # Missing radius/thickness → None
    assert by_name["disk_defaults"].type == ObjectiveCSDParams(
        electrode_type=ElectrodeType.OBJECTIVE_CSD_DISK, radius=None, thickness=None
    )


def test_make_electrode_dict_invalid_type(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(",x,y,z,type\nbad,1,2,3,TotallyInvalid\n")
    with pytest.raises(ValueError):
        Electrode.from_csv(str(csv_path))


def test_electrode_file_structure(tmp_path):
    electrodes = make_electrodes()
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as h5file:
        _write_electrode_metadata_to_h5(h5file, GIDS, electrodes, POPULATION_NAME)
    e = electrodes[0]
    with h5py.File(path, "r") as f:
        np.testing.assert_equal(f["electrodes/name/position"][:], e.position)
        np.testing.assert_equal(f["electrodes/name/type"][()].decode(), e.type.value)
        np.testing.assert_equal(f["electrodes/name/region"][()].decode(), e.region)
        np.testing.assert_equal(f["electrodes/name/layer"][()].decode(), e.layer)
        np.testing.assert_equal(f[f"{POPULATION_NAME}/node_ids"][:], GIDS)


def test_electrode_file_structure_objective(tmp_path):
    electrodes = make_electrodes_objective()
    path = tmp_path / "test.h5"
    with h5py.File(path, "w") as h5file:
        _write_electrode_metadata_to_h5(h5file, GIDS, electrodes, POPULATION_NAME)
    e = electrodes[0]
    with h5py.File(path, "r") as f:
        np.testing.assert_equal(f["electrodes/name/position"][:], e.position)
        np.testing.assert_equal(f["electrodes/name/type"][()].decode(), e.type.electrode_type.value)
        np.testing.assert_equal(f["electrodes/name/type"].attrs.get("radius"), e.type.radius)
        np.testing.assert_equal(f["electrodes/name/type"].attrs.get("thickness"), e.type.thickness)
        np.testing.assert_equal(f["electrodes/name/region"][()].decode(), e.region)
        np.testing.assert_equal(f["electrodes/name/layer"][()].decode(), e.layer)
        np.testing.assert_equal(f[f"{POPULATION_NAME}/node_ids"][:], GIDS)


def test_offset():
    sec_counts = make_sec_counts()
    offsets = _get_offsets(sec_counts)
    np.testing.assert_equal(offsets, np.array([0, 19, 25]))


# ---------------------------------------------------------------------------
# Precompute segment geometry tests
# ---------------------------------------------------------------------------


def test_precompute_segment_geometry_basic():
    """Test _precompute_segment_geometry with soma + one line-source segment."""
    positions = make_two_section_positions()
    # positions has columns: (1,0), (1,1), (1,1)
    # (1,0) is soma, (1,1)+(1,1) is a line-source segment pair

    result = _precompute_segment_geometry(positions)

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

    # Length should be 1 µm = 1e-6 m
    assert result.seg_lengths.shape == (1,)
    np.testing.assert_almost_equal(result.seg_lengths[0], 1e-6)

    # Direction should be along z-axis
    assert result.seg_dirs.shape == (1, 3)
    np.testing.assert_array_almost_equal(result.seg_dirs[0], [0.0, 0.0, 1.0])


def test_precompute_segment_geometry_multi_neuron():
    """Test _precompute_segment_geometry with multiple neurons and sections."""
    # Build a positions DataFrame with 2 neurons:
    # neuron 1: soma + 2 segments (section 1 has 3 boundaries → 2 segments)
    # neuron 2: soma + 1 segment
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

    result = _precompute_segment_geometry(positions)

    # Expected: 2 somas, 3 line-source segments
    # Order: soma(1,0), seg(1→2), seg(2→3), soma(2,0), seg(y1→y2)
    assert np.sum(result.is_soma) == 2
    assert np.sum(~result.is_soma) == 3

    assert result.soma_positions.shape == (2, 3)
    np.testing.assert_array_equal(result.soma_positions[0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.soma_positions[1], [10.0, 0.0, 0.0])

    assert result.start_pos.shape == (3, 3)
    assert result.end_pos.shape == (3, 3)

    # All segments are 1 µm long along x or y axis
    np.testing.assert_array_almost_equal(result.seg_lengths, [1e-6, 1e-6, 1e-6])


def test_precompute_segment_geometry_no_soma():
    """Test with no soma segments (all line-source)."""
    columns = [[1, 1], [1, 1]]
    idx = list(zip(*columns, strict=False))
    mi = pd.MultiIndex.from_tuples(idx, names=["id", "section"])
    values = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).T
    positions = pd.DataFrame(data=values, columns=mi)

    result = _precompute_segment_geometry(positions)

    assert np.sum(result.is_soma) == 0
    assert result.soma_positions.shape == (0, 3)
    assert result.start_pos.shape == (1, 3)
    assert result.end_pos.shape == (1, 3)
    np.testing.assert_almost_equal(result.seg_lengths[0], 1e-6)


def test_precompute_segment_geometry_only_soma():
    """Test with only soma segments (no line-source)."""
    columns = [[1], [0]]
    idx = list(zip(*columns, strict=False))
    mi = pd.MultiIndex.from_tuples(idx, names=["id", "section"])
    values = np.array([[5.0, 3.0, 1.0]]).T
    positions = pd.DataFrame(data=values, columns=mi)

    result = _precompute_segment_geometry(positions)

    assert np.sum(result.is_soma) == 1
    assert np.sum(~result.is_soma) == 0
    assert result.soma_positions.shape == (1, 3)
    np.testing.assert_array_equal(result.soma_positions[0], [5.0, 3.0, 1.0])
    assert result.start_pos.shape == (0, 3)
    assert result.seg_lengths.shape == (0,)


def test_vectorized_matches_scalar():
    """Verify vectorized implementation matches scalar for non-trivial input."""
    rng = np.random.default_rng(42)

    # Build positions DataFrame with ~10 segments across 2-3 neurons:
    # neuron 1 (gid=1): soma + section 1 with 4 boundary points (3 segments)
    # neuron 2 (gid=2): soma + section 1 with 3 boundary points (2 segments)
    #                         + section 2 with 4 boundary points (3 segments)
    # Total: 2 soma + 8 line-source segments = 10 output coefficients

    # Generate random positions (deterministic seed)
    # neuron 1 soma
    n1_soma = rng.uniform(-50, 50, size=3)
    # neuron 1 section 1: 4 boundary points (forming 3 segments)
    n1_s1_p0 = n1_soma + rng.uniform(5, 20, size=3)
    n1_s1_p1 = n1_s1_p0 + rng.uniform(5, 20, size=3)
    n1_s1_p2 = n1_s1_p1 + rng.uniform(5, 20, size=3)
    n1_s1_p3 = n1_s1_p2 + rng.uniform(5, 20, size=3)

    # neuron 2 soma
    n2_soma = rng.uniform(-50, 50, size=3)
    # neuron 2 section 1: 3 boundary points (2 segments)
    n2_s1_p0 = n2_soma + rng.uniform(5, 20, size=3)
    n2_s1_p1 = n2_s1_p0 + rng.uniform(5, 20, size=3)
    n2_s1_p2 = n2_s1_p1 + rng.uniform(5, 20, size=3)
    # neuron 2 section 2: 4 boundary points (3 segments)
    n2_s2_p0 = n2_soma + rng.uniform(-20, -5, size=3)
    n2_s2_p1 = n2_s2_p0 + rng.uniform(5, 20, size=3)
    n2_s2_p2 = n2_s2_p1 + rng.uniform(5, 20, size=3)
    n2_s2_p3 = n2_s2_p2 + rng.uniform(5, 20, size=3)

    # Construct positions DataFrame with MultiIndex columns (gid, section_id)
    # Column layout: (gid, section_id) pairs
    columns_tuples = [
        # neuron 1
        (1, 0),  # soma
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),  # section 1: 4 boundary points
        # neuron 2
        (2, 0),  # soma
        (2, 1),
        (2, 1),
        (2, 1),  # section 1: 3 boundary points
        (2, 2),
        (2, 2),
        (2, 2),
        (2, 2),  # section 2: 4 boundary points
    ]
    mi = pd.MultiIndex.from_tuples(columns_tuples, names=["id", "section"])

    # Each column is a 3D position (x, y, z) as the rows
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

    # Output columns: one per segment (soma or line-source segment)
    # neuron 1: soma(1,0), seg(1,1), seg(1,1), seg(1,1)
    # neuron 2: soma(2,0), seg(2,1), seg(2,1), seg(2,2), seg(2,2), seg(2,2)
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

    # 5 electrodes at various positions (some close, some far)
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
                dist = np.linalg.norm(soma_pos - epos) * 1e-6
                scalar_coeffs.append(1 / (4 * np.pi * sigma * dist) * 1e-9)
                i += 1
            elif i + 1 < n_cols and col_section_ids[i] == col_section_ids[i + 1]:
                # Line-source segment: start at i, end at i+1
                start = positions.iloc[:, i].values
                end = positions.iloc[:, i + 1].values
                scalar_coeffs.append(_get_line_coeffs(start, end, epos, sigma))
                i += 1
            else:
                # Last boundary point of a section (no next pair) — skip
                i += 1

        # Vectorized computation
        vec_result = get_coeffs_line_source(positions, output_mi, epos, sigma)

        np.testing.assert_allclose(
            vec_result.values.flatten(),
            np.array(scalar_coeffs),
            rtol=1e-10,
            err_msg=f"Mismatch for electrode at {epos}",
        )

    # Also test batch function
    batch_result = get_coeffs_line_source_batch(positions, output_mi, electrode_positions, sigma, verbose=False)
    for i, epos in enumerate(electrode_positions):
        single_result = get_coeffs_line_source(positions, output_mi, epos, sigma)
        np.testing.assert_allclose(
            batch_result.iloc[i].values,
            single_result.values.flatten(),
            rtol=1e-10,
            err_msg=f"Batch mismatch for electrode {i} at {epos}",
        )
