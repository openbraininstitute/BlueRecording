# SPDX-License-Identifier: GPL-3.0-or-later
import h5py
import numpy as np
import pandas as pd
import pytest

from bluerecording.weights import (
    Electrode,
    ElectrodeType,
    ObjectiveCSDParams,
    _add_data,
    _distances_in_planar_coords,
    _get_array_spacing,
    _get_coeffs_dipole_reciprocity,
    _get_coeffs_line_source,
    _get_coeffs_objective_csd_disk,
    _get_coeffs_objective_csd_plane,
    _get_coeffs_objective_csd_sphere,
    _get_coeffs_point_source,
    _get_coeffs_reciprocity,
    _get_line_coeffs,
    _get_objective_csd_array,
    _get_offsets,
    _get_segment_midpts,
    _get_thickness,
    _sort_electrode_names,
)
from tests.helpers import (
    GIDS,
    POPULATION_NAME,
    create_e_field,
    create_electrode_file,
    create_neuron_file,
    create_potential_field,
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
    path = create_neuron_file(tmp_path / "weights.h5")
    with h5py.File(path, "r") as f:
        np.testing.assert_equal(f[f"electrodes/{POPULATION_NAME}/scaling_factors"][:], np.ones((25, 2)))
        np.testing.assert_equal(f[f"{POPULATION_NAME}/offsets"][:], np.array([0, 19, 25]))


def test_add_coeffs(tmp_path):
    path = create_neuron_file(tmp_path / "weights.h5")
    data = make_report_data()
    with h5py.File(path, "r+") as h5:
        test_data = pd.DataFrame(data=np.arange(25)[np.newaxis, :], columns=data.columns)
        _add_data(h5, GIDS, test_data, POPULATION_NAME)
        expected = np.array([np.arange(25), np.ones(25)]).T
        np.testing.assert_equal(h5[f"electrodes/{POPULATION_NAME}/scaling_factors"][:], expected)


def test_add_coeffs_backwards(tmp_path):
    path = create_neuron_file(tmp_path / "weights.h5")
    data_bw = make_report_data_backwards()
    with h5py.File(path, "r+") as h5:
        test_data = pd.DataFrame(data=np.arange(25)[np.newaxis, :], columns=data_bw.columns)
        _add_data(h5, GIDS, test_data, POPULATION_NAME)
        expected = np.array([np.hstack((np.arange(6, 25), np.arange(6))), np.ones(25)]).T
        np.testing.assert_equal(h5[f"electrodes/{POPULATION_NAME}/scaling_factors"][:], expected)


def test_get_coeffs_line_source():
    positions = make_two_section_positions()
    data = make_two_section_data()
    electrode_pos = np.array([10, 10, 10])
    sigma = 1

    coeffs = _get_coeffs_line_source(positions, data.columns, electrode_pos, sigma)

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


def test_get_coeffs_point_source():
    positions = make_two_section_positions()
    electrode_pos = np.array([10, 10, 10])
    sigma = 1
    midpts = _get_segment_midpts(positions, GIDS)
    coeffs = _get_coeffs_point_source(midpts, electrode_pos, sigma)

    soma_dist = np.sqrt(3 * 10**2) * 1e-6
    expected_soma = 1 / (4 * np.pi * sigma * soma_dist) * 1e-9
    seg_dist = np.sqrt(10**2 + 10**2 + (10 - 0.5) ** 2) * 1e-6
    expected_seg = 1 / (4 * np.pi * sigma * seg_dist) * 1e-9
    expected = pd.DataFrame(data=np.hstack((expected_soma, expected_seg))[np.newaxis, :], columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def test_get_coeffs_reciprocity(tmp_path):
    positions = make_two_section_positions()
    field_path = create_potential_field(tmp_path / "potential.h5")
    midpts = _get_segment_midpts(positions, GIDS)
    potentials = _get_coeffs_reciprocity(midpts, field_path)

    columns = [[1, 1], [0, 1]]
    mi = pd.MultiIndex.from_tuples(list(zip(*columns, strict=False)), names=["id", "section"])
    expected = pd.DataFrame(data=np.array([0, 0.5e-6])[np.newaxis, :], columns=mi)
    pd.testing.assert_frame_equal(potentials, expected)


def test_get_coeffs_dipole_reciprocity(tmp_path):
    positions = make_two_section_positions()
    field_path = create_e_field(tmp_path / "efield.h5")
    midpts = _get_segment_midpts(positions, GIDS)
    center = midpts.mean(axis=1)
    potentials = _get_coeffs_dipole_reciprocity(midpts, field_path, center)

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

    coeffs = _get_coeffs_objective_csd_sphere(midpts, all_epos[0], all_epos)
    expected = pd.DataFrame(data=np.array([[1, 1]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = _get_coeffs_objective_csd_sphere(midpts, all_epos[0], all_epos, radius=0.1)
    expected = pd.DataFrame(data=np.array([[1, 0]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def test_objective_csd_disk():
    positions = make_two_section_positions()
    all_epos = np.array([[0, 0, 0], [1, 0, 0]])
    midpts = _get_segment_midpts(positions, GIDS)

    coeffs = _get_coeffs_objective_csd_disk(midpts, all_epos[0], all_epos)
    expected = pd.DataFrame(data=np.array([[1, 1]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = _get_coeffs_objective_csd_disk(midpts, all_epos[1], all_epos)
    expected = pd.DataFrame(data=np.array([[0, 0]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = _get_coeffs_objective_csd_disk(midpts, all_epos[0], all_epos, radius=0.1)
    expected = pd.DataFrame(data=np.array([[1, 0]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = _get_coeffs_objective_csd_disk(midpts, all_epos[0], all_epos, diskThickness=10)
    expected = pd.DataFrame(data=np.array([[1, 1]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def test_objective_csd_plane():
    positions = make_two_section_positions()
    all_epos = np.array([[0, 0, 0], [1, 0, 0]])
    midpts = _get_segment_midpts(positions, GIDS)

    coeffs = _get_coeffs_objective_csd_plane(midpts, all_epos[0], all_epos)
    expected = pd.DataFrame(data=np.array([[1, 1]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = _get_coeffs_objective_csd_plane(midpts, all_epos[1], all_epos)
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
    path = create_electrode_file(tmp_path / "test.h5", electrodes)
    e = electrodes[0]
    with h5py.File(path, "r") as f:
        np.testing.assert_equal(f["electrodes/name/position"][:], e.position)
        np.testing.assert_equal(f["electrodes/name/type"][()].decode(), e.type.value)
        np.testing.assert_equal(f["electrodes/name/region"][()].decode(), e.region)
        np.testing.assert_equal(f["electrodes/name/layer"][()].decode(), e.layer)
        np.testing.assert_equal(f[f"{POPULATION_NAME}/node_ids"][:], GIDS)


def test_electrode_file_structure_objective(tmp_path):
    electrodes = make_electrodes_objective()
    path = create_electrode_file(tmp_path / "test.h5", electrodes)
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
