# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
import pandas as pd
import numpy as np
import h5py

from bluerecording.weights import (
    ElectrodeFileStructure, electrode_type,
    add_data, get_coeffs_lineSource, get_coeffs_pointSource,
    get_coeffs_reciprocity, get_coeffs_dipoleReciprocity,
    get_coeffs_objectiveCSD_Sphere, get_coeffs_objectiveCSD_Disk,
    get_coeffs_objectiveCSD_Plane, get_line_coeffs,
    get_segment_midpts, get_array_spacing, get_thickness,
    distances_in_planar_coords, sort_electrode_names,
    get_objectiveCSD_array, get_offsets,
    make_electrode_dict, check_input_type_objectiveCSD, process_objectiveCSD,
    initialize_h5_file, write_h5_file,
)

from tests.helpers import (
    GIDS, POPULATION_NAME,
    make_report_data, make_report_data_backwards,
    make_electrodes, make_electrodes_objective, make_electrodes_objective_array,
    make_two_section_positions, make_two_section_data, make_sec_counts,
    create_electrode_file, create_neuron_file,
    create_potential_field, create_e_field,
)


# ---------------------------------------------------------------------------
# Weight computation tests
# ---------------------------------------------------------------------------

def test_get_segment_midpts():
    positions = make_two_section_positions()
    columns = [[1, 1], [0, 1]]
    idx = list(zip(*columns))
    mi = pd.MultiIndex.from_tuples(idx, names=['id', 'section'])
    expected = pd.DataFrame(
        data=np.array([[0., 0., 0.], [0., 0., .5]]).T, columns=mi)
    expected.index = range(len(expected))

    result = get_segment_midpts(positions, GIDS)
    pd.testing.assert_frame_equal(result, expected)


def test_write_neuron(tmp_path):
    path, _ = create_neuron_file(tmp_path / "weights.h5")
    with h5py.File(path, 'r') as f:
        np.testing.assert_equal(
            f[f'electrodes/{POPULATION_NAME}/scaling_factors'][:],
            np.ones((25, 2)))
        np.testing.assert_equal(
            f[f'{POPULATION_NAME}/offsets'][:],
            np.array([0, 19, 25]))


def test_add_coeffs(tmp_path):
    path, _ = create_neuron_file(tmp_path / "weights.h5")
    data = make_report_data()
    with h5py.File(path, 'r+') as h5:
        test_data = pd.DataFrame(data=np.arange(25)[np.newaxis, :], columns=data.columns)
        add_data(h5, GIDS, test_data, POPULATION_NAME)
        expected = np.array([np.arange(25), np.ones(25)]).T
        np.testing.assert_equal(
            h5[f'electrodes/{POPULATION_NAME}/scaling_factors'][:], expected)


def test_add_coeffs_backwards(tmp_path):
    path, _ = create_neuron_file(tmp_path / "weights.h5")
    data_bw = make_report_data_backwards()
    with h5py.File(path, 'r+') as h5:
        test_data = pd.DataFrame(data=np.arange(25)[np.newaxis, :], columns=data_bw.columns)
        add_data(h5, GIDS, test_data, POPULATION_NAME)
        expected = np.array([np.hstack((np.arange(6, 25), np.arange(6))), np.ones(25)]).T
        np.testing.assert_equal(
            h5[f'electrodes/{POPULATION_NAME}/scaling_factors'][:], expected)


def test_get_coeffs_line_source():
    positions = make_two_section_positions()
    data = make_two_section_data()
    electrode_pos = np.array([10, 10, 10])
    sigma = 1

    coeffs = get_coeffs_lineSource(positions, data.columns, electrode_pos, sigma)

    soma_dist = np.sqrt(3 * 10**2) * 1e-6
    expected_soma = 1 / (4 * np.pi * sigma * soma_dist) * 1e-9
    expected_line = get_line_coeffs(np.array([0, 0, 0]), np.array([0, 0, 1]), electrode_pos, sigma)
    expected = pd.DataFrame(
        data=np.hstack((expected_soma, expected_line))[np.newaxis, :], columns=data.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def test_line_source():
    seg = [np.array([0, 0, 0]), np.array([1, 0, 0])]
    epos = np.array([2, 0, 1])
    sigma = 1
    ds = 1e-6
    h, r, l = 1e-6, 1e-6, 2e-6
    expected = 1 / (4 * np.pi * sigma * ds) * np.log(
        np.abs((np.sqrt(h**2 + r**2) - h) / (np.sqrt(l**2 + r**2) - l)))
    result = get_line_coeffs(seg[0], seg[1], epos, sigma)
    np.testing.assert_almost_equal(result, expected * 1e-9)


def test_line_source_2():
    seg = [np.array([0, 0, 0]), np.array([1, 0, 0])]
    epos = np.array([-2, 0, 1])
    sigma = 1
    ds = 1e-6
    h, r, l = -3e-6, 1e-6, -2e-6
    expected = 1 / (4 * np.pi * sigma * ds) * np.log(
        np.abs((np.sqrt(h**2 + r**2) - h) / (np.sqrt(l**2 + r**2) - l)))
    result = get_line_coeffs(seg[0], seg[1], epos, sigma)
    np.testing.assert_almost_equal(result, expected * 1e-9)


def test_line_source_3():
    seg = [np.array([0, 0, 0]), np.array([1, 0, 0])]
    epos = np.array([0.5, 0, 1])
    sigma = 1
    ds = 1e-6
    h, r, l = -0.5e-6, 1e-6, 0.5e-6
    expected = 1 / (4 * np.pi * sigma * ds) * np.log(
        np.abs((np.sqrt(h**2 + r**2) - h) / (np.sqrt(l**2 + r**2) - l)))
    result = get_line_coeffs(seg[0], seg[1], epos, sigma)
    np.testing.assert_almost_equal(result, expected * 1e-9)


def test_get_coeffs_point_source():
    positions = make_two_section_positions()
    electrode_pos = np.array([10, 10, 10])
    sigma = 1
    midpts = get_segment_midpts(positions, GIDS)
    coeffs = get_coeffs_pointSource(midpts, electrode_pos, sigma)

    soma_dist = np.sqrt(3 * 10**2) * 1e-6
    expected_soma = 1 / (4 * np.pi * sigma * soma_dist) * 1e-9
    seg_dist = np.sqrt(10**2 + 10**2 + (10 - .5)**2) * 1e-6
    expected_seg = 1 / (4 * np.pi * sigma * seg_dist) * 1e-9
    expected = pd.DataFrame(
        data=np.hstack((expected_soma, expected_seg))[np.newaxis, :], columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def test_get_coeffs_reciprocity(tmp_path):
    positions = make_two_section_positions()
    field_path = create_potential_field(tmp_path / "potential.h5")
    midpts = get_segment_midpts(positions, GIDS)
    potentials = get_coeffs_reciprocity(midpts, field_path)

    columns = [[1, 1], [0, 1]]
    mi = pd.MultiIndex.from_tuples(list(zip(*columns)), names=['id', 'section'])
    expected = pd.DataFrame(data=np.array([0, 0.5e-6])[np.newaxis, :], columns=mi)
    pd.testing.assert_frame_equal(potentials, expected)


def test_get_coeffs_dipole_reciprocity(tmp_path):
    positions = make_two_section_positions()
    field_path = create_e_field(tmp_path / "efield.h5")
    midpts = get_segment_midpts(positions, GIDS)
    center = midpts.mean(axis=1)
    potentials = get_coeffs_dipoleReciprocity(midpts, field_path, center)

    columns = [[1, 1], [0, 1]]
    mi = pd.MultiIndex.from_tuples(list(zip(*columns)), names=['id', 'section'])
    expected = pd.DataFrame(data=-1 * np.array([0.5e-6, 0])[np.newaxis, :]**2, columns=mi)
    pd.testing.assert_frame_equal(potentials, expected)


def test_sort_electrode_names():
    keys = [0, 1, 10, 'S1nonbarrel_neurons', 3]
    result = sort_electrode_names(keys, 'S1nonbarrel_neurons')
    assert np.array_equal(result, [0, 1, 3, 10])


def test_electrode_type():
    valid = ['PointSource', 'LineSource', 'Reciprocity', 'DipoleReciprocity',
             'ObjectiveCSD_Sphere', 'ObjectiveCSD_Disk']
    for t in valid:
        assert electrode_type(t) == 0
    with pytest.raises(AssertionError):
        electrode_type('sadasd')


def test_objective_csd_sphere():
    positions = make_two_section_positions()
    all_epos = np.array([[0, 0, 0], [2, 0, 0]])
    midpts = get_segment_midpts(positions, GIDS)

    coeffs = get_coeffs_objectiveCSD_Sphere(midpts, all_epos[0], all_epos)
    expected = pd.DataFrame(data=np.array([[1, 1]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = get_coeffs_objectiveCSD_Sphere(midpts, all_epos[0], all_epos, radius=.1)
    expected = pd.DataFrame(data=np.array([[1, 0]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def test_objective_csd_disk():
    positions = make_two_section_positions()
    all_epos = np.array([[0, 0, 0], [1, 0, 0]])
    midpts = get_segment_midpts(positions, GIDS)

    coeffs = get_coeffs_objectiveCSD_Disk(midpts, all_epos[0], all_epos)
    expected = pd.DataFrame(data=np.array([[1, 1]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = get_coeffs_objectiveCSD_Disk(midpts, all_epos[1], all_epos)
    expected = pd.DataFrame(data=np.array([[0, 0]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = get_coeffs_objectiveCSD_Disk(midpts, all_epos[0], all_epos, radius=.1)
    expected = pd.DataFrame(data=np.array([[1, 0]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = get_coeffs_objectiveCSD_Disk(midpts, all_epos[0], all_epos, diskThickness=10)
    expected = pd.DataFrame(data=np.array([[1, 1]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def test_objective_csd_plane():
    positions = make_two_section_positions()
    all_epos = np.array([[0, 0, 0], [1, 0, 0]])
    midpts = get_segment_midpts(positions, GIDS)

    coeffs = get_coeffs_objectiveCSD_Plane(midpts, all_epos[0], all_epos)
    expected = pd.DataFrame(data=np.array([[1, 1]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)

    coeffs = get_coeffs_objectiveCSD_Plane(midpts, all_epos[1], all_epos)
    expected = pd.DataFrame(data=np.array([[0, 0]]), columns=midpts.columns)
    pd.testing.assert_frame_equal(coeffs, expected)


def test_array_spacing():
    all_epos = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    main_axis, spacing = get_array_spacing(all_epos)
    np.testing.assert_equal(main_axis, np.array([0, 0, 1])[:, np.newaxis])
    np.testing.assert_equal(spacing, np.array([1, 1]))


def test_array_thickness():
    assert get_thickness(np.array([1, 1])) == 0.5


def test_planar_coords():
    positions = make_two_section_positions()
    all_epos = np.array([[0, 0, 0], [1, 0, 0]])
    midpts = get_segment_midpts(positions, GIDS)
    main_axis, _ = get_array_spacing(all_epos)

    axial, radial = distances_in_planar_coords(midpts, all_epos[0], main_axis)
    np.testing.assert_equal(axial, np.array([0, 0])[:, np.newaxis])
    np.testing.assert_equal(radial, np.array([0, 0.5]))

    axial, radial = distances_in_planar_coords(midpts, all_epos[1], main_axis)
    np.testing.assert_equal(axial, np.array([1, 1])[:, np.newaxis])
    np.testing.assert_equal(radial, np.array([0, 0.5]))


def test_get_objective_csd_array(tmp_path):
    electrodes = make_electrodes_objective_array()
    path, _ = create_electrode_file(tmp_path / "obj.h5", electrodes)
    h5 = h5py.File(path, 'r+')
    names = ['a', 'b', 'name', 'name1', 'name2', 'name3']

    idx, count = get_objectiveCSD_array('ObjectiveCSD_Disk', None, 0, names, h5, 0)
    assert idx == [2, 3, 4, 5]
    assert count == 0

    idx, count = get_objectiveCSD_array('ObjectiveCSD_Disk', ['2:3', '4:5'], 0, names, h5, 4)
    np.testing.assert_equal(idx, np.arange(4, 5))
    assert count == 1
    h5.close()


# ---------------------------------------------------------------------------
# H5 initialization tests
# ---------------------------------------------------------------------------

def test_make_electrode_dict():
    electrodes = make_electrodes()
    np.testing.assert_equal(make_electrode_dict('tests/data/electrode.csv')['name'], electrodes['name'])


def test_electrode_file_structure(tmp_path):
    electrodes = make_electrodes()
    path, _ = create_electrode_file(tmp_path / "test.h5", electrodes)
    with h5py.File(path, 'r') as f:
        for key, value in electrodes['name'].items():
            if key == 'position':
                np.testing.assert_equal(f['electrodes/name/' + key][:], value)
            else:
                np.testing.assert_equal(f['electrodes/name/' + key][()].decode(), value)
        np.testing.assert_equal(f[f'{POPULATION_NAME}/node_ids'][:], GIDS)


def test_electrode_file_structure_objective(tmp_path):
    electrodes = make_electrodes_objective()
    path, _ = create_electrode_file(tmp_path / "test.h5", electrodes)
    with h5py.File(path, 'r') as f:
        for key, value in electrodes['name'].items():
            if key == 'position':
                np.testing.assert_equal(f['electrodes/name/' + key][:], value)
            elif key == 'type':
                np.testing.assert_equal(f['electrodes/name/type'][()].decode(), value['type'])
                np.testing.assert_equal(f['electrodes/name/type'].attrs.get('radius'), value['radius'])
                np.testing.assert_equal(f['electrodes/name/type'].attrs.get('thickness'), value['thickness'])
            else:
                np.testing.assert_equal(f['electrodes/name/' + key][()].decode(), value)
        np.testing.assert_equal(f[f'{POPULATION_NAME}/node_ids'][:], GIDS)


def test_offset():
    sec_counts = make_sec_counts()
    offsets = get_offsets(sec_counts)
    np.testing.assert_equal(offsets, np.array([0, 19, 25]))


def test_check_input_type_objective_csd():
    assert check_input_type_objectiveCSD('ObjectiveCSD_Sphere', 'ObjectiveCSD_Sphere_4'.split('_')) == 0

    with pytest.raises(ValueError, match='must provide either no numerical'):
        check_input_type_objectiveCSD('ObjectiveCSD_Sphere', 'ObjectiveCSD_Sphere_4_2'.split('_'))

    with pytest.raises(ValueError, match='Invalid numerical parameter'):
        check_input_type_objectiveCSD('ObjectiveCSD_Sphere', 'ObjectiveCSD_Sphere_twss'.split('_'))

    assert check_input_type_objectiveCSD('ObjectiveCSD_Plane', 'ObjectiveCSD_Plane_4'.split('_')) == 0

    with pytest.raises(ValueError, match='must provide either no numerical'):
        check_input_type_objectiveCSD('ObjectiveCSD_Plane', 'ObjectiveCSD_Plane_4_2'.split('_'))

    assert check_input_type_objectiveCSD('ObjectiveCSD_Disk', 'ObjectiveCSD_Disk_4'.split('_')) == 0
    assert check_input_type_objectiveCSD('ObjectiveCSD_Disk', 'ObjectiveCSD_Disk_4.1_200'.split('_')) == 0

    with pytest.raises(ValueError, match='Invalid numerical parameter'):
        check_input_type_objectiveCSD('ObjectiveCSD_Disk', 'ObjectiveCSD_Disk_4.1_2as'.split('_'))


def test_process_objective_csd():
    with pytest.raises(ValueError, match='is an invalid objective electrode type'):
        process_objectiveCSD('LineSource')

    assert process_objectiveCSD('ObjectiveCSD_Disk') == 'ObjectiveCSD_Disk'
    assert process_objectiveCSD('ObjectiveCSD_Sphere_2') == {'type': 'ObjectiveCSD_Sphere', 'radius': 2.0}
    assert process_objectiveCSD('ObjectiveCSD_Disk_2') == {'type': 'ObjectiveCSD_Disk', 'radius': 2.0}
    assert process_objectiveCSD('ObjectiveCSD_Disk_2.1_44') == {'type': 'ObjectiveCSD_Disk', 'radius': 2.1, 'thickness': 44.}
    assert process_objectiveCSD('ObjectiveCSD_Plane_44') == {'type': 'ObjectiveCSD_Plane', 'thickness': 44.}


# ---------------------------------------------------------------------------
# Integration tests (require data download)
# ---------------------------------------------------------------------------

@pytest.mark.skip_in_ci
def test_circuit_write_weights(tmp_path):
    """Full write_weights pipeline for sscx_100_cells."""
    from bluerecording.circuit import init_circuit
    from bluerecording import positions

    simconfig = "examples/sscx_100_cells/simulation_config.json"
    csv = "examples/sscx_100_cells/electrodes.csv"
    ref = "examples/sscx_100_cells/reference/weights_ref.h5"
    out = str(tmp_path / "weights.h5")

    nm, ids, cols, pop, pop_name, morphologies_dir = init_circuit(simconfig)
    pos_df, cols, _ = positions.get_positions(nm, ids, cols, pop, morphologies_dir=morphologies_dir)
    initialize_h5_file(cols, pop_name, out, csv)
    write_h5_file(pos_df, cols, pop_name, out)

    with h5py.File(ref, "r") as r, h5py.File(out, "r") as n:
        np.testing.assert_array_equal(r[f"{pop_name}/node_ids"][:], n[f"{pop_name}/node_ids"][:])
        np.testing.assert_array_equal(r[f"{pop_name}/offsets"][:], n[f"{pop_name}/offsets"][:])
        dset = f"electrodes/{pop_name}/scaling_factors"
        np.testing.assert_allclose(r[dset][:], n[dset][:], rtol=1e-6, atol=1e-9)

    with open(ref, "rb") as f1, open(out, "rb") as f2:
        assert f1.read() == f2.read(), "Output differs from reference at byte level"


@pytest.mark.skip_in_ci
def test_single_cell_write_weights(tmp_path):
    """Write_weights for single_cell_l5_tpc (near electrodes)."""
    from bluerecording.circuit import init_circuit
    from bluerecording import positions

    simconfig = "examples/single_cell_l5_tpc/simulation_config_near.json"
    csv = "examples/single_cell_l5_tpc/near_electrodes.csv"
    ref = "examples/single_cell_l5_tpc/reference/weights_near_ref.h5"
    field = "examples/single_cell_l5_tpc/Infinite_Close_HighRes_SmallSphere.h5"
    out = str(tmp_path / "weights.h5")

    nm, ids, cols, pop, pop_name, morphologies_dir = init_circuit(simconfig)
    pos_df, cols, _ = positions.get_positions(nm, ids, cols, pop, morphologies_dir=morphologies_dir)
    initialize_h5_file(cols, pop_name, out, csv)
    write_h5_file(pos_df, cols, pop_name, out, path_to_fields=[field, field])

    with h5py.File(ref, "r") as r, h5py.File(out, "r") as n:
        np.testing.assert_array_equal(r[f"{pop_name}/node_ids"][:], n[f"{pop_name}/node_ids"][:])
        np.testing.assert_array_equal(r[f"{pop_name}/offsets"][:], n[f"{pop_name}/offsets"][:])
        dset = f"electrodes/{pop_name}/scaling_factors"
        np.testing.assert_allclose(r[dset][:], n[dset][:], rtol=1e-6, atol=1e-9)


@pytest.mark.skip_in_ci
def test_single_cell_write_weights_distant(tmp_path):
    """Write_weights for single_cell_l5_tpc (distant electrodes)."""
    from bluerecording.circuit import init_circuit
    from bluerecording import positions

    simconfig = "examples/single_cell_l5_tpc/simulation_config_near.json"
    csv = "examples/single_cell_l5_tpc/distant_electrodes.csv"
    ref = "examples/single_cell_l5_tpc/reference/weights_distant_ref.h5"
    field = "examples/single_cell_l5_tpc/Infinite_VeryFar_HighRes.h5"
    out = str(tmp_path / "weights.h5")

    nm, ids, cols, pop, pop_name, morphologies_dir = init_circuit(simconfig)
    pos_df, cols, _ = positions.get_positions(nm, ids, cols, pop, morphologies_dir=morphologies_dir)
    initialize_h5_file(cols, pop_name, out, csv)
    write_h5_file(pos_df, cols, pop_name, out, path_to_fields=[field, field])

    with h5py.File(ref, "r") as r, h5py.File(out, "r") as n:
        np.testing.assert_array_equal(r[f"{pop_name}/node_ids"][:], n[f"{pop_name}/node_ids"][:])
        np.testing.assert_array_equal(r[f"{pop_name}/offsets"][:], n[f"{pop_name}/offsets"][:])
        dset = f"electrodes/{pop_name}/scaling_factors"
        np.testing.assert_allclose(r[dset][:], n[dset][:], rtol=1e-6, atol=1e-9)


@pytest.mark.skip_in_ci
def test_single_cell_neurite_types(tmp_path):
    """Write weights with --with-neurite-type and verify types independently."""
    from bluerecording.circuit import init_circuit
    from bluerecording import positions
    from neurodamus.metype import BaseCell

    simconfig = "examples/single_cell_l5_tpc/simulation_config_near.json"
    csv = "examples/single_cell_l5_tpc/near_electrodes.csv"
    field = "examples/single_cell_l5_tpc/Infinite_Close_HighRes_SmallSphere.h5"
    out = str(tmp_path / "weights.h5")

    nm, ids, cols, pop, pop_name, morphologies_dir = init_circuit(simconfig)
    pos_df, cols, neurite_types = positions.get_positions(
        nm, ids, cols, pop, morphologies_dir=morphologies_dir,
    )
    initialize_h5_file(cols, pop_name, out, csv, with_neurite_type=True)
    write_h5_file(pos_df, cols, pop_name, out,
                  path_to_fields=[field, field],
                  neurite_types=neurite_types)

    # --- Independent verification ---
    # Build expected type codes by iterating SectionLists on the cell directly,
    # without using resolve_neurite_types.
    type_to_code = {st: idx for idx, (st, _) in enumerate(BaseCell.SECTION_TYPES)}

    for gid in ids:
        cell = nm.get_cell(gid)
        counts = cell.get_section_counts()

        # Build a section_id → type_code map from the counts
        expected_map = {}
        offset = 0
        for (sec_type, _), count in zip(BaseCell.SECTION_TYPES, counts):
            for local_idx in range(count):
                expected_map[offset + local_idx] = type_to_code[sec_type]
            offset += count

        cols_for_gid = cols[cols[:, 0] == gid]
        expected_codes = np.array(
            [expected_map[int(sec_id)] for sec_id in cols_for_gid[:, 1]],
            dtype=np.int32,
        )

        gid_mask = cols[:, 0] == gid
        actual_codes = neurite_types[gid_mask]
        np.testing.assert_array_equal(actual_codes, expected_codes)

    # Also verify the H5 dataset was written correctly
    with h5py.File(out, "r") as h5:
        assert f"{pop_name}/neurite_types" in h5
        stored = h5[f"{pop_name}/neurite_types"][:]
        np.testing.assert_array_equal(stored, neurite_types)
