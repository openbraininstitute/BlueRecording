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
    _get_objective_csd_array,
    _get_offsets,
    _get_segment_midpts,
    _iter_cell_chunks,
    _sort_electrode_names,
    _write_electrode_metadata_to_h5,
    get_and_save_weights,
    get_weights,
    save_weights,
)
from tests.helpers import (
    GIDS,
    POPULATION_NAME,
    create_weights_file,
    make_cells,
    make_electrodes,
    make_electrodes_mixed,
    make_electrodes_objective,
    make_electrodes_objective_array,
    make_report_data,
    make_report_data_backwards,
    make_sec_counts,
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
        np.testing.assert_equal(f[f"electrodes/{POPULATION_NAME}/scaling_factors"][:], np.ones((25, 1)))
        np.testing.assert_equal(f[f"{POPULATION_NAME}/offsets"][:], np.array([0, 19, 25]))


def test_write_neuron_creates_missing_directory(tmp_path):
    """save_weights creates parent directories if they don't exist."""
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
        _add_data(h5, test_data, POPULATION_NAME, start=0)
        expected = np.arange(25)[:, np.newaxis].astype(np.float64)
        np.testing.assert_equal(h5[f"electrodes/{POPULATION_NAME}/scaling_factors"][:], expected)


def test_add_coeffs_backwards(tmp_path):
    """Test that _add_data writes correctly when coeffs columns are reordered.

    In the new rank-ordered architecture, the file layout always matches cols
    order (which matches coeffs column order). This test verifies that writing
    with start=0 using a different column ordering produces the expected layout
    (data appears in the order of the coeffs columns, not GID-sorted).
    """
    path = create_weights_file(tmp_path / "weights.h5")
    data_bw = make_report_data_backwards()
    with h5py.File(path, "r+") as h5:
        test_data = pd.DataFrame(data=np.arange(25)[np.newaxis, :], columns=data_bw.columns)
        _add_data(h5, test_data, POPULATION_NAME, start=0)
        # With start=0, data is written in coeffs column order (node 2 first, then node 1)
        # So rows 0-24 get values 0-24 sequentially
        expected = np.arange(25)[:, np.newaxis].astype(np.float64)
        np.testing.assert_equal(h5[f"electrodes/{POPULATION_NAME}/scaling_factors"][:], expected)


def test_iter_cell_chunks():
    """Chunks hold whole cells up to the segment limit; a bigger cell gets a chunk of its own."""
    cols, positions = make_cells()  # 5, 9, 3 and 7 segments

    chunks = list(_iter_cell_chunks(positions, cols, segment_chunk_size=12))
    assert [(rows.start, rows.stop) for rows, _ in chunks] == [(0, 5), (5, 17), (17, 24)]
    assert [(points.start, points.stop) for _, points in chunks] == [(0, 7), (7, 24), (24, 34)]

    chunks = list(_iter_cell_chunks(positions, cols, segment_chunk_size=8))
    assert [(rows.start, rows.stop) for rows, _ in chunks] == [(0, 5), (5, 14), (14, 17), (17, 24)]

    for rows, points in chunks:
        assert set(cols[rows, 0]) == set(positions.columns[points].get_level_values(0))


def test_get_and_save_weights_in_chunks(tmp_path):
    """get_and_save_weights writes the same file as get_weights + save_weights, whatever the chunk size."""
    cols, positions = make_cells()
    electrodes = make_electrodes_mixed()
    expected = str(tmp_path / "expected.h5")
    save_weights(get_weights(positions, cols, electrodes), cols, POPULATION_NAME, expected, electrodes)
    datasets = [
        f"electrodes/{POPULATION_NAME}/scaling_factors",
        f"{POPULATION_NAME}/node_ids",
        f"{POPULATION_NAME}/offsets",
    ]

    for segment_chunk_size in (1, 12, 1000):
        out = str(tmp_path / f"weights_{segment_chunk_size}.h5")
        get_and_save_weights(positions, cols, POPULATION_NAME, out, electrodes, segment_chunk_size=segment_chunk_size)
        with h5py.File(expected, "r") as e, h5py.File(out, "r") as o:
            for path in datasets:
                np.testing.assert_array_equal(o[path][:], e[path][:])


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
    result = Electrode.from_json("tests/data/electrode.json")
    e = result[0]
    np.testing.assert_equal(e.position, expected[0].position)
    assert e.type == expected[0].type
    assert e.region == expected[0].region
    assert e.layer == expected[0].layer


def test_make_electrode_dict_objective_csd():
    result = Electrode.from_json("tests/data/electrode_objective.json")
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
    import json

    json_path = tmp_path / "bad.json"
    json_path.write_text(json.dumps([{"name": "bad", "x": 1, "y": 2, "z": 3, "type": "TotallyInvalid"}]))
    with pytest.raises(ValueError):
        Electrode.from_json(str(json_path))


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


def test_electrode_json_roundtrip(tmp_path):
    """Write electrodes to JSON, read them back, verify they match."""
    electrodes = [
        Electrode(name="e0", position=np.array([1.0, 2.0, 3.0]), type=ElectrodeType.POINT_SOURCE),
        Electrode(
            name="e1",
            position=np.array([4.0, 5.0, 6.0]),
            type=ElectrodeType.LINE_SOURCE,
            region="S1",
            layer="L5",
        ),
    ]
    json_path = str(tmp_path / "electrodes.json")
    Electrode.to_json(electrodes, json_path)

    loaded = Electrode.from_json(json_path)

    assert len(loaded) == 2
    for orig, read in zip(electrodes, loaded, strict=True):
        assert orig.name == read.name
        np.testing.assert_array_almost_equal(orig.position, read.position)
        assert orig.type == read.type
        assert orig.region == read.region
        assert orig.layer == read.layer


def test_electrode_json_roundtrip_objective_csd(tmp_path):
    """Write ObjectiveCSD electrodes to JSON and verify roundtrip."""
    electrodes = [
        Electrode(
            name="ocsd0",
            position=np.array([10.0, 20.0, 30.0]),
            type=ObjectiveCSDParams(electrode_type=ElectrodeType.OBJECTIVE_CSD_DISK, radius=500.0, thickness=10.0),
        ),
    ]
    json_path = str(tmp_path / "electrodes_ocsd.json")
    Electrode.to_json(electrodes, json_path)

    loaded = Electrode.from_json(json_path)

    assert len(loaded) == 1
    assert loaded[0].name == "ocsd0"
    np.testing.assert_array_almost_equal(loaded[0].position, np.array([10.0, 20.0, 30.0]))
    assert isinstance(loaded[0].type, ObjectiveCSDParams)
    assert loaded[0].type.radius == 500.0
    assert loaded[0].type.thickness == 10.0
