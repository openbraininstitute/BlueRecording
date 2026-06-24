# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for bluerecording.compare."""
import h5py
import numpy as np

from bluerecording.compare import compare_weights

POPULATION = "testPop"


def _create_weights_file(path, node_ids, offsets, scaling_factors):
    """Create a minimal weights H5 file for testing."""
    with h5py.File(path, "w") as f:
        f.create_dataset(f"{POPULATION}/node_ids", data=node_ids)
        f.create_dataset(f"{POPULATION}/offsets", data=offsets)
        f.create_dataset(f"electrodes/{POPULATION}/scaling_factors", data=scaling_factors)


def test_matching_files(tmp_path):
    """Identical files should match."""
    node_ids = np.array([10, 20])
    offsets = np.array([0, 3, 5])
    sf = np.ones((5, 3))

    _create_weights_file(tmp_path / "a.h5", node_ids, offsets, sf)
    _create_weights_file(tmp_path / "b.h5", node_ids, offsets, sf)

    match, report = compare_weights(str(tmp_path / "a.h5"), str(tmp_path / "b.h5"))
    assert match
    assert "2 nodes match" in report


def test_matching_different_order(tmp_path):
    """Files with same data but different node order should match."""
    sf_a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    sf_b = np.array([[5.0, 6.0], [7.0, 8.0], [1.0, 2.0], [3.0, 4.0]])

    # File A: node 10 → rows 0-1, node 20 → rows 2-3
    _create_weights_file(tmp_path / "a.h5", np.array([10, 20]), np.array([0, 2, 4]), sf_a)
    # File B: node 20 → rows 0-1, node 10 → rows 2-3
    _create_weights_file(tmp_path / "b.h5", np.array([20, 10]), np.array([0, 2, 4]), sf_b)

    match, report = compare_weights(str(tmp_path / "a.h5"), str(tmp_path / "b.h5"))
    assert match


def test_mismatched_scaling_factors(tmp_path):
    """Different scaling factors should fail."""
    node_ids = np.array([1])
    offsets = np.array([0, 2])
    sf_a = np.ones((2, 3))
    sf_b = np.ones((2, 3))
    sf_b[1, 0] = 999.0

    _create_weights_file(tmp_path / "a.h5", node_ids, offsets, sf_a)
    _create_weights_file(tmp_path / "b.h5", node_ids, offsets, sf_b)

    match, report = compare_weights(str(tmp_path / "a.h5"), str(tmp_path / "b.h5"))
    assert not match
    assert "node_id 1" in report
    assert "max abs diff" in report


def test_mismatched_node_ids(tmp_path):
    """Different node_id sets should fail."""
    sf = np.ones((3, 2))
    _create_weights_file(tmp_path / "a.h5", np.array([1, 2]), np.array([0, 1, 3]), sf)
    _create_weights_file(tmp_path / "b.h5", np.array([1, 3]), np.array([0, 1, 3]), sf)

    match, report = compare_weights(str(tmp_path / "a.h5"), str(tmp_path / "b.h5"))
    assert not match
    assert "node_ids mismatch" in report
