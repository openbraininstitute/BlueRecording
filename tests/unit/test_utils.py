# SPDX-License-Identifier: GPL-3.0-or-later
import pytest

from bluerecording.utils import get_circuit_path, resolve_output_path


def test_getCircuitPath(path_to_simconfig_with_output, expected_circuit_path):
    assert get_circuit_path(path_to_simconfig_with_output) == expected_circuit_path


def test_resolve_output_path_directory(tmp_path):
    """When given a directory, appends weights.h5."""
    result = resolve_output_path(tmp_path)
    assert result == tmp_path / "weights.h5"


def test_resolve_output_path_no_suffix(tmp_path):
    """When given a path without extension, treats it as a directory."""
    target = tmp_path / "my_output"
    result = resolve_output_path(target)
    assert result == target / "weights.h5"
    assert target.is_dir()


def test_resolve_output_path_h5_file(tmp_path):
    """When given a .h5 path, returns it directly."""
    target = tmp_path / "custom.h5"
    result = resolve_output_path(target)
    assert result == target


def test_resolve_output_path_custom_default_filename(tmp_path):
    """Respects a custom default_filename."""
    result = resolve_output_path(tmp_path, default_filename="output.h5")
    assert result == tmp_path / "output.h5"


def test_resolve_output_path_invalid_extension(tmp_path):
    """Raises ValueError for non-.h5 extensions."""
    with pytest.raises(ValueError, match="must be a directory or an .h5 file"):
        resolve_output_path(tmp_path / "bad.txt")


def test_resolve_output_path_creates_parent(tmp_path):
    """Creates parent directories for a .h5 file path."""
    target = tmp_path / "nested" / "dir" / "weights.h5"
    result = resolve_output_path(target)
    assert result == target
    assert target.parent.is_dir()
