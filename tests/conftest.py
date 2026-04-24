# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).parent
ROOT_DIR = TESTS_DIR.parent
EXAMPLE_SINGLE_CELL = ROOT_DIR / "examples" / "single_cell_l5_tpc"
EXAMPLE_SSCX_100 = ROOT_DIR / "examples" / "sscx_100_cells"


@pytest.fixture(scope="session")
def path_to_simconfig_with_atlas():
    return str(TESTS_DIR / "data" / "simulation_config.json")


@pytest.fixture(scope="session")
def path_to_example_folder():
    return EXAMPLE_SINGLE_CELL


@pytest.fixture(scope="session")
def path_to_simconfig_with_output(path_to_example_folder):
    return str(path_to_example_folder / "simulation_config_near.json")


@pytest.fixture(scope="session")
def expected_path_to_morph(path_to_example_folder):
    morph_path = "configuration/components/morphologies/ascii"
    morph_file = "dend-rat_P16_S1_RH3_20140129_axon-C060110A5_-_Scale_x1.000_y0.975_z1.000_-_Clone_0.asc"
    return str(path_to_example_folder / morph_path / morph_file)


@pytest.fixture(scope="session")
def expected_circuit_path(path_to_example_folder):
    return str(path_to_example_folder / "configuration" / "circuit_config.json")
