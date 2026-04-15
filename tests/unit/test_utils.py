

# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
from bluerecording.utils import *

def test_getCircuitPath(path_to_simconfig_with_output,expected_circuit_path):

    assert get_circuit_path(path_to_simconfig_with_output)==expected_circuit_path

