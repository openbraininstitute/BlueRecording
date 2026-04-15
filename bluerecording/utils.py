# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path


def get_circuit_path(path_to_simconfig: str | Path) -> str:
    """Return the absolute path to the circuit config for a given simulation config.

    The circuit path is resolved relative to the simulation config file location.

    Args:
        path_to_simconfig: Path to the simulation configuration JSON file.

    Returns:
        Absolute path to the circuit configuration file.
    """
    with open(path_to_simconfig) as f:
        circuitpath = json.load(f)['network']
    
    circuitpath = str((Path(path_to_simconfig).parent / circuitpath).resolve())
    return circuitpath


    
