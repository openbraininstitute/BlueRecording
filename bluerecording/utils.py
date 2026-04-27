# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path

import libsonata


def is_circuit_config(path: str | Path) -> bool:
    """Check whether a JSON file is a circuit config (as opposed to a simulation config).

    A circuit config contains a top-level "networks" key with "nodes"/"edges",
    while a simulation config contains a "network" key pointing to a circuit config.
    """
    with open(path) as f:
        data = json.load(f)
    return "networks" in data


def get_circuit_path(path_to_config: str | Path) -> str:
    """Return the absolute path to the circuit config.

    Accepts either a simulation config or a circuit config directly.
    If the file is already a circuit config, its resolved path is returned.
    If it's a simulation config, the referenced network path is extracted.

    Args:
        path_to_config: Path to a simulation or circuit configuration JSON file.

    Returns:
        Absolute path to the circuit configuration file.
    """
    path_to_config = Path(path_to_config).resolve()
    if is_circuit_config(path_to_config):
        return str(path_to_config)
    sim_conf = libsonata.SimulationConfig.from_file(str(path_to_config))
    return sim_conf.network
