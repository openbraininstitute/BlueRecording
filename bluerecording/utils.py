# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path

import libsonata


def get_circuit_path(path_to_simconfig: str | Path) -> str:
    """Return the absolute path to the circuit config for a given simulation config.

    Uses libsonata to resolve the network path, including manifest variable
    expansion and relative path resolution.

    Args:
        path_to_simconfig: Path to the simulation configuration JSON file.

    Returns:
        Absolute path to the circuit configuration file.
    """
    sim_conf = libsonata.SimulationConfig.from_file(str(path_to_simconfig))
    return sim_conf.network
