# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared circuit initialization via neurodamus.

Provides the entry point for loading a circuit model and extracting
the discretization info (node IDs, compartment structure, morphology access)
needed by both get_positions and write_weights.
"""

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import libsonata
import numpy as np

from .utils import get_circuit_path, is_circuit_config

_MINIMAL_SIM_TEMPLATE = {
    "run": {"tstop": 0, "dt": 0.025, "random_seed": 1},
    "conditions": {"v_init": -65},
}


@contextmanager
def _ensure_simconfig(path_to_config: str):
    """Yield a path to a simulation config, creating a temporary one if needed.

    If *path_to_config* is already a simulation config, yield it as-is.
    If it's a circuit config, generate a minimal temporary simulation config
    that references it so neurodamus can load the circuit, and clean it up
    on exit.
    """
    path = Path(path_to_config).resolve()
    if not is_circuit_config(path):
        yield str(path)
        return

    sim = dict(_MINIMAL_SIM_TEMPLATE)
    sim["network"] = str(path)

    with tempfile.NamedTemporaryFile(mode="w", suffix="_sim_config.json", delete=False, dir=path.parent) as tmp:
        json.dump(sim, tmp, indent=2)
    try:
        yield tmp.name
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def init_circuit(path_to_config: str):
    """Initialize neurodamus and extract circuit discretization info.

    Accepts either a SONATA simulation config or a circuit config directly.
    When a circuit config is provided, a minimal simulation config is
    generated automatically so that neurodamus can load the circuit.

    Args:
        path_to_config: Path to a SONATA simulation or circuit configuration file.

    Returns:
        node_manager: The neurodamus node manager for the single population.
        ids: GIDs assigned to this MPI rank.
        cols: (N, 2) int64 array of (gid, section) pairs describing every
            compartment on this rank.
        population: libsonata NodePopulation, needed for morphology file
            resolution.
        population_name: Name of the SONATA node population.
        morphologies_dir: Fully resolved path to the morphologies directory,
            as provided by libsonata.
    """
    # Lazy import: neurodamus pulls in NEURON, which is not available
    # in lightweight installs (e.g. CI with --quick).
    import neurodamus

    with _ensure_simconfig(path_to_config) as simconfig_path:
        nd = neurodamus.Neurodamus(
            simconfig_path,
            disable_reports=True,
            direct_mode=True,
            build_model=True,
            enable_coord_mapping=True,
            keep_build=False,
            simulator="NEURON",
        )
        assert len(nd.circuits.node_managers) == 1, "Multiple or no node managers are not allowed for the moment"
        node_manager = next(iter(nd.circuits.node_managers.values()))

        ids = node_manager.get_final_gids()
        points = node_manager.target_manager.get_target(None).get_point_list(
            node_manager,
            libsonata.SimulationConfig.Report.Sections.all,
            libsonata.SimulationConfig.Report.Compartments.all,
        )
        cols = np.array(
            [(p.gid, s) for p in points for s in sorted(p.sclst_ids)],
            dtype=np.int64,
        ).reshape(-1, 2)

        population_name = node_manager.population_name

        circuit_conf = libsonata.CircuitConfig.from_file(get_circuit_path(path_to_config))
        population = circuit_conf.node_population(population_name)
        morphologies_dir = circuit_conf.node_population_properties(population_name).morphologies_dir

    return node_manager, ids, cols, population, population_name, morphologies_dir
