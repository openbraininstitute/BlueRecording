# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared circuit initialization via neurodamus.

Provides the entry point for loading a circuit model and extracting
the discretization info (node IDs, compartment structure, morphology access)
needed by both get_positions and write_weights.
"""
import json
from contextlib import contextmanager
from pathlib import Path

import bluepysnap as bp
import libsonata
import neurodamus
import numpy as np


@contextmanager
def _prepare_simconfig(path_to_simconfig: str):
    """Context manager yielding a simulation config path for circuit initialization.

    If the config contains ``run.electrodes_file``, a temporary copy is written
    next to the original (reports removed, tstop set to dt) so that neurodamus
    builds the circuit without trying to load electrode weights. The temporary
    file is always removed on exit, even if neurodamus raises.

    Otherwise the original path is yielded unchanged.
    """
    config_path = Path(path_to_simconfig).resolve()

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    if "electrodes_file" not in config.get("run", {}):
        yield str(config_path)
        return

    del config["run"]["electrodes_file"]
    config.pop("reports", None)
    config["run"]["tstop"] = config["run"].get("dt", 0.025)

    stem = config_path.stem
    suffix = config_path.suffix
    base_path = config_path.parent / f".{stem}_bluerecording_base{suffix}"

    if base_path.exists():
        raise FileExistsError(
            f"Temporary config file already exists: {base_path}. "
            "Remove it manually if it is not needed."
        )

    try:
        with open(base_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        yield str(base_path)
    finally:
        base_path.unlink(missing_ok=True)


def init_circuit(path_to_simconfig: str):
    """Initialize neurodamus and extract circuit discretization info.

    Args:
        path_to_simconfig: Path to the SONATA simulation configuration file.

    Returns:
        node_manager: The neurodamus node manager for the single population.
        ids: GIDs assigned to this MPI rank.
        cols: (N, 2) int64 array of (gid, section) pairs describing every
            compartment on this rank.
        population: bluepysnap NodePopulation, needed for morphology file
            resolution.
        population_name: Name of the SONATA node population.
    """
    with _prepare_simconfig(path_to_simconfig) as effective_config:
        nd = neurodamus.Neurodamus(
            effective_config,
            disable_reports=True,
            direct_mode=True,
            build_model=True,
            enable_coord_mapping=True,
        )
    assert len(nd.circuits.node_managers) == 1, (
        "Multiple or no node managers are not allowed for the moment"
    )
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
    )

    r_sim = bp.Simulation(path_to_simconfig)
    population_name = node_manager.population_name
    population = r_sim.circuit.nodes[population_name]

    return node_manager, ids, cols, population, population_name
