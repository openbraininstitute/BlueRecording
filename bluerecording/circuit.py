# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared circuit initialization via neurodamus.

Provides the entry point for loading a circuit model and extracting
the discretization info (node IDs, compartment structure, morphology access)
"""

import libsonata
import numpy as np

from .utils import resolve_simulation_config


def init_circuit(path_to_config: str):
    """Initialize neurodamus and extract circuit discretization info.

    Accepts either a simulation config or a circuit config path. Internally
    builds a ``libsonata.SimulationConfig`` object in-memory (no temp files)
    and passes it directly to neurodamus.

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

    with resolve_simulation_config(path_to_config) as sim_config_obj:
        nd = neurodamus.Neurodamus(
            sim_config_obj,
            disable_reports=True,
            enable_coord_mapping=True,
            simulator="NEURON",
        )
        node_managers = nd.circuits.node_managers
        if len(node_managers) == 1:
            node_manager = next(iter(node_managers.values()))
        elif len(node_managers) > 1:
            # Multiple populations loaded — pick the one with cells on this rank
            managers_with_cells = {name: mgr for name, mgr in node_managers.items() if len(mgr.get_final_gids()) > 0}
            if len(managers_with_cells) != 1:
                raise RuntimeError(
                    f"Expected exactly one population with cells, got: "
                    f"{list(managers_with_cells.keys())}. "
                    f"Use a node_set that targets a single population."
                )
            node_manager = next(iter(managers_with_cells.values()))
        else:
            raise RuntimeError("No node managers found.")

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

        circuit_conf = libsonata.CircuitConfig.from_file(sim_config_obj.network)
        population = circuit_conf.node_population(population_name)
        morphologies_dir = circuit_conf.node_population_properties(population_name).morphologies_dir
        alt_morphs = circuit_conf.node_population_properties(population_name).alternate_morphology_formats
        if alt_morphs:
            if "neurolucida-asc" in alt_morphs:
                morphologies_dir = alt_morphs["neurolucida-asc"]
            elif "h5v1" in alt_morphs:
                morphologies_dir = alt_morphs["h5v1"]

    return node_manager, ids, cols, population, population_name, morphologies_dir
