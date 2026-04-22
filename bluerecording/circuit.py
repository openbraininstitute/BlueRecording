# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared circuit initialization via neurodamus.

Provides the entry point for loading a circuit model and extracting
the discretization info (node IDs, compartment structure, morphology access)
needed by both get_positions and write_weights.
"""
import libsonata
import numpy as np

from .utils import get_circuit_path


def init_circuit(path_to_simconfig: str):
    """Initialize neurodamus and extract circuit discretization info.

    Args:
        path_to_simconfig: Path to the SONATA simulation configuration file.

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

    nd = neurodamus.Neurodamus(
        path_to_simconfig,
        disable_reports=True,
        direct_mode=True,
        build_model=True,
        enable_coord_mapping=True,
        keep_build=False,
        simulator="NEURON",
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
    ).reshape(-1, 2)

    population_name = node_manager.population_name

    circuit_conf = libsonata.CircuitConfig.from_file(
        get_circuit_path(path_to_simconfig)
    )
    population = circuit_conf.node_population(population_name)
    morphologies_dir = circuit_conf.node_population_properties(population_name).morphologies_dir

    return node_manager, ids, cols, population, population_name, morphologies_dir
