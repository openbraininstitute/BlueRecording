# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared circuit initialization via neurodamus.

Provides the entry point for loading a circuit model and extracting
the discretization info (cell objects, compartment structure, morphology access)
"""

import libsonata
import numpy as np
from mpi4py import MPI

from .utils import log_rank0, resolve_simulation_config

_NEURODAMUS_OPTIONS = {
    "disable_reports": True,
    "enable_coord_mapping": True,
    "restrict_connectivity": 2,
    "simulator": "NEURON",
}


def _new_node(sim_config_obj):
    """Create a neurodamus Node for the config and load its targets.

    Only cells are then built, with ``Node.create_cells``: no synapses,
    stimuli or reports, which the weights do not depend on.
    """
    # Lazy import: neurodamus pulls in NEURON, which is not available
    # in lightweight installs (e.g. CI with --quick).
    import neurodamus

    node = neurodamus.Node(sim_config_obj, dict(_NEURODAMUS_OPTIONS))
    node.load_targets()
    return node


def _target_gids(node, circuit) -> np.ndarray:
    """Return the 0-based node IDs of the circuit's population that the simulation targets."""
    from neurodamus.target_manager import TargetSpec

    if circuit.NodesetName:
        target = node.target_manager.get_target(TargetSpec(circuit.NodesetName, circuit.PopulationName))
        return np.asarray(target.gids(raw_gids=True), dtype=np.uint32)
    size = libsonata.NodeStorage(circuit.CellLibraryFile).open_population(circuit.PopulationName).size
    return np.arange(size, dtype=np.uint32)


def _population_circuit(node):
    """Return the circuit config of the one population with targeted cells, and their node IDs."""
    targeted = {}
    for circuit in node.all_circuits():
        if circuit.get("PopulationType") == "virtual" or not isinstance(circuit.CellLibraryFile, str):
            continue
        gids = _target_gids(node, circuit)
        if len(gids):
            targeted[circuit.PopulationName] = circuit, gids
    if len(targeted) != 1:
        raise RuntimeError(
            f"Expected exactly one population with cells, got: {list(targeted)}. "
            f"Use a node_set that targets a single population."
        )
    return next(iter(targeted.values()))


def _round_robin_batches(gids: np.ndarray, cells_per_batch: int | None, n_ranks: int) -> list[np.ndarray]:
    """Split gids into batches of at most cells_per_batch cells made of whole round-robin rounds.

    neurodamus gives the i-th cell of a target to rank ``i % n_ranks``, so
    batches of whole rounds of ``n_ranks`` cells give every rank the same cells
    as a single build. Rounds are dealt to the batches in turn, so that each
    batch samples the whole circuit. A batch holds at least one round.
    """
    if not cells_per_batch or cells_per_batch >= len(gids):
        return [gids]
    rounds = np.arange(len(gids)) // n_ranks
    rounds_per_batch = max(1, cells_per_batch // n_ranks)
    n_batches = -(-(int(rounds[-1]) + 1) // rounds_per_batch)
    return [gids[rounds % n_batches == i] for i in range(n_batches)]


def _local_cells(node, circuit, sim_config_obj):
    """Return init_circuit's tuple for the cells that node has built on this rank."""
    population_name = circuit.PopulationName
    node_manager = node.circuits.get_node_manager(population_name)
    offset = node_manager.local_nodes.offset

    points = node_manager.target_manager.get_target(None).get_point_list(
        node_manager,
        libsonata.SimulationConfig.Report.Sections.all,
        libsonata.SimulationConfig.Report.Compartments.all,
    )
    # Build compartment layout with 0-based SONATA node IDs
    cols = np.array(
        [(p.gid - offset, s) for p in points for s in sorted(p.sclst_ids)],
        dtype=np.uint64,
    ).reshape(-1, 2)

    cells = list(node_manager.gid2cell.values())

    circuit_conf = libsonata.CircuitConfig.from_file(sim_config_obj.network)
    population = circuit_conf.node_population(population_name)

    return cells, cols, population, population_name, circuit.MorphologyPath


def init_circuit(path_to_config: str):
    """Initialize neurodamus and extract circuit discretization info.

    Accepts either a simulation config or a circuit config path. Internally
    builds a ``libsonata.SimulationConfig`` object in-memory (no temp files)
    and passes it directly to neurodamus.

    Args:
        path_to_config: Path to a SONATA simulation or circuit configuration file.

    Returns:
        cells: List of neurodamus cell objects on this MPI rank. Each cell
            exposes ``.raw_gid`` (0-based SONATA node ID),
            ``.local_to_global_coord_mapping``, ``.local_to_global_matrix``,
            and ``.get_sec(section_id)``.
        cols: (N, 2) uint64 array of (node_id, section) pairs describing
            every compartment on this rank. Node IDs are 0-based SONATA IDs
            (no neurodamus offset).
        population: libsonata NodePopulation, needed for morphology file
            resolution.
        population_name: Name of the SONATA node population.
        morphologies_dir: Fully resolved path to the morphologies directory,
            as provided by libsonata.
    """
    with resolve_simulation_config(path_to_config) as sim_config_obj:
        node = _new_node(sim_config_obj)
        circuit, _ = _population_circuit(node)
        node.create_cells()
        return _local_cells(node, circuit, sim_config_obj)


def iter_circuit_batches(path_to_config: str, cells_per_batch: int | None = None):
    """Build the circuit a batch of cells at a time, yielding each batch like init_circuit.

    A batch's cells are freed before the next batch is built, so at most
    ``cells_per_batch`` cells exist at once across all ranks. Each rank builds
    the same cells as with init_circuit, in node ID order within a batch.

    Args:
        path_to_config: Path to a SONATA simulation or circuit configuration file.
        cells_per_batch: Max cells built at once across all ranks (limits peak
            memory). None builds them all at once.

    Yields:
        init_circuit's (cells, cols, population, population_name, morphologies_dir)
        for this rank's cells of one batch. The cells list is emptied when the
        next batch is requested, as those cells no longer exist.
    """
    from neurodamus.core.nodeset import SelectionNodeSet
    from neurodamus.target_manager import NodesetTarget

    with resolve_simulation_config(path_to_config) as sim_config_obj:
        node = _new_node(sim_config_obj)
        circuit, gids = _population_circuit(node)
        population_name = circuit.PopulationName
        # Register every targeted cell first, so that gid offsets are those of a single build
        SelectionNodeSet(gids).register_global(population_name)

        batches = _round_robin_batches(gids, cells_per_batch, MPI.COMM_WORLD.Get_size())
        for i, batch_gids in enumerate(batches):
            if i:
                node.clear_model()
            log_rank0(f"iter_circuit_batches: building batch {i + 1}/{len(batches)} ({len(batch_gids)} cells)")
            nodeset = SelectionNodeSet(batch_gids).register_global(population_name)
            target = NodesetTarget(f"{population_name}__batch_{i}", [nodeset])
            node.target_manager.register_target(target)
            circuit.NodesetName = target.name
            node.create_cells()

            batch = _local_cells(node, circuit, sim_config_obj)
            yield batch
            batch[0].clear()

        node.clear_model()
