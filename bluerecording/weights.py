# SPDX-License-Identifier: GPL-3.0-or-later
import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from mpi4py import MPI

from . import positions as _positions
from .circuit import init_circuit
from .physics import (
    SegmentGeometry,
    get_coeffs_dipole_reciprocity,
    get_coeffs_line_source,
    get_coeffs_objective_csd_disk,
    get_coeffs_objective_csd_plane,
    get_coeffs_objective_csd_sphere,
    get_coeffs_point_source,
    get_coeffs_reciprocity,
)
from .utils import log_rank0

DEFAULT_SIGMA = 0.277  # Extracellular conductivity in S/m
DEFAULT_ELECTRODE_CHUNK_SIZE = 50  # Max electrodes per physics call to limit peak memory


class ElectrodeType(StrEnum):
    """Recognized electrode types."""

    LINE_SOURCE = "LineSource"
    POINT_SOURCE = "PointSource"
    DIPOLE_RECIPROCITY = "DipoleReciprocity"
    RECIPROCITY = "Reciprocity"
    OBJECTIVE_CSD_SPHERE = "ObjectiveCSD_Sphere"
    OBJECTIVE_CSD_DISK = "ObjectiveCSD_Disk"
    OBJECTIVE_CSD_PLANE = "ObjectiveCSD_Plane"


@dataclass
class ObjectiveCSDParams:
    """Parameters for an objective CSD electrode type."""

    electrode_type: ElectrodeType
    radius: float | None = None
    thickness: float | None = None


@dataclass
class Electrode:
    """Metadata for a single electrode."""

    name: str
    position: np.ndarray
    type: ElectrodeType | ObjectiveCSDParams
    region: str = "NA"
    layer: str = "NA"

    @classmethod
    def from_csv(cls, electrode_csv: str) -> list["Electrode"]:
        """Read electrode metadata from a CSV file.

        The CSV must have columns ``x``, ``y``, ``z``.  Optional columns:
        ``type`` (default ``LineSource``), ``layer``, ``region``,
        ``radius``, ``thickness``.  The last two are only used for
        ObjectiveCSD electrode types.
        """
        electrode_df = pd.read_csv(electrode_csv, header=0, index_col=0)

        electrodes: list[Electrode] = []

        for i in range(len(electrode_df)):
            name = str(electrode_df.index.values[i])
            position = np.array(
                [
                    electrode_df["x"].iloc[i],
                    electrode_df["y"].iloc[i],
                    electrode_df["z"].iloc[i],
                ]
            )
            layer = electrode_df["layer"].iloc[i] if "layer" in electrode_df.columns else "NA"
            region = electrode_df["region"].iloc[i] if "region" in electrode_df.columns else "NA"

            if "type" in electrode_df.columns:
                etype = ElectrodeType(electrode_df["type"].iloc[i])
            else:
                etype = ElectrodeType.LINE_SOURCE

            if "ObjectiveCSD" in etype:
                radius = (
                    float(electrode_df["radius"].iloc[i])
                    if "radius" in electrode_df.columns and pd.notna(electrode_df["radius"].iloc[i])
                    else None
                )
                thickness = (
                    float(electrode_df["thickness"].iloc[i])
                    if "thickness" in electrode_df.columns and pd.notna(electrode_df["thickness"].iloc[i])
                    else None
                )
                electrodes.append(
                    cls(
                        name=name,
                        position=position,
                        type=ObjectiveCSDParams(electrode_type=etype, radius=radius, thickness=thickness),
                        region=region,
                        layer=layer,
                    )
                )
            else:
                electrodes.append(
                    cls(
                        name=name,
                        position=position,
                        type=etype,
                        region=region,
                        layer=layer,
                    )
                )

        return electrodes


# ---------------------------------------------------------------------------
# H5 file initialization (formerly writeH5_prelim.py)
# ---------------------------------------------------------------------------


def _write_electrode_metadata_to_h5(
    h5: h5py.File,
    node_ids: np.ndarray,
    electrodes: list[Electrode],
    population_name: str,
) -> None:
    """Write electrode metadata into an HDF5 file.

    Creates the ``node_ids`` dataset and one group per electrode containing
    its position, type, region, and layer.

    Args:
        h5: HDF5 file handle opened for writing.
        node_ids: Node IDs.
        electrodes: List of ``Electrode`` objects.
        population_name: SONATA population name.
    """
    h5.create_dataset(f"{population_name}/node_ids", data=node_ids)

    for index, electrode in enumerate(electrodes):
        prefix = f"electrodes/{electrode.name}"
        h5.create_dataset(f"{prefix}/{population_name}", data=index)
        h5.create_dataset(f"{prefix}/position", data=electrode.position)

        if isinstance(electrode.type, ObjectiveCSDParams):
            dset = h5.create_dataset(f"{prefix}/type", data=electrode.type.electrode_type.value)
            if electrode.type.radius is not None:
                dset.attrs.create("radius", electrode.type.radius)
            if electrode.type.thickness is not None:
                dset.attrs.create("thickness", electrode.type.thickness)
        else:
            h5.create_dataset(f"{prefix}/type", data=electrode.type.value)

        h5.create_dataset(f"{prefix}/region", data=electrode.region)
        h5.create_dataset(f"{prefix}/layer", data=electrode.layer)


def _get_offsets(section_ids_frame: pd.DataFrame) -> np.ndarray:
    """Compute per-node offsets into the flat segment array.

    Returns an array of length ``n_nodes + 1``: entry *i* is the index
    of the first segment for the *i*-th node, and the last entry is the
    total number of segments.
    """
    counts = section_ids_frame.groupby("id", sort=False).size().to_numpy()
    return np.hstack(([0], np.cumsum(counts)))


def _init_scaling_factors_and_offsets(
    section_ids_frame: pd.DataFrame,
    population_name: str,
    h5file: h5py.File,
    electrodes: list,
) -> None:
    """Create the scaling_factors and offsets datasets in the H5 file.

    ``scaling_factors`` is initialized to ones with shape
    ``(n_segments, n_electrodes + 1)``.  ``offsets`` maps each node to
    its segment range inside ``scaling_factors``.
    """
    n_segments = len(section_ids_frame)
    n_electrodes = len(electrodes)
    n_cols = n_electrodes + 1
    h5file.create_dataset(
        f"electrodes/{population_name}/scaling_factors",
        shape=(n_segments, n_cols),
        dtype=np.float64,
    )
    h5file.create_dataset(
        f"{population_name}/offsets",
        data=_get_offsets(section_ids_frame),
    )


def _init_weights(
    cols: np.ndarray,
    population_name: str,
    outputfile: str,
    electrodes: list[Electrode],
    with_neurite_type: bool = False,
) -> None:
    """Initialize the HDF5 electrode weights file on rank 0.

    Gathers rank-local cols via MPI, builds a rank-ordered file layout
    (rank 0's segments first, then rank 1's, etc.), and writes electrode
    metadata and offsets. The file is closed before returning.

    Args:
        cols: Rank-local (N, 2) int64 array of (gid, section) pairs.
        population_name: SONATA population name.
        outputfile: Path to the output HDF5 file.
        electrodes: Mapping of electrode name to ``Electrode`` objects.
        with_neurite_type: If True, pre-allocate a neurite_types dataset.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_count = len(cols)
    counts = np.array(comm.gather(local_count, root=0))

    all_cols = None
    if rank == 0:
        total = int(counts.sum())
        all_cols = np.empty((total, 2), dtype=np.uint64)
        recvcounts = counts * 2  # each row has 2 uint32 elements
        displacements = np.zeros(size, dtype=np.intp)
        np.cumsum(recvcounts[:-1], out=displacements[1:])
    else:
        recvcounts = None
        displacements = None

    comm.Gatherv(
        [cols, MPI.UINT64_T],
        [all_cols, recvcounts, displacements, MPI.UINT64_T] if rank == 0 else None,
        root=0,
    )

    if rank == 0:
        # Split all_cols back into per-rank arrays for node_ids construction
        offsets_per_rank = np.zeros(size + 1, dtype=np.intp)
        np.cumsum(counts, out=offsets_per_rank[1:])

        # Build node_ids in rank order: for each rank's cols, extract unique GIDs
        # (locally sorted within that rank), concatenate in rank order.
        # GIDs are disjoint across ranks (round-robin distribution).
        node_ids_parts = []
        for r in range(size):
            start_r = int(offsets_per_rank[r])
            end_r = int(offsets_per_rank[r + 1])
            if end_r > start_r:
                node_ids_parts.append(np.unique(all_cols[start_r:end_r, 0]))
        node_ids = np.concatenate(node_ids_parts) if node_ids_parts else np.array([], dtype=np.uint64)

        # Build section_ids_frame preserving rank-concatenation order
        section_ids_frame = pd.DataFrame(all_cols, columns=["id", "section"])

        Path(outputfile).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(outputfile, "w") as h5file:
            # Tune HDF5 metadata cache for faster writes
            h5id = h5file.id
            cc = h5id.get_mdc_config()
            cc.max_size = 1024 * 1024 * 124  # 124 MiB
            h5id.set_mdc_config(cc)

            _write_electrode_metadata_to_h5(h5file, node_ids, electrodes, population_name)

            _init_scaling_factors_and_offsets(section_ids_frame, population_name, h5file, electrodes)

            if with_neurite_type:
                n_compartments = len(all_cols)
                h5file.create_dataset(
                    f"{population_name}/neurite_types",
                    shape=(n_compartments,),
                    dtype=np.int32,
                )

    comm.Barrier()


# ---------------------------------------------------------------------------
# Weight computation (formerly writeH5.py)
# ---------------------------------------------------------------------------


def _add_data(
    h5: h5py.File,
    coeffs: pd.DataFrame,
    population_name: str,
    start: int,
) -> None:
    """Write computed coefficients into the scaling_factors dataset.

    Writes the full coefficient block in a single contiguous slice
    starting at row ``start``.

    Args:
        h5: HDF5 file handle opened for writing.
        coeffs: DataFrame of shape (N_electrodes, N_local_segments).
        population_name: SONATA population name.
        start: Starting row index for this rank's contiguous block.
    """
    dset = f"electrodes/{population_name}/scaling_factors"
    block = coeffs.to_numpy().T  # shape: (N_local_segments, N_electrodes)
    end = start + block.shape[0]

    h5[dset][start:end, :-1] = block
    h5[dset][start:end, -1] = 1.0


def _get_neuron_segment_midpts(position: pd.DataFrame) -> pd.DataFrame:
    """Compute segment midpoints for a single neuron.

    Soma columns (section id 0) are kept as-is. For other sections,
    consecutive boundary positions are averaged to produce midpoints.
    Single-point sections are kept unchanged.
    """
    sec_ids = np.array(list(position.columns))[:, 1]
    unique_sec_ids = np.unique(sec_ids)

    parts = []
    for sid in unique_sec_ids:
        pos = position.iloc[:, np.where(sid == sec_ids)[0]]

        if sid == 0 or pos.shape[1] == 1:
            parts.append(pos)
        else:
            parts.append((pos.iloc[:, :-1] + pos.iloc[:, 1:]) / 2)

    return pd.concat(parts, axis=1)


def _get_segment_midpts(positions: pd.DataFrame, node_ids: np.ndarray) -> pd.DataFrame:
    """Compute segment midpoints for all neurons in the position DataFrame."""
    return positions.T.groupby(level=0, group_keys=False).apply(lambda g: _get_neuron_segment_midpts(g.T).T).T


def _sort_electrode_names(electrode_keys, population_name: str):
    """Return electrode names sorted, excluding the population's scaling_factors key."""
    electrode_names = np.array(list(electrode_keys))
    electrode_names = electrode_names[electrode_names != population_name]

    electrode_list = []
    for e in electrode_names:
        try:
            name = int(e)
        except ValueError:
            name = e
        electrode_list.append(name)

    return np.sort(electrode_list)


def _parse_index_range(spec: str) -> range:
    """Parse a 'start:end' string into a range."""
    start, end = spec.split(":")
    return range(int(start), int(end))


def _get_objective_csd_array(
    electrode_type: ElectrodeType,
    objective_csd_array_indices: list[str] | None,
    objective_csd_count: int,
    electrodes_ordered: list[Electrode],
    electrode_idx: int,
) -> tuple[list[int] | range, int]:
    """Determine which electrodes belong to the objective CSD array.

    If no explicit indices are given, all electrodes matching the type
    are used. Otherwise the provided subsampling indices are applied.

    Args:
        electrode_type: The ObjectiveCSD electrode type to match.
        objective_csd_array_indices: Optional list of 'start:end' range specs.
        objective_csd_count: Running count of CSD arrays encountered so far.
        electrodes_ordered: List of Electrode objects in sorted order.
        electrode_idx: Index of the current electrode in the sorted list.
    """
    if objective_csd_array_indices is None:
        all_types = [
            e.type.electrode_type if isinstance(e.type, ObjectiveCSDParams) else e.type for e in electrodes_ordered
        ]
        array_idx = [i for i, t in enumerate(all_types) if t == electrode_type]
    else:
        array_idx = _parse_index_range(objective_csd_array_indices[objective_csd_count])
        if electrode_idx not in array_idx:
            objective_csd_count += 1
            array_idx = _parse_index_range(objective_csd_array_indices[objective_csd_count])
            if electrode_idx not in array_idx:
                raise ValueError("Electrode arrays used in objective CSD must be sequential in electrode file")

    return array_idx, objective_csd_count


def _get_electrode_type(electrode: Electrode) -> ElectrodeType:
    """Extract the ElectrodeType from an Electrode (handles ObjectiveCSDParams)."""
    if isinstance(electrode.type, ObjectiveCSDParams):
        return electrode.type.electrode_type
    return electrode.type


def get_weights(
    positions: pd.DataFrame,
    cols: np.ndarray,
    electrodes: list[Electrode] | str,
    sigma: list[float] | None = None,
    path_to_fields: list[str] | None = None,
    objective_csd_array_indices: list[str] | None = None,
    electrode_chunk_size: int = DEFAULT_ELECTRODE_CHUNK_SIZE,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """Compute electrode transfer coefficients from pre-computed positions.

    Groups electrodes by type, computes each group in chunks,
    then reassembles results in original electrode order.
    Pure computation — no file I/O.

    Args:
        positions: DataFrame of segment boundary positions (from
            ``compute_positions``).
        cols: (N, 2) int64 array of (gid, section) pairs.
        electrodes: Electrode metadata (dict or path to CSV).
        sigma: Extracellular conductivity value(s) in S/m.
        path_to_fields: Path(s) to potential/E-field files for reciprocity.
        objective_csd_array_indices: Subsampling indices for objective CSD.
        electrode_chunk_size: Max electrodes per physics call (limits peak memory).
        verbose: If True, print progress information on rank 0.

    Returns:
        DataFrame of transfer coefficients, or None if this rank has no nodes.
    """
    if sigma is None:
        sigma = [DEFAULT_SIGMA]

    if isinstance(electrodes, str):
        electrodes = Electrode.from_csv(electrodes)

    node_ids = np.unique(cols[:, 0])
    columns = pd.MultiIndex.from_arrays([cols[:, 0], cols[:, 1]], names=["id", "section"])

    if len(node_ids) == 0:
        return None

    n_electrodes = len(electrodes)
    n_segments = len(cols)

    all_coeffs = np.empty((n_electrodes, n_segments))

    line_source_indices: list[int] = []
    point_source_indices: list[int] = []
    other_indices: list[int] = []

    sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (n_electrodes,))

    for idx, elec in enumerate(electrodes):
        etype = _get_electrode_type(elec)
        if etype is ElectrodeType.LINE_SOURCE:
            line_source_indices.append(idx)
        elif etype is ElectrodeType.POINT_SOURCE:
            point_source_indices.append(idx)
        else:
            other_indices.append(idx)

    # --- Compute LINE_SOURCE ---
    if line_source_indices:
        epos_array = np.array([electrodes[i].position for i in line_source_indices])
        group_sigma = sigma_arr[line_source_indices]
        log_rank0(f"Computing line-source weights: {len(line_source_indices)} electrodes", verbose)
        geom = SegmentGeometry.from_positions(positions)
        for chunk_start in range(0, len(line_source_indices), electrode_chunk_size):
            chunk_end = min(chunk_start + electrode_chunk_size, len(line_source_indices))
            chunk_coeffs = get_coeffs_line_source(
                geom, columns, epos_array[chunk_start:chunk_end], group_sigma[chunk_start:chunk_end]
            )
            all_coeffs[line_source_indices[chunk_start:chunk_end]] = chunk_coeffs.to_numpy()

    # --- Compute POINT_SOURCE ---
    mid_positions = None
    if point_source_indices:
        mid_positions = _get_segment_midpts(positions, node_ids)
        epos_array = np.array([electrodes[i].position for i in point_source_indices])
        group_sigma = sigma_arr[point_source_indices]
        log_rank0(f"Computing point-source weights: {len(point_source_indices)} electrodes", verbose)
        for chunk_start in range(0, len(point_source_indices), electrode_chunk_size):
            chunk_end = min(chunk_start + electrode_chunk_size, len(point_source_indices))
            chunk_coeffs = get_coeffs_point_source(
                mid_positions, epos_array[chunk_start:chunk_end], group_sigma[chunk_start:chunk_end]
            )
            all_coeffs[point_source_indices[chunk_start:chunk_end]] = chunk_coeffs.to_numpy()

    reciprocity_idx = 0
    objective_csd_count = 0

    for idx in other_indices:
        electrode = electrodes[idx]
        epos = electrode.position
        electrode_type = _get_electrode_type(electrode)

        if mid_positions is None:
            mid_positions = _get_segment_midpts(positions, node_ids)

        if "ObjectiveCSD" in electrode_type:
            array_idx, objective_csd_count = _get_objective_csd_array(
                electrode_type,
                objective_csd_array_indices,
                objective_csd_count,
                electrodes,
                idx,
            )
            all_epos = [electrodes[i].position for i in array_idx]

            if isinstance(electrode.type, ObjectiveCSDParams):
                radius = electrode.type.radius
                thickness = electrode.type.thickness
            else:
                radius = None
                thickness = None

            if electrode_type is ElectrodeType.OBJECTIVE_CSD_SPHERE:
                coeffs = get_coeffs_objective_csd_sphere(mid_positions, epos, all_epos, radius)
            elif electrode_type is ElectrodeType.OBJECTIVE_CSD_DISK:
                coeffs = get_coeffs_objective_csd_disk(mid_positions, epos, all_epos, radius, thickness)
            elif electrode_type is ElectrodeType.OBJECTIVE_CSD_PLANE:
                coeffs = get_coeffs_objective_csd_plane(mid_positions, epos, all_epos, thickness)

        elif electrode_type is ElectrodeType.DIPOLE_RECIPROCITY:
            coeffs = get_coeffs_dipole_reciprocity(mid_positions, path_to_fields[reciprocity_idx])
            reciprocity_idx += 1

        else:
            coeffs = get_coeffs_reciprocity(mid_positions, path_to_fields[reciprocity_idx])
            reciprocity_idx += 1

        all_coeffs[idx] = coeffs.to_numpy().ravel()

    result = pd.DataFrame(data=all_coeffs, columns=columns)
    return result


def _write_neurite_types(
    h5: h5py.File,
    neurite_types: np.ndarray,
    population_name: str,
    start: int,
) -> None:
    """Write neurite type codes into the H5 file.

    Writes the full local neurite_types array in a single contiguous slice.

    Args:
        h5: HDF5 file handle opened for writing.
        neurite_types: (N,) int32 array of neurite type codes.
        population_name: SONATA population name.
        start: Starting row index for this rank's contiguous block.
    """
    end = start + len(neurite_types)
    h5[f"{population_name}/neurite_types"][start:end] = neurite_types


def compute_weights(
    path_to_config: str | Path,
    electrodes: list[Electrode] | str,
    replace_axons: bool = True,
    sigma: list[float] | None = None,
    path_to_fields: list[str] | None = None,
    objective_csd_array_indices: list[str] | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame, np.ndarray, np.ndarray, str]:
    """High-level API: compute weights and positions from a config file.

    Handles circuit initialization, position computation, and weight
    computation in one call. Mirrors ``positions.compute_positions``.

    Args:
        path_to_config: Path to a SONATA simulation or circuit configuration
            file.
        electrodes: Electrode metadata (dict or path to CSV).
        replace_axons: If True, replace morphological axons with a standardized
            stub.
        sigma: Extracellular conductivity value(s) in S/m.
        path_to_fields: Path(s) to potential/E-field files for reciprocity.
        objective_csd_array_indices: Subsampling indices for objective CSD.

    Returns:
        weights: DataFrame of transfer coefficients, or None if this rank
            has no nodes.
        positions_df: DataFrame of segment boundary positions.
        cols: (N, 2) int64 array of (gid, section) pairs.
        neurite_types: (N,) int32 array of neurite type codes per compartment.
        population_name: SONATA population name (needed by ``save_weights``).
    """
    node_manager, ids, cols, population, population_name, morphologies_dir = init_circuit(str(path_to_config))

    positions_df, cols, neurite_types = _positions.get_positions(
        node_manager,
        ids,
        cols,
        population,
        morphologies_dir=morphologies_dir,
        replace_axons=replace_axons,
    )

    weights = get_weights(
        positions_df,
        cols,
        electrodes,
        sigma=sigma,
        path_to_fields=path_to_fields,
        objective_csd_array_indices=objective_csd_array_indices,
    )

    return weights, positions_df, cols, neurite_types, population_name


@dataclass
class ComputeWeightsTask:
    """Specification for a single electrode weights computation.

    Used with :func:`compute_and_save_weights` to process multiple electrode
    configurations in a single circuit load.

    Args:
        electrodes: Path to an electrode CSV file, or a pre-built list of
            :class:`Electrode` objects.
        output: Path to the output HDF5 weights file.
        sigma: Extracellular conductivity value(s) in S/m. If None, uses
            the default (0.277 S/m).
        path_to_fields: Path(s) to potential/E-field files for reciprocity
            electrode types.
        with_neurite_type: If True, include a neurite_types dataset in the
            output file.
        objective_csd_array_indices: Subsampling indices for objective CSD
            electrode types.
    """

    electrodes: str | list[Electrode]
    output: str
    sigma: list[float] | None = None
    path_to_fields: list[str] | None = None
    with_neurite_type: bool = False
    objective_csd_array_indices: list[str] | None = None


def compute_and_save_weights(
    path_to_config: str | Path,
    tasks: list[ComputeWeightsTask],
    replace_axons: bool = True,
) -> None:
    """Compute and save weights for multiple electrode configurations.

    Loads the circuit and computes segment positions once, then iterates
    over each task to compute and save electrode weights. This avoids
    redundant circuit loading and position computation when generating
    weights for multiple electrode setups on the same circuit.

    Equivalent to calling :func:`compute_weights` + :func:`save_weights`
    for each electrode file, but without re-initializing NEURON each time
    (which is impossible within a single process).

    Args:
        path_to_config: Path to a SONATA simulation or circuit configuration
            file.
        tasks: List of :class:`ComputeWeightsTask` objects, each specifying an
            electrode configuration and output path.
        replace_axons: If True, replace morphological axons with a standardized
            stub (two 30 µm AIS sections + 1000 µm myelinated section).
    """
    n_tasks = len(tasks)
    log_rank0(f"compute_and_save_weights: {n_tasks} task(s)")

    node_manager, ids, cols, population, population_name, morphologies_dir = init_circuit(str(path_to_config))

    positions_df, cols, neurite_types = _positions.get_positions(
        node_manager,
        ids,
        cols,
        population,
        morphologies_dir=morphologies_dir,
        replace_axons=replace_axons,
    )

    for i, task in enumerate(tasks):
        electrodes = task.electrodes
        if isinstance(electrodes, str):
            label = Path(electrodes).name
        else:
            label = f"{len(electrodes)} electrodes"
        log_rank0(f"compute_and_save_weights: task {i + 1}/{n_tasks} — {label}")

        weights = get_weights(
            positions_df,
            cols,
            electrodes,
            sigma=task.sigma,
            path_to_fields=task.path_to_fields,
            objective_csd_array_indices=task.objective_csd_array_indices,
        )

        save_weights(
            weights,
            cols,
            population_name,
            task.output,
            electrodes,
            neurite_types=neurite_types if task.with_neurite_type else None,
        )

    log_rank0(f"compute_and_save_weights: all {n_tasks} task(s) complete")


def save_weights(
    weights: pd.DataFrame | None,
    cols: np.ndarray,
    population_name: str,
    outputfile: str,
    electrodes: list[Electrode] | str,
    neurite_types: np.ndarray | None = None,
) -> None:
    """Initialize the HDF5 weights file and write pre-computed coefficients.

    Handles MPI gather (for file structure) and parallel write using
    collective I/O. Each rank writes its contiguous block in a single
    operation. The file layout is rank-ordered (rank 0's segments first,
    then rank 1's, etc.).

    Args:
        weights: DataFrame of transfer coefficients returned by
            ``compute_weights``, or None for empty ranks.
        cols: (N, 2) array of (gid, section) pairs for this rank.
        population_name: SONATA population name.
        outputfile: Path to the output HDF5 weights file.
        electrodes: Electrode metadata (dict or path to CSV).
        neurite_types: (N,) int32 array; if provided, populates the
            neurite_types dataset.
    """
    if isinstance(electrodes, str):
        electrodes = Electrode.from_csv(electrodes)

    comm = MPI.COMM_WORLD
    t0 = MPI.Wtime()

    # 1. Initialize the file (gather + rank 0 creates structure + barrier)
    log_rank0("save_weights: initializing HDF5 file structure...")
    _init_weights(
        cols,
        population_name,
        outputfile,
        electrodes,
        with_neurite_type=neurite_types is not None,
    )
    t1 = MPI.Wtime()
    log_rank0(f"save_weights: file initialized. ({t1 - t0:.1f}s, includes MPI sync)")

    # 2. Compute each rank's contiguous row offset using MPI_Scan
    local_segments = len(cols)

    # Exclusive scan: each rank's start = sum of all previous ranks' segments
    start = comm.scan(local_segments, op=MPI.SUM) - local_segments

    # 3. Open file for parallel write — ALL ranks must participate
    if comm.Get_size() > 1 and not h5py.get_config().mpi:
        warnings.warn(
            "h5py was not built with MPI support. Parallel writes are unavailable; falling back to serial I/O.",
            stacklevel=2,
        )

    if comm.Get_size() > 1:
        h5 = h5py.File(outputfile, "a", driver="mpio", comm=comm)
    else:
        h5 = h5py.File(outputfile, "a")
    t2 = MPI.Wtime()

    # 4. Write coefficients — each rank writes its own contiguous slice.
    # All ranks must participate in the dataset access to avoid MPI-IO
    # deadlocks (even ranks with no data perform a zero-length read).
    dset = h5[f"electrodes/{population_name}/scaling_factors"]
    if weights is not None and local_segments > 0:
        block = weights.to_numpy().T  # (N_local_segments, N_electrodes)
        # Append a column of ones (the identity/normalization column)
        full_block = np.empty((block.shape[0], block.shape[1] + 1), dtype=np.float64)
        full_block[:, :-1] = block
        full_block[:, -1] = 1.0
        dset[start : start + full_block.shape[0], :] = full_block
    t3 = MPI.Wtime()

    # 5. Write neurite types if requested
    if neurite_types is not None and local_segments > 0:
        _write_neurite_types(h5, neurite_types, population_name, start=start)

    comm.Barrier()
    h5.close()
    t4 = MPI.Wtime()

    total_segments = comm.allreduce(local_segments, op=MPI.SUM)
    n_electrodes = len(electrodes)
    file_size_gb = total_segments * (n_electrodes + 1) * 8 / 1e9
    log_rank0(
        f"save_weights: done. "
        f"{total_segments:,} segments × {n_electrodes} electrodes = {file_size_gb:.1f} GB | "
        f"init {t1 - t0:.1f}s, open {t2 - t1:.1f}s, write {t3 - t2:.1f}s, close {t4 - t3:.1f}s, "
        f"total {t4 - t0:.1f}s"
    )
