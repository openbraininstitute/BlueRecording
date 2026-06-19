# SPDX-License-Identifier: GPL-3.0-or-later
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
    get_coeffs_dipole_reciprocity,
    get_coeffs_line_source_batch,
    get_coeffs_objective_csd_disk,
    get_coeffs_objective_csd_plane,
    get_coeffs_objective_csd_sphere,
    get_coeffs_point_source_batch,
    get_coeffs_reciprocity,
)

DEFAULT_SIGMA = 0.277  # Extracellular conductivity in S/m


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

    Counts segments per node in the order they appear (preserving
    rank-concatenation order) and returns their prefix sum (partial sum),
    with a leading zero.  The result has length ``n_nodes + 1``: entry *i*
    is the index of the first segment for the *i*-th node, and the last
    entry is the total number of segments.

    Uses pandas groupby to preserve appearance order (stable), unlike
    np.unique which sorts globally.
    """
    # Group by 'id' in order of first appearance, count segments per node
    counts = section_ids_frame.groupby("id", sort=False).size().values
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
    h5file.create_dataset(
        f"electrodes/{population_name}/scaling_factors",
        data=np.ones((n_segments, n_electrodes + 1)),
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

    # Gather all rank-local cols to rank 0
    all_cols_list = comm.gather(cols, root=0)

    if rank == 0:
        # Keep all_cols in rank-concatenation order (rank 0 first, rank 1 next, etc.)
        all_cols = np.concatenate(all_cols_list, axis=0)

        # Build node_ids in rank order: for each rank's cols, extract unique GIDs
        # (locally sorted within that rank), concatenate in rank order.
        # Use dict.fromkeys to preserve first-appearance order across ranks.
        seen = {}
        for rank_cols in all_cols_list:
            if len(rank_cols) > 0:
                # Unique GIDs for this rank, locally sorted
                rank_gids = np.unique(rank_cols[:, 0])
                for gid in rank_gids:
                    if gid not in seen:
                        seen[gid] = True
        node_ids = np.array(list(seen.keys()), dtype=np.int64)

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
    block = coeffs.values.T  # shape: (N_local_segments, N_electrodes)
    end = start + block.shape[0]

    try:
        with h5[dset].collective:
            h5[dset][start:end, :-1] = block
    except AttributeError:
        # Non-parallel h5py (single-rank / unit tests without mpio)
        h5[dset][start:end, :-1] = block


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
    verbose: bool = True,
) -> pd.DataFrame | None:
    """Compute electrode transfer coefficients from pre-computed positions.

    Groups electrodes by (type, sigma), computes each group as a batch,
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

    # Result array: (N_electrodes, N_segments), filled per group then reordered
    all_coeffs = np.empty((n_electrodes, n_segments))

    # --- Group electrodes by type ---
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

    # --- Batch compute LINE_SOURCE ---
    if line_source_indices:
        epos_array = np.array([electrodes[i].position for i in line_source_indices])
        group_sigma = sigma_arr[line_source_indices]
        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Computing line-source weights: {len(line_source_indices)} electrodes")
        batch_coeffs = get_coeffs_line_source_batch(positions, columns, epos_array, group_sigma, verbose=verbose)
        all_coeffs[line_source_indices] = batch_coeffs.values

    # --- Batch compute POINT_SOURCE ---
    mid_positions = None
    if point_source_indices:
        mid_positions = _get_segment_midpts(positions, node_ids)
        epos_array = np.array([electrodes[i].position for i in point_source_indices])
        group_sigma = sigma_arr[point_source_indices]
        if verbose and MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Computing point-source weights: {len(point_source_indices)} electrodes")
        batch_coeffs = get_coeffs_point_source_batch(mid_positions, columns, epos_array, group_sigma, verbose=verbose)
        all_coeffs[point_source_indices] = batch_coeffs.values

    # --- Process remaining electrode types one by one ---
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
            center = mid_positions.mean(axis=1)
            coeffs = get_coeffs_dipole_reciprocity(mid_positions, path_to_fields[reciprocity_idx], center)
            reciprocity_idx += 1

        else:
            coeffs = get_coeffs_reciprocity(mid_positions, path_to_fields[reciprocity_idx])
            reciprocity_idx += 1

        all_coeffs[idx] = coeffs.values.ravel()

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
    try:
        with h5[f"{population_name}/neurite_types"].collective:
            h5[f"{population_name}/neurite_types"][start:end] = neurite_types
    except AttributeError:
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

    # 1. Initialize the file (gather + rank 0 creates structure + barrier)
    _init_weights(
        cols,
        population_name,
        outputfile,
        electrodes,
        with_neurite_type=neurite_types is not None,
    )

    # 2. Compute each rank's contiguous row offset using MPI_Scan
    comm = MPI.COMM_WORLD
    local_segments = len(cols)

    # Exclusive scan: each rank's start = sum of all previous ranks' segments
    start = comm.scan(local_segments, op=MPI.SUM) - local_segments

    # 3. Open file for parallel write — ALL ranks must participate
    if comm.Get_size() > 1:
        h5 = h5py.File(outputfile, "a", driver="mpio", comm=comm)
    else:
        h5 = h5py.File(outputfile, "a")

    # 4. Write coefficients — all ranks participate (empty ranks do zero-length write)
    if weights is not None and local_segments > 0:
        _add_data(h5, weights, population_name, start=start)
    else:
        # Empty rank: still must participate in collective I/O
        dset = f"electrodes/{population_name}/scaling_factors"
        try:
            with h5[dset].collective:
                h5[dset][start:start, :-1] = np.empty((0, h5[dset].shape[1] - 1))
        except AttributeError:
            pass  # Single-rank, no-op

    # 5. Write neurite types if requested
    if neurite_types is not None:
        if local_segments > 0:
            _write_neurite_types(h5, neurite_types, population_name, start=start)
        else:
            # Empty rank: participate in collective I/O
            try:
                with h5[f"{population_name}/neurite_types"].collective:
                    h5[f"{population_name}/neurite_types"][start:start] = np.empty((0,), dtype=np.int32)
            except AttributeError:
                pass

    h5.close()
