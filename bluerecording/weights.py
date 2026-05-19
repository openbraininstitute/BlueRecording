# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA

from . import positions as _positions
from .circuit import init_circuit

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
    h5.create_dataset(f"{population_name}/node_ids", data=sorted(node_ids))

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

    Counts segments per node and returns their prefix sum (partial sum),
    with a leading zero.  The result has length ``n_nodes + 1``: entry *i*
    is the index of the first segment for the *i*-th node, and the last
    entry is the total number of segments.
    """
    _, counts = np.unique(section_ids_frame["id"].values, return_counts=True)
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
    electrodes: dict[str, Electrode],
    with_neurite_type: bool = False,
) -> None:
    """Initialize the HDF5 electrode weights file on rank 0.

    Gathers rank-local cols via MPI, builds the global structure, and
    writes electrode metadata and offsets. The file is closed before
    returning.

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
        all_cols = np.concatenate(all_cols_list, axis=0)
        node_ids = np.unique(all_cols[:, 0])

        section_ids_frame = pd.DataFrame(all_cols, columns=["id", "section"])

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
    ids: np.ndarray,
    coeffs: pd.DataFrame,
    population_name: str,
) -> None:
    """Write computed coefficients into the scaling_factors dataset.

    Looks up each node's offset range and writes the corresponding
    coefficient rows into the HDF5 dataset.
    """
    dset = f"electrodes/{population_name}/scaling_factors"
    node_ids = h5[f"{population_name}/node_ids"][:]
    offsets = h5[f"{population_name}/offsets"][:]

    is_in_input = np.isin(node_ids, ids)
    nodes_in_input = node_ids[is_in_input]
    id_index = np.where(is_in_input)[0]

    offset0 = offsets[id_index]
    offset1 = np.zeros_like(offset0)

    last_offset_idx = len(offsets) - 1

    if np.any(id_index == last_offset_idx):
        last_node_idx = np.where(id_index == last_offset_idx)[0]
        offset1[last_node_idx] = len(h5[dset])

    not_last = np.where(id_index != last_offset_idx)[0]
    offset1[not_last] = offsets[id_index[not_last] + 1]

    for i, node_id in enumerate(nodes_in_input):
        h5[dset][offset0[i] : offset1[i], :-1] = coeffs.loc[:, node_id].values.T


def _line_source_cases(h: float, r2: float, l: float) -> float:
    """Return the line-source potential term for the given geometry case.

    Selects the appropriate logarithmic formula depending on the signs
    of the axial projections *h* (segment end) and *l* (segment start).

    Args:
        h: Axial projection of the electrode onto the segment direction
            relative to the segment end (m).
        r2: Squared perpendicular distance from the electrode to the
            segment axis (m²).
        l: Axial projection relative to the segment start; always
            ``h + segment_length``, so ``l > h``.
    """
    if h < 0 and l < 0:
        return np.log(((h**2 + r2) ** 0.5 - h) / ((l**2 + r2) ** 0.5 - l))
    elif h < 0 and l > 0:
        return np.log(((h**2 + r2) ** 0.5 - h) * (l + (l**2 + r2) ** 0.5) / r2)
    elif h > 0 and l > 0:
        return np.log((l + (l**2 + r2) ** 0.5) / ((r2 + h**2) ** 0.5 + h))
    else:
        raise ValueError(f"Unhandled line-source geometry: h={h}, l={l} (expected l > h with segment_length > 0)")


def _get_line_coeffs(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    electrode_pos: np.ndarray,
    sigma: float,
) -> float:
    """Compute the line-source coefficient for a single segment.

    All positions are in µm and are converted to m internally.
    The returned coefficient converts a current in nA to a potential in V.

    Args:
        start_pos: Starting position of the segment (µm).
        end_pos: Ending position of the segment (µm).
        electrode_pos: Electrode position (µm).
        sigma: Extracellular conductivity (S/m).
    """
    start_pos = start_pos * 1e-6
    end_pos = end_pos * 1e-6
    electrode_pos = electrode_pos * 1e-6

    seg_length = np.linalg.norm(start_pos - end_pos)

    # Vector from segment end to electrode
    delta = electrode_pos - end_pos
    # Segment direction (end - start)
    seg_dir = end_pos - start_pos

    h = np.dot(delta, seg_dir) / seg_length
    l = h + seg_length

    r2 = np.abs(np.dot(delta, delta) - h**2)

    line_source_term = _line_source_cases(h, r2, l)

    seg_coeff = 1 / (4 * np.pi * sigma * seg_length) * line_source_term
    seg_coeff *= 1e-9

    return seg_coeff


def _get_coeffs_line_source(
    positions: pd.DataFrame,
    columns: pd.MultiIndex,
    electrode_pos: np.ndarray,
    sigma: float,
) -> pd.DataFrame:
    """Compute line-source coefficients for all segments.

    Soma segments are treated as point sources; other segments use the
    line-source approximation between consecutive position endpoints.

    Args:
        positions: DataFrame of segment boundary positions.
        columns: MultiIndex of (gid, section) pairs for the output.
        electrode_pos: Electrode position (µm).
        sigma: Extracellular conductivity (S/m).
    """
    coeff_list = []

    for i in range(len(positions.columns) - 1):
        if positions.columns[i][-1] == 0:
            soma_pos = positions.iloc[:, i]
            dist = np.linalg.norm(soma_pos - electrode_pos) * 1e-6
            soma_coeff = 1 / (4 * np.pi * sigma * dist)
            soma_coeff *= 1e-9
            coeff_list.append(soma_coeff)

        elif positions.columns[i][-1] == positions.columns[i + 1][-1]:
            coeff_list.append(_get_line_coeffs(positions.iloc[:, i], positions.iloc[:, i + 1], electrode_pos, sigma))

    coeffs = pd.DataFrame(data=np.array(coeff_list)[np.newaxis, :])
    coeffs.columns = columns
    return coeffs


def _get_coeffs_point_source(
    positions: pd.DataFrame,
    electrode_pos: np.ndarray,
    sigma: float,
) -> pd.DataFrame:
    """Compute point-source coefficients for all segments.

    Each segment is treated as a point source. Distances are converted
    from µm to m and currents from nA to A.

    Args:
        positions: DataFrame of segment midpoint positions (µm).
        electrode_pos: Electrode position (µm).
        sigma: Extracellular conductivity (S/m).
    """
    distances = np.linalg.norm(positions.values - electrode_pos[:, np.newaxis], axis=0) * 1e-6
    coeffs = 1 / (4 * np.pi * sigma * distances)
    coeffs *= 1e-9
    return pd.DataFrame(data=coeffs[np.newaxis, :], columns=positions.columns)


def _get_array_spacing(all_epos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the main axis and inter-electrode spacing of an array.

    Uses PCA to find the principal axis, projects electrode positions
    onto it, and returns the axis and the spacing between consecutive
    projections (ignoring co-planar electrodes).

    Args:
        all_epos: Electrode positions, shape ``(n_electrodes, 3)``.

    Returns:
        main_axis: Unit vector along the principal axis, shape ``(3, 1)``.
        array_spacing: Distances between consecutive projections (> 1e-3).
    """
    pca = PCA(n_components=1)
    pca.fit(all_epos)
    main_axis = pca.components_[0] / np.linalg.norm(pca.components_[0])
    main_axis = main_axis[:, np.newaxis]

    projected = np.matmul(all_epos, main_axis).flatten()
    spacing = np.abs(np.diff(projected))
    spacing = spacing[spacing > 1e-3]

    return main_axis, spacing


def _get_coeffs_objective_csd_sphere(
    positions: pd.DataFrame,
    electrode_pos: np.ndarray,
    all_epos: np.ndarray,
    radius: float | None = None,
) -> pd.DataFrame:
    """Compute objective CSD coefficients using a spherical region.

    A segment's coefficient is 1 if it lies within the given radius
    of the electrode, 0 otherwise.

    Args:
        positions: DataFrame of segment midpoint positions (µm).
        electrode_pos: Electrode position (µm).
        all_epos: All electrode positions in the array (unused for sphere,
            kept for API consistency with disk/plane).
        radius: Sphere radius in µm (default: 10).
    """
    if radius is None:
        radius = 10

    distances = np.linalg.norm(positions.values - electrode_pos[:, np.newaxis], axis=0)
    coeffs = (distances <= radius).astype(int)
    return pd.DataFrame(data=coeffs[np.newaxis, :], columns=positions.columns)


def _get_coeffs_objective_csd_plane(
    compartment_positions: pd.DataFrame,
    electrode_pos: np.ndarray,
    all_epos: np.ndarray,
    plane_thickness: float | None = None,
) -> pd.DataFrame:
    """Compute objective CSD coefficients using an infinite plane.

    A segment's coefficient is 1 if its axial distance from the electrode
    plane is within the thickness, 0 otherwise. If no thickness is given,
    it is estimated from the inter-electrode spacing.

    Args:
        compartment_positions: DataFrame of segment midpoint positions (µm).
        electrode_pos: Electrode position (µm).
        all_epos: All electrode positions in the array.
        plane_thickness: Half-thickness of the plane in µm (default:
            estimated from electrode spacing).
    """
    main_axis, spacing = _get_array_spacing(all_epos)

    if plane_thickness is None:
        plane_thickness = _get_thickness(spacing)

    axial_distances, _ = _distances_in_planar_coords(compartment_positions, electrode_pos, main_axis)
    coeffs = (axial_distances <= plane_thickness).astype(int).flatten()
    return pd.DataFrame(data=coeffs[np.newaxis, :], columns=compartment_positions.columns)


def _get_thickness(spacing: np.ndarray) -> float:
    """Estimate plane/disk thickness as half the mean electrode spacing."""
    return np.abs(np.mean(spacing) / 2)


def _calculate_axial_vectors(
    axial_distances: np.ndarray,
    main_axis: np.ndarray,
) -> np.ndarray:
    """Build per-compartment axial displacement vectors along the main axis."""
    return np.tile(main_axis.T, (len(axial_distances), 1)) * axial_distances


def _distances_in_planar_coords(
    compartment_positions: pd.DataFrame,
    electrode_pos: np.ndarray,
    main_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose compartment positions into axial and radial distances.

    Projects each compartment's displacement from the electrode onto the
    array's main axis (axial) and the perpendicular plane (radial).

    Args:
        compartment_positions: DataFrame of segment midpoint positions (µm).
        electrode_pos: Electrode position (µm).
        main_axis: Unit vector along the array axis, shape ``(3, 1)``.

    Returns:
        axial_distances: Absolute axial distances, shape ``(n_segments, 1)``.
        radial_distances: Radial distances, shape ``(n_segments,)``.
    """
    diff_vectors = compartment_positions.values - electrode_pos[:, np.newaxis]
    axial_distances = np.matmul(diff_vectors.T, main_axis)
    axial_vectors = _calculate_axial_vectors(axial_distances, main_axis)
    radial_vectors = diff_vectors - axial_vectors.T
    radial_distances = np.linalg.norm(radial_vectors, axis=0)
    return np.abs(axial_distances), radial_distances


def _get_coeffs_objective_csd_disk(
    compartment_positions: pd.DataFrame,
    electrode_pos: np.ndarray,
    all_epos: np.ndarray,
    radius: float | None = None,
    diskThickness: float | None = None,
) -> pd.DataFrame:
    """Compute objective CSD coefficients using a disk region.

    A segment's coefficient is 1 if it lies within both the disk radius
    and thickness, 0 otherwise.

    Args:
        compartment_positions: DataFrame of segment midpoint positions (µm).
        electrode_pos: Electrode position (µm).
        all_epos: All electrode positions in the array.
        radius: Disk radius in µm (default: 500).
        diskThickness: Half-thickness of the disk in µm (default:
            estimated from electrode spacing).
    """
    if radius is None:
        radius = 500

    main_axis, spacing = _get_array_spacing(all_epos)

    if diskThickness is None:
        diskThickness = _get_thickness(spacing)

    axial_distances, radial_distances = _distances_in_planar_coords(
        compartment_positions,
        electrode_pos,
        main_axis,
    )

    radial_mask = (radial_distances <= radius).astype(int).flatten()
    axial_mask = (axial_distances <= diskThickness).astype(int).flatten()
    coeffs = radial_mask * axial_mask
    return pd.DataFrame(data=coeffs[np.newaxis, :], columns=compartment_positions.columns)


def _get_h5_dataset(h5f: str, group_name: str, dataset_name: str) -> np.ndarray:
    """Find and return a dataset from an HDF5 file.

    Searches recursively under *group_name* for the first object whose
    path contains *dataset_name*.

    Args:
        h5f: Path to the HDF5 file.
        group_name: Group to search from (``'/'`` for root).
        dataset_name: Name of the dataset to find.
    """

    def find_dataset(name):
        if dataset_name in name:
            return name

    with h5py.File(h5f, "r") as f:
        k = f[group_name].visit(find_dataset)
        return f[f"{group_name}/{k}"][()]


def _get_coeffs_dipole_reciprocity(
    compartment_positions: pd.DataFrame,
    path_to_fields: str,
    center: pd.Series,
) -> pd.DataFrame:
    """Compute dipole-reciprocity coefficients from a Sim4Life E-field file.

    Interpolates the E-field at the neural center and computes the
    transfer coefficient for each compartment via the dipole approximation.

    Args:
        compartment_positions: DataFrame of segment positions (µm).
        path_to_fields: Path to the HDF5 file with the E-field.
        center: Center of the neuron population (µm).
    """
    position_columns = compartment_positions.columns
    compartment_positions = compartment_positions.values

    with h5py.File(path_to_fields, "r") as f:
        for i in f["FieldGroups"]:
            field_group = f"FieldGroups/{i}/AllFields/EM E(x,y,z,f0)/_Object/Snapshots/0/"

        ex = _get_h5_dataset(path_to_fields, field_group, "comp0")
        ey = _get_h5_dataset(path_to_fields, field_group, "comp1")
        ez = _get_h5_dataset(path_to_fields, field_group, "comp2")

        for i in f["Meshes"]:
            mesh_group = f"Meshes/{i}"
            break
        x = _get_h5_dataset(path_to_fields, mesh_group, "axis_x")
        y = _get_h5_dataset(path_to_fields, mesh_group, "axis_y")
        z = _get_h5_dataset(path_to_fields, mesh_group, "axis_z")

        x_center = (x[:-1] + x[1:]) / 2
        y_center = (y[:-1] + y[1:]) / 2
        z_center = (z[:-1] + z[1:]) / 2

        current_applied = f["CurrentApplied"][()]

    compartment_positions = compartment_positions * 1e-6
    center = center * 1e-6
    relative_positions = compartment_positions - center.values[:, np.newaxis]

    interp_x = RegularGridInterpolator((x_center, y, z), ex[:, :, :, 0], method="linear")
    interp_y = RegularGridInterpolator((x, y_center, z), ey[:, :, :, 0], method="linear")
    interp_z = RegularGridInterpolator((x, y, z_center), ez[:, :, :, 0], method="linear")

    e_at_center = np.array(
        [
            interp_x(center)[0],
            interp_y(center)[0],
            interp_z(center)[0],
        ]
    )

    potential = (
        relative_positions[0] * e_at_center[0]
        + relative_positions[1] * e_at_center[1]
        + relative_positions[2] * e_at_center[2]
    )

    return pd.DataFrame(data=(-potential / current_applied)[np.newaxis, :], columns=position_columns)


def _get_coeffs_reciprocity(
    compartment_positions: pd.DataFrame,
    path_to_fields: str,
) -> pd.DataFrame:
    """Compute reciprocity coefficients from a Sim4Life potential field.

    Interpolates the potential at each compartment position and scales
    by the applied current.

    Args:
        compartment_positions: DataFrame of segment positions (µm).
        path_to_fields: Path to the HDF5 file with the potential field.
    """
    position_columns = compartment_positions.columns
    positions_m = compartment_positions.values * 1e-6

    with h5py.File(path_to_fields, "r") as f:
        for i in f["FieldGroups"]:
            field_group = f"FieldGroups/{i}/AllFields/EM Potential(x,y,z,f0)/_Object/Snapshots/0/"
        pot = _get_h5_dataset(path_to_fields, field_group, "comp0")
        for i in f["Meshes"]:
            mesh_group = f"Meshes/{i}"
            break
        x = _get_h5_dataset(path_to_fields, mesh_group, "axis_x")
        y = _get_h5_dataset(path_to_fields, mesh_group, "axis_y")
        z = _get_h5_dataset(path_to_fields, mesh_group, "axis_z")

        current_applied = f["CurrentApplied"][()]

    selections = positions_m.T
    interp = RegularGridInterpolator((x, y, z), pot[:, :, :, 0], method="linear")
    potential = interp(selections)[np.newaxis]

    return pd.DataFrame(data=(potential / current_applied), columns=position_columns)


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


def get_weights(
    positions: pd.DataFrame,
    cols: np.ndarray,
    electrodes: dict[str, Electrode] | str,
    sigma: list[float] | None = None,
    path_to_fields: list[str] | None = None,
    objective_csd_array_indices: list[str] | None = None,
) -> pd.DataFrame | None:
    """Compute electrode transfer coefficients from pre-computed positions.

    Dispatches to the appropriate coefficient function based on each
    electrode's type and returns the concatenated result.
    Pure computation — no file I/O.

    Args:
        positions: DataFrame of segment boundary positions (from
            ``compute_positions``).
        cols: (N, 2) int64 array of (gid, section) pairs.
        electrodes: Electrode metadata (dict or path to CSV).
        sigma: Extracellular conductivity value(s) in S/m.
        path_to_fields: Path(s) to potential/E-field files for reciprocity.
        objective_csd_array_indices: Subsampling indices for objective CSD.

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

    coeff_list = []
    electrodes_ordered = electrodes

    reciprocity_idx = 0
    sigma_idx = 0
    objective_csd_count = 0

    for electrode_idx, electrode in enumerate(electrodes_ordered):
        epos = electrode.position

        if isinstance(electrode.type, ObjectiveCSDParams):
            electrode_type = electrode.type.electrode_type
        else:
            electrode_type = electrode.type

        if electrode_type is ElectrodeType.LINE_SOURCE:
            coeffs = _get_coeffs_line_source(positions, columns, epos, sigma[sigma_idx])
            if len(sigma) > 1:
                sigma_idx += 1

        else:
            mid_positions = _get_segment_midpts(positions, node_ids)

            if electrode_type is ElectrodeType.POINT_SOURCE:
                coeffs = _get_coeffs_point_source(mid_positions, epos, sigma[sigma_idx])
                if len(sigma) > 1:
                    sigma_idx += 1

            elif "ObjectiveCSD" in electrode_type:
                array_idx, objective_csd_count = _get_objective_csd_array(
                    electrode_type,
                    objective_csd_array_indices,
                    objective_csd_count,
                    electrodes_ordered,
                    electrode_idx,
                )
                all_epos = [electrodes_ordered[i].position for i in array_idx]

                if isinstance(electrode.type, ObjectiveCSDParams):
                    radius = electrode.type.radius
                    thickness = electrode.type.thickness
                else:
                    radius = None
                    thickness = None

                if electrode_type is ElectrodeType.OBJECTIVE_CSD_SPHERE:
                    coeffs = _get_coeffs_objective_csd_sphere(mid_positions, epos, all_epos, radius)
                elif electrode_type is ElectrodeType.OBJECTIVE_CSD_DISK:
                    coeffs = _get_coeffs_objective_csd_disk(mid_positions, epos, all_epos, radius, thickness)
                elif electrode_type is ElectrodeType.OBJECTIVE_CSD_PLANE:
                    coeffs = _get_coeffs_objective_csd_plane(mid_positions, epos, all_epos, thickness)

            else:
                if electrode_type is ElectrodeType.DIPOLE_RECIPROCITY:
                    center = mid_positions.mean(axis=1)
                    coeffs = _get_coeffs_dipole_reciprocity(mid_positions, path_to_fields[reciprocity_idx], center)
                else:
                    coeffs = _get_coeffs_reciprocity(mid_positions, path_to_fields[reciprocity_idx])
                reciprocity_idx += 1

        coeff_list.append(coeffs)

    return pd.concat(coeff_list) if len(coeff_list) > 1 else coeff_list[0]


def _write_neurite_types(
    h5: h5py.File,
    cols: np.ndarray,
    node_ids: np.ndarray,
    neurite_types: np.ndarray,
    population_name: str,
) -> None:
    """Write neurite type codes into the H5 file for each node."""
    offsets = h5[f"{population_name}/offsets"][:]
    all_node_ids = h5[f"{population_name}/node_ids"][:]

    for gid in node_ids:
        gid_mask = cols[:, 0] == gid
        ntypes = neurite_types[gid_mask]

        id_index = np.where(all_node_ids == gid)[0][0]
        offset0 = offsets[id_index]
        if id_index == len(offsets) - 1:
            offset1 = h5[f"electrodes/{population_name}/scaling_factors"].shape[0]
        else:
            offset1 = offsets[id_index + 1]

        h5[f"{population_name}/neurite_types"][offset0:offset1] = ntypes


def compute_weights(
    path_to_config: str | Path,
    electrodes: dict[str, Electrode] | str,
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
    electrodes: dict[str, Electrode] | str,
    neurite_types: np.ndarray | None = None,
) -> None:
    """Initialize the HDF5 weights file and write pre-computed coefficients.

    Handles MPI gather (for file structure) and parallel write.

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

    node_ids = np.unique(cols[:, 0])

    # 1. Initialize the file (gather + rank 0 creates structure + barrier)
    _init_weights(
        cols,
        population_name,
        outputfile,
        electrodes,
        with_neurite_type=neurite_types is not None,
    )

    # 2. Write coefficients in parallel
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        h5 = h5py.File(outputfile, "a", driver="mpio", comm=comm)
    else:
        h5 = h5py.File(outputfile, "a")

    if len(node_ids) == 0:
        h5.close()
        return

    assert weights is not None
    _add_data(h5, node_ids, weights, population_name)

    if neurite_types is not None:
        _write_neurite_types(h5, cols, node_ids, neurite_types, population_name)

    h5.close()
