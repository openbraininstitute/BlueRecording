# SPDX-License-Identifier: GPL-3.0-or-later
import warnings
from dataclasses import dataclass
from enum import StrEnum

import h5py
import numpy as np
import pandas as pd
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA

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

    type: ElectrodeType
    radius: float | None = None
    thickness: float | None = None


@dataclass
class Electrode:
    """Metadata for a single electrode."""

    position: np.ndarray
    type: ElectrodeType | ObjectiveCSDParams
    region: str = "NA"
    layer: str = "NA"

    @classmethod
    def from_csv(cls, electrode_csv: str) -> dict[str, "Electrode"]:
        """Read electrode metadata from a CSV file.

        The CSV must have columns ``x``, ``y``, ``z``.  Optional columns:
        ``type`` (default ``LineSource``), ``layer``, ``region``,
        ``radius``, ``thickness``.  The last two are only used for
        ObjectiveCSD electrode types.
        """
        electrode_df = pd.read_csv(electrode_csv, header=0, index_col=0)

        electrodes: dict[str, Electrode] = {}

        for i in range(len(electrode_df)):
            name = electrode_df.index.values[i]
            position = np.array([
                electrode_df['x'].iloc[i],
                electrode_df['y'].iloc[i],
                electrode_df['z'].iloc[i],
            ])
            layer = electrode_df['layer'].iloc[i] if 'layer' in electrode_df.columns else "NA"
            region = electrode_df['region'].iloc[i] if 'region' in electrode_df.columns else "NA"

            if 'type' in electrode_df.columns:
                etype = ElectrodeType(electrode_df['type'].iloc[i])
            else:
                etype = ElectrodeType.LINE_SOURCE

            if 'ObjectiveCSD' in etype:
                radius = (
                    float(electrode_df['radius'].iloc[i])
                    if 'radius' in electrode_df.columns and pd.notna(electrode_df['radius'].iloc[i])
                    else None
                )
                thickness = (
                    float(electrode_df['thickness'].iloc[i])
                    if 'thickness' in electrode_df.columns and pd.notna(electrode_df['thickness'].iloc[i])
                    else None
                )
                electrodes[name] = cls(
                    position=position,
                    type=ObjectiveCSDParams(type=etype, radius=radius, thickness=thickness),
                    region=region,
                    layer=layer,
                )
            else:
                electrodes[name] = cls(
                    position=position, type=etype, region=region, layer=layer,
                )

        return electrodes


# ---------------------------------------------------------------------------
# H5 file initialization (formerly writeH5_prelim.py)
# ---------------------------------------------------------------------------

def write_electrode_metadata_to_h5(
    h5: h5py.File,
    node_ids: np.ndarray,
    electrodes: dict[str, Electrode],
    population_name: str,
) -> None:
    """Write electrode metadata into an HDF5 file.

    Creates the ``node_ids`` dataset and one group per electrode containing
    its position, type, region, and layer.

    Args:
        h5: HDF5 file handle opened for writing.
        node_ids: Node IDs.
        electrodes: Mapping of electrode name to ``Electrode``.
        population_name: SONATA population name.
    """
    h5.create_dataset(f"{population_name}/node_ids", data=sorted(node_ids))

    for index, (key, electrode) in enumerate(electrodes.items()):
        prefix = f"electrodes/{key}"
        h5.create_dataset(f"{prefix}/{population_name}", data=index)
        h5.create_dataset(f"{prefix}/position", data=electrode.position)

        if isinstance(electrode.type, ObjectiveCSDParams):
            dset = h5.create_dataset(f"{prefix}/type", data=electrode.type.type.value)
            if electrode.type.radius is not None:
                dset.attrs.create("radius", electrode.type.radius)
            if electrode.type.thickness is not None:
                dset.attrs.create("thickness", electrode.type.thickness)
        else:
            h5.create_dataset(f"{prefix}/type", data=electrode.type.value)

        h5.create_dataset(f"{prefix}/region", data=electrode.region)
        h5.create_dataset(f"{prefix}/layer", data=electrode.layer)

def get_offsets(section_ids_frame: pd.DataFrame) -> np.ndarray:
    """Compute per-node offsets into the flat segment array.

    Counts segments per node and returns their prefix sum (partial sum),
    with a leading zero.  The result has length ``n_nodes + 1``: entry *i*
    is the index of the first segment for the *i*-th node, and the last
    entry is the total number of segments.
    """
    _, counts = np.unique(section_ids_frame['id'].values, return_counts=True)
    return np.hstack(([0], np.cumsum(counts)))

def _init_scaling_factors_and_offsets(
    section_ids_frame: pd.DataFrame,
    population_name: str,
    h5file: h5py.File,
    electrodes: dict,
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
        data=get_offsets(section_ids_frame),
    )


def initialize_h5_file(
    cols: np.ndarray,
    population_name: str,
    outputfile: str,
    electrode_csv: str,
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
        electrode_csv: Path to the electrode CSV file.
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

        electrodes = Electrode.from_csv(electrode_csv)

        with h5py.File(outputfile, 'w') as h5file:
            # Tune HDF5 metadata cache for faster writes
            h5id = h5file.id
            cc = h5id.get_mdc_config()
            cc.max_size = 1024 * 1024 * 124  # 124 MiB
            h5id.set_mdc_config(cc)

            write_electrode_metadata_to_h5(h5file, node_ids, electrodes, population_name)

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

def add_data(
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
        h5[dset][offset0[i]:offset1[i], :-1] = coeffs.loc[:, node_id].values.T

def line_source_cases(h: float, r2: float, l: float) -> float:
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
        return np.log(((h**2 + r2)**0.5 - h) / ((l**2 + r2)**0.5 - l))
    elif h < 0 and l > 0:
        return np.log(((h**2 + r2)**0.5 - h) * (l + (l**2 + r2)**0.5) / r2)
    elif h > 0 and l > 0:
        return np.log((l + (l**2 + r2)**0.5) / ((r2 + h**2)**0.5 + h))
    else:
        raise ValueError(
            f"Unhandled line-source geometry: h={h}, l={l} "
            "(expected l > h with segment_length > 0)"
        )

def get_line_coeffs(
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

    line_source_term = line_source_cases(h, r2, l)

    seg_coeff = 1 / (4 * np.pi * sigma * seg_length) * line_source_term
    seg_coeff *= 1e-9

    return seg_coeff


def get_coeffs_line_source(
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
            coeff_list.append(
                get_line_coeffs(positions.iloc[:, i], positions.iloc[:, i + 1], electrode_pos, sigma)
            )

    coeffs = pd.DataFrame(data=np.array(coeff_list)[np.newaxis, :])
    coeffs.columns = columns
    return coeffs

def get_coeffs_point_source(
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

def get_array_spacing(all_epos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

def get_coeffs_objective_csd_sphere(
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


def get_coeffs_objective_csd_plane(
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
    main_axis, spacing = get_array_spacing(all_epos)

    if plane_thickness is None:
        plane_thickness = get_thickness(spacing)

    axial_distances, _ = distances_in_planar_coords(compartment_positions, electrode_pos, main_axis)
    coeffs = (axial_distances <= plane_thickness).astype(int).flatten()
    return pd.DataFrame(data=coeffs[np.newaxis, :], columns=compartment_positions.columns)

def get_thickness(spacing: np.ndarray) -> float:
    """Estimate plane/disk thickness as half the mean electrode spacing."""
    return np.abs(np.mean(spacing) / 2)

def calculate_axial_vectors(
    axial_distances: np.ndarray,
    main_axis: np.ndarray,
) -> np.ndarray:
    """Build per-compartment axial displacement vectors along the main axis."""
    return np.tile(main_axis.T, (len(axial_distances), 1)) * axial_distances

def distances_in_planar_coords(
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
    axial_vectors = calculate_axial_vectors(axial_distances, main_axis)
    radial_vectors = diff_vectors - axial_vectors.T
    radial_distances = np.linalg.norm(radial_vectors, axis=0)
    return np.abs(axial_distances), radial_distances


def get_coeffs_objectiveCSD_Disk(compartment_positions,electrodePos,allEpos,radius=None,diskThickness=None):
    """Compute objective CSD coefficients using a disk region.

    A segment's coefficient is 1 if it lies within both the disk radius
    and thickness, 0 otherwise. Default radius is 500 um; thickness is
    estimated from electrode spacing if not provided.
    """
    if radius is None:
        radius = 500

    main_axis, arraySpacing = get_array_spacing(allEpos)

    if diskThickness is None:
        diskThickness = get_thickness(arraySpacing)

    axialDistances, radialDistances = distances_in_planar_coords(compartment_positions,electrodePos,main_axis)

    coeffs1 = np.array((radialDistances <= radius).astype(int)).flatten()
    coeffs2 = np.array((axialDistances <= diskThickness).astype(int)).flatten()

    coeffs = coeffs1 * coeffs2

    coeffs = pd.DataFrame(data=coeffs[np.newaxis,:])

    coeffs.columns = compartment_positions.columns

    return coeffs

def get_h5_dataset(h5f, group_name, dataset_name):
    """Find and return a dataset from an HDF5 file.

    Args:
        h5f: Path to the HDF5 file.
        group_name: Group to search from (``'/'`` for root).
        dataset_name: Name of the dataset to find.

    Returns:
        numpy.ndarray: The dataset contents.
    """

    def find_dataset(name):
        """Find first object with dataset_name anywhere in the name."""
        if dataset_name in name:
            return name

    with h5py.File(h5f, 'r') as f:
        k = f[group_name].visit(find_dataset)
        return f[group_name + '/' + k][()]

def get_coeffs_dipoleReciprocity(compartment_positions, path_to_fields,center):
    """Compute dipole-reciprocity coefficients from a Sim4Life E-field file.

    Interpolates the E-field at the neural center and computes the
    transfer coefficient for each compartment via the dipole approximation.

    Args:
        compartment_positions: DataFrame of segment positions (um).
        path_to_fields: Path to the HDF5 file with the E-field.
        center: Center of the neuron population.
    """
    positionColumns = compartment_positions.columns
    compartment_positions = compartment_positions.values

    with h5py.File(path_to_fields, 'r') as f:
        for i in f['FieldGroups']:
            tmp = 'FieldGroups/' + i + '/AllFields/EM E(x,y,z,f0)/_Object/Snapshots/0/'

        Ex = get_h5_dataset(path_to_fields, tmp, 'comp0')
        Ey = get_h5_dataset(path_to_fields, tmp, 'comp1')
        Ez = get_h5_dataset(path_to_fields, tmp, 'comp2')

        for i in f['Meshes']:
            tmp = 'Meshes/'+i
            break
        x = get_h5_dataset(path_to_fields, tmp, 'axis_x')
        y = get_h5_dataset(path_to_fields, tmp, 'axis_y')
        z = get_h5_dataset(path_to_fields, tmp, 'axis_z')

        xCenter = (x[:-1]+x[1:])/2
        yCenter = (y[:-1]+y[1:])/2
        zCenter = (z[:-1]+z[1:])/2

        currentApplied = f['CurrentApplied'][()]


    compartment_positions = compartment_positions * 1e-6

    center = center * 1e-6

    compartment_positions_New = compartment_positions - center.values[:,np.newaxis]


    InterpFcnX = RegularGridInterpolator((xCenter, y, z), Ex[:, :, :, 0], method='linear')
    InterpFcnY = RegularGridInterpolator((x, yCenter, z), Ey[:, :, :, 0], method='linear')
    InterpFcnZ = RegularGridInterpolator((x, y, zCenter), Ez[:, :, :, 0], method='linear')

    XComp = InterpFcnX(center)[np.newaxis]

    YComp = InterpFcnY(center)[np.newaxis]

    ZComp = InterpFcnZ(center)[np.newaxis]


    out2rat = compartment_positions_New[0]*XComp + compartment_positions_New[1]*YComp + compartment_positions_New[2]*ZComp


    outdf = pd.DataFrame(data=(-out2rat / currentApplied), columns=positionColumns)

    return outdf

def get_coeffs_reciprocity(compartment_positions, path_to_fields):
    """Compute reciprocity coefficients from a Sim4Life potential field.

    Interpolates the potential at each compartment position and scales
    by the applied current.

    Args:
        compartment_positions: DataFrame of segment positions (um).
        path_to_fields: Path to the HDF5 file with the potential field.
    """

    with h5py.File(path_to_fields, 'r') as f:
        for i in f['FieldGroups']:
            tmp = 'FieldGroups/' + i + '/AllFields/EM Potential(x,y,z,f0)/_Object/Snapshots/0/'
        pot = get_h5_dataset(path_to_fields, tmp, 'comp0')
        for i in f['Meshes']:
            tmp = 'Meshes/'+i
            break
        x = get_h5_dataset(path_to_fields, tmp, 'axis_x')
        y = get_h5_dataset(path_to_fields, tmp, 'axis_y')
        z = get_h5_dataset(path_to_fields, tmp, 'axis_z')

        currentApplied = f['CurrentApplied'][()]

    compartment_positions *= 1e-6

    xSelect = compartment_positions.values[0]
    ySelect = compartment_positions.values[1]
    zSelect = compartment_positions.values[2]


    selections = np.array([xSelect, ySelect, zSelect]).T


    InterpFcn = RegularGridInterpolator((x, y, z), pot[:, :, :, 0], method='linear')

    out2rat = InterpFcn(selections)[np.newaxis]

    outdf = pd.DataFrame(data=(out2rat / currentApplied), columns=compartment_positions.columns)

    return outdf

def get_neuron_segment_midpts(position):
    """Compute segment midpoints for a single neuron."""


    secIds = np.array(list(position.columns))[:,1]
    uniqueSecIds = np.unique(secIds)

    for sId in uniqueSecIds:

        pos = position.iloc[:,np.where(sId == secIds)[0]]

        if sId == 0:

            newPos = pos

        elif np.shape(pos.values)[-1] == 1:
            newPos = pd.concat((newPos,pos),axis=1)

        else:
            pos = (pos.iloc[:,:-1]+pos.iloc[:,1:])/2

            newPos = pd.concat((newPos,pos),axis=1)

    return newPos

def get_segment_midpts(positions,node_ids):
    """Compute segment midpoints for all neurons in the position DataFrame."""
    newPos = (
    positions.T
        .groupby(level=0, group_keys=False)
        .apply(lambda g: get_neuron_segment_midpts(g.T).T)
        .T
    )

    return newPos



def sort_electrode_names(electrodeKeys,population_name):
    """Return electrode names sorted, excluding the population's scaling_factors key."""
    electrodeNames = np.array(list(electrodeKeys))

    electrodeNames = electrodeNames[np.where(electrodeNames!=population_name)]

    electrode_list = []

    for e in electrodeNames:

        try:
            name = int(e)

        except:
            name = e

        electrode_list.append(name)

    electrode_list = np.sort(electrode_list)

    return electrode_list


def _parse_index_range(spec):
    """Parse a 'start:end' string into a range."""
    start, end = spec.split(':')
    return range(int(start), int(end))


def get_objectiveCSD_array(electrodeType, objective_csd_array_indices,
                           objectiveCSD_count, electrodeNames, h5, electrodeIdx):
    """Determine which electrodes belong to the objective CSD array.

    If no explicit indices are given, all electrodes matching the type
    are used. Otherwise the provided subsampling indices are applied.
    """
    if objective_csd_array_indices is None:
        all_types = [
            h5['electrodes'][str(e)]['type'][()].decode()
            for e in electrodeNames
        ]
        arrayIdx = [i for i, t in enumerate(all_types) if t == electrodeType]
    else:
        arrayIdx = _parse_index_range(objective_csd_array_indices[objectiveCSD_count])
        if electrodeIdx not in arrayIdx:
            objectiveCSD_count += 1
            arrayIdx = _parse_index_range(objective_csd_array_indices[objectiveCSD_count])
            if electrodeIdx not in arrayIdx:
                raise ValueError(
                    'Electrode arrays used in objective CSD must be sequential in electrode file'
                )

    return arrayIdx, objectiveCSD_count

def write_h5_file(positions, cols, population_name, outputfile, sigma=None, path_to_fields=None, objective_csd_array_indices=None, neurite_types=None):
    """Compute and write electrode coefficients to the HDF5 weights file.

    Args:
        positions: DataFrame of segment boundary positions.
        cols: (N, 2) array of (gid, section) pairs for this rank.
        population_name: SONATA population name.
        outputfile: Path to the HDF5 weights file.
        sigma: Extracellular conductivity value(s) in S/m.
        path_to_fields: Path(s) to potential/E-field files for reciprocity.
        objective_csd_array_indices: Subsampling indices for objective CSD.
        neurite_types: (N,) int32 array from get_positions; if provided,
            populates the neurite_types dataset.
    """

    if sigma is None:
        sigma = [DEFAULT_SIGMA]

    node_ids = np.unique(cols[:, 0])
    columns = pd.MultiIndex.from_arrays([cols[:, 0], cols[:, 1]], names=["id", "section"])

    h5 = h5py.File(outputfile, 'a',driver='mpio',comm=MPI.COMM_WORLD)

    if len(node_ids)==0:

        warnings.warn('No nodes are processed on rank '+str(MPI.COMM_WORLD.Get_rank())+' Either increase or reduce the number of ranks such that it is an integer multiple of the number of position files')

        h5.close()

        return 1


    coeffList = []

    electrodeNames = sort_electrode_names(h5['electrodes'].keys(),population_name)

    reciprocityIdx = 0
    sigmaIdx = 0
    objectiveCSD_count = 0

    for electrodeIdx, electrode in enumerate(electrodeNames):

        epos = h5['electrodes'][str(electrode)]['position'][:]

        electrodeType = ElectrodeType(h5['electrodes'][str(electrode)]['type'][()].decode())

        if electrodeType is ElectrodeType.LINE_SOURCE:

            coeffs = get_coeffs_line_source(positions,columns,epos,sigma[sigmaIdx])

            if len(sigma) > 1:
                sigmaIdx += 1

        else:

            newPositions = get_segment_midpts(positions,node_ids) # For other methods, we need the segment centers, not the endpoints


            if electrodeType is ElectrodeType.POINT_SOURCE:

                coeffs = get_coeffs_point_source(newPositions, epos, sigma[sigmaIdx])

                if len(sigma) > 1:
                    sigmaIdx += 1

            elif 'ObjectiveCSD' in electrodeType:

                arrayIdx, objectiveCSD_count = get_objectiveCSD_array(electrodeType, objective_csd_array_indices, objectiveCSD_count, electrodeNames, h5, electrodeIdx)

                allEpos = []

                for e in electrodeNames[arrayIdx]:
                    allEpos.append( h5['electrodes'][str(e)]['position'][:] )

                radius = h5['electrodes'][str(electrode)]['type'].attrs.get('radius',None)
                thickness = h5['electrodes'][str(electrode)]['type'].attrs.get('thickness', None)

                if electrodeType is ElectrodeType.OBJECTIVE_CSD_SPHERE:
                    coeffs = get_coeffs_objective_csd_sphere(newPositions,epos,allEpos,radius)

                elif electrodeType is ElectrodeType.OBJECTIVE_CSD_DISK:
                    coeffs = get_coeffs_objectiveCSD_Disk(newPositions,epos,allEpos,radius,thickness)

                elif electrodeType is ElectrodeType.OBJECTIVE_CSD_PLANE:
                    coeffs = get_coeffs_objective_csd_plane(newPositions,epos,allEpos,thickness)


            else:

                if electrodeType is ElectrodeType.DIPOLE_RECIPROCITY:

                    center = newPositions.mean(axis=1)

                    coeffs = get_coeffs_dipoleReciprocity(newPositions,path_to_fields[reciprocityIdx],center)

                else:

                    coeffs = get_coeffs_reciprocity(newPositions,path_to_fields[reciprocityIdx])

                reciprocityIdx += 1


        if electrodeIdx == 0:
            coeffList = coeffs
        else:
            coeffList = pd.concat((coeffList,coeffs))

    add_data(h5,node_ids,coeffList,population_name)

    if neurite_types is not None:
        offsets = h5[population_name + '/offsets'][:]
        all_node_ids = h5[population_name + '/node_ids'][:]

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

    h5.close()

    return 0
