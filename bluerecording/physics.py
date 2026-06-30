# SPDX-License-Identifier: GPL-3.0-or-later
"""Physics models for extracellular potential computation.

Contains the mathematical formulations for computing transfer coefficients
between neural segments and electrodes: line-source, point-source,
reciprocity, dipole-reciprocity, and objective CSD methods.
"""

from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .utils import log_rank0
from sklearn.decomposition import PCA


@dataclass
class SegmentGeometry:
    """Precomputed segment geometry for vectorized line-source computation."""

    start_pos: np.ndarray  # (N_line_segments, 3) segment start positions (µm)
    end_pos: np.ndarray  # (N_line_segments, 3) segment end positions (µm)
    seg_lengths: np.ndarray  # (N_line_segments,) lengths (µm)
    seg_dirs: np.ndarray  # (N_line_segments, 3) unit direction vectors
    is_soma: np.ndarray  # (N_total_segments,) bool, True for soma entries
    soma_positions: np.ndarray  # (N_soma, 3) soma positions (µm)

    @classmethod
    def from_positions(cls, positions: pd.DataFrame) -> "SegmentGeometry":
        """Build geometry from a positions DataFrame.

        Extracts start/end positions for line-source segments and soma
        positions for point-source segments, along with derived quantities
        (lengths, direction vectors).

        The positions DataFrame has a MultiIndex column ``(gid, section_id)``.
        - If section_id == 0, it's a soma (point source).
        - If consecutive columns share the same section_id, they form a
          line-source segment boundary pair.

        Args:
            positions: DataFrame of segment boundary positions (µm), shape
                ``(3, N_columns)`` with MultiIndex columns ``(gid, section_id)``.
        """
        n_cols = len(positions.columns)
        col_section_ids = np.array([c[-1] for c in positions.columns])

        soma_positions_list = []
        start_positions_list = []
        end_positions_list = []

        is_soma_list = []

        i = 0
        while i < n_cols:
            section_id = col_section_ids[i]

            if section_id == 0:
                soma_positions_list.append(positions.iloc[:, i].values)
                is_soma_list.append(True)
                i += 1
            elif i + 1 < n_cols and col_section_ids[i] == col_section_ids[i + 1]:
                start_positions_list.append(positions.iloc[:, i].values)
                end_positions_list.append(positions.iloc[:, i + 1].values)
                is_soma_list.append(False)
                i += 1
            else:
                i += 1

        is_soma = np.array(is_soma_list, dtype=bool)

        if soma_positions_list:
            soma_positions = np.array(soma_positions_list)
        else:
            soma_positions = np.empty((0, 3), dtype=np.float64)

        if start_positions_list:
            start_pos = np.array(start_positions_list)
            end_pos = np.array(end_positions_list)

            diff = end_pos - start_pos
            seg_lengths = np.linalg.norm(diff, axis=1)

            safe_lengths = np.where(seg_lengths > 0, seg_lengths, 1.0)
            seg_dirs = diff / safe_lengths[:, np.newaxis]
        else:
            start_pos = np.empty((0, 3), dtype=np.float64)
            end_pos = np.empty((0, 3), dtype=np.float64)
            seg_lengths = np.empty((0,), dtype=np.float64)
            seg_dirs = np.empty((0, 3), dtype=np.float64)

        return cls(
            start_pos=start_pos,
            end_pos=end_pos,
            seg_lengths=seg_lengths,
            seg_dirs=seg_dirs,
            is_soma=is_soma,
            soma_positions=soma_positions,
        )


def precompute_segment_geometry(positions: pd.DataFrame) -> SegmentGeometry:
    """Precompute segment geometry arrays. Use ``SegmentGeometry.from_positions()`` instead."""
    return SegmentGeometry.from_positions(positions)



def get_coeffs_line_source(
    geom: SegmentGeometry,
    columns: pd.MultiIndex,
    electrode_positions: np.ndarray,
    sigma: float | np.ndarray,
    verbose: bool = True,
    log_every: int = 50,
) -> pd.DataFrame:
    """Compute line-source coefficients for one or more electrodes.

    Soma segments are treated as point sources; other segments use the
    line-source approximation. Operates on precomputed geometry so that
    the caller can parse positions once and reuse across electrodes.

    Accepts a single electrode position ``(3,)`` or multiple ``(N_elec, 3)``.

    Args:
        geom: Precomputed segment geometry from ``SegmentGeometry.from_positions()``.
        columns: MultiIndex of (gid, section) pairs for the output.
        electrode_positions: Electrode position(s) in µm. Shape ``(3,)`` for a
            single electrode or ``(N_elec, 3)`` for multiple.
        sigma: Extracellular conductivity (S/m). Scalar or array of shape ``(N_elec,)``.
        verbose: If True, print progress on rank 0 (only for multiple electrodes).
        log_every: Print progress every N electrodes.

    Returns:
        DataFrame of shape ``(N_elec, N_segments)`` with columns matching ``columns``.
        For a single electrode, N_elec is 1.
    """
    electrode_positions = np.asarray(electrode_positions, dtype=np.float64)
    if electrode_positions.ndim == 1:
        electrode_positions = electrode_positions[np.newaxis, :]

    n_elec = len(electrode_positions)
    sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (n_elec,))

    is_soma = geom.is_soma
    n_total = len(is_soma)
    all_coeffs = np.empty((n_elec, n_total))

    if np.any(is_soma):
        all_coeffs[:, is_soma] = _point_source_coeffs(geom.soma_positions, electrode_positions, sigma_arr)

    line_mask = ~is_soma
    if np.any(line_mask):
        all_coeffs[:, line_mask] = _line_source_coeffs(
            geom.start_pos, geom.end_pos, geom.seg_lengths, electrode_positions, sigma_arr
        )

    # No unit conversion needed: µm + S/m → mV/nA directly

    if n_elec > 1:
        log_rank0(f"  Line-source: computed {n_elec} electrodes", verbose)

    return pd.DataFrame(data=all_coeffs, columns=columns)


def _line_source_coeffs(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    seg_lengths: np.ndarray,
    electrode_positions: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Compute line-source coefficients for segments × electrodes.

    All positions in µm, sigma in S/m. Result in mV/nA.

    Args:
        start_pos: Segment start positions (µm), shape ``(N_line, 3)``.
        end_pos: Segment end positions (µm), shape ``(N_line, 3)``.
        seg_lengths: Segment lengths (µm), shape ``(N_line,)``.
        electrode_positions: Electrode positions (µm), shape ``(N_elec, 3)``.
        sigma: Conductivity per electrode (S/m), shape ``(N_elec,)``.

    Returns:
        Coefficients array (mV/nA), shape ``(N_elec, N_line)``.
    """
    seg_dir = end_pos - start_pos

    # (N_line, N_elec, 3) = (1, N_elec, 3) - (N_line, 1, 3)
    delta = electrode_positions[np.newaxis, :, :] - end_pos[:, np.newaxis, :]

    h = np.sum(delta * seg_dir[:, np.newaxis, :], axis=2) / seg_lengths[:, np.newaxis]
    l = h + seg_lengths[:, np.newaxis]

    delta_sq = np.sum(delta * delta, axis=2)
    r2 = np.abs(delta_sq - h**2)

    sqrt_h2_r2 = np.sqrt(h**2 + r2)
    sqrt_l2_r2 = np.sqrt(l**2 + r2)

    # Case 1: h < 0, l < 0
    case1 = np.log((sqrt_h2_r2 - h) / (sqrt_l2_r2 - l))
    # Case 2: h < 0, l > 0
    case2 = np.log((sqrt_h2_r2 - h) * (l + sqrt_l2_r2) / r2)
    # Case 3: h > 0, l > 0
    case3 = np.log((l + sqrt_l2_r2) / (sqrt_h2_r2 + h))

    line_source_term = np.where(
        h < 0,
        np.where(l < 0, case1, case2),
        case3,
    )

    coeffs = 1 / (4 * np.pi * sigma[np.newaxis, :] * seg_lengths[:, np.newaxis]) * line_source_term
    return coeffs.T


def _point_source_coeffs(
    positions_um: np.ndarray,
    electrode_positions_um: np.ndarray,
    sigma: float | np.ndarray,
) -> np.ndarray:
    """Compute point-source coefficients for positions × electrodes.

    All positions in µm, sigma in S/m. Result in mV/nA.

    Args:
        positions_um: Segment positions (µm), shape ``(N_points, 3)``.
        electrode_positions_um: Electrode positions (µm), shape ``(N_elec, 3)``.
        sigma: Extracellular conductivity (S/m). Scalar or array of shape ``(N_elec,)``.

    Returns:
        Coefficients array (mV/nA), shape ``(N_elec, N_points)``.
    """
    # (N_points, 1, 3) - (1, N_elec, 3) → (N_points, N_elec, 3)
    delta = positions_um[:, np.newaxis, :] - electrode_positions_um[np.newaxis, :, :]
    dist = np.linalg.norm(delta, axis=2)
    if np.ndim(sigma) == 0:
        coeffs = 1 / (4 * np.pi * sigma * dist)
    else:
        sigma_row = np.asarray(sigma)[np.newaxis, :]
        coeffs = 1 / (4 * np.pi * sigma_row * dist)
    return coeffs.T


def get_coeffs_point_source(
    positions: pd.DataFrame,
    electrode_positions: np.ndarray,
    sigma: float | np.ndarray,
) -> pd.DataFrame:
    """Compute point-source coefficients for one or more electrodes.

    Each segment is treated as a point source at its midpoint.
    Accepts a single electrode position ``(3,)`` or multiple ``(N_elec, 3)``.

    Args:
        positions: DataFrame of segment midpoint positions (µm), shape ``(3, N_segments)``.
        electrode_positions: Electrode position(s) in µm. Shape ``(3,)`` for a
            single electrode or ``(N_elec, 3)`` for multiple.
        sigma: Extracellular conductivity (S/m). Scalar or array of shape ``(N_elec,)``.

    Returns:
        DataFrame of shape ``(N_elec, N_segments)``.
        For a single electrode, N_elec is 1.
    """
    electrode_positions = np.asarray(electrode_positions, dtype=np.float64)
    if electrode_positions.ndim == 1:
        electrode_positions = electrode_positions[np.newaxis, :]

    positions_um = positions.values.T
    coeffs = _point_source_coeffs(positions_um, electrode_positions, sigma)
    return pd.DataFrame(data=coeffs, columns=positions.columns)


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


def get_coeffs_objective_csd_sphere(
    positions: pd.DataFrame,
    electrode_pos: np.ndarray,
    all_epos: np.ndarray,
    radius: float | None = None,
) -> pd.DataFrame:
    """Compute objective CSD coefficients using a spherical region.

    A segment's coefficient is 1 if it lies within the given radius
    of the electrode, 0 otherwise.
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
    plane is within the thickness, 0 otherwise.
    """
    main_axis, spacing = _get_array_spacing(all_epos)

    if plane_thickness is None:
        plane_thickness = _get_thickness(spacing)

    axial_distances, _ = _distances_in_planar_coords(compartment_positions, electrode_pos, main_axis)
    coeffs = (axial_distances <= plane_thickness).astype(int).flatten()
    return pd.DataFrame(data=coeffs[np.newaxis, :], columns=compartment_positions.columns)


def get_coeffs_objective_csd_disk(
    compartment_positions: pd.DataFrame,
    electrode_pos: np.ndarray,
    all_epos: np.ndarray,
    radius: float | None = None,
    diskThickness: float | None = None,
) -> pd.DataFrame:
    """Compute objective CSD coefficients using a disk region.

    A segment's coefficient is 1 if it lies within both the disk radius
    and thickness, 0 otherwise.
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
    """

    def find_dataset(name):
        if dataset_name in name:
            return name

    with h5py.File(h5f, "r") as f:
        k = f[group_name].visit(find_dataset)
        return f[f"{group_name}/{k}"][()]


def get_coeffs_dipole_reciprocity(
    compartment_positions: pd.DataFrame,
    path_to_fields: str,
) -> pd.DataFrame:
    """Compute dipole-reciprocity coefficients from a Sim4Life E-field file.

    Interpolates the E-field at the neural centroid and computes the
    transfer coefficient for each compartment via the dipole approximation.
    """
    position_columns = compartment_positions.columns
    center = compartment_positions.mean(axis=1)
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


def get_coeffs_reciprocity(
    compartment_positions: pd.DataFrame,
    path_to_fields: str,
) -> pd.DataFrame:
    """Compute reciprocity coefficients from a Sim4Life potential field.

    Interpolates the potential at each compartment position and scales
    by the applied current.
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
