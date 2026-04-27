# SPDX-License-Identifier: GPL-3.0-or-later

import libsonata
import numpy as np
import pandas as pd
from collections.abc import Callable
from morphio import Morphology, SectionType
from mpi4py import MPI
from pathlib import Path
from scipy.interpolate import interp1d

from .circuit import init_circuit

rank = MPI.COMM_WORLD.Get_rank()


class PositionedMorphology:
    """A morphology with points transformed to global (circuit) coordinates.

    Wraps an immutable MorphIO morphology, applying an optional coordinate
    transform at construction time. Provides lazy per-section point indexing
    and a convenience accessor for section points.

    Args:
        morph: An immutable MorphIO Morphology object.
        transform: Optional callable that maps the (N, 3) points array to
            global coordinates (e.g. ``cell.local_to_global_coord_mapping``).
    """

    def __init__(self, morph: Morphology, transform: Callable[[np.ndarray], np.ndarray] | None = None):
        self._morph: Morphology = morph
        all_points: np.ndarray = np.concatenate([s.points for s in morph.sections])
        self._points: np.ndarray = transform(all_points) if transform else all_points
        self._offsets: np.ndarray | None = None

    @property
    def sections(self):
        """The morphology's section list, delegated to the underlying MorphIO object."""
        return self._morph.sections

    @property
    def num_sections(self) -> int:
        """Number of sections (excluding soma)."""
        return len(self._morph.sections)

    @property
    def points(self) -> np.ndarray:
        """Flat (N, 3) array of all section points in global coordinates."""
        return self._points

    @property
    def offsets(self) -> np.ndarray:
        """Section boundary offsets into the flat ``points`` array.

        ``offsets`` has length ``num_sections + 1``.  The points for section *i*
        are ``points[offsets[i]:offsets[i+1]]``.
        Built lazily on first access.
        """
        if self._offsets is None:
            counts = [len(sec.points) for sec in self._morph.sections]
            self._offsets = np.zeros(len(counts) + 1, dtype=np.intp)
            np.cumsum(counts, out=self._offsets[1:])
        return self._offsets

    def section_points(self, sec_id: int) -> np.ndarray:
        """Return the (possibly transformed) points for a given section."""
        o = self.offsets
        return self._points[o[sec_id]:o[sec_id + 1]]


def interp_points(coords: np.ndarray, ncomps: int) -> np.ndarray:
    """Interpolate segment boundary points along a dendritic section.

    Given the 3D points of a section and a number of compartments, returns
    equally-spaced boundary points by linear interpolation along the arc length.
    Consecutive duplicate points (which can arise from float32 rotation
    precision) are removed before interpolation.

    Args:
        coords: (P, 3) array of 3D section points.
        ncomps: Number of compartments (segments) in the section.

    Returns:
        (ncomps + 1, 3) array of interpolated boundary positions.
    """

    # --- 1. Remove consecutive near-duplicate points (float32 rotation artefacts) ---
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    mask = np.concatenate(([True], diffs > 0))  # exact dedup only
    coords = coords[mask]

    # --- 2. Interpolate equally-spaced boundary points along the arc length ---
    arc = np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    arc = np.insert(arc, 0, 0)
    arc /= arc[-1]  # normalise to [0, 1]

    targets = np.linspace(0, 1, ncomps + 1)
    xyz = np.column_stack([
        interp1d(arc, coords[:, dim], kind='linear')(targets)
        for dim in range(coords.shape[1])
    ])

    return xyz


def _get_cumulative_length(
    m: PositionedMorphology, sec, soma_pos: np.ndarray, cache: dict[int, float]
) -> float:
    """Return cumulative arc length from soma to the end of a section.

    Computes lazily and caches results so each section is measured at most once.

    Args:
        m: Mutable morphology with rotated/translated points and section indices.
        sec: A MorphIO section object.
        soma_pos: Soma position as a column vector, shape (3, 1).
        cache: Dict mapping section id to cumulative length (mutated in place).

    Returns:
        Cumulative arc length (µm) from soma to the end of this section.
    """
    if sec.id in cache:
        return cache[sec.id]

    pts = m.section_points(sec.id)
    arc = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

    if sec.is_root:
        gap = np.linalg.norm(pts[0][:, np.newaxis] - soma_pos)
        parent_len = 0
    else:
        parent_pts = m.section_points(sec.parent.id)
        gap = np.linalg.norm(parent_pts[-1] - pts[0])
        parent_len = _get_cumulative_length(m, sec.parent, soma_pos, cache)

    cache[sec.id] = parent_len + gap + arc
    return cache[sec.id]


def _get_branch_section_ids(sec) -> list[int]:
    """Walk from a leaf section back to the root, returning section IDs root→tip.

    Note: 'root' here means the first neurite section attached to the soma,
    not the soma itself (which is not a MorphIO section).
    """
    idxs = []
    this_sec = sec
    while True:
        idxs.append(this_sec.id)
        if this_sec.is_root:
            break
        this_sec = this_sec.parent
    idxs.reverse()
    return idxs


def _find_best_axon_branch(
    m: PositionedMorphology, soma_pos: np.ndarray, target_length: float
) -> tuple[list[int], bool]:
    """Find the best axonal branch for simulated-axon position reconstruction.

    Lazily computes cumulative lengths for visited sections, then picks the
    first axonal leaf whose branch length >= target_length. If none qualifies,
    returns the longest branch and signals that extrapolation is needed.

    Args:
        m: Mutable morphology with rotated/translated points and section indices.
        soma_pos: Soma position as a column vector, shape (3, 1).
        target_length: Required branch length in µm.

    Returns:
        section_ids: Section IDs ordered from soma to tip.
        need_extension: True if the branch is shorter than target_length.
    """
    cumulative = {}

    longest_length = 0
    best_idx = []
    length = 0

    for sec in m.sections:
        if sec.type == SectionType.axon and len(sec.children) == 0:
            length = _get_cumulative_length(m, sec, soma_pos, cumulative)
            idxs = _get_branch_section_ids(sec)
            if length > longest_length:
                longest_length = length
                best_idx = idxs
            if length > target_length:
                break

    need_extension = False
    if length < target_length:
        idxs = best_idx
        need_extension = True

    return idxs, need_extension


def _collect_branch_points(
    m: PositionedMorphology,
    section_ids: list[int],
    soma_pos: np.ndarray,
    target_length: float,
) -> tuple[np.ndarray, list[float], float]:
    """Walk a branch from soma to tip, collecting 3D points and arc lengths.

    Stops as soon as the cumulative arc length exceeds target_length.

    Args:
        m: Mutable morphology with rotated/translated points and section indices.
        section_ids: Ordered section IDs from soma to tip.
        soma_pos: Soma position as a column vector, shape (3, 1).
        target_length: Maximum arc length to collect (µm).

    Returns:
        points: (N, 3) array of 3D positions along the branch.
        running_len: List of cumulative arc lengths, one per point.
        current_len: Final cumulative arc length.
    """
    last_pt = soma_pos.flatten()

    point_list = [last_pt.reshape(1, 3)]
    running_len = [0]
    current_len = 0

    for x in section_ids:
        sec = m.sections[x]
        pts = m.section_points(sec.id)

        for pt in pts:
            current_len += np.linalg.norm(pt - last_pt)
            running_len.append(current_len)
            point_list.append(pt.reshape(1, 3))
            last_pt = pt
            if current_len > target_length:
                break
        if current_len > target_length:
            break

    return np.vstack(point_list), running_len, current_len


def _extrapolate_branch(
    points: np.ndarray,
    running_len: list[float],
    current_len: float,
    target_length: float,
) -> tuple[np.ndarray, list[float]]:
    """Linearly extrapolate a branch that is shorter than target_length.

    Extends the branch by adding a single point along the direction defined
    by the last two existing points.

    Args:
        points: (N, 3) array of branch positions.
        running_len: List of cumulative arc lengths.
        current_len: Current total arc length.
        target_length: Desired total length in µm.

    Returns:
        points: (N+1, 3) array with the extrapolated point appended.
        running_len: List with the new arc length appended.
    """
    to_add = target_length - current_len
    slopes = (points[-1] - points[-2]) / (running_len[-1] - running_len[-2])
    new_pt = points[-1] + slopes * to_add
    points = np.vstack((points, new_pt))
    current_len += np.linalg.norm(new_pt - points[-2])
    running_len.append(current_len)
    return points, running_len


def get_axon_points(m: PositionedMorphology, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract 3D positions and cumulative lengths along the simulated axon.

    The simulated axon consists of two AIS sections (30 µm each) and a 1000 µm
    myelinated section, totalling 1060 µm.  Since the simulator does not define
    the spatial positions of these sections, we walk the morphology tree to find
    the first axonal branch that is at least 1060 µm long and extract its 3D
    points.  If no branch is long enough, the longest one is linearly
    extrapolated.

    Args:
        m: PositionedMorphology with points in global coordinates.
        center: Soma position as a 1D array of shape (3,).

    Returns:
        axon_points: Unique 3D positions along the selected axonal branch,
            shape (N, 3).
        running_lengths: Cumulative arc length at each point, shape (N,).
    """

    target_length = 1060
    soma_pos = center[:, np.newaxis]
    section_ids, need_extension = _find_best_axon_branch(m, soma_pos, target_length)
    points, running_len, current_len = _collect_branch_points(m, section_ids, soma_pos, target_length)

    if need_extension:
        points, running_len = _extrapolate_branch(points, running_len, current_len, target_length)
    
    # Remove duplicate points (morphology formats may repeat section boundaries)
    axon_points, indices = np.unique(np.array(points), axis=0, return_index=True)
    return axon_points, np.array(running_len)[indices]


def interp_points_axon(
    axon_points: np.ndarray,
    running_lens: np.ndarray,
    sec_id: int,
    num_compartments: int,
) -> np.ndarray:
    """Interpolate segment boundary points for a simulated-axon section.

    These are global section IDs after axon replacement.

    For each section the relevant subset of *axon_points* is selected by
    cumulative arc length, then linearly interpolated to produce equally-spaced
    segment boundary positions.

    Args:
        axon_points: (N, 3) array of 3D positions along the axonal branch.
        running_lens: (N,) array of cumulative arc lengths aligned with
            *axon_points*.
        sec_id: Section identifier (1 = first AIS, 2 = second AIS,
            ≥3 = myelinated).
        num_compartments: Number of compartments (segments) in this section.

    Returns:
        (num_compartments + 1, 3) array of interpolated boundary positions.
    """

    # --- 1. Determine section geometry and select relevant points ---

    if sec_id == 1:  # First AIS section (0–30 µm)
        sec_len = 30
        start, end = 0, 30

        idx = np.where(running_lens <= end)
        axon_relevant = axon_points[idx]
        lens_relevant = running_lens[idx] / sec_len

        if len(axon_relevant) < 2:
            axon_relevant = axon_points[:2]
            lens_relevant = running_lens[:2] / sec_len

    elif sec_id == 2:  # Second AIS section (30–60 µm)
        sec_len = 30
        start, end = 30, 60

        idx = np.where((running_lens >= start) & (running_lens <= end))[0]
        axon_relevant = axon_points[idx]
        lens_relevant = (running_lens[idx] - start) / sec_len

        if len(axon_relevant) < 2:
            idx_lo = np.argmin(np.abs(running_lens - start))
            idx_hi = np.argmin(np.abs(running_lens - end))

            if idx_lo == idx_hi:
                if idx_hi < len(running_lens) - 1:
                    idx_hi += 1
                else:
                    idx_lo -= 1

            idx = [idx_lo, idx_hi]
            axon_relevant = axon_points[idx]
            lens_relevant = (running_lens[idx] - start) / sec_len

    else:  # Myelinated section (60–1060 µm)
        sec_len = 1000
        start, end = 60, 1060

        idx = np.where(running_lens >= start)[0]

        if len(idx) == 1:
            idx = [idx[0] - 1, idx[0]]

        axon_relevant = axon_points[idx]
        lens_relevant = (running_lens[idx] - start) / sec_len

    # --- 2. Interpolate equally-spaced boundary points ---

    seg_len = sec_len / num_compartments
    targets = np.array([(i * seg_len) / sec_len for i in range(num_compartments + 1)])

    seg_pos = np.column_stack([
        interp1d(lens_relevant, axon_relevant[:, dim], kind='linear',
                 fill_value='extrapolate')(targets)
        for dim in range(3)
    ])

    return seg_pos

def get_new_index(cols: np.ndarray) -> pd.MultiIndex:
    """Build a new MultiIndex by duplicating certain (id, section) column tuples.

    Each column is kept once. Non-somatic columns (section != 0) are
    duplicated when the next column tuple differs, to represent the end
    point of that section.  The last column is always repeated.

    Args:
        cols: (N, 2) array of (id, section) pairs.

    Returns:
        MultiIndex with levels ["id", "section"].
    """
    new_idx = []
    cols_list = [tuple(c) for c in cols]

    for i, col in enumerate(cols_list):
        new_idx.append(col)

        # Last column: repeat to account for end point
        if i == len(cols_list) - 1:
            new_idx.append(col)

        # Non-somatic segments: add extra entry if next col is different
        elif col[-1] != 0:
            if cols_list[i + 1] != col:
                new_idx.append(col)

    return pd.MultiIndex.from_tuples(new_idx, names=["id", "section"])

def _get_cell_positions(
    m: PositionedMorphology,
    center: np.ndarray,
    cols: np.ndarray,
    gid: int,
    replace_axons: bool,
) -> np.ndarray:
    """Compute the 3D segment boundary positions for a single cell.

    For each section the morphology points are interpolated to produce
    equally-spaced segment boundaries.  When axon replacement is active,
    sections 1–2 (AIS) and any section beyond the morphology are
    interpolated along the simulated axon branch instead.

    Args:
        m: PositionedMorphology with points in global coordinates.
        center: Soma position, shape (3,).
        cols: (N, 2) array of (gid, section) pairs for all cells.
        gid: GID of the cell to process.
        replace_axons: If True, use the simulated axon for AIS/myelinated
            sections.

    Returns:
        (3, M) array where each column is an x/y/z segment boundary position.
    """
    soma_pos = center[:, np.newaxis]
    gid_mask = cols[:, 0] == gid

    axon_points, running_lens = None, None
    if replace_axons:
        axon_points, running_lens = get_axon_points(m, center)

    sections = np.unique(cols[np.where(gid_mask), 1:].flatten())

    # Build soma columns — all somatic segments share the same position
    num_somas = np.sum(gid_mask & (cols[:, 1] == 0))
    xyz = np.tile(soma_pos.reshape(3, 1), num_somas)

    for sec_id in sections[1:]:
        num_compartments = np.sum(gid_mask & (cols[:, 1] == sec_id))

        if sec_id < 3 and replace_axons:
            seg_pos = interp_points_axon(axon_points, running_lens, sec_id, num_compartments)
        else:
            morpho_sec_id = sec_id - 1
            if morpho_sec_id >= m.num_sections:
                # Beyond morphology sections → myelinated AIS
                seg_pos = interp_points_axon(axon_points, running_lens, sec_id, num_compartments)
            else:
                sec_pts = np.array(m.section_points(morpho_sec_id))
                seg_pos = interp_points(sec_pts, num_compartments)

        xyz = np.hstack((xyz, seg_pos.T))

    return xyz

def resolve_neurite_types(cols_for_gid: np.ndarray, cell) -> np.ndarray:
    """Return an int array of neurite-type codes for one neuron's compartments.

    Queries the actual NEURON section via ``cell.get_sec()`` and maps the
    section-type string (e.g. ``"soma"``, ``"axon"``, ``"myelin"``) to its
    index in ``neurodamus.metype.BaseCell.SECTION_TYPES``.  This reflects
    the simulator's SectionList membership rather than the coarser MorphIO
    classification, and automatically picks up any future types added to
    neurodamus.

    Args:
        cols_for_gid: (M, 2) array of (gid, section_index) pairs for this neuron.
        cell: Neurodamus cell object (e.g. ``Cell_V6``).

    Returns:
        (M,) int32 array where each element is the index into
        ``BaseCell.SECTION_TYPES`` for that compartment's section.
    """
    from neurodamus.metype import BaseCell

    type_to_code = {st: idx for idx, (st, _) in enumerate(BaseCell.SECTION_TYPES)}

    section_indices = cols_for_gid[:, 1]
    result = np.empty(len(section_indices), dtype=np.int32)
    for i, sec_idx in enumerate(section_indices):
        sec = cell.get_sec(int(sec_idx))
        sec_type = sec.name().rsplit(".", 1)[-1].rsplit("[", 1)[0]
        result[i] = type_to_code[sec_type]
    return result


def _find_morph_file(morph_name: str, morph_dir: str) -> str:
    """Locate a morphology file by trying common subdirectory/extension combos.

    Args:
        morph_name: Morphology name (without extension) from the population.
        morph_dir: Absolute path to the morphologies directory (from libsonata).

    Returns:
        Absolute path to the first existing morphology file found.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    base = Path(morph_dir)
    candidates = [
        base / "ascii" / f"{morph_name}.asc",
        base / f"{morph_name}.asc",
        base / "morphologies_asc" / f"{morph_name}.asc",
        base / "swc" / f"{morph_name}.swc",
        base / f"{morph_name}.swc",
        base / "morphologies_swc" / f"{morph_name}.swc",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"Morphology '{morph_name}' not found in {morph_dir}"
    )


def get_positions(
    node_manager,
    ids: np.ndarray,
    cols: np.ndarray,
    population: libsonata.NodePopulation,
    morphologies_dir: str,
    replace_axons: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute segment boundary positions for all cells on this rank.

    Pure computation — no file I/O. Returns the positions DataFrame,
    the cols array, and per-compartment neurite types for downstream use.

    Args:
        node_manager: Neurodamus node manager.
        ids: GIDs assigned to this MPI rank.
        cols: (N, 2) int64 array of (gid, section) pairs.
        population: libsonata NodePopulation for morphology resolution.
        morphologies_dir: Fully resolved path to the morphologies directory.
        replace_axons: If True, replace morphological axons with a standardized
            stub (two 30 µm AIS sections + 1000 µm myelinated section).

    Returns:
        positions_df: DataFrame with MultiIndex columns (id, section),
            shape (3, M) where M includes segment boundary duplicates.
        cols: The input cols array, passed through for convenience.
        neurite_types: (N,) int32 array of neurite type codes per compartment,
            aligned with cols row order.
    """
    cell_arrays = []
    neurite_type_arrays = []

    for i in ids:
        cell = node_manager.get_cell(i)
        # Load morphology, transform to global coordinates, extract soma pos.
        # center is the raw placement position (translation column of the
        # transform matrix), not the neurodamus soma centroid which can
        # differ by up to ~1.8 µm.
        morph_name = population.get_attribute("morphology", cell.raw_gid)
        morph_path = _find_morph_file(morph_name, morphologies_dir)
        m = PositionedMorphology(
            Morphology(morph_path),
            transform=cell.local_to_global_coord_mapping,
        )
        center = cell.local_to_global_matrix[:, 3]

        cell_arrays.append(_get_cell_positions(m, center, cols, i, replace_axons))

        cols_for_gid = cols[cols[:, 0] == i]
        neurite_type_arrays.append(resolve_neurite_types(cols_for_gid, cell))

    if not cell_arrays:
        empty_idx = pd.MultiIndex.from_tuples([], names=["id", "section"])
        positions_df = pd.DataFrame(np.empty((3, 0)), columns=empty_idx)
        return positions_df, cols, np.array([], dtype=np.int32)

    xyz = np.hstack(cell_arrays)
    new_cols = get_new_index(cols)
    positions_df = pd.DataFrame(xyz, columns=new_cols)
    neurite_types = np.concatenate(neurite_type_arrays)

    return positions_df, cols, neurite_types

def save_positions(positions_df: pd.DataFrame, path_to_positions_folder: str | Path) -> None:
    """Write positions DataFrame to a pickle file for this MPI rank.

    Args:
        positions_df: DataFrame returned by get_positions.
        path_to_positions_folder: Output directory.
    """
    path_to_positions_folder = Path(path_to_positions_folder)
    path_to_positions_folder.mkdir(parents=True, exist_ok=True)
    positions_df.to_pickle(path_to_positions_folder / f"positions{rank}.pkl")
