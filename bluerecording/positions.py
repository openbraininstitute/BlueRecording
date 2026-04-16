# SPDX-License-Identifier: GPL-3.0-or-later
import json
import os

import libsonata
import numpy as np
import pandas as pd
from collections.abc import Callable
from morphio import Morphology, SectionType
from mpi4py import MPI
from pathlib import Path
from scipy.interpolate import interp1d

from .utils import get_circuit_path

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
        self._indices: list[list[int]] | None = None

    @property
    def sections(self):
        """The morphology's section list, delegated to the underlying MorphIO object."""
        return self._morph.sections

    @property
    def points(self) -> np.ndarray:
        """Flat (N, 3) array of all section points in global coordinates."""
        return self._points

    @property
    def indices(self) -> list[list[int]]:
        """Per-section index mapping into the flat ``points`` array.

        ``indices[i]`` is a list of integer offsets such that
        ``points[indices[i]]`` gives the 3D points belonging to section *i*.
        The soma is not included (sections are numbered as in MorphIO).
        Built lazily on first access.
        """
        if self._indices is None:
            self._indices = []
            idx = 0
            for sec in self._morph.sections:
                n = len(sec.points)
                self._indices.append(list(range(idx, idx + n)))
                idx += n
        return self._indices

    def section_points(self, sec_id: int) -> np.ndarray:
        """Return the (possibly transformed) points for a given section."""
        return self._points[self.indices[sec_id]]


def interp_points(coords, ncomps):

    '''
    For a given dendritic section with 3d points coords and a number of segments ncomps, we interpolate the start and end points of each segment,
    with each segment having equal length
    '''

    # Remove consecutive duplicate points that can arise from float32 rotation precision
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    mask = np.concatenate(([True], diffs > 0))
    coords = coords[mask]

    xyz = np.array([]).reshape(ncomps + 1, 0)

    distances = np.cumsum(np.linalg.norm(np.diff(coords,axis=0),axis=1))
    distances /= distances[-1]
    distances = np.insert(distances,0,0)

    for dim in range(coords.shape[1]):

        f = interp1d(distances, coords[:, dim], kind='linear')
        ic = f(np.linspace(0, 1, ncomps + 1)).reshape(ncomps + 1, 1)
        xyz = np.hstack((xyz, ic))

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

    pts = m.points[m.indices[sec.id]]
    arc = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

    if sec.is_root:
        gap = np.linalg.norm(pts[0][:, np.newaxis] - soma_pos)
        parent_len = 0
    else:
        parent_pts = m.points[m.indices[sec.parent.id]]
        gap = np.linalg.norm(parent_pts[-1] - pts[0])
        parent_len = _get_cumulative_length(m, sec.parent, soma_pos, cache)

    cache[sec.id] = parent_len + gap + arc
    return cache[sec.id]


def _get_branch_section_ids(sec) -> list[int]:
    """Walk from a leaf section back to the root, returning section IDs soma→tip."""
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
        pts = m.points[m.indices[sec.id]]

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


def interp_points_axon(axonPoints, runningLens, secName, numCompartments, somaPos):

    segPos = []


    if secName == 1: # First AIS section

        secLen = 30 # By construction, has length of 30 um
        segLen = secLen / numCompartments # Assumes each segment has the same length

        startPoint = 0
        endPoint = 30

        idx = np.where(runningLens <= endPoint) # Finds indices of axon 3d points where cumulative length < 30 um

        axonRelevant = axonPoints[idx]


        lensRelevant = runningLens[idx] / secLen # Gets fraction of the total section length for each 3d point


        if len(axonRelevant) < 2: # If there are not enough points, we use the soma position (which would be included in the axon point list) and the first real point in the axon
            idx = 0

            axonRelevant = axonPoints[:2]

            lensRelevant = runningLens[:2] / secLen


    elif secName == 2: # Second AIS section

        secLen = 30 # Length is 30 unm, by construction
        segLen = secLen / numCompartments

        startPoint = 30 # Cumulative length of first AIS section
        endPoint = 60 # Cumulative length of both AIS sections

        idx = np.intersect1d(np.where(runningLens <= endPoint), np.where(runningLens >= startPoint)) # Finds indices 3d points falling in this length bin

        axonRelevant = axonPoints[idx]

        lensRelevant = (runningLens[idx] - startPoint) / secLen

        if len(axonRelevant) < 2: # If there aren't enough points, we estimate

            idxSmall = np.argmin(np.abs(runningLens - startPoint)) # Index closest to 30 um

            idxBig = np.argmin(np.abs(runningLens - endPoint)) # Index closest to 60 um

            if idxSmall == idxBig: # If these two points are the same, we use different points
                if idxBig < len(runningLens)-1:
                    idxBig += 1
                else:
                    idxSmall -= 1 # If the two points are identical, then idxSmall can never be zero, since otherwise this would imply a one-point axon

            idx = [idxSmall, idxBig]

            axonRelevant = axonPoints[idx]
            lensRelevant = (runningLens[idx] - startPoint) / secLen


    else: # Myelinated section

        secLen = 1000
        segLen = secLen / numCompartments

        startPoint = 60
        endPoint = 1060

        idx = np.where(runningLens >= startPoint)[0] # Get indices of 3d points that are beyond the AIS

        if len(idx) == 1:
            idx = [idx[0] - 1, idx[0]]

        axonRelevant = axonPoints[idx]

        lensRelevant = (runningLens[idx] - startPoint) / secLen

    for i in range(numCompartments+1): # Interpolates segment positions

        frac = (i * segLen) / secLen


        fx = interp1d(lensRelevant, axonRelevant[:, 0], kind='linear', fill_value='extrapolate')
        fy = interp1d(lensRelevant, axonRelevant[:, 1], kind='linear', fill_value='extrapolate')
        fz = interp1d(lensRelevant, axonRelevant[:, 2], kind='linear', fill_value='extrapolate')

        newx = fx(frac)
        newy = fy(frac)
        newz = fz(frac)

        segPos.append([newx, newy, newz])


    segPos = np.array(segPos)
    return segPos

def remove_variables(js, finalmorphpath):

    '''
    Removes references to variables in path to morphology
    Assumes that all references are in the manifest section
    '''

    while '$' in finalmorphpath:

        elements = finalmorphpath.split('/')

        for i, element in enumerate(elements):

            if '$' in element:
                element = js['manifest'][element]
                
            if i == 0:
                finalmorphpath = element
            else:
                finalmorphpath = finalmorphpath + '/' + element

    return finalmorphpath

def tryFileNames(morphName, finalmorphpath):

    asc = finalmorphpath+'/ascii/'+morphName+'.asc'
    asc1 = finalmorphpath+'/'+morphName+'.asc'
    asc2 = finalmorphpath+'/morphologies_asc/'+morphName+'.asc'

    swc = finalmorphpath+'/swc/'+morphName+'.swc'
    swc1 = finalmorphpath+'/'+morphName+'.swc'
    swc2 = finalmorphpath+'/morphologies_swc/'+morphName+'.swc'

    options = [asc, asc1, asc2, swc, swc1, swc2]

    for option in options:
        if os.path.exists(option):
            fileName = option
            break

    return fileName

def get_morph_path(population, i, path_to_simconfig):

    morphName = population.get_attribute('morphology', i) # Gets name of the morphology file for node_id i

    circuitpath = get_circuit_path(path_to_simconfig) # path to circuit_config file

    with open(circuitpath) as f: # Gets path to morphology file from circuit_config

        js = json.load(f)

        if 'components' in js.keys() and 'morphologies_dir' in js['components'].keys():
            finalmorphpath = js['components']['morphologies_dir']

        else:
            finalmorphpath = js['manifest']['$MORPHOLOGIES']

        finalmorphpath = remove_variables(js, finalmorphpath)

        finalmorphpath = str((Path(circuitpath).parent / finalmorphpath).resolve())

    fileName = tryFileNames(morphName, finalmorphpath)

    return fileName


def get_morphology(
    population: libsonata.NodePopulation,
    i: int,
    path_to_simconfig: str,
    cell,
) -> tuple[PositionedMorphology, np.ndarray]:
    """Load and transform a morphology into circuit (global) coordinates.

    Args:
        population: libsonata NodePopulation.
        i: Node index within the population.
        path_to_simconfig: Path to the SONATA simulation configuration file.
        cell: Neurodamus cell object with coordinate mapping.

    Returns:
        m: PositionedMorphology with points transformed to global coordinates.
        center: (3,) float32 array — the soma position taken from the sonata
            node x/y/z (translation column of the transform matrix).  This is
            the BlueRecording convention (raw placement position), not the
            neurodamus soma centroid (mean of NEURON soma section boundary
            points), which can differ by up to ~1.8 µm.
    """

    finalmorphpath = get_morph_path(population, i, path_to_simconfig)

    mImmutable = Morphology(finalmorphpath) # Immutable MorphIO morphology object

    m = PositionedMorphology(mImmutable, transform=cell.local_to_global_coord_mapping)

    center = cell.local_to_global_matrix[:, 3]

    return m, center


def getNewIndex(cols):
    """Build a new MultiIndex by duplicating certain (id, section) column tuples.

    Rules:
    - Every column is kept once.
    - The last column is repeated to represent the end point.
    - Columns with section != 0 are duplicated if the next column tuple differs.

    Returns a pandas MultiIndex with levels ["id", "section"].
    """
    newIdx = []

    # Ensure cols is a list of tuples
    cols_list = [tuple(c) for c in cols]

    for i, col in enumerate(cols_list):
        newIdx.append(col)

        # Last column: repeat to account for end point
        if i == len(cols_list) - 1:
            newIdx.append(col)

        # Non-somatic segments: add extra entry if next col is different
        elif col[-1] != 0:  # section != 0
            if cols_list[i + 1] != col:  # now comparing tuples
                newIdx.append(col)

    newCols = pd.MultiIndex.from_tuples(newIdx, names=["id", "section"])

    return newCols


def get_cell_positions(m, center, cols, gid, replace_axons):
    """Compute the 3D segment boundary positions for a single cell.

    Returns a (3, N) array where each column is the x/y/z position of a segment
    boundary (start points, plus the end point of the last segment in each section).
    """

    soma_pos = center[:,np.newaxis]

    axon_points, running_lens = None, None
    if replace_axons: # If the axons are replaced by a stub axon, we need to get the positions thereof
        axon_points, running_lens = get_axon_points(m,center) # Gets 3d positions and cumulative length of the axon

    sections = np.unique(cols[np.where(cols[:,0]==gid),1:].flatten()) # List of sections for the given neuron

    # Start with soma position(s)
    xyz = soma_pos.reshape(3,1)

    num_somas = np.sum((cols[:,0] == gid) & (cols[:,1] == 0))

    if num_somas > 1: # If there is more than one somatic segment, we assume that they all have the same position
        for k in np.arange(1,num_somas):
            xyz = np.hstack((xyz,soma_pos.reshape(3,1)))

    for sec_name in list(sections[1:]):

        num_compartments = np.sum((cols[:,0] == gid) & (cols[:,1] == sec_name))

        # Section 1 and 2 are always axonal when axons are replaced (AIS)
        if sec_name < 3 and replace_axons:
            seg_pos = interp_points_axon(axon_points,running_lens,sec_name,num_compartments,soma_pos)
        else:
            sec_id = sec_name - 1
            if sec_id >= len(m.indices): # Beyond morphology sections → myelinated AIS
                seg_pos = interp_points_axon(axon_points,running_lens,sec_name,num_compartments,soma_pos)
            else:
                sec_pts = np.array(m.section_points(sec_id))
                seg_pos = interp_points(sec_pts,num_compartments)

        xyz = np.hstack((xyz,seg_pos.T))

    return xyz



def resolve_neurite_types(cols_for_gid, cell):
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


def get_positions(node_manager, ids, cols, population, path_to_simconfig, replace_axons=True):
    """Compute segment boundary positions for all cells on this rank.

    Pure computation — no file I/O. Returns the positions DataFrame,
    the cols array, and per-compartment neurite types for downstream use.

    Args:
        node_manager: Neurodamus node manager.
        ids: GIDs assigned to this MPI rank.
        cols: (N, 2) int64 array of (gid, section) pairs.
        population: libsonata NodePopulation for morphology resolution.
        path_to_simconfig: Path to the SONATA simulation config.
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
        m, center = get_morphology(population, i, path_to_simconfig, cell)

        cell_arrays.append(get_cell_positions(m, center, cols, i, replace_axons))

        cols_for_gid = cols[cols[:, 0] == i]
        neurite_type_arrays.append(resolve_neurite_types(cols_for_gid, cell))

    if not cell_arrays:
        empty_idx = pd.MultiIndex.from_tuples([], names=["id", "section"])
        positions_df = pd.DataFrame(np.empty((3, 0)), columns=empty_idx)
        return positions_df, cols, np.array([], dtype=np.int32)

    xyz = np.hstack(cell_arrays)
    new_cols = getNewIndex(cols)
    positions_df = pd.DataFrame(xyz, columns=new_cols)
    neurite_types = np.concatenate(neurite_type_arrays)

    return positions_df, cols, neurite_types




def save_positions(positions_df, path_to_positions_folder):
    """Write positions DataFrame to a pickle file for this MPI rank.

    Args:
        positions_df: DataFrame returned by get_positions.
        path_to_positions_folder: Output directory.
    """
    path_to_positions_folder = Path(path_to_positions_folder)
    path_to_positions_folder.mkdir(parents=True, exist_ok=True)
    positions_df.to_pickle(path_to_positions_folder / f"positions{rank}.pkl")
