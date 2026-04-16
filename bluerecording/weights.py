# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import h5py
import os
import pandas as pd
import sys
import json
import datetime
import warnings

from mpi4py import MPI
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator
from sklearn.decomposition import PCA

from .utils import *

DEFAULT_SIGMA = 0.277  # Extracellular conductivity in S/m


# ---------------------------------------------------------------------------
# H5 file initialization (formerly writeH5_prelim.py)
# ---------------------------------------------------------------------------

class ElectrodeFileStructure(object):

    """Write datasets to the HDF5 electrode file."""

    def __init__(self, h5, lst_ids, electrodes, population_name):
        """Initialize electrode file structure and write metadata to HDF5.

        Args:
            h5: HDF5 file handle returned by h5py.File(filename, 'w').
            lst_ids: Node IDs.
            electrodes: Dictionary with metadata about electrodes.
            population_name: SONATA population name.
        """
        dset = h5.create_dataset(population_name+"/node_ids", data=sorted(lst_ids))

        for index, (key, electrode) in enumerate(electrodes.items()):
            h5.create_dataset("electrodes/" + str(key) + '/'+population_name,data=index)

            for item in electrode.items():
                if item[0] == 'type' and isinstance(item[1],dict):
                    dset = h5.create_dataset("electrodes/" + str(key) + '/' + item[0],
                                      data=item[1]['type'])

                    for entry in item[1].items():
                        if entry[0] !='type':
                            dset.attrs.create(entry[0],entry[1])

                else:
                    h5.create_dataset("electrodes/" + str(key) + '/' + item[0],
                              data=item[1])

        self._ids = np.array(lst_ids)

def get_offsets(sectionIdsFrame):
    """Compute per-node offsets into the flat segment array.

    Returns an array where entry *i* is the index of the first segment
    belonging to the *i*-th unique node ID.
    """
    _unique, counts = np.unique(sectionIdsFrame['id'].values,return_counts=True)

    out_offsets = np.hstack((np.array([0]),np.cumsum(counts)))

    return out_offsets

def write_all_neuron(sectionIdsFrame, population_name, h5file, electrode_struc):
    """Initialize scaling_factors with ones and write per-node offsets.

    Creates a dataset of shape (nSegments, nElectrodes+1) filled with ones,
    plus the offset array that maps each node to its range in scaling_factors.
    """
    h5file.create_dataset('/electrodes/'+population_name+'/scaling_factors', data=np.ones([len(sectionIdsFrame['id'].values),len(electrode_struc.items())+1]))

    out_offsets = get_offsets(sectionIdsFrame)

    h5file.create_dataset(population_name+"/offsets", data=out_offsets)


def make_electrode_dict(electrode_csv):
    """Read electrode metadata from a CSV file and return it as a dictionary."""

    electrode_df = pd.read_csv(electrode_csv,header=0,index_col=0)

    electrodes = {}

    for i in range(len(electrode_df.values)):

        name = electrode_df.index.values[i]

        position = np.array([electrode_df['x'].iloc[i],electrode_df['y'].iloc[i],electrode_df['z'].iloc[i]])

        if 'layer' in electrode_df.columns:

            layer = electrode_df['layer'].iloc[i]

        else:

            layer = "NA"

        if 'region' in electrode_df.columns:
            region = electrode_df['region'].iloc[i]
        else:
            region = 'NA'

        if 'type' in electrode_df.columns:
            electrodeType = electrode_df['type'].iloc[i]

            if 'ObjectiveCSD' in electrodeType:

                electrodeType = process_objectiveCSD(electrodeType)

        else:
            electrodeType = 'LineSource'

        electrodes[name] = {'position': position,'type': electrodeType,
        'region':region,'layer':layer}


    return electrodes

def check_input_type_objectiveCSD(objectiveType,input):
    """Validate the numerical parameters for an objective CSD electrode type."""
    if objectiveType == 'ObjectiveCSD_Sphere' or objectiveType == 'ObjectiveCSD_Plane':
        try:
            assert len(input) == 3
        except:
            raise ValueError(objectiveType + ' must provide either no numerical parameters or exactly one')
    elif objectiveType =='ObjectiveCSD_Disk':
        try:
            assert len(input) == 3 or len(input)==4
        except:
            raise ValueError(objectiveType + ' must provide one or two numerical parameters')
    else:
        raise ValueError('Invalid electrode type')

    for numericalParameter in input[2:]:
        try:
            float(numericalParameter)
        except:
            raise ValueError('Invalid numerical parameter provided to objective CSD electrode')

    return 0

def process_objectiveCSD(electrodeType):
    """Process an objective CSD electrode type string.

    Parses the electrode type to extract geometry parameters (radius,
    thickness) when provided. Returns a dict with the parameters, or
    the plain type string for backwards compatibility.

    Format: ``ObjectiveCSD_Method[_X[_Y]]``
      - Sphere: X = radius (um)
      - Plane:  X = thickness (um)
      - Disk:   X = radius (um), Y = thickness (um)
    """


    input = electrodeType.split('_')

    if len(input) < 2:
        raise ValueError(electrodeType + ' is an invalid objective electrode type')

    elif len(input)==2: # If no other options are provided, returns string, for backwards compatibility with previous versions
        return electrodeType

    else:

        objectiveType = input[0] + '_' + input[1]
        objectiveDict = {'type':objectiveType}

        check_input_type_objectiveCSD(objectiveType, input)

        if objectiveType == 'ObjectiveCSD_Sphere':

            radius = float(input[2])
            objectiveDict['radius'] = radius

        elif objectiveType == 'ObjectiveCSD_Plane':

            thickness = float(input[2])
            objectiveDict['thickness'] = thickness

        elif objectiveType == 'ObjectiveCSD_Disk':

            radius = float(input[2])
            objectiveDict['radius'] = radius

            if len(input)==4:
                thickness = float(input[3])
                objectiveDict['thickness'] = thickness
        else:
            raise ValueError("Invalid electrode type value")

        return objectiveDict

def initialize_h5_file(cols, population_name, outputfile, electrode_csv, with_neurite_type=False):
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

        electrodes = make_electrode_dict(electrode_csv)

        h5file = h5py.File(outputfile, 'w')

        # Tune HDF5 metadata cache for faster writes
        h5id = h5file.id
        cc = h5id.get_mdc_config()
        cc.max_size = 1024 * 1024 * 124
        h5id.set_mdc_config(cc)

        h5 = ElectrodeFileStructure(h5file, node_ids, electrodes, population_name)

        write_all_neuron(section_ids_frame, population_name, h5file, electrodes)

        if with_neurite_type:
            n_compartments = len(all_cols)
            h5file.create_dataset(
                f"{population_name}/neurite_types",
                shape=(n_compartments,),
                dtype=np.int32,
            )

        h5file.close()

    comm.Barrier()


# ---------------------------------------------------------------------------
# Weight computation (formerly writeH5.py)
# ---------------------------------------------------------------------------

def add_data(h5, ids, coeffs, population_name):
    """Write computed coefficients into the scaling_factors dataset.

    Looks up each node's offset range and writes the corresponding
    coefficient rows into the HDF5 dataset.
    """
    dset = 'electrodes/'+population_name+'/scaling_factors'

    node_ids = h5[population_name+'/node_ids'][:]

    isInInput = np.isin(node_ids,ids)
    nodesInInput = node_ids[isInInput]
    idIndex = np.where(isInInput)[0]

    offset0 = h5[population_name+'/offsets'][idIndex]

    offset1 = np.zeros_like(offset0)

    if np.any(idIndex ==  len(h5[population_name+'/offsets'][:])-1):

        lastNodeIdx =  np.where(idIndex == len(h5[population_name+'/offsets'][:])-1)[0]
        offset1[lastNodeIdx] = len(h5[dset][:])

    notLastNodeIdx = np.where(idIndex != len(h5[population_name+'/offsets'][:])-1)[0]
    offset1[notLastNodeIdx] =  h5[population_name+'/offsets'][idIndex[notLastNodeIdx]+1]

    for i, id in enumerate(nodesInInput):

        h5[dset][offset0[i]:offset1[i],:-1] = coeffs.loc[:,id].values.T

def line_source_cases(h,r2,l):
    """Return the line-source potential term for the given geometry case.

    Selects the appropriate formula depending on the signs of h and l.
    """
    if h < 0 and l < 0:

        lineSourceTerm = np.log(((h**2+r2)**.5-h)/((l**2+r2)**.5-l))

    elif h < 0 and l > 0:

        lineSourceTerm = np.log( ( ((h**2+r2)**.5-h)* (l + (l**2+r2)**.5 ) ) / r2  )

    elif h > 0 and l > 0:

        lineSourceTerm = np.log( ( (l + (l**2+r2)**.5 ) ) / ( (r2+h**2)**.5 + h)  )


    return lineSourceTerm

def get_line_coeffs(startPos,endPos,electrodePos,sigma):
    """Compute the line-source coefficient for a single segment.

    Args:
        startPos: Starting position of the segment (um).
        endPos: Ending position of the segment (um).
        electrodePos: Electrode position (um).
        sigma: Extracellular conductivity (S/m).
    """
    startPos = startPos * 1e-6
    endPos = endPos * 1e-6
    electrodePos = electrodePos * 1e-6

    segLength = np.linalg.norm(startPos-endPos)

    x1 = electrodePos[0]-endPos[0]
    y1 = electrodePos[1]-endPos[1]
    z1 = electrodePos[2]-endPos[2]



    xdiff = endPos[0]-startPos[0]
    ydiff = endPos[1]-startPos[1]
    zdiff = endPos[2]-startPos[2]


    h = 1/segLength * (x1*xdiff + y1*ydiff + z1*zdiff)

    l = h + segLength

    subtractionTerm = h**2

    r2 = (electrodePos[0]-endPos[0])**2 + (electrodePos[1]-endPos[1])**2 + (electrodePos[2]-endPos[2])**2 - subtractionTerm

    r2 = np.abs(r2)


    lineSourceTerm = line_source_cases(h,r2,l)

    segCoeff = 1/(4*np.pi*sigma*segLength)*lineSourceTerm

    segCoeff *= 1e-9

    return segCoeff


def get_coeffs_lineSource(positions,columns,electrodePos,sigma):
    """Compute line-source coefficients for all segments.

    Soma segments are treated as point sources; other segments use the
    line-source approximation between consecutive position endpoints.
    """
    for i in range(len(positions.columns)-1):

        if positions.columns[i][-1]==0:

            somaPos = positions.iloc[:,i]

            distance = np.linalg.norm(somaPos-electrodePos)

            distance *= 1e-6

            somaCoeff = 1/(4*np.pi*sigma*distance)

            somaCoeff *= 1e-9

            if i == 0:
                coeffs = somaCoeff
            else:

                coeffs = np.hstack((coeffs,somaCoeff))

        elif positions.columns[i][-1]==positions.columns[i+1][-1]:

            segCoeff = get_line_coeffs(positions.iloc[:,i],positions.iloc[:,i+1],electrodePos,sigma)

            coeffs = np.hstack((coeffs,segCoeff))


    coeffs = pd.DataFrame(data=coeffs[np.newaxis,:])

    coeffs.columns = columns

    return coeffs

def get_coeffs_pointSource(positions,electrodePos,sigma):
    """Compute point-source coefficients for all segments.

    Each segment is treated as a point source. Distances are converted
    from um to m and currents from nA to A.
    """
    distances = np.linalg.norm(positions.values-electrodePos[:,np.newaxis],axis=0)

    distances *= 1e-6

    coeffs = 1/(4*np.pi*sigma*distances)

    coeffs *= 1e-9

    coeffs = pd.DataFrame(data=coeffs[np.newaxis,:])

    coeffs.columns = positions.columns

    return coeffs

def get_array_spacing(allEpos):
    """Compute the main axis and inter-electrode spacing of an array.

    Uses PCA to find the principal axis, projects electrode positions
    onto it, and returns the axis and the spacing between consecutive
    projections (ignoring co-planar electrodes).
    """
    pca = PCA(n_components=1)
    pca.fit(allEpos)
    main_axis = pca.components_[0]/np.linalg.norm(pca.components_[0])
    main_axis = main_axis[:,np.newaxis]

    allEpos_projected = np.matmul(allEpos,main_axis).flatten()

    arraySpacing = np.abs(np.diff(allEpos_projected))

    arraySpacing = arraySpacing[arraySpacing >1e-3]

    return main_axis, arraySpacing

def get_coeffs_objectiveCSD_Sphere(positions,electrodePos,allEpos,radius=None):
    """Compute objective CSD coefficients using a spherical region.

    A segment's coefficient is 1 if it lies within the given radius
    of the electrode, 0 otherwise. Default radius is 10 um.
    """
    _, arraySpacing = get_array_spacing(allEpos)

    if radius is None:
        radius = 10

    distances = np.linalg.norm(positions.values-electrodePos[:,np.newaxis],axis=0)

    coeffs = np.array((distances <= radius).astype(int))

    coeffs = pd.DataFrame(data=coeffs[np.newaxis,:])

    coeffs.columns = positions.columns

    return coeffs


def get_coeffs_objectiveCSD_Plane(compartment_positions,electrodePos,allEpos,planeThickness=None):
    """Compute objective CSD coefficients using an infinite plane.

    A segment's coefficient is 1 if its axial distance from the electrode
    plane is within the thickness, 0 otherwise. If no thickness is given,
    it is estimated from the inter-electrode spacing.
    """
    main_axis, arraySpacing = get_array_spacing(allEpos)

    if planeThickness is None:
        planeThickness = get_thickness(arraySpacing)

    axialDistances, _ = distances_in_planar_coords(compartment_positions,electrodePos,main_axis)

    coeffs = np.array((axialDistances <= planeThickness).astype(int)).flatten()

    coeffs = pd.DataFrame(data=coeffs[np.newaxis,:])

    coeffs.columns = compartment_positions.columns

    return coeffs

def get_thickness(arraySpacing):
    """Estimate plane/disk thickness as half the mean electrode spacing."""
    return np.abs(np.mean(arraySpacing)/2)

def calculate_axial_vectors(axialDistances,main_axis):
    """Build per-compartment axial displacement vectors along the main axis."""
    axialVectors = main_axis.T
    for i in range(len(axialDistances)-1):
        axialVectors = np.vstack((axialVectors,main_axis.T))

    axialVectors = axialVectors * axialDistances

    return axialVectors


def distances_in_planar_coords(compartment_positions, electrodePos, main_axis):
    """Decompose compartment positions into axial and radial distances.

    Projects each compartment's displacement from the electrode onto the
    array's main axis (axial) and the perpendicular plane (radial).
    """
    differenceVectors = compartment_positions.values - electrodePos[:,np.newaxis]

    axialDistances = np.matmul(differenceVectors.T,main_axis)

    axialVectors = calculate_axial_vectors(axialDistances,main_axis)

    radialVectors = differenceVectors - axialVectors.T

    radialDistances = np.linalg.norm(radialVectors,axis=0)

    return np.abs(axialDistances), radialDistances


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

def electrode_type(electrodeType):
    """Validate that the electrode type is recognized."""
    if electrodeType == 'LineSource' or electrodeType == 'PointSource' or electrodeType == 'DipoleReciprocity' or electrodeType == 'Reciprocity' or electrodeType == 'ObjectiveCSD_Sphere' or electrodeType == 'ObjectiveCSD_Disk' or electrodeType == 'ObjectiveCSD_Plane':
        return 0
    else:
        raise AssertionError("Electrode type not recognized")

def get_objectiveCSD_array(electrodeType,objective_csd_array_indices,objectiveCSD_count,electrodeNames, h5, electrodeIdx):
    """Determine which electrodes belong to the objective CSD array.

    If no explicit indices are given, all electrodes matching the type
    are used. Otherwise the provided subsampling indices are applied.
    """
    if objective_csd_array_indices is None:

        allTypes = []
        for electrode in electrodeNames:
            allTypes.append( h5['electrodes'][str(electrode)]['type'][()].decode() )

        arrayIdx = [i for i, e in enumerate(allTypes) if e==electrodeType]

    else:

        arrayIdx = processSubsampling(objective_csd_array_indices[objectiveCSD_count])

        if electrodeIdx not in arrayIdx:
            objectiveCSD_count += 1
            arrayIdx = processSubsampling(objective_csd_array_indices[objectiveCSD_count])
            if electrodeIdx not in arrayIdx:
                raise AssertionError('Electrode arrays used in objective CSD must be sequential in eletcrode file')

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

        electrodeType = h5['electrodes'][str(electrode)]['type'][()].decode()


        electrode_type(electrodeType)

        if electrodeType == 'LineSource':

            coeffs = get_coeffs_lineSource(positions,columns,epos,sigma[sigmaIdx])

            if len(sigma) > 1:
                sigmaIdx += 1

        else:

            newPositions = get_segment_midpts(positions,node_ids) # For other methods, we need the segment centers, not the endpoints


            if electrodeType == 'PointSource':

                coeffs = get_coeffs_pointSource(newPositions, epos, sigma[sigmaIdx])

                if len(sigma) > 1:
                    sigmaIdx += 1

            elif 'ObjectiveCSD' in electrodeType:

                arrayIdx, objectiveCSD_count = get_objectiveCSD_array(electrodeType, objective_csd_array_indices, objectiveCSD_count, electrodeNames, h5, electrodeIdx)

                allEpos = []

                for e in electrodeNames[arrayIdx]:
                    allEpos.append( h5['electrodes'][str(e)]['position'][:] )

                radius = h5['electrodes'][str(electrode)]['type'].attrs.get('radius',None)
                thickness = h5['electrodes'][str(electrode)]['type'].attrs.get('thickness', None)

                if electrodeType == 'ObjectiveCSD_Sphere':
                    coeffs = get_coeffs_objectiveCSD_Sphere(newPositions,epos,allEpos,radius)

                elif electrodeType == 'ObjectiveCSD_Disk':
                    coeffs = get_coeffs_objectiveCSD_Disk(newPositions,epos,allEpos,radius,thickness)

                elif electrodeType == 'ObjectiveCSD_Plane':
                    coeffs = get_coeffs_objectiveCSD_Plane(newPositions,epos,allEpos,thickness)


            else:

                if electrodeType == 'DipoleReciprocity':

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
