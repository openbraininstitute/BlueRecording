# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import h5py
import os
import pandas as pd
import sys
import bluepysnap as bp
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

    '''
    This class writes datasets to the h5 file
    '''

    def __init__(self, h5, lst_ids, electrodes, population_name):

        '''
        h5: h5 file returned by h5py.File(filename,'w')
        lst_ids: node ids
        electrodes: Dictionary with metadata about electrodes
        population_name: Sonata population
        '''


        dset = h5.create_dataset(population_name+"/node_ids", data=sorted(lst_ids))

        index = 0

        ### Iterates through electrode dictionary to write metadata
        for key, electrode in electrodes.items(): # Iterates through electrodes

            h5.create_dataset("electrodes/" + str(key) + '/'+population_name,data=index) # Index of the column corresponding to this electrode in /electrodes/{population_name}/scaling_factors
            index += 1

            for item in electrode.items(): # Iterates through metadata fields for each electrode

                if item[0] == 'type' and isinstance(item[1],dict): # If electrodetype is a dict produced by process_ObjectiveCSD(), write the real electrode type string and the metadata

                    dset = h5.create_dataset("electrodes/" + str(key) + '/' + item[0],
                                      data=item[1]['type'])

                    for entry in item[1].items(): # Write parameters for objective calculation

                        if entry[0] !='type':
                            dset.attrs.create(entry[0],entry[1])

                else:

                    h5.create_dataset("electrodes/" + str(key) + '/' + item[0],
                              data=item[1])
        ####

        self._ids = np.array(lst_ids)

    def file(self):
        return h5py.File(self._fn, "r+")

    def lengths(self, gid):
        if gid not in self._ids:
            raise AssertionError("gid not present")

        return "lengths/" + str(int(gid))

    def offsets(self,population_name):
        return population_name+"/offsets"

    def weights(self, population_name):

        return '/electrodes/'+population_name+'/scaling_factors'

def get_offsets(sectionIdsFrame):

    unique, counts = np.unique(sectionIdsFrame['id'].values,return_counts=True) # Unique node_ids and number of segments per node id

    out_offsets = np.hstack((np.array([0]),np.cumsum(counts))) # Offset from start of list for each node id

    return out_offsets

def write_all_neuron(sectionIdsFrame, population_name, h5, file, electrode_struc):

    file.create_dataset(h5.weights(population_name), data=np.ones([len(sectionIdsFrame['id'].values),len(electrode_struc.items())+1])) # Initializes /electrodes/{population_name}/scaling_factors with array of ones of size nSegments x (nElectrodes+1)

    out_offsets = get_offsets(sectionIdsFrame)

    file.create_dataset(h5.offsets(population_name), data=out_offsets) # The offset for each node in the scaling_factors field


def makeElectrodeDict(electrode_csv):

    '''
    Reads electrode metadata from input csv file and writes it to a dictionary
    '''

    electrode_df = pd.read_csv(electrode_csv,header=0,index_col=0)

    electrodes = {}

    for i in range(len(electrode_df.values)): # Iterates through each electrode in array

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

                electrodeType = process_objectiveCSD(electrodeType) # Returns a dict containing parameters for objective CSD electrodes

        else:
            electrodeType = 'LineSource'

        electrodes[name] = {'position': position,'type': electrodeType,
        'region':region,'layer':layer}


    return electrodes

def check_input_type_objectiveCSD(objectiveType,input):

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

    '''
    If the electrode is one of the objective CSD electrodes, this function processes the input string to determine the radius (for a sphere) or thickness (for disk and plane) and diameter (for disk)
    If these items are provided, returns them in a dict
    If these items aren't provided, then returns the electrode type as a string

    electrodeType is of format objectveCSD_CalculationMethod, objectveCSD_CalculationMethod_X, or objectveCSD_Disk_X_Y
    if calculationMethod == Sphere, X == radius, in um
    If calculationMethod == Plane, X == thickness, in um
    if calculationMethod == Disk, X == radius, Y == thickess, both in um
    '''


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

def initializeH5File(cols, population_name, outputfile, electrode_csv):

    '''
    Initializes the H5 electrode weights file on rank 0.

    Gathers rank-local cols via MPI, builds the global structure, and writes
    electrode metadata + offsets. The file is closed before returning.

    cols: rank-local (N, 2) int64 array of (gid, section) pairs
    population_name: SONATA population name
    outputfile: path to the output H5 file
    electrode_csv: path to the electrode CSV file
    '''

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Gather all rank-local cols to rank 0
    all_cols_list = comm.gather(cols, root=0)

    if rank == 0:
        all_cols = np.concatenate(all_cols_list, axis=0)
        node_ids = np.unique(all_cols[:, 0])

        section_ids_frame = pd.DataFrame(all_cols, columns=["id", "section"])

        electrodes = makeElectrodeDict(electrode_csv)

        h5file = h5py.File(outputfile, 'w')

        # Tune HDF5 metadata cache for faster writes
        h5id = h5file.id
        cc = h5id.get_mdc_config()
        cc.max_size = 1024 * 1024 * 124
        h5id.set_mdc_config(cc)

        h5 = ElectrodeFileStructure(h5file, node_ids, electrodes, population_name)

        write_all_neuron(section_ids_frame, population_name, h5, h5file, electrodes)

        h5file.close()

    comm.Barrier()


# ---------------------------------------------------------------------------
# Weight computation (formerly writeH5.py)
# ---------------------------------------------------------------------------

def add_data(h5, ids, coeffs ,population_name):



    dset = 'electrodes/'+population_name+'/scaling_factors'

    node_ids = h5[population_name+'/node_ids'][:]

    isInInput = np.isin(node_ids,ids)
    nodesInInput = node_ids[isInInput] # This contains the same values as the variable ids, but in the order as in node_ids
    idIndex = np.where(isInInput)[0]

    offset0 = h5[population_name+'/offsets'][idIndex] # Finds offset in  'electrodes/'+population_name+'/scaling_factors' for this particular node id

    offset1 = np.zeros_like(offset0)

    if np.any(idIndex ==  len(h5[population_name+'/offsets'][:])-1):

        lastNodeIdx =  np.where(idIndex == len(h5[population_name+'/offsets'][:])-1)[0]
        offset1[lastNodeIdx] = len(h5[dset][:]) # If this is the last node in the list, we write the coefficients up to the end of the coefficient array

    notLastNodeIdx = np.where(idIndex != len(h5[population_name+'/offsets'][:])-1)[0]
    offset1[notLastNodeIdx] =  h5[population_name+'/offsets'][idIndex[notLastNodeIdx]+1] # Otherwise, we write up to the offset for the next node

    for i, id in enumerate(nodesInInput):


        h5[dset][offset0[i]:offset1[i],:-1] = coeffs.loc[:,id].values.T

def line_source_cases(h,r2,l):


    if h < 0 and l < 0:

        lineSourceTerm = np.log(((h**2+r2)**.5-h)/((l**2+r2)**.5-l))

    elif h < 0 and l > 0:

        lineSourceTerm = np.log( ( ((h**2+r2)**.5-h)* (l + (l**2+r2)**.5 ) ) / r2  )

    elif h > 0 and l > 0:

        lineSourceTerm = np.log( ( (l + (l**2+r2)**.5 ) ) / ( (r2+h**2)**.5 + h)  )


    return lineSourceTerm

def get_line_coeffs(startPos,endPos,electrodePos,sigma):

    '''
    startPos and endPos are the starting and ending positions of the segment
    sigma is the extracellular conductivity
    '''

    ### Convert from um to m
    startPos = startPos * 1e-6
    endPos = endPos * 1e-6
    electrodePos = electrodePos * 1e-6
    ###

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

    segCoeff *= 1e-9 # Convert from nA to A

    return segCoeff


def get_coeffs_lineSource(positions,columns,electrodePos,sigma):

    for i in range(len(positions.columns)-1):

        if positions.columns[i][-1]==0: # Implies that it is a soma

            somaPos = positions.iloc[:,i]

            distance = np.linalg.norm(somaPos-electrodePos)

            distance *= 1e-6 # Converts from um to m

            somaCoeff = 1/(4*np.pi*sigma*distance) # We treat the soma as a point, so the contribution at the electrode follows the formula for the potential from a point source

            somaCoeff *= 1e-9 # Converts from nA to A

            if i == 0:
                coeffs = somaCoeff
            else:

                coeffs = np.hstack((coeffs,somaCoeff))

        elif positions.columns[i][-1]==positions.columns[i+1][-1]: # Ensures we are not at the far end of a section

            segCoeff = get_line_coeffs(positions.iloc[:,i],positions.iloc[:,i+1],electrodePos,sigma)

            coeffs = np.hstack((coeffs,segCoeff))


    coeffs = pd.DataFrame(data=coeffs[np.newaxis,:])

    coeffs.columns = columns

    return coeffs

def get_coeffs_pointSource(positions,electrodePos,sigma):

    distances = np.linalg.norm(positions.values-electrodePos[:,np.newaxis],axis=0)

    distances *= 1e-6 # Converts from um to m

    coeffs = 1/(4*np.pi*sigma*distances)

    coeffs *= 1e-9 # Converts from nA to A

    coeffs = pd.DataFrame(data=coeffs[np.newaxis,:])

    coeffs.columns = positions.columns

    return coeffs

def getArraySpacing(allEpos):

    ### Finds main axis of electrode array
    pca = PCA(n_components=1)
    pca.fit(allEpos)
    main_axis = pca.components_[0]/np.linalg.norm(pca.components_[0])
    main_axis = main_axis[:,np.newaxis]
    ###

    allEpos_projected = np.matmul(allEpos,main_axis).flatten()

    arraySpacing = np.abs(np.diff(allEpos_projected))

    arraySpacing = arraySpacing[arraySpacing >1e-3] # removes zeros in order to not take into account electrodes on the same plane

    return main_axis, arraySpacing

def get_coeffs_objectiveCSD_Sphere(positions,electrodePos,allEpos,radius=None):

    _, arraySpacing = getArraySpacing(allEpos)

    if radius is None:
        radius = 10 # Default value of 10 um

    distances = np.linalg.norm(positions.values-electrodePos[:,np.newaxis],axis=0) # in microns

    coeffs = np.array((distances <= radius).astype(int)) # Coeff is 1 if segment is within radius, zero otherwise

    coeffs = pd.DataFrame(data=coeffs[np.newaxis,:])

    coeffs.columns = positions.columns

    return coeffs


def get_coeffs_objectiveCSD_Plane(compartment_positions,electrodePos,allEpos,planeThickness=None):

    main_axis, arraySpacing = getArraySpacing(allEpos)

    if planeThickness is None: # If no value provided, estimates value from spacing between electrodes
        planeThickness = getThickness(arraySpacing) # Assumes that all electrodes are evenly spaced. TODO: Relax this assumption

    axialDistances, _ = distances_in_planar_coords(compartment_positions,electrodePos,main_axis)

    coeffs = np.array((axialDistances <= planeThickness).astype(int)).flatten() ### Coeff is 1 if segment is within infinite plane, zero otherwise

    coeffs = pd.DataFrame(data=coeffs[np.newaxis,:])

    coeffs.columns = compartment_positions.columns

    return coeffs

def getThickness(arraySpacing):

    # Given the spacing between electrodes in an array, calculates the thickness for objective plane and objective disk

    return np.abs(np.mean(arraySpacing)/2)

def calculate_axial_vectors(axialDistances,main_axis):

    axialVectors = main_axis.T # Size 1x3
    for i in range(len(axialDistances)-1):
        axialVectors = np.vstack((axialVectors,main_axis.T))

    axialVectors = axialVectors * axialDistances

    return axialVectors


def distances_in_planar_coords(compartment_positions, electrodePos, main_axis):

    '''
    For a disk or plane perpendicular to main_axis, returns the axial and radial coordinates of each of the compartment positions
    '''

    ### Projects compartment positions onto plane, containing the point electrodePos, normal to electrode array
    differenceVectors = compartment_positions.values - electrodePos[:,np.newaxis]

    axialDistances = np.matmul(differenceVectors.T,main_axis) # Size len(compartment_positions)x1

    axialVectors = calculate_axial_vectors(axialDistances,main_axis) # Projection of diffence vector onto the main axis of the electrode array

    radialVectors = differenceVectors - axialVectors.T

    radialDistances = np.linalg.norm(radialVectors,axis=0) # in microns

    return np.abs(axialDistances), radialDistances


def get_coeffs_objectiveCSD_Disk(compartment_positions,electrodePos,allEpos,radius=None,diskThickness=None):

    if radius is None:
        radius = 500 # Default radius of 500 um

    main_axis, arraySpacing = getArraySpacing(allEpos)

    if diskThickness is None: # If no disk thickness provided, estimate value from spacing between electrodes
        diskThickness = getThickness(arraySpacing) # Assumes that all electrodes are evenly spaced. TODO: Relax this assumption

    axialDistances, radialDistances = distances_in_planar_coords(compartment_positions,electrodePos,main_axis)

    ###

    ### Coeff is 1 if segment is within disk, zero otherwise
    coeffs1 = np.array((radialDistances <= radius).astype(int)).flatten()
    coeffs2 = np.array((axialDistances <= diskThickness).astype(int)).flatten()

    coeffs = coeffs1 * coeffs2
    ###

    coeffs = pd.DataFrame(data=coeffs[np.newaxis,:])

    coeffs.columns = compartment_positions.columns

    return coeffs

def geth5Dataset(h5f, group_name, dataset_name):
    """
    Find and get dataset from h5 file.
    out = geth5Dataset(h5f, group_name, dataset_name)
    h5f - string - h5 file path and name
    group_name - string - where to initiate search, '/' for root
    dataset_name - string - dataset to be found
    return - numpy array
    """

    def find_dataset(name):
        """ Find first object with dataset_name anywhere in the name """
        if dataset_name in name:
            return name

    with h5py.File(h5f, 'r') as f:
        k = f[group_name].visit(find_dataset)
        return f[group_name + '/' + k][()]

def get_coeffs_dipoleReciprocity(compartment_positions, path_to_fields,center):

    '''
    path_to_fields is the path to the h5 file containing the potential field, outputted from Sim4Life
    '''


    positionColumns = compartment_positions.columns
    compartment_positions = compartment_positions.values


    # Get new output file potential field

    with h5py.File(path_to_fields, 'r') as f:
        for i in f['FieldGroups']:
            tmp = 'FieldGroups/' + i + '/AllFields/EM E(x,y,z,f0)/_Object/Snapshots/0/'

        Ex = geth5Dataset(path_to_fields, tmp, 'comp0')
        Ey = geth5Dataset(path_to_fields, tmp, 'comp1')
        Ez = geth5Dataset(path_to_fields, tmp, 'comp2')

        for i in f['Meshes']:
            tmp = 'Meshes/'+i
            break
        x = geth5Dataset(path_to_fields, tmp, 'axis_x')
        y = geth5Dataset(path_to_fields, tmp, 'axis_y')
        z = geth5Dataset(path_to_fields, tmp, 'axis_z')

        ### E field is cell-centered, so we need to take midpoints of mesh
        xCenter = (x[:-1]+x[1:])/2
        yCenter = (y[:-1]+y[1:])/2
        zCenter = (z[:-1]+z[1:])/2
        ####

        currentApplied = f['CurrentApplied'][()] # The potential field should have a current, but if not, just assume it is 1


    compartment_positions = compartment_positions * 1e-6 # Converts um to m, to match the potential field file

    center = center * 1e-6

    compartment_positions_New = compartment_positions - center.values[:,np.newaxis]


    InterpFcnX = RegularGridInterpolator((xCenter, y, z), Ex[:, :, :, 0], method='linear')
    InterpFcnY = RegularGridInterpolator((x, yCenter, z), Ey[:, :, :, 0], method='linear')
    InterpFcnZ = RegularGridInterpolator((x, y, zCenter), Ez[:, :, :, 0], method='linear')

    XComp = InterpFcnX(center)[np.newaxis]  # Interpolate E field at location of neural center

    YComp = InterpFcnY(center)[np.newaxis]  # Interpolate E field at location of neural center

    ZComp = InterpFcnZ(center)[np.newaxis]  # Interpolate E field at location of neural center


    out2rat = compartment_positions_New[0]*XComp + compartment_positions_New[1]*YComp + compartment_positions_New[2]*ZComp


    outdf = pd.DataFrame(data=(-out2rat / currentApplied), columns=positionColumns) # Scale potential field by applied current

    return outdf

def get_coeffs_reciprocity(compartment_positions, path_to_fields):

    '''
    path_to_fields is the path to the h5 file containing the potential field, outputted from Sim4Life
    path_to_positions is the path to the output from the position-finding script
    '''

    # Get new output file potential field

    with h5py.File(path_to_fields, 'r') as f:
        for i in f['FieldGroups']:
            tmp = 'FieldGroups/' + i + '/AllFields/EM Potential(x,y,z,f0)/_Object/Snapshots/0/'
        pot = geth5Dataset(path_to_fields, tmp, 'comp0')
        for i in f['Meshes']:
            tmp = 'Meshes/'+i
            break
        x = geth5Dataset(path_to_fields, tmp, 'axis_x')
        y = geth5Dataset(path_to_fields, tmp, 'axis_y')
        z = geth5Dataset(path_to_fields, tmp, 'axis_z')


        currentApplied = f['CurrentApplied'][()] # The potential field should have a current, but if not, just assume it is 1


    compartment_positions *= 1e-6 # Converts um to m, to match the potential field file

    xSelect = compartment_positions.values[0]
    ySelect = compartment_positions.values[1]
    zSelect = compartment_positions.values[2]


    selections = np.array([xSelect, ySelect, zSelect]).T


    InterpFcn = RegularGridInterpolator((x, y, z), pot[:, :, :, 0], method='linear')

    out2rat = InterpFcn(selections)[np.newaxis]  # Interpolate potential field at location of neural segments


    outdf = pd.DataFrame(data=(out2rat / currentApplied), columns=compartment_positions.columns) # Scale potential field by applied current

    return outdf

def getNeuronSegmentMidpts(position):
    '''
    Gets midpoints for a single neuron
    '''


    secIds = np.array(list(position.columns))[:,1]
    uniqueSecIds = np.unique(secIds)

    for sId in uniqueSecIds: # Iterates through sections

        pos = position.iloc[:,np.where(sId == secIds)[0]]

        if sId == 0: # Implies that section is a soma, so we just take the position from the file

            newPos = pos

        elif np.shape(pos.values)[-1] == 1: # If there is only one point in the section, we just take the value
            newPos = pd.concat((newPos,pos),axis=1)

        else: # We take the midpoints of the values in the file, which are the endpoints of the segments
            pos = (pos.iloc[:,:-1]+pos.iloc[:,1:])/2

            newPos = pd.concat((newPos,pos),axis=1)

    return newPos

def getSegmentMidpts(positions,node_ids):

    newPos = (
    positions.T
        .groupby(level=0, group_keys=False)
        .apply(lambda g: getNeuronSegmentMidpts(g.T).T)
        .T
    )

    return newPos



def sort_electrode_names(electrodeKeys,population_name):

    electrodeNames = np.array(list(electrodeKeys))

    electrodeNames = electrodeNames[np.where(electrodeNames!=population_name)] # The field /electrodes/{population_name} contains the scaling factors, not the metadata

    electrode_list = []

    for e in electrodeNames:

        try:
            name = int(e)

        except:
            name = e

        electrode_list.append(name)

    electrode_list = np.sort(electrode_list)

    return electrode_list

def ElectrodeType(electrodeType):

    if electrodeType == 'LineSource' or electrodeType == 'PointSource' or electrodeType == 'DipoleReciprocity' or electrodeType == 'Reciprocity' or electrodeType == 'ObjectiveCSD_Sphere' or electrodeType == 'ObjectiveCSD_Disk' or electrodeType == 'ObjectiveCSD_Plane':
        return 0
    else:
        raise AssertionError("Electrode type not recognized")

def get_objectiveCSD_array(electrodeType,objective_csd_array_indices,objectiveCSD_count,electrodeNames, h5, electrodeIdx):

    if objective_csd_array_indices is None: # Assume all electrodes of given type are used to calculate CSD

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

def writeH5File(positions, cols, population_name, outputfile, sigma=None, path_to_fields=None, objective_csd_array_indices=None):

    '''
    positions is the DataFrame of segment boundary positions (output of get_positions)
    cols is the (N, 2) array of (gid, section) pairs describing every compartment on this rank
    population_name is the SONATA population name
    outputfile is the h5 file containing the compartment weights
    '''

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

    reciprocityIdx = 0 # Keeps track of number of non-analytical electrodes
    sigmaIdx = 0 # Keeps track of number of analytical electrodes
    objectiveCSD_count = 0 # Keeps track of number of objective CSD electrodes

    for electrodeIdx, electrode in enumerate(electrodeNames):


        epos = h5['electrodes'][str(electrode)]['position'][:] # Gets position for each electrode


        electrodeType = h5['electrodes'][str(electrode)]['type'][()].decode() # Gets type for each electrode


        ElectrodeType(electrodeType)

        if electrodeType == 'LineSource':

            coeffs = get_coeffs_lineSource(positions,columns,epos,sigma[sigmaIdx])

            if len(sigma) > 1:
                sigmaIdx += 1

        else:

            newPositions = getSegmentMidpts(positions,node_ids) # For other methods, we need the segment centers, not the endpoints


            if electrodeType == 'PointSource':

                coeffs = get_coeffs_pointSource(newPositions, epos, sigma[sigmaIdx])

                if len(sigma) > 1:
                    sigmaIdx += 1

            elif 'ObjectiveCSD' in electrodeType:

                arrayIdx, objectiveCSD_count = get_objectiveCSD_array(electrodeType, objective_csd_array_indices, objectiveCSD_count, electrodeNames, h5, electrodeIdx)

                allEpos = [] # List of electrode positions used to calculate CSD

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

    h5.close()

    return 0
