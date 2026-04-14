# SPDX-License-Identifier: GPL-3.0-or-later
import json
import numpy as np
import os
from voxcell.nexus.voxelbrain import Atlas
from sklearn.decomposition import PCA

def concretize_path(known_path, newpath):

    '''
    Given a path to a particular file known_path, and a different path newpath which is defined relative to the file in known_path, returns an absolute path to newpath
    '''

    absolute_path = os.path.abspath(known_path)

    known_filename = known_path.split('/')[-1]

    path_to_dir = absolute_path.rstrip(known_filename)

    if newpath[0] != '/': # Checks that newpath is not already an absolute path

        newpath = path_to_dir+newpath

    newpath = os.path.normpath(newpath)
    
    return newpath
    

def getCircuitPath(path_to_simconfig):

    '''
    circuit: Path to the circuit used to generate the time steps. Gets written to the h5 file and is checked by neurodamus when and LFP simulation is run. LFP simulation will fail if it uses a different circuit than the one in the h5 file
    '''

    with open(path_to_simconfig) as f:

        circuitpath = json.load(f)['network']
    
    circuitpath =  concretize_path(path_to_simconfig, circuitpath)
    

    return circuitpath

def getAtlasInfo(path_to_simconfig,electrodePositions):

    '''
    For an array of electrode positions, returns brain region and layer in which each electrode is located. 
    '''
    
    circuitpath = getCircuitPath(path_to_simconfig)

    with open(circuitpath) as f:
        path_to_atlas = json.load(f)['components']['provenance']['atlas_dir']

    path_to_atlas = concretize_path(circuitpath,path_to_atlas)

    atlas = Atlas.open(path_to_atlas)
    brain_regions = atlas.load_data('brain_regions')

    region_map = atlas.load_region_map()

    regionList = []
    layerList = []


    for position in electrodePositions:

        try:

            for id_ in brain_regions.lookup([position]):

                region = region_map.get(id_, 'acronym')
                regionList.append(region.split(';')[0])
                layerList.append(region.split(';')[1])

        except:

            regionList.append('Outside')
            layerList.append('Outside')

    return regionList, layerList

def alignmentInfo(path_to_simconfig,target):

    '''
    Gets loction and angle information in order to align a probe with long axis of of the specified target (typically a cortical column)
    '''
    
    population = getPopulationObject(path_to_simconfig)

    somaPos = population.get(properties=['x','y','z'],group=target) # Gets soma position

    center = np.mean(somaPos,axis=0).values

    pca = PCA(n_components=3)
    pca.fit(somaPos)
    main_axis = pca.components_[0]

    elevation = np.arctan2(np.sqrt(main_axis[0]**2+main_axis[1]**2),main_axis[2])
    azimuth = np.arctan2(main_axis[1],main_axis[0])

    return center, azimuth, elevation
    
