# SPDX-License-Identifier: GPL-3.0-or-later
import json
import MEAutility as MEA
import pandas as pd
import bluepysnap as bp
import sys
import numpy as np
from pathlib import Path
from voxcell.nexus.voxelbrain import Atlas
from sklearn.decomposition import PCA
from bluerecording.utils import get_circuit_path


def getAtlasInfo(path_to_simconfig,electrodePositions):

    '''
    For an array of electrode positions, returns brain region and layer in which each electrode is located. 
    '''
    
    circuitpath = get_circuit_path(path_to_simconfig)

    with open(circuitpath) as f:
        path_to_atlas = json.load(f)['components']['provenance']['atlas_dir']

    path_to_atlas = str((Path(circuitpath).parent / path_to_atlas).resolve())

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

def repositionElectrode(probe,center,azimuth,elevation):

    probe.rotate([0,1,0],elevation*180/np.pi)
    probe.rotate([0,0,1],azimuth*180/np.pi)
    probe.move(center)

    return(probe)


if __name__=='__main__':

    probe_name = sys.argv[1]
    path_to_simconfig = sys.argv[2]
    electrode_csv = sys.argv[3]

    probe = MEA.return_mea(probe_name)

    center, azimuth, elevation = alignmentInfo(path_to_simconfig,'hex0')

    repositionElectrode(probe, center, azimuth, elevation)

    electrodePositions = probe.positions

    regionList, layerList = getAtlasInfo(path_to_simconfig, electrodePositions)

    electrodeData = pd.DataFrame(data=electrodePositions,columns=['x','y','z'])
    
    electrodeTypeList = []
    for p in electrodePositions:
        electrodeTypeList.append('LineSource')

    layerData = pd.DataFrame(data=layerList,columns=['layer'])

    regionData = pd.DataFrame(data=regionList,columns=['region'])
    
    electrodeTypeData = pd.DataFrame(data=electrodeTypeList,columns=['type'])

    data = pd.concat((electrodeData,layerData),axis=1)
    data = pd.concat((data,regionData),axis=1)
    data = pd.concat((data,electrodeTypeData),axis=1)

    data.to_csv(electrode_csv)
