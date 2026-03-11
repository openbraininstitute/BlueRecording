# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as R

from bluerecording import getPositions

def test_MutableMorph(morphology):

    morph = getPositions.MutableMorph(morphology)

    assert morph.indices == [[0,1,2,3],[4,5],[6,7,8]]

@pytest.mark.skip_in_ci
def test_get_morph_path(path_to_simconfig_with_output,expected_path_to_morph):

    neuron_id = 0

    _, _, population = getPositions.getSimulationInfo(path_to_simconfig_with_output)

    morph_path = getPositions.get_morph_path(population, neuron_id, path_to_simconfig_with_output)

    assert morph_path==expected_path_to_morph

def test_get_axon_points(morphology, somaPos):

    morphology = getPositions.MutableMorph(morphology)

    points, lengths = getPositions.get_axon_points(morphology, somaPos)
    expectedLengths = np.array([0,1,2,3,1073])

    expectedPoints = np.array([[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,1073]])
    np.testing.assert_almost_equal(lengths,expectedLengths,decimal=2)
    np.testing.assert_almost_equal(points,expectedPoints,decimal=2)

def test_apply_transform(morphology_trivial):

    center = np.array([1,1,1])

    r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])

    expectedPoints = np.array([[1.,1.,1.],[1.,2.,1.]])

    m = getPositions.apply_transform(getPositions.MutableMorph(morphology_trivial), center, r)

    np.testing.assert_almost_equal(m.points, expectedPoints)

def test_get_axon_points_extrapolate(morphology_short, somaPos):

    morphology_short = getPositions.MutableMorph(morphology_short)

    points, lengths = getPositions.get_axon_points(morphology_short, somaPos)
    expectedLengths = np.array([0,1,2,3,4,1060])

    expectedPoints = np.array([[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4],[0,0,1060]])
    np.testing.assert_almost_equal(lengths,expectedLengths,decimal=2)

    np.testing.assert_almost_equal(points,expectedPoints,decimal=2)

def test_checkAxonsFirst(morphology_short,morphology_short_dendFirst):

    assert getPositions.checkAxonsFirst(morphology_short)
    assert not getPositions.checkAxonsFirst(morphology_short_dendFirst)

def test_getNewIdx(data):

    colIdx = data.columns
    expectedColumns = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,2, 2, 2, 2, 2, 2],[0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,10,10,10,10,10,10,0,1,1,1,1,1,1]]

    expectedIdx = list(zip(*expectedColumns))

    expectedMultiIndex = pd.MultiIndex.from_tuples(expectedIdx,names=['id','section'])

    newIdx = getPositions.getNewIndex(colIdx)

    pd.testing.assert_index_equal(newIdx, expectedMultiIndex)

def test_interpolate_dendrite(data, morphology):

    morphology = getPositions.MutableMorph(morphology)

    colIdx = data.columns # GID and Section IDs for each cell
    cols = np.array(list(data.columns))

    i = 1 # gid

    sections = np.unique(cols[np.where(cols[:,0]==i),1:].flatten()) # List of sections for the given neuron

    secName = sections[3]

    numCompartments = np.shape(data[i][secName])[-1]

    sec = morphology.sections[secName-1] # If the section is not dendritic, this will trhow an Exception
    pts = sec.points

    secPts = np.array(pts)

    segPos = getPositions.interp_points(secPts,numCompartments)

    expectedSegPos = np.array([[0,0,0],[33.33,0,0],[66.66,0,0],[100,0,0]])

    np.testing.assert_almost_equal(segPos,expectedSegPos,decimal=2)

def test_interpolate_dendrite_dedFirst(data, morphology_dendFirst):

    morphology = getPositions.MutableMorph(morphology_dendFirst)

    colIdx = data.columns # GID and Section IDs for each cell
    cols = np.array(list(data.columns))

    i = 1 # gid

    sections = np.unique(cols[np.where(cols[:,0]==i),1:].flatten()) # List of sections for the given neuron

    secName = sections[3]

    numCompartments = np.shape(data[i][secName])[-1]

    morphSectionIdx = 0

    if morphology.sections[morphSectionIdx].type == 3 or morphology.sections[morphSectionIdx].type == 4: # If dendrite

        ptIdx = morphology.indices[morphSectionIdx]
        pts = morphology.points[ptIdx]

        secPts = np.array(pts)

        segPos = getPositions.interp_points(secPts,numCompartments)

    expectedSegPos = np.array([[0,0,0],[33.33,0,0],[66.66,0,0],[100,0,0]])

    np.testing.assert_almost_equal(segPos,expectedSegPos,decimal=2)


def test_interpolate_AIS(data,morphology, somaPos):

    morphology = getPositions.MutableMorph(morphology)

    colIdx = data.columns # GID and Section IDs for each cell
    cols = np.array(list(data.columns))

    i = 1 # gid

    sections = np.unique(cols[np.where(cols[:,0]==i),1:].flatten()) # List of sections for the given neuron

    secName = sections[1]

    numCompartments = np.shape(data[i][secName])[-1]

    axonPoints, runningLens = getPositions.get_axon_points(morphology, somaPos)

    segPos = getPositions.interp_points_axon(axonPoints,runningLens,secName,numCompartments,somaPos)

    expectedSegPos = np.array([[0,0,0],[0,0,6],[0,0,12],[0,0,18],[0,0,24],[0,0,30]])

    np.testing.assert_almost_equal(segPos,expectedSegPos,decimal=2)

def test_interpolate_AIS_farAxon(data,morphology_farAxon, somaPos):

    '''
    Makes sure that edge case in which only the soma itself is less than 30 um away from the soma is properly handled
    '''

    morphology_farAxon = getPositions.MutableMorph(morphology_farAxon)

    colIdx = data.columns # GID and Section IDs for each cell
    cols = np.array(list(data.columns))

    i = 1 # gid

    sections = np.unique(cols[np.where(cols[:,0]==i),1:].flatten()) # List of sections for the given neuron

    secName = sections[1]

    numCompartments = np.shape(data[i][secName])[-1]

    axonPoints, runningLens = getPositions.get_axon_points(morphology_farAxon, somaPos)


    segPos = getPositions.interp_points_axon(axonPoints,runningLens,secName,numCompartments,somaPos)

    expectedSegPos = np.array([[0,0,0],[0,0,6],[0,0,12],[0,0,18],[0,0,24],[0,0,30]])

    np.testing.assert_almost_equal(segPos,expectedSegPos,decimal=2)

def test_interpolate_AIS_short(data,morphology_short, somaPos):

    '''
    Tests the case where no point is greater than 30 um away from the soma
    '''

    morphology_short = getPositions.MutableMorph(morphology_short)

    colIdx = data.columns # GID and Section IDs for each cell
    cols = np.array(list(data.columns))

    i = 1 # gid

    sections = np.unique(cols[np.where(cols[:,0]==i),1:].flatten()) # List of sections for the given neuron

    secName = sections[1]

    numCompartments = np.shape(data[i][secName])[-1]

    axonPoints, runningLens = getPositions.get_axon_points(morphology_short, somaPos)


    segPos = getPositions.interp_points_axon(axonPoints,runningLens,secName,numCompartments,somaPos)

    expectedSegPos = np.array([[0,0,0],[0,0,6],[0,0,12],[0,0,18],[0,0,24],[0,0,30]])

    np.testing.assert_almost_equal(segPos,expectedSegPos,decimal=2)

def test_interpolate_AIS_2(data,morphology, somaPos):

    '''
    Tests the case where no point is between 30 and 60 um from the soma, but there is one farther than 60 um
    '''

    morphology = getPositions.MutableMorph(morphology)

    colIdx = data.columns # GID and Section IDs for each cell
    cols = np.array(list(data.columns))

    i = 1 # gid

    sections = np.unique(cols[np.where(cols[:,0]==i),1:].flatten()) # List of sections for the given neuron

    secName = sections[2]

    numCompartments = np.shape(data[i][secName])[-1]

    axonPoints, runningLens = getPositions.get_axon_points(morphology, somaPos)

    segPos = getPositions.interp_points_axon(axonPoints,runningLens,secName,numCompartments,somaPos)

    expectedSegPos = np.array([[0,0,30],[0,0,36],[0,0,42],[0,0,48],[0,0,54],[0,0,60]])

    np.testing.assert_almost_equal(segPos,expectedSegPos,decimal=2)

def test_interpolate_AIS_2_short(data,morphology_short, somaPos):

    '''
    Tests the case where no points greater than 30 um from the soma
    '''

    morphology_short = getPositions.MutableMorph(morphology_short)

    colIdx = data.columns # GID and Section IDs for each cell
    cols = np.array(list(data.columns))

    i = 1 # gid

    sections = np.unique(cols[np.where(cols[:,0]==i),1:].flatten()) # List of sections for the given neuron

    secName = sections[2]

    numCompartments = np.shape(data[i][secName])[-1]

    axonPoints, runningLens = getPositions.get_axon_points(morphology_short, somaPos)

    segPos = getPositions.interp_points_axon(axonPoints,runningLens,secName,numCompartments,somaPos)

    expectedSegPos = np.array([[0,0,30],[0,0,36],[0,0,42],[0,0,48],[0,0,54],[0,0,60]])

    np.testing.assert_almost_equal(segPos,expectedSegPos,decimal=2)

def test_interpolate_myelin(data,morphology, somaPos):

    morphology = getPositions.MutableMorph(morphology)

    colIdx = data.columns # GID and Section IDs for each cell
    cols = np.array(list(data.columns))

    i = 1 # gid

    sections = np.unique(cols[np.where(cols[:,0]==i),1:].flatten()) # List of sections for the given neuron

    secName = sections[-1]

    numCompartments = np.shape(data[i][secName])[-1]

    axonPoints, runningLens = getPositions.get_axon_points(morphology, somaPos)

    segPos = getPositions.interp_points_axon(axonPoints,runningLens,secName,numCompartments,somaPos)

    expectedSegPos = np.array([[0,0,60],[0,0,260],[0,0,460],[0,0,660],[0,0,860],[0,0,1060]])

    np.testing.assert_almost_equal(segPos,expectedSegPos,decimal=2)

def test_interpolate_myelin_short(data,morphology_short, somaPos):

    morphology_short = getPositions.MutableMorph(morphology_short)

    colIdx = data.columns # GID and Section IDs for each cell
    cols = np.array(list(data.columns))

    i = 1 # gid

    sections = np.unique(cols[np.where(cols[:,0]==i),1:].flatten()) # List of sections for the given neuron

    secName = sections[-1]

    numCompartments = np.shape(data[i][secName])[-1]

    axonPoints, runningLens = getPositions.get_axon_points(morphology_short, somaPos)

    segPos = getPositions.interp_points_axon(axonPoints,runningLens,secName,numCompartments,somaPos)

    expectedSegPos = np.array([[0,0,60],[0,0,260],[0,0,460],[0,0,660],[0,0,860],[0,0,1060]])

    np.testing.assert_almost_equal(segPos,expectedSegPos,decimal=2)

@pytest.mark.skip_in_ci
def test_circuit_get_positions():
    path_to_simconfig = "examples/circuitTest/data/simulation/simulation_config.json"
    path_to_positions_folder = "examples/circuitTest/data/getPositions/positions"
    getPositions.getPositions(path_to_simconfig=path_to_simconfig, 
                 files_per_folder=50, path_to_positions_folder=path_to_positions_folder)

    ref_path = "examples/circuitTest/data/getPositions/positions0_ref.pkl"
    new_path = "examples/circuitTest/data/getPositions/positions/0/positions0.pkl"

    df_ref = pd.read_pickle(ref_path)
    df_new = pd.read_pickle(new_path)

    # Index and column structure must match exactly
    assert df_ref.index.equals(df_new.index)
    assert df_ref.columns.equals(df_new.columns)

    # Allow small numerical differences
    pd.testing.assert_frame_equal(
        df_ref,
        df_new,
        check_exact=False,
        rtol=1e-5,
        atol=1e-8,
    )
