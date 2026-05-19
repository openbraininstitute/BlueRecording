# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared test helpers — plain functions, not fixtures."""

import h5py
import numpy as np
import pandas as pd
from morphio import Morphology

from bluerecording.weights import (
    Electrode,
    ElectrodeType,
    ObjectiveCSDParams,
    _init_scaling_factors_and_offsets,
    _write_electrode_metadata_to_h5,
)

# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------

GIDS = [1, 2]
POPULATION_NAME = "testPopulation"
SOMA_POS = np.array([0, 0, 0])


def make_report_columns():
    """Columns mimicking a voltage report (gid, section pairs)."""
    columns = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 10, 10, 10, 10, 10, 0, 1, 1, 1, 1, 1],
    ]
    idx = list(zip(*columns, strict=False))
    return pd.MultiIndex.from_tuples(idx, names=["id", "section"])


def make_report_data():
    cols = make_report_columns()
    return pd.DataFrame(data=np.zeros([1, len(cols)]), columns=cols)


def make_report_data_backwards():
    columns = [
        [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 10, 10, 10, 10, 10],
    ]
    idx = list(zip(*columns, strict=False))
    mi = pd.MultiIndex.from_tuples(idx, names=["id", "section"])
    return pd.DataFrame(data=np.zeros([1, len(mi)]), columns=mi)


def make_sec_counts():
    data = make_report_data()
    frame = data.columns.to_frame()
    frame.index = range(len(frame))
    return frame


def make_electrodes():
    return [
        Electrode(
            name="name",
            position=np.array([1, 2, 3]),
            type=ElectrodeType.RECIPROCITY,
            region="Outside",
            layer="Outside",
        )
    ]


def make_electrodes_objective():
    return [
        Electrode(
            name="name",
            position=np.array([1, 2, 3]),
            type=ObjectiveCSDParams(electrode_type=ElectrodeType.OBJECTIVE_CSD_DISK, radius=500, thickness=10),
            region="Outside",
            layer="Outside",
        )
    ]


def make_electrodes_objective_array():
    return [
        Electrode(
            name="a", position=np.array([1, 0, 0]), type=ElectrodeType.RECIPROCITY, region="Outside", layer="Outside"
        ),
        Electrode(
            name="b", position=np.array([1, 0, 0]), type=ElectrodeType.RECIPROCITY, region="Outside", layer="Outside"
        ),
        Electrode(
            name="name",
            position=np.array([1, 0, 0]),
            type=ElectrodeType.OBJECTIVE_CSD_DISK,
            region="Outside",
            layer="Outside",
        ),
        Electrode(
            name="name1",
            position=np.array([2, 0, 0]),
            type=ElectrodeType.OBJECTIVE_CSD_DISK,
            region="Outside",
            layer="Outside",
        ),
        Electrode(
            name="name2",
            position=np.array([1, 0, 0]),
            type=ElectrodeType.OBJECTIVE_CSD_DISK,
            region="Outside",
            layer="Outside",
        ),
        Electrode(
            name="name3",
            position=np.array([2, 0, 0]),
            type=ElectrodeType.OBJECTIVE_CSD_DISK,
            region="Outside",
            layer="Outside",
        ),
    ]


def make_two_section_positions():
    """Position dataframe: soma + one section with start/end."""
    columns = [[1, 1, 1], [0, 1, 1]]
    idx = list(zip(*columns, strict=False))
    mi = pd.MultiIndex.from_tuples(idx, names=["id", "section"])
    values = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return pd.DataFrame(data=values, columns=mi)


def make_two_section_data():
    columns = [[1, 1], [0, 1]]
    idx = list(zip(*columns, strict=False))
    mi = pd.MultiIndex.from_tuples(idx, names=["id", "section"])
    return pd.DataFrame(data=np.zeros([1, 2]), columns=mi)


# ---------------------------------------------------------------------------
# H5 file builders
# ---------------------------------------------------------------------------


def create_electrode_file(path, electrodes, gids=GIDS, population=POPULATION_NAME):
    """Create an initialized electrode H5 file."""
    with h5py.File(path, "w") as h5file:
        _write_electrode_metadata_to_h5(h5file, gids, electrodes, population)
    return path


def create_neuron_file(path, electrodes=None, gids=GIDS, population=POPULATION_NAME):
    """Create electrode file with neuron weights initialized to ones."""
    if electrodes is None:
        electrodes = make_electrodes()
    path = create_electrode_file(path, electrodes, gids, population)
    sec_counts = make_sec_counts()
    h5file = h5py.File(path, "r+")
    _init_scaling_factors_and_offsets(sec_counts, population, h5file, electrodes)
    h5file.close()
    return path


# ---------------------------------------------------------------------------
# Morphology builders
# ---------------------------------------------------------------------------


def create_morphology(path, structure, points):
    """Write an H5 morphology file and return a Morphology object."""
    with h5py.File(path, "w") as f:
        f.create_dataset("structure", data=np.array(structure))
        f.create_dataset("points", data=np.array(points))
    return Morphology(str(path))


def make_morphology(path):
    """Standard morphology: soma + 2 axon sections + 1 dendrite."""
    return create_morphology(
        path,
        structure=[[0, 1, -1], [3, 2, 0], [7, 2, 1], [9, 3, 0]],
        points=[
            [-1, 0, 0, 1],
            [0, -1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 0.3],
            [0, 0, 2, 0.3],
            [0, 0, 3, 1],
            [0, 0, 3, 1],
            [0, 0, 1073, 1],
            [0, 0, 0, 1],
            [10, 0, 0, 5],
            [100, 0, 0, 5],
        ],
    )


def make_morphology_short(path):
    """Short morphology: no axon point > 30 um from soma."""
    return create_morphology(
        path,
        structure=[[0, 1, -1], [3, 2, 0], [7, 2, 1], [9, 3, 0]],
        points=[
            [-1, 0, 0, 1],
            [0, -1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 0.3],
            [0, 0, 2, 0.3],
            [0, 0, 3, 1],
            [0, 0, 3, 1],
            [0, 0, 4, 1],
            [0, 0, 0, 1],
            [10, 0, 0, 5],
            [100, 0, 0, 5],
        ],
    )


def make_morphology_far_axon(path):
    """Morphology where only the soma is < 30 um from soma."""
    return create_morphology(
        path,
        structure=[[0, 1, -1], [3, 2, 0], [5, 3, 0]],
        points=[
            [-1, 0, 0, 1],
            [0, -1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 1073, 1],
            [0, 0, 0, 1],
            [10, 0, 0, 5],
            [100, 0, 0, 5],
        ],
    )


def make_morphology_two_axon_branches(path):
    """Two short axonal branches: first leaf is longer than second.

    Both branches are shorter than 1060 µm, so extrapolation is needed.
    The first leaf (section 2, 100 µm) is longer than the second
    (section 3, 10 µm).  The bug fixed in 699fa38 would have picked the
    last leaf (shorter) instead of the longest one.

    Structure::

        soma (sec 0)
          └─ axon root (sec 1): 0→5 µm along z
               ├─ branch A (sec 2, leaf): 5→100 µm along z  (longest)
               └─ branch B (sec 3, leaf): 5→10 µm along z   (shorter)
          └─ dendrite (sec 4): 0→100 µm along x
    """
    return create_morphology(
        path,
        structure=[
            [0, 1, -1],  # sec 0: soma, 3 points starting at idx 0
            [3, 2, 0],  # sec 1: axon root, 2 points starting at idx 3
            [5, 2, 1],  # sec 2: axon branch A (leaf), 2 points at idx 5
            [7, 2, 1],  # sec 3: axon branch B (leaf), 2 points at idx 7
            [9, 3, 0],  # sec 4: dendrite, 3 points at idx 9
        ],
        points=[
            # soma (3 pts)
            [-1, 0, 0, 1],
            [0, -1, 0, 1],
            [0, 0, 0, 1],
            # sec 1: axon root (2 pts)
            [0, 0, 0, 1],
            [0, 0, 5, 0.3],
            # sec 2: axon branch A — longer leaf (2 pts)
            [0, 0, 5, 0.3],
            [0, 0, 100, 0.3],
            # sec 3: axon branch B — shorter leaf (2 pts)
            [0, 0, 5, 0.3],
            [0, 0, 10, 0.3],
            # sec 4: dendrite (3 pts)
            [0, 0, 0, 1],
            [10, 0, 0, 5],
            [100, 0, 0, 5],
        ],
    )


# ---------------------------------------------------------------------------
# Field file builders
# ---------------------------------------------------------------------------


def create_potential_field(path):
    """Create a potential field H5 file for reciprocity tests."""
    with h5py.File(path, "w") as f:
        f.create_dataset("CurrentApplied", data=1)
        xaxis = np.linspace(-10, 10) * 1e-6
        yaxis = np.linspace(-10, 10) * 1e-6
        zaxis = np.linspace(-10, 10) * 1e-6
        real_imag = np.array([0, 1])

        mesh = f.create_group("Meshes/FirstDataField")
        mesh.create_dataset("axis_x", data=xaxis)
        mesh.create_dataset("axis_y", data=yaxis)
        mesh.create_dataset("axis_z", data=zaxis)

        field0 = f.create_group("FieldGroups/randomname/AllFields/EM Potential(x,y,z,f0)/_Object/Snapshots/0")
        _, _, zd, _ = np.meshgrid(xaxis, yaxis, zaxis, real_imag, indexing="ij")
        field0.create_dataset("comp0", data=zd)
    return str(path)


def create_e_field(path):
    """Create an E-field H5 file for dipole reciprocity tests."""
    with h5py.File(path, "w") as f:
        f.create_dataset("CurrentApplied", data=1)
        xaxis = np.linspace(-10, 10) * 1e-6
        yaxis = np.linspace(-10, 10) * 1e-6
        zaxis = np.linspace(-10, 10) * 1e-6
        xc = (xaxis[:-1] + xaxis[1:]) / 2
        yc = (yaxis[:-1] + yaxis[1:]) / 2
        zc = (zaxis[:-1] + zaxis[1:]) / 2
        real_imag = np.array([0, 1])

        mesh = f.create_group("Meshes/FirstDataField")
        mesh.create_dataset("axis_x", data=xaxis)
        mesh.create_dataset("axis_y", data=yaxis)
        mesh.create_dataset("axis_z", data=zaxis)

        field0 = f.create_group("FieldGroups/randomname/AllFields/EM E(x,y,z,f0)/_Object/Snapshots/0")
        xd, _, _, _ = np.meshgrid(xc, yaxis, zaxis, real_imag, indexing="ij")
        _, yd, _, _ = np.meshgrid(xaxis, yc, zaxis, real_imag, indexing="ij")
        _, _, zd, _ = np.meshgrid(xaxis, yaxis, zc, real_imag, indexing="ij")
        field0.create_dataset("comp0", data=xd)
        field0.create_dataset("comp1", data=yd)
        field0.create_dataset("comp2", data=zd)
    return str(path)
