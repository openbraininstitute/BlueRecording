# Single Cell L5 TPC

Extracellular recordings from a single layer 5 thick-tufted pyramidal cell placed in a large homogeneous medium.

## Prerequisites

Run `source setup.sh --dev --data` from the project root. This installs all dependencies and downloads the required datasets.

This will download the files `Infinite_VeryFar_HighRes.h5` and `Infinite_Close_HighRes_SmallSphere.h5` from [our Zenodo repository](https://zenodo.org/records/10927050) into this folder.

## Calculating Segment Positions

From the root folder:

```bash
bluerecording write_positions examples/single_cell_l5_tpc/simulation_config_near.json <positions_folder>
```

## Computing Electrode Weights

Two electrode configurations are provided: `near_electrodes.csv` (close to the neuron) and `distant_electrodes.csv` (far from the neuron). The only difference is the location of the recording electrode; the reference electrode is in the same position.

From the root folder:

```bash
bluerecording write_weights examples/single_cell_l5_tpc/simulation_config_near.json examples/single_cell_l5_tpc/near_electrodes.csv <weights_folder> --path-to-fields examples/single_cell_l5_tpc/Infinite_Close_HighRes_SmallSphere.h5 examples/single_cell_l5_tpc/Infinite_Close_HighRes_SmallSphere.h5 
```

or

```bash
bluerecording write_weights examples/single_cell_l5_tpc/simulation_config_near.json examples/single_cell_l5_tpc/near_electrodes.csv <weights_folder> --path-to-fields examples/single_cell_l5_tpc/Infinite_VeryFar_HighRes.h5 examples/single_cell_l5_tpc/Infinite_VeryFar_HighRes.h5 
```

## Running the Simulation

To run the simulation you need NEURON compiled with [libsonatareport](https://github.com/openbraininstitute/libsonatareport) support. The NEURON installed by `setup.sh` is the PyPI wheel, which does not include libsonatareport.

Once your environment is ready, run:

```bash
neurodamus examples/single_cell_l5_tpc/simulation_config_near.json
```

or

```bash
neurodamus examples/single_cell_l5_tpc/simulation_config_distant.json
```

## Analysis

After the simulation completes, open the notebook `make_figures.ipynb` in this folder to visualize and compare the extracellular signals against reference solutions. This notebook produces Figure 2 from the BlueRecording paper.

