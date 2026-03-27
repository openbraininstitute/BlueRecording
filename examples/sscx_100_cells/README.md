# SSCx 100 Cells

Extracellular recordings from a network of 100 SSCx cells.

## Prerequisites

Run `source setup.sh --dev --data` from the project root. This installs all dependencies and downloads the required datasets (networks and atlas).

## Calculating Segment Positions

From the root folder:

```bash
bluerecording write_positions examples/sscx_100_cells/simulation_config.json <positions_folder>
```

## Computing Electrode Weights

From the root folder:

```bash
bluerecording write_weights examples/sscx_100_cells/simulation_config.json examples/sscx_100_cells/electrodes.csv <weights_folder>
```

## Reference Outputs

The `reference/` folder contains `positions0_ref.pkl` and `weights_ref.h5` for verifying results. These are also used by the unit and MPI tests.
