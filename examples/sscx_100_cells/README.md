# SSCx 100 Cells

Extracellular recordings from a network of 100 SSCx cells.

## Prerequisites

Run `./dev_setup.sh` from the project root to install all dependencies, then `./download_examples_data.sh` to download the required datasets (networks and atlas). Finally, `source env.sh` to set up the project environment.

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

## Inspecting Weights

You can use the [analyze_weights](../analyze_weights.ipynb) notebook to inspect the generated weights file — it validates the H5 structure and shows weight distributions per electrode. One can see this broken down by section type by adding `--with-neurite-type` option when executing the `write_weights` operation.
