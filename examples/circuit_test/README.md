# Circuit Test Example

This example produces extracellular signals from a network of 100 cells.

## Prerequisites

Run `source setup.sh --dev --data` from the project root. This installs all dependencies and downloads the required datasets (networks and atlas).

### Calculating Segment Positions

Run:

```bash
bluerecording write_positions examples/circuit_test/simulation_config.json <positions_folder>
```

and the segment positions will appear in `<positions_folder>`.

### Electrode File

The electrode csv file is provided at `electrodes.csv`.

### Reference outputs

The folder `reference/` contains reference outputs (`positions0_ref.pkl` and `weights_ref.h5`) that can be used to verify your results.
