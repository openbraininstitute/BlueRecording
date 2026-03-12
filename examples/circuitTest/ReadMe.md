# Examples

Here, we provide examples of extracellular signals produced by a network of 100 cells

## System requirements

We assume that you are running these examples on a Linux system with slurm. If this is not the case, you will have to modify the provided bash scripts accordingly.

## Instructions

### Download data
Download the file networks.zip from 10.5281/zenodo.10927050 and unzip it into the folder *data/simulation/configuration/networks*

### Calculating Segment Positions

As the same neuron is used in both examples, segment positions only need to be calculated once. Just run:

```bash
bluerecording get_positions examples/circuitTest/data/simulation_config.json <positions_folder>
```

and the segment positions will apear in `<positions_folder>`.

### Electrode File

In subfolder **test/electrodeFile**, the the electrode weights h5 file is created by running the script **WriteH5.sh**. 

### Online signal calculation

In subfolder **test/simulation**, the simulation is launched by running the script **launch.sh**. 

## Important note

The user should note that in the circuit and simulation configuration files, the absolute paths listed should be modified to match the paths in the user's system. In the bash scripts, the slurm commands must be modified according to the user's system.
