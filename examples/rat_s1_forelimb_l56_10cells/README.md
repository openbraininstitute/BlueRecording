# rat_s1_forelimb_l56_10cells_simplified

Lightweight 10-cell rat somatosensory cortex (S1) forelimb circuit, covering
layers 5 and 6. Used as a fast integration-test fixture that runs in CI without
requiring large downloaded datasets or FEM field files.

## Origin

This example was extracted from a larger non-barrel S1 circuit
(`nbS1-O1-sSub-pre-dim2-nCN-HEX0-L6-01`) using **obi-one** (v2025.6.1,
`MorphologyDecontainerization` workflow). The source circuit config lived at:

```
obi-output/extracted_small_microcircuits_simplex/dim2-target/5/circuit_config.json
```

The extraction produced a self-contained SONATA circuit with:

- 10 biophysical neurons (population `S1nonbarrel_neurons`)
- SWC morphologies and HOC e-models
- Chemical synaptic connectivity (internal + projections from VPM and POm)

## Added in

Commit `ccd1990` ("feat: circuit config support, test reorg, compute_positions
API") introduced this example to enable CI tests for `circuit_config.json`
support, MPI positions, and write-weights without external data dependencies.
