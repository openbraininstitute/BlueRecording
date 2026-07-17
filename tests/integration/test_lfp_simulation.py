# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration test: neurodamus LFP simulation using bluerecording weights.

Uses the pre-computed reference weights for the rat_s1_forelimb_l56_10cells
circuit and runs a neurodamus/CoreNEURON simulation with an LFP report.
Compares the resulting LFP output against a stored reference.

Requires CoreNEURON (CORENEURONLIB set). Not run in CI.
"""

import json
from pathlib import Path

import libsonata
import numpy as np
import pytest

from tests.conftest import EXAMPLE_RAT_S1

REFERENCE_DIR = EXAMPLE_RAT_S1 / "reference"
WEIGHTS_FILE = str(REFERENCE_DIR / "weights_ref.h5")
LFP_REFERENCE = str(REFERENCE_DIR / "lfp_report_ref.h5")


def _build_simulation_config(tmp_path: Path, weights_file: str) -> str:
    """Create a minimal simulation config for LFP reporting."""
    sim_config = {
        "network": str(EXAMPLE_RAT_S1 / "circuit_config.json"),
        "node_sets_file": str(EXAMPLE_RAT_S1 / "node_sets.json"),
        "target_simulator": "CORENEURON",
        "node_set": "Mosaic",
        "output": {
            "output_dir": str(tmp_path / "output"),
        },
        "run": {
            "random_seed": 42,
            "dt": 0.025,
            "tstop": 5.0,
        },
        "conditions": {
            "celsius": 34,
            "v_init": -80,
        },
        "inputs": {
            "stimulus_pulse": {
                "module": "pulse",
                "input_type": "current_clamp",
                "delay": 1.0,
                "duration": 3.0,
                "node_set": "Mosaic",
                "represents_physical_electrode": True,
                "amp_start": 0.5,
                "width": 1.0,
                "frequency": 50.0,
            }
        },
        "reports": {
            "lfp_report": {
                "type": "lfp",
                "cells": "Mosaic",
                "electrodes_file": weights_file,
                "dt": 0.1,
                "start_time": 0.0,
                "end_time": 5.0,
            }
        },
    }
    config_path = str(tmp_path / "simulation_config.json")
    with open(config_path, "w") as f:
        json.dump(sim_config, f, indent=2)
    return config_path


@pytest.mark.skip_in_ci
def test_lfp_simulation(tmp_path):
    """Run neurodamus LFP simulation with bluerecording weights and compare to reference."""
    from neurodamus import Neurodamus

    config_path = _build_simulation_config(tmp_path, WEIGHTS_FILE)

    nd = Neurodamus(config_path)
    nd.run()

    # Read output
    lfp_file = tmp_path / "output" / "lfp_report.h5"
    assert lfp_file.exists(), f"LFP report not produced at {lfp_file}"

    report = libsonata.ElementReportReader(str(lfp_file))
    pop_names = report.get_population_names()
    assert "S1nonbarrel_neurons" in pop_names

    lfp_pop = report["S1nonbarrel_neurons"]
    result_ids = lfp_pop.get_node_ids()
    result_data = lfp_pop.get()

    assert len(result_ids) > 0, "No node IDs in LFP report"
    assert np.all(np.isfinite(result_data.data)), "LFP data contains NaN/Inf"
    assert np.any(result_data.data != 0.0), "LFP data is all zeros"

    # Compare against reference if it exists
    if Path(LFP_REFERENCE).exists():
        ref_report = libsonata.ElementReportReader(LFP_REFERENCE)
        ref_pop = ref_report["S1nonbarrel_neurons"]
        ref_ids = ref_pop.get_node_ids()
        ref_data = ref_pop.get()

        np.testing.assert_array_equal(result_ids, ref_ids)
        np.testing.assert_allclose(
            result_data.data,
            ref_data.data,
            rtol=1e-5,
            err_msg="LFP output differs from reference",
        )
