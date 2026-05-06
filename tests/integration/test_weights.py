# SPDX-License-Identifier: GPL-3.0-or-later
import h5py
import numpy as np
import pytest

from bluerecording import positions
from bluerecording.circuit import init_circuit
from bluerecording.weights import Electrode, get_weights, save_weights


@pytest.mark.skip_in_ci
def test_sscx_100_cells_write_weights(tmp_path):
    """Full write_weights pipeline for sscx_100_cells."""
    simconfig = "examples/sscx_100_cells/simulation_config.json"
    csv = "examples/sscx_100_cells/electrodes.csv"
    ref = "examples/sscx_100_cells/reference/weights_ref.h5"
    out = str(tmp_path / "weights.h5")

    nm, ids, cols, pop, pop_name, morphologies_dir = init_circuit(simconfig)
    pos_df, cols, _ = positions.get_positions(nm, ids, cols, pop, morphologies_dir=morphologies_dir)
    electrodes = Electrode.from_csv(csv)
    weights = get_weights(pos_df, cols, electrodes=electrodes)
    save_weights(weights, cols, pop_name, out, electrodes=electrodes)

    with h5py.File(ref, "r") as r, h5py.File(out, "r") as n:
        np.testing.assert_array_equal(r[f"{pop_name}/node_ids"][:], n[f"{pop_name}/node_ids"][:])
        np.testing.assert_array_equal(r[f"{pop_name}/offsets"][:], n[f"{pop_name}/offsets"][:])
        dset = f"electrodes/{pop_name}/scaling_factors"
        np.testing.assert_allclose(r[dset][:], n[dset][:], rtol=1e-6, atol=1e-9)


def test_rat_s1_write_weights(tmp_path):
    """Write_weights pipeline for rat_s1_forelimb_l56_10cells using circuit_config.json."""
    from tests.conftest import EXAMPLE_RAT_S1

    circuit_config = str(EXAMPLE_RAT_S1 / "circuit_config.json")
    csv = str(EXAMPLE_RAT_S1 / "electrodes.csv")
    ref = str(EXAMPLE_RAT_S1 / "reference" / "weights_ref.h5")
    out = str(tmp_path / "weights.h5")

    nm, ids, cols, pop, pop_name, morphologies_dir = init_circuit(circuit_config)
    pos_df, cols, _ = positions.get_positions(nm, ids, cols, pop, morphologies_dir=morphologies_dir)
    electrodes = Electrode.from_csv(csv)
    weights = get_weights(pos_df, cols, electrodes=electrodes)
    save_weights(weights, cols, pop_name, out, electrodes=electrodes)

    with h5py.File(ref, "r") as r, h5py.File(out, "r") as n:
        np.testing.assert_array_equal(r[f"{pop_name}/node_ids"][:], n[f"{pop_name}/node_ids"][:])
        np.testing.assert_array_equal(r[f"{pop_name}/offsets"][:], n[f"{pop_name}/offsets"][:])
        dset = f"electrodes/{pop_name}/scaling_factors"
        np.testing.assert_allclose(r[dset][:], n[dset][:], rtol=1e-6, atol=1e-9)


@pytest.mark.skip_in_ci
def test_single_cell_write_weights_near(tmp_path):
    """Write_weights for single_cell_l5_tpc (near electrodes)."""
    simconfig = "examples/single_cell_l5_tpc/simulation_config_near.json"
    csv = "examples/single_cell_l5_tpc/near_electrodes.csv"
    ref = "examples/single_cell_l5_tpc/reference/weights_near_ref.h5"
    field = "examples/single_cell_l5_tpc/Infinite_Close_HighRes_SmallSphere.h5"
    out = str(tmp_path / "weights.h5")

    nm, ids, cols, pop, pop_name, morphologies_dir = init_circuit(simconfig)
    pos_df, cols, _ = positions.get_positions(nm, ids, cols, pop, morphologies_dir=morphologies_dir)
    electrodes = Electrode.from_csv(csv)
    weights = get_weights(pos_df, cols, electrodes=electrodes, path_to_fields=[field, field])
    save_weights(weights, cols, pop_name, out, electrodes=electrodes)

    with h5py.File(ref, "r") as r, h5py.File(out, "r") as n:
        np.testing.assert_array_equal(r[f"{pop_name}/node_ids"][:], n[f"{pop_name}/node_ids"][:])
        np.testing.assert_array_equal(r[f"{pop_name}/offsets"][:], n[f"{pop_name}/offsets"][:])
        dset = f"electrodes/{pop_name}/scaling_factors"
        np.testing.assert_allclose(r[dset][:], n[dset][:], rtol=1e-6, atol=1e-9)


@pytest.mark.skip_in_ci
def test_single_cell_write_weights_distant(tmp_path):
    """Write_weights for single_cell_l5_tpc (distant electrodes)."""
    simconfig = "examples/single_cell_l5_tpc/simulation_config_near.json"
    csv = "examples/single_cell_l5_tpc/distant_electrodes.csv"
    ref = "examples/single_cell_l5_tpc/reference/weights_distant_ref.h5"
    field = "examples/single_cell_l5_tpc/Infinite_VeryFar_HighRes.h5"
    out = str(tmp_path / "weights.h5")

    nm, ids, cols, pop, pop_name, morphologies_dir = init_circuit(simconfig)
    pos_df, cols, _ = positions.get_positions(nm, ids, cols, pop, morphologies_dir=morphologies_dir)
    electrodes = Electrode.from_csv(csv)
    weights = get_weights(pos_df, cols, electrodes=electrodes, path_to_fields=[field, field])
    save_weights(weights, cols, pop_name, out, electrodes=electrodes)

    with h5py.File(ref, "r") as r, h5py.File(out, "r") as n:
        np.testing.assert_array_equal(r[f"{pop_name}/node_ids"][:], n[f"{pop_name}/node_ids"][:])
        np.testing.assert_array_equal(r[f"{pop_name}/offsets"][:], n[f"{pop_name}/offsets"][:])
        dset = f"electrodes/{pop_name}/scaling_factors"
        np.testing.assert_allclose(r[dset][:], n[dset][:], rtol=1e-6, atol=1e-9)


def test_rat_s1_neurite_types(tmp_path):
    """Write weights with --with-neurite-type and verify types independently."""
    from neurodamus.metype import BaseCell

    from tests.conftest import EXAMPLE_RAT_S1

    circuit_config = str(EXAMPLE_RAT_S1 / "circuit_config.json")
    csv = str(EXAMPLE_RAT_S1 / "electrodes.csv")
    out = str(tmp_path / "weights.h5")

    nm, ids, cols, pop, pop_name, morphologies_dir = init_circuit(circuit_config)
    pos_df, cols, neurite_types = positions.get_positions(nm, ids, cols, pop, morphologies_dir=morphologies_dir)
    electrodes = Electrode.from_csv(csv)
    weights = get_weights(pos_df, cols, electrodes=electrodes)
    save_weights(weights, cols, pop_name, out, electrodes=electrodes, neurite_types=neurite_types)

    # --- Independent verification ---
    type_to_code = {st: idx for idx, (st, _) in enumerate(BaseCell.SECTION_TYPES)}

    for gid in ids:
        cell = nm.get_cell(gid)
        counts = cell.get_section_counts()

        expected_map = {}
        offset = 0
        for (sec_type, _), count in zip(BaseCell.SECTION_TYPES, counts, strict=False):
            for local_idx in range(count):
                expected_map[offset + local_idx] = type_to_code[sec_type]
            offset += count

        cols_for_gid = cols[cols[:, 0] == gid]
        expected_codes = np.array(
            [expected_map[int(sec_id)] for sec_id in cols_for_gid[:, 1]],
            dtype=np.int32,
        )

        gid_mask = cols[:, 0] == gid
        actual_codes = neurite_types[gid_mask]
        np.testing.assert_array_equal(actual_codes, expected_codes)

    with h5py.File(out, "r") as h5:
        assert f"{pop_name}/neurite_types" in h5
        stored = h5[f"{pop_name}/neurite_types"][:]
        np.testing.assert_array_equal(stored, neurite_types)
