# SPDX-License-Identifier: GPL-3.0-or-later
"""Order-agnostic comparison of weights H5 files.

This module does NOT import NEURON or neurodamus, so it can be used
without mpirun for quick validation of weights files.
"""

import h5py
import numpy as np


def _find_population(h5: h5py.File) -> str:
    """Auto-detect the population name (first group with node_ids)."""
    for key in h5.keys():
        if key != "electrodes" and isinstance(h5[key], h5py.Group) and "node_ids" in h5[key]:
            return key
    raise ValueError("Could not find a population group with node_ids in the file")


def _get_node_scaling_factors(
    h5: h5py.File, population_name: str, node_id: int
) -> np.ndarray:
    """Extract scaling_factors rows for a given node_id using offsets."""
    node_ids = h5[f"{population_name}/node_ids"][:]
    offsets = h5[f"{population_name}/offsets"][:]
    dset_path = f"electrodes/{population_name}/scaling_factors"

    idx = np.where(node_ids == node_id)[0]
    if len(idx) == 0:
        raise KeyError(f"node_id {node_id} not found")
    i = idx[0]
    start = offsets[i]
    end = offsets[i + 1] if i + 1 < len(offsets) else h5[dset_path].shape[0]
    return h5[dset_path][start:end, :]


def compare_weights(
    reference: str,
    target: str,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    population_name: str | None = None,
) -> tuple[bool, str]:
    """Compare two weights H5 files in a node-order-agnostic manner.

    Matches nodes by ID regardless of their position in the file, then
    compares per-node scaling factors within the given tolerances.

    Args:
        reference: Path to the reference weights H5 file.
        target: Path to the target weights H5 file to validate.
        rtol: Relative tolerance for allclose comparison.
        atol: Absolute tolerance for allclose comparison.
        population_name: SONATA population name. Auto-detected if None.

    Returns:
        Tuple of (match, report):
        - match: True if all node scaling factors match within tolerance.
        - report: Human-readable summary. On mismatch, includes the first
          failing node_id and max difference.
    """
    with h5py.File(reference, "r") as ref, h5py.File(target, "r") as tgt:
        if population_name is None:
            population_name = _find_population(ref)

        ref_ids = ref[f"{population_name}/node_ids"][:]
        tgt_ids = tgt[f"{population_name}/node_ids"][:]

        ref_set = set(ref_ids.tolist())
        tgt_set = set(tgt_ids.tolist())

        if ref_set != tgt_set:
            missing = ref_set - tgt_set
            extra = tgt_set - ref_set
            parts = []
            if missing:
                parts.append(f"missing in target: {sorted(list(missing))[:10]}")
            if extra:
                parts.append(f"extra in target: {sorted(list(extra))[:10]}")
            return False, f"node_ids mismatch: {'; '.join(parts)}"

        n_nodes = len(ref_ids)
        for i, node_id in enumerate(sorted(ref_set)):
            ref_sf = _get_node_scaling_factors(ref, population_name, node_id)
            tgt_sf = _get_node_scaling_factors(tgt, population_name, node_id)

            if ref_sf.shape != tgt_sf.shape:
                return False, (
                    f"node_id {node_id}: shape mismatch "
                    f"(reference={ref_sf.shape}, target={tgt_sf.shape})"
                )

            if not np.allclose(ref_sf, tgt_sf, rtol=rtol, atol=atol):
                abs_diff = np.abs(ref_sf - tgt_sf)
                max_abs = float(np.max(abs_diff))
                max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
                ref_val = float(ref_sf[max_idx])
                tgt_val = float(tgt_sf[max_idx])
                return False, (
                    f"node_id {node_id} (node {i + 1}/{n_nodes}): "
                    f"max abs diff = {max_abs:.2e} at index {max_idx} "
                    f"(ref={ref_val:.6e}, target={tgt_val:.6e})"
                )

    return True, f"OK: {n_nodes} nodes match (rtol={rtol}, atol={atol})"
