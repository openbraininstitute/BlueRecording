# SPDX-License-Identifier: GPL-3.0-or-later
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import libsonata
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_circuit_path(path_to_simconfig: str | Path) -> str:
    """Return the absolute path to the circuit config for a given simulation config.

    Uses libsonata to resolve the network path, including manifest variable
    expansion and relative path resolution.

    Args:
        path_to_simconfig: Path to the simulation configuration JSON file.

    Returns:
        Absolute path to the circuit configuration file.
    """
    sim_conf = libsonata.SimulationConfig.from_file(str(path_to_simconfig))
    return sim_conf.network


def _is_circuit_config(path: str | Path) -> bool:
    """Detect whether a JSON file is a circuit config (vs. a simulation config).

    A circuit config contains a top-level ``"networks"`` key (plural).
    A simulation config contains ``"network"`` (singular) and ``"run"``.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        True if the file looks like a circuit config, False otherwise.
    """
    with open(path) as f:
        data = json.load(f)
    return "networks" in data


def _make_simulation_config(circuit_config_path: str | Path) -> dict:
    """Build a minimal SONATA simulation config dict pointing at *circuit_config_path*.

    The generated config contains only the fields required by neurodamus to
    load the circuit in ``direct_mode`` with ``disable_reports``.
    """
    circuit_config_path = Path(circuit_config_path).resolve()
    return {
        "network": str(circuit_config_path),
        "run": {
            "dt": 0.025,
            "tstop": 0,
            "random_seed": 1,
        },
    }


@contextmanager
def resolve_simulation_config(path: str | Path):
    """Context manager that yields a path to a simulation config.

    If *path* already points to a simulation config, it is yielded as-is.
    If *path* points to a circuit config, a temporary simulation config is
    created (in the same directory, so relative paths inside the circuit
    config keep working) and its path is yielded.  The temporary file is
    removed on exit.

    MPI-safe: only rank 0 creates and removes the temporary file.
    A barrier ensures all ranks see the file before proceeding and
    wait before it is deleted.

    Args:
        path: Path to either a simulation or circuit configuration file.

    Yields:
        Resolved path (str) to a simulation configuration file.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    if not _is_circuit_config(path):
        yield str(path)
    else:
        tmp_name = None
        try:
            if rank == 0:
                sim_cfg = _make_simulation_config(path)
                tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
                    mode="w",
                    suffix=".json",
                    prefix=".bluerecording_sim_",
                    dir=path.parent,
                    delete=False,
                )
                json.dump(sim_cfg, tmp, indent=2)
                tmp.close()
                tmp_name = tmp.name

            tmp_name = comm.bcast(tmp_name, root=0)
            comm.Barrier()

            yield tmp_name
        finally:
            comm.Barrier()
            if rank == 0 and tmp_name is not None:
                Path(tmp_name).unlink(missing_ok=True)
