# SPDX-License-Identifier: GPL-3.0-or-later
"""Lightweight electrode metadata module.

This module can be imported without NEURON, MPI, or any heavy dependencies.
Only requires numpy for position arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np


class ElectrodeType(StrEnum):
    """Recognized electrode types."""

    LINE_SOURCE = "LineSource"
    POINT_SOURCE = "PointSource"
    DIPOLE_RECIPROCITY = "DipoleReciprocity"
    RECIPROCITY = "Reciprocity"
    OBJECTIVE_CSD_SPHERE = "ObjectiveCSD_Sphere"
    OBJECTIVE_CSD_DISK = "ObjectiveCSD_Disk"
    OBJECTIVE_CSD_PLANE = "ObjectiveCSD_Plane"


@dataclass
class ObjectiveCSDParams:
    """Parameters for an objective CSD electrode type."""

    electrode_type: ElectrodeType
    radius: float | None = None
    thickness: float | None = None


@dataclass
class Electrode:
    """Metadata for a single electrode."""

    name: str
    position: np.ndarray
    type: ElectrodeType | ObjectiveCSDParams
    region: str = "NA"
    layer: str = "NA"

    @classmethod
    def from_json(cls, electrode_json: str | Path) -> list[Electrode]:
        """Read electrode metadata from a JSON file.

        The JSON must be a list of objects with keys ``name``, ``x``, ``y``, ``z``.
        Optional keys: ``type`` (default ``LineSource``), ``layer``, ``region``,
        ``radius``, ``thickness``.
        """
        with open(electrode_json) as f:
            data = json.load(f)

        electrodes: list[Electrode] = []
        for entry in data:
            name = str(entry["name"])
            position = np.array([entry["x"], entry["y"], entry["z"]], dtype=float)
            layer = entry.get("layer", "NA")
            region = entry.get("region", "NA")
            etype = ElectrodeType(entry["type"]) if "type" in entry else ElectrodeType.LINE_SOURCE

            if "ObjectiveCSD" in etype:
                radius = entry.get("radius")
                thickness = entry.get("thickness")
                electrodes.append(
                    cls(
                        name=name,
                        position=position,
                        type=ObjectiveCSDParams(electrode_type=etype, radius=radius, thickness=thickness),
                        region=region,
                        layer=layer,
                    )
                )
            else:
                electrodes.append(cls(name=name, position=position, type=etype, region=region, layer=layer))

        return electrodes

    @staticmethod
    def to_json(electrodes: list[Electrode], output_path: str | Path) -> None:
        """Write a list of electrodes to a JSON file.

        Produces a JSON file compatible with :meth:`from_json`.
        """
        data = []
        for e in electrodes:
            if isinstance(e.type, ObjectiveCSDParams):
                entry = {
                    "name": e.name,
                    "x": float(e.position[0]),
                    "y": float(e.position[1]),
                    "z": float(e.position[2]),
                    "type": e.type.electrode_type.value,
                    "region": e.region,
                    "layer": e.layer,
                    "radius": e.type.radius,
                    "thickness": e.type.thickness,
                }
            else:
                entry = {
                    "name": e.name,
                    "x": float(e.position[0]),
                    "y": float(e.position[1]),
                    "z": float(e.position[2]),
                    "type": e.type.value,
                    "region": e.region,
                    "layer": e.layer,
                }
            data.append(entry)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
