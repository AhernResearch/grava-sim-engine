from __future__ import annotations

from ..simulator import PDMSimulator
from ..types import VehicleParams


class ScorerBase:
    """Vehicle and simulator configuration shared by driving scorers."""

    def __init__(
        self,
        vehicle: VehicleParams | None = None,
        simulator: PDMSimulator | None = None,
        discretization_time: float = 0.1,
    ) -> None:
        self._vehicle = vehicle or VehicleParams()
        self._simulator = simulator or PDMSimulator(
            discretization_time=discretization_time, vehicle=self._vehicle
        )
