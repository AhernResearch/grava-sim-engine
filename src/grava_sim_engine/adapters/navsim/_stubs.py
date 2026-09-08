"""Stub classes for deserializing NavSim/nuPlan MetricCache pickles without navsim installed.

Architecture:
  MetricCacheUnpickler overrides find_class() to resolve navsim/nuplan classes:
  1. Try real class (if navsim is installed)
  2. Fall back to explicit stubs (classes with __reduce__ whose __init__ is called on unpickle)
  3. Fall back to generic stubs (dataclass / __dict__-based classes)
"""

from __future__ import annotations

import logging
import math
import pickle
from enum import IntEnum
from typing import Any, Dict, List, Tuple, Type


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. Explicit stubs — classes with __reduce__ (pickle calls __init__)
# ═══════════════════════════════════════════════════════════════════════


class InterpolatedTrajectory:
    """Stub for nuplan.planning.simulation.trajectory.interpolated_trajectory.InterpolatedTrajectory.

    Real __init__ builds scipy interpolators; stub just stores the trajectory list.
    """

    __module__ = "nuplan.planning.simulation.trajectory.interpolated_trajectory"

    def __init__(self, trajectory: list) -> None:
        self._trajectory = list(trajectory)

    def get_sampled_trajectory(self) -> list:
        return list(self._trajectory)


class VehicleParameters:
    """Stub for nuplan.common.actor_state.vehicle_parameters.VehicleParameters.

    Real __reduce__ returns (cls, (width, front_length, rear_length,
    cog_position_from_rear_axle, wheel_base, vehicle_name, vehicle_type, height)).
    """

    __module__ = "nuplan.common.actor_state.vehicle_parameters"

    def __init__(
        self,
        width: float,
        front_length: float,
        rear_length: float,
        cog_position_from_rear_axle: float,
        wheel_base: float,
        vehicle_name: str,
        vehicle_type: str,
        height: float | None = None,
    ) -> None:
        self.width = width
        self.front_length = front_length
        self.rear_length = rear_length
        self.length = front_length + rear_length
        self.cog_position_from_rear_axle = cog_position_from_rear_axle
        self.wheel_base = wheel_base
        self.vehicle_name = vehicle_name
        self.vehicle_type = vehicle_type
        self.height = height

    @property
    def half_length(self) -> float:
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    @property
    def rear_axle_to_center(self) -> float:
        return self.half_length - self.rear_length


class PDMOccupancyMap:
    """Stub for navsim PDMOccupancyMap. Stores tokens/geometries, skips STRtree."""

    __module__ = "navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map"

    def __init__(
        self, tokens: List[str], geometries: Any, node_capacity: int = 10
    ) -> None:
        self._tokens = list(tokens)
        self._geometries = geometries
        self._node_capacity = node_capacity

    def __len__(self) -> int:
        return len(self._tokens)


class PDMDrivableMap(PDMOccupancyMap):
    """Stub for navsim PDMDrivableMap. Adds _map_types on top of PDMOccupancyMap."""

    # Same __module__ as PDMOccupancyMap — both live in the same navsim module.
    __module__ = "navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map"

    def __init__(
        self,
        tokens: List[str],
        map_types: list,
        geometries: Any,
        node_capacity: int = 10,
    ) -> None:
        super().__init__(tokens, geometries, node_capacity)
        self._map_types = list(map_types)


class PDMPath:
    """Stub for navsim PDMPath. Stores discrete_path, skips scipy/shapely interpolation."""

    __module__ = "navsim.planning.simulation.planner.pdm_planner.utils.pdm_path"

    def __init__(self, discrete_path: list) -> None:
        self._discrete_path = list(discrete_path)


# ═══════════════════════════════════════════════════════════════════════
# 2. Property stubs — __dict__-pickle classes whose properties we access
# ═══════════════════════════════════════════════════════════════════════


def _translate_longitudinally(pose: Any, distance: float) -> Any:
    """Pure-math translate along heading — replaces nuplan.common.geometry.transform."""
    x = pose.x + distance * math.cos(pose.heading)
    y = pose.y + distance * math.sin(pose.heading)
    # Return a simple namespace that has x, y, heading
    return _SimpleState(x, y, pose.heading)


class _SimpleState:
    """Minimal (x, y, heading) container for CarFootprint.rear_axle fallback."""

    __slots__ = ("x", "y", "heading")

    def __init__(self, x: float, y: float, heading: float) -> None:
        self.x = x
        self.y = y
        self.heading = heading


class CarFootprint:
    """Stub for nuplan.common.actor_state.car_footprint.CarFootprint.

    Pickle restores __dict__ (center, _vehicle_parameters, width, length, height).
    cache_loader accesses .rear_axle — we compute it from center + vehicle_parameters.
    """

    __module__ = "nuplan.common.actor_state.car_footprint"

    @property
    def rear_axle(self) -> Any:
        # Check if already computed (cached_property pattern)
        if "rear_axle" in self.__dict__:
            return self.__dict__["rear_axle"]
        # Compute: translate center backwards by rear_axle_to_center
        center = self.__dict__.get("_center") or self.__dict__.get("center")
        vp = self.__dict__.get("_vehicle_parameters")
        if center is not None and vp is not None:
            dist = -(vp.half_length - vp.rear_length)
            result = _translate_longitudinally(center, dist)
            self.__dict__["rear_axle"] = result
            return result
        raise AttributeError("CarFootprint stub: cannot compute rear_axle")

    @property
    def vehicle_parameters(self) -> Any:
        return self.__dict__.get("_vehicle_parameters")


class EgoState:
    """Stub for nuplan.common.actor_state.ego_state.EgoState.

    Pickle restores __dict__: _car_footprint, _dynamic_car_state,
    _tire_steering_angle, _is_in_auto_mode, _time_point.
    """

    __module__ = "nuplan.common.actor_state.ego_state"

    @property
    def rear_axle(self) -> Any:
        return self.car_footprint.rear_axle

    @property
    def car_footprint(self) -> Any:
        return self.__dict__.get("_car_footprint")

    @property
    def dynamic_car_state(self) -> Any:
        return self.__dict__.get("_dynamic_car_state")

    @property
    def tire_steering_angle(self) -> float:
        return self.__dict__.get("_tire_steering_angle", 0.0)

    @property
    def time_point(self) -> Any:
        return self.__dict__.get("_time_point")


class DynamicCarState:
    """Stub for nuplan.common.actor_state.dynamic_car_state.DynamicCarState.

    Pickle restores __dict__: _rear_axle_to_center_dist,
    _rear_axle_velocity_2d, _rear_axle_acceleration_2d,
    _angular_velocity, _angular_acceleration, _tire_steering_rate.
    """

    __module__ = "nuplan.common.actor_state.dynamic_car_state"

    @property
    def rear_axle_velocity_2d(self) -> Any:
        return self.__dict__.get("_rear_axle_velocity_2d")

    @property
    def rear_axle_acceleration_2d(self) -> Any:
        return self.__dict__.get("_rear_axle_acceleration_2d")

    @property
    def angular_velocity(self) -> float:
        return self.__dict__.get("_angular_velocity", 0.0)

    @property
    def angular_acceleration(self) -> float:
        return self.__dict__.get("_angular_acceleration", 0.0)

    @property
    def tire_steering_rate(self) -> float:
        return self.__dict__.get("_tire_steering_rate", 0.0)


class StateVector2D:
    """Stub for nuplan.common.actor_state.state_representation.StateVector2D.

    Original uses __slots__ = ("_x", "_y", "_array") with @property x/y.
    Pickle restores these into __dict__ when using generic stub.
    """

    __module__ = "nuplan.common.actor_state.state_representation"

    @property
    def x(self) -> float:
        return self.__dict__.get("_x", 0.0)

    @property
    def y(self) -> float:
        return self.__dict__.get("_y", 0.0)

    @property
    def array(self) -> Any:
        return self.__dict__.get("_array")

    @array.setter
    def array(self, value: Any) -> None:
        self.__dict__["_array"] = value


# ═══════════════════════════════════════════════════════════════════════
# 3. IntEnum stubs
# ═══════════════════════════════════════════════════════════════════════


class SemanticMapLayer(IntEnum):
    """Stub for nuplan.common.maps.maps_datatypes.SemanticMapLayer."""

    LANE = 0
    INTERSECTION = 1
    STOP_LINE = 2
    TURN_STOP = 3
    CROSSWALK = 4
    DRIVABLE_AREA = 5
    YIELD = 6
    TRAFFIC_LIGHT = 7
    STOP_SIGN = 8
    EXTENDED_PUDO = 9
    SPEED_BUMP = 10
    LANE_CONNECTOR = 11
    BASELINE_PATHS = 12
    BOUNDARIES = 13
    WALKWAYS = 14
    CARPARK_AREA = 15
    PUDO = 16
    ROADBLOCK = 17
    ROADBLOCK_CONNECTOR = 18


SemanticMapLayer.__module__ = "nuplan.common.maps.maps_datatypes"


class TrackedObjectType(IntEnum):
    """Stub for nuplan.common.actor_state.tracked_objects_types.TrackedObjectType."""

    VEHICLE = 0
    PEDESTRIAN = 1
    BICYCLE = 2
    TRAFFIC_CONE = 3
    BARRIER = 4
    CZONE_SIGN = 5
    GENERIC_OBJECT = 6
    EGO = 7


TrackedObjectType.__module__ = "nuplan.common.actor_state.tracked_objects_types"


class SceneFrameType(IntEnum):
    """Stub for navsim.common.enums.SceneFrameType."""

    ORIGINAL = 0
    SYNTHETIC = 1


SceneFrameType.__module__ = "navsim.common.enums"


# ═══════════════════════════════════════════════════════════════════════
# 4. Generic stub factory
# ═══════════════════════════════════════════════════════════════════════

_GENERIC_STUB_CACHE: Dict[Tuple[str, str], type] = {}
_ENUM_STUB_CACHE: Dict[Tuple[str, str], type] = {}


def _make_enum_stub(module: str, name: str) -> type | None:
    """Create a dynamic IntEnum stub that accepts any integer value on unpickling.

    Pickle stores enums as (EnumClass, (value,)). The stub needs to accept
    arbitrary values. We use a custom __new__ that auto-creates missing members.
    """
    key = (module, name)
    if key in _ENUM_STUB_CACHE:
        return _ENUM_STUB_CACHE[key]

    # Build a minimal enum class with a __new__ that handles unknown values
    stub = IntEnum(name, {"_placeholder": -999})
    stub.__module__ = module

    # Override __new__ to handle unknown values
    original_new = stub.__new__

    def _flexible_new(cls, value):
        try:
            return original_new(cls, value)
        except ValueError:
            # Unknown value — add it dynamically
            member = int.__new__(cls, value)
            member._name_ = f"_auto_{value}"
            member._value_ = value
            return member

    stub.__new__ = _flexible_new
    stub.__module__ = module
    _ENUM_STUB_CACHE[key] = stub
    return stub


def _make_generic_stub(module: str, name: str) -> type:
    """Create a dynamic stub class for __dict__-based pickle deserialization."""
    key = (module, name)
    if key in _GENERIC_STUB_CACHE:
        return _GENERIC_STUB_CACHE[key]

    stub_cls = type(name, (), {"__module__": module})
    _GENERIC_STUB_CACHE[key] = stub_cls
    return stub_cls


# ═══════════════════════════════════════════════════════════════════════
# 5. Registry — maps (module, name) → stub class
# ═══════════════════════════════════════════════════════════════════════

_STUB_REGISTRY: Dict[Tuple[str, str], type] = {
    # __reduce__ stubs
    (
        "nuplan.planning.simulation.trajectory.interpolated_trajectory",
        "InterpolatedTrajectory",
    ): InterpolatedTrajectory,
    (
        "nuplan.common.actor_state.vehicle_parameters",
        "VehicleParameters",
    ): VehicleParameters,
    (
        "navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map",
        "PDMOccupancyMap",
    ): PDMOccupancyMap,
    (
        "navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map",
        "PDMDrivableMap",
    ): PDMDrivableMap,
    (
        "navsim.planning.simulation.planner.pdm_planner.utils.pdm_path",
        "PDMPath",
    ): PDMPath,
    # Property stubs
    ("nuplan.common.actor_state.ego_state", "EgoState"): EgoState,
    ("nuplan.common.actor_state.car_footprint", "CarFootprint"): CarFootprint,
    ("nuplan.common.actor_state.dynamic_car_state", "DynamicCarState"): DynamicCarState,
    ("nuplan.common.actor_state.state_representation", "StateVector2D"): StateVector2D,
    # IntEnum stubs
    ("nuplan.common.maps.maps_datatypes", "SemanticMapLayer"): SemanticMapLayer,
    ("navsim.common.enums", "SceneFrameType"): SceneFrameType,
    ("nuplan.common.actor_state.tracked_objects_types", "TrackedObjectType"): TrackedObjectType,
}


# ═══════════════════════════════════════════════════════════════════════
# 6. MetricCacheUnpickler
# ═══════════════════════════════════════════════════════════════════════


# Module paths that may contain enum types we haven't explicitly stubbed.
_ENUM_MODULE_PREFIXES = (
    "nuplan.common.actor_state.tracked_objects_types",
    "nuplan.common.maps.maps_datatypes",
    "navsim.common.enums",
)


class MetricCacheUnpickler(pickle.Unpickler):
    """Custom unpickler that resolves navsim/nuplan classes via stubs when not installed."""

    def find_class(self, module: str, name: str) -> type:
        # 1. Try real class
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            logger.debug("Falling back to MetricCache stub for %s.%s", module, name)

        # 2. Explicit stub
        key = (module, name)
        stub = _STUB_REGISTRY.get(key)
        if stub is not None:
            return stub

        # 3. Try enum stub for known enum modules
        if any(module.startswith(prefix) for prefix in _ENUM_MODULE_PREFIXES):
            enum_stub = _make_enum_stub(module, name)
            if enum_stub is not None:
                _STUB_REGISTRY[key] = enum_stub  # cache for next time
                return enum_stub

        # 4. Generic stub
        return _make_generic_stub(module, name)


__all__ = ["MetricCacheUnpickler"]
