from __future__ import annotations

from enum import IntEnum


class StateIndex(IntEnum):
    X = 0
    Y = 1
    HEADING = 2
    VELOCITY_X = 3
    VELOCITY_Y = 4
    ACCELERATION_X = 5
    ACCELERATION_Y = 6
    STEERING_ANGLE = 7
    STEERING_RATE = 8
    ANGULAR_VELOCITY = 9
    ANGULAR_ACCELERATION = 10

    @classmethod
    def size(cls) -> int:
        return 11


class BBCoordsIndex(IntEnum):
    FRONT_LEFT = 0
    REAR_LEFT = 1
    REAR_RIGHT = 2
    FRONT_RIGHT = 3
    CENTER = 4


class CollisionType(IntEnum):
    STOPPED_EGO_OPEN = 0
    STOPPED_TRACK_OPEN = 1
    ACTIVE_FRONT_BUMPER = 2
    ACTIVE_REAR_BUMPER = 3
    ACTIVE_LATERAL = 4


class EgoAreaIndex(IntEnum):
    MULTIPLE_LANES = 0
    NON_DRIVABLE_AREA = 1
    ONCOMING_TRAFFIC = 2
    IN_INTERSECTION = 3


class MultiMetricIndex(IntEnum):
    NO_COLLISION = 0
    DRIVABLE_AREA = 1
    TRAFFIC_LIGHT = 2
    DRIVING_DIRECTION = 3


class WeightedMetricIndex(IntEnum):
    PROGRESS = 0
    TTC = 1
    LANE_KEEPING = 2
    COMFORT = 3


class SemanticMapLayer(IntEnum):
    LANE = 0
    INTERSECTION = 1
    STOP_LINE = 2
    CROSSWALK = 3
    ROADBLOCK = 4
    LANE_CONNECTOR = 5
    DRIVABLE_AREA = 6
    CARPARK_AREA = 7


DRIVABLE_LAYERS: frozenset[SemanticMapLayer] = frozenset({
    SemanticMapLayer.LANE,
    SemanticMapLayer.LANE_CONNECTOR,
    SemanticMapLayer.INTERSECTION,
    SemanticMapLayer.ROADBLOCK,
    SemanticMapLayer.DRIVABLE_AREA,
    SemanticMapLayer.CARPARK_AREA,
})
