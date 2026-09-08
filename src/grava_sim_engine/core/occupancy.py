from __future__ import annotations

from collections import defaultdict
from functools import cached_property
from typing import DefaultDict, List, Sequence

import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .enums import DRIVABLE_LAYERS, SemanticMapLayer


def _normalize_layer_name(layer_type: str) -> str:
    return layer_type.strip().lower().replace("-", "_").replace(" ", "_")


_LAYER_NAME_TO_ENUM = {
    _normalize_layer_name(layer.name): layer for layer in SemanticMapLayer
}


def _coerce_layer_name(layer_type: str | int | SemanticMapLayer) -> str:
    if isinstance(layer_type, SemanticMapLayer):
        return _normalize_layer_name(layer_type.name)
    if isinstance(layer_type, (int, np.integer)):
        return _normalize_layer_name(SemanticMapLayer(int(layer_type)).name)
    return _normalize_layer_name(layer_type)


def _as_geometry_array(
    geometries: BaseGeometry | Sequence[BaseGeometry] | np.ndarray,
) -> np.ndarray:
    if isinstance(geometries, BaseGeometry):
        return np.asarray([geometries], dtype=object)

    geometry_array = np.asarray(geometries, dtype=object)
    if geometry_array.ndim == 0:
        return geometry_array.reshape(1)
    return geometry_array


class OccupancyMap:
    """Shapely STRtree spatial index wrapper."""

    def __init__(self, tokens: List[str], polygons: Sequence[BaseGeometry] | np.ndarray):
        polygon_array = _as_geometry_array(polygons)
        if len(tokens) != len(polygon_array):
            raise ValueError(
                f"tokens/polygons length mismatch: {len(tokens)} != {len(polygon_array)}"
            )

        self._tokens = list(tokens)
        self._polygons = polygon_array
        self._tree = STRtree(self._polygons) if len(self._polygons) > 0 else None
        self._token_to_idx = {token: idx for idx, token in enumerate(self._tokens)}

    @property
    def tokens(self) -> List[str]:
        return self._tokens

    @property
    def token_to_idx(self) -> dict[str, int]:
        return self._token_to_idx

    def intersects(self, query_polygons: BaseGeometry | Sequence[BaseGeometry] | np.ndarray) -> np.ndarray:
        """Return whether each query polygon intersects any indexed polygon."""
        queries = _as_geometry_array(query_polygons)
        output = np.zeros(len(queries), dtype=bool)

        if self._tree is None or len(queries) == 0:
            return output

        if len(queries) == 1:
            indices = self._tree.query(queries[0], predicate="intersects")
            output[0] = len(indices) > 0
            return output

        intersection_pairs = self._tree.query(queries, predicate="intersects")
        if intersection_pairs.size == 0:
            return output

        output[np.unique(intersection_pairs[0])] = True
        return output

    def get_colliding_tokens(self, query_polygon: BaseGeometry) -> List[str]:
        """Return all tokens whose polygons intersect with the query polygon."""
        if self._tree is None:
            return []

        indices = self._tree.query(query_polygon, predicate="intersects")
        return [self._tokens[idx] for idx in indices.tolist()]

    def __getitem__(self, token: str):
        return self._polygons[self._token_to_idx[token]]

    def __len__(self) -> int:
        return len(self._tokens)


class DrivableMap(OccupancyMap):
    """Drivable-area occupancy map grouped by semantic layer."""

    def __init__(
        self,
        tokens: List[str],
        types: List[str | int | SemanticMapLayer],
        polygons: Sequence[BaseGeometry] | np.ndarray,
    ):
        if len(tokens) != len(types):
            raise ValueError(f"tokens/types length mismatch: {len(tokens)} != {len(types)}")

        super().__init__(tokens, polygons)
        self._types = [_coerce_layer_name(layer_type) for layer_type in types]

        self._type_to_indices: DefaultDict[str, List[int]] = defaultdict(list)
        for index, layer_type in enumerate(self._types):
            self._type_to_indices[layer_type].append(index)

    @property
    def types(self) -> List[str]:
        return self._types

    @cached_property
    def drivable_union(self) -> BaseGeometry:
        """Union polygon of all drivable semantic layers."""
        drivable_names = {_normalize_layer_name(layer.name) for layer in DRIVABLE_LAYERS}
        indices: list[int] = []
        for layer_name in drivable_names:
            indices.extend(self._type_to_indices.get(layer_name, []))

        if not indices:
            return Polygon()

        polygons = [self._polygons[index] for index in indices]
        return unary_union(polygons)

    def points_in_polygons(self, points: np.ndarray) -> np.ndarray:
        """Check if points are inside polygons, aggregated by semantic layer."""
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1] != 2:
            raise ValueError(f"points last dim must be 2, got {points.shape}")

        input_shape = points.shape[:-1]
        flat_points = points.reshape(-1, 2)
        output = np.zeros((len(flat_points), len(SemanticMapLayer)), dtype=bool)

        if len(self._polygons) == 0 or len(flat_points) == 0:
            return output.reshape((*input_shape, len(SemanticMapLayer)))

        x_coords = flat_points[:, 0]
        y_coords = flat_points[:, 1]

        for layer_name, polygon_indices in self._type_to_indices.items():
            layer = _LAYER_NAME_TO_ENUM.get(layer_name)
            if layer is None:
                continue

            layer_mask = np.zeros(len(flat_points), dtype=bool)
            for polygon_index in polygon_indices:
                layer_mask |= shapely.contains_xy(
                    self._polygons[polygon_index],
                    x_coords,
                    y_coords,
                )
            output[:, layer.value] = layer_mask

        return output.reshape((*input_shape, len(SemanticMapLayer)))

    def is_in_layer(self, point: np.ndarray, layer_type: str | int | SemanticMapLayer) -> bool:
        """Check whether a single point lies inside any polygon of the requested layer."""
        layer_name = _coerce_layer_name(layer_type)
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (2,):
            raise ValueError(f"point must have shape (2,), got {point.shape}")

        if self._tree is None:
            return False

        polygon_indices = self._tree.query(Point(point[0], point[1]), predicate="within")
        return any(self._types[index] == layer_name for index in polygon_indices.tolist())
