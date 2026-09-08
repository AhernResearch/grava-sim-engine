from __future__ import annotations

import json
import socket
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from grava_common.coordinates import CoordinateConverter


class SimEngineClientError(RuntimeError):
    """Raised when the sim engine returns an invalid HTTP response."""


class SimEngineClient:
    """Lightweight urllib-based HTTP client for the grava-sim-engine service.

    Public API defaults to NOUS ego frame for trajectory inputs. The client handles
    NOUS -> NUPLAN conversion for requests, and converts selected geometry outputs
    (e.g. local_centerline_points) back to NOUS-friendly semantics.
    """

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # ── PDM Scoring (NavSim-compatible) ─────────────────────────────────

    def score(
        self,
        trajectory: List[List[float]],
        scene_token: str,
        log_name: str,
        dataset: str,
        trajectory_frame: Literal["nous", "nuplan"] = "nous",
    ) -> Tuple[float, dict[str, Any]]:
        """Score a single trajectory and return ``(pdm_score, full_result_dict)``."""
        try:
            response = self._request_json(
                method="POST",
                path="/v1/score",
                payload={
                    "trajectory": self._normalize_trajectory_input(trajectory, trajectory_frame),
                    "scene_token": scene_token,
                    "log_name": log_name,
                    "dataset": dataset,
                },
            )
        except SimEngineClientError as exc:
            error_result = self._error_result(str(exc))
            return 0.0, error_result

        result = self._normalize_result(response, endpoint="/v1/score")
        result = self._normalize_output(result)
        pdm_score = self._result_score(result)
        if result.get("error"):
            return 0.0, result
        return pdm_score, result

    def score_batch(
        self,
        trajectories: List[List[List[float]]],
        scene_token: str,
        log_name: str,
        dataset: str,
        trajectory_frame: Literal["nous", "nuplan"] = "nous",
    ) -> List[dict[str, Any]]:
        """Score a batch of trajectories and return the raw result dicts."""
        try:
            response = self._request_json(
                method="POST",
                path="/v1/score/batch",
                payload={
                    "trajectories": [self._normalize_trajectory_input(t, trajectory_frame) for t in trajectories],
                    "scene_token": scene_token,
                    "log_name": log_name,
                    "dataset": dataset,
                },
            )
        except SimEngineClientError as exc:
            return [self._error_result(str(exc)) for _ in trajectories]

        if not isinstance(response, dict):
            return [
                self._error_result("Invalid JSON payload from /v1/score/batch: expected object")
                for _ in trajectories
            ]

        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(trajectories):
            return [
                self._error_result("Invalid JSON payload from /v1/score/batch: malformed results")
                for _ in trajectories
            ]

        return [self._normalize_output(self._normalize_result(item, endpoint="/v1/score/batch")) for item in results]

    # ── Control Signal Scoring (bypass LQR) ────────────────────────────

    def score_control(
        self,
        control_signals: List[List[float]],
        scene_token: str,
        log_name: str,
        dataset: str,
    ) -> Tuple[float, dict[str, Any]]:
        """Score control signals (accel, heading_rate) via direct kinematics."""
        payload = {
            "control_signals": control_signals,
            "scene_token": scene_token,
            "log_name": log_name,
            "dataset": dataset,
        }
        try:
            response = self._request_json(method="POST", path="/v1/score/control", payload=payload)
        except SimEngineClientError as exc:
            return 0.0, {"error": str(exc), "pdm_score": 0.0}
        result = self._normalize_result(response, endpoint="/v1/score/control")
        result = self._normalize_output(result)
        score = result.get("pdm_score", 0.0)
        if result.get("error"):
            return 0.0, result
        return score, result

    def score_batch_control(
        self,
        control_signals_batch: List[List[List[float]]],
        scene_token: str,
        log_name: str,
        dataset: str,
    ) -> List[dict[str, Any]]:
        """Score a batch of control signals."""
        payload = {
            "control_signals_batch": control_signals_batch,
            "scene_token": scene_token,
            "log_name": log_name,
            "dataset": dataset,
        }
        try:
            response = self._request_json(method="POST", path="/v1/score/control/batch", payload=payload)
        except SimEngineClientError as exc:
            return [{"error": str(exc), "pdm_score": 0.0} for _ in control_signals_batch]
        results = response.get("results", [])
        return [self._normalize_output(self._normalize_result(item, endpoint="/v1/score/control/batch")) for item in results]

    # ── RL Scoring (continuous / discrete) ──────────────────────────────

    def score_rl(
        self,
        trajectory: List[List[float]],
        scene_token: str,
        log_name: str,
        dataset: str,
        scoring_mode: Literal["continuous", "discrete"] = "continuous",
        config_overrides: Optional[Dict[str, float]] = None,
        trajectory_frame: Literal["nous", "nuplan"] = "nous",
    ) -> Tuple[float, dict[str, Any]]:
        """Score a single trajectory with RL reward and return ``(rl_score, full_result_dict)``."""
        payload: dict[str, Any] = {
            "trajectory": self._normalize_trajectory_input(trajectory, trajectory_frame),
            "scene_token": scene_token,
            "log_name": log_name,
            "dataset": dataset,
            "scoring_mode": scoring_mode,
        }
        if config_overrides:
            payload["config_overrides"] = config_overrides

        try:
            response = self._request_json(method="POST", path="/v1/score/rl", payload=payload)
        except SimEngineClientError as exc:
            error_result = self._rl_error_result(str(exc))
            return 0.0, error_result

        result = self._normalize_rl_result(response, endpoint="/v1/score/rl")
        result = self._normalize_output(result)
        rl_score = self._rl_result_score(result)
        if result.get("error"):
            return 0.0, result
        return rl_score, result

    def score_batch_rl(
        self,
        trajectories: List[List[List[float]]],
        scene_token: str,
        log_name: str,
        dataset: str,
        scoring_mode: Literal["continuous", "discrete"] = "continuous",
        config_overrides: Optional[Dict[str, float]] = None,
        trajectory_frame: Literal["nous", "nuplan"] = "nous",
    ) -> List[dict[str, Any]]:
        """Score a batch of trajectories with RL reward."""
        payload: dict[str, Any] = {
            "trajectories": [self._normalize_trajectory_input(t, trajectory_frame) for t in trajectories],
            "scene_token": scene_token,
            "log_name": log_name,
            "dataset": dataset,
            "scoring_mode": scoring_mode,
        }
        if config_overrides:
            payload["config_overrides"] = config_overrides

        try:
            response = self._request_json(method="POST", path="/v1/score/rl/batch", payload=payload)
        except SimEngineClientError as exc:
            return [self._rl_error_result(str(exc)) for _ in trajectories]

        if not isinstance(response, dict):
            return [
                self._rl_error_result("Invalid JSON payload from /v1/score/rl/batch: expected object")
                for _ in trajectories
            ]

        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(trajectories):
            return [
                self._rl_error_result("Invalid JSON payload from /v1/score/rl/batch: malformed results")
                for _ in trajectories
            ]

        return [self._normalize_output(self._normalize_rl_result(item, endpoint="/v1/score/rl/batch")) for item in results]

    # ── Internal Key-Action Scoring ────────────────────────────────────────

    def score_internal_rl(
        self,
        trajectory: List[List[float]],
        scene_token: str,
        log_name: str,
        dataset: str,
        config_overrides: Optional[Dict[str, float]] = None,
        trajectory_frame: Literal["nous", "nuplan"] = "nous",
    ) -> Tuple[float, dict[str, Any]]:
        """Score one trajectory with the internal hard key-action metric."""
        payload: dict[str, Any] = {
            "trajectory": self._normalize_trajectory_input(trajectory, trajectory_frame),
            "scene_token": scene_token,
            "log_name": log_name,
            "dataset": dataset,
        }
        if config_overrides:
            payload["config_overrides"] = config_overrides

        try:
            response = self._request_json(
                method="POST",
                path="/v1/score/internal/rl",
                payload=payload,
            )
        except SimEngineClientError as exc:
            error_result = self._internal_error_result(str(exc))
            return 0.0, error_result

        result = self._normalize_internal_result(response, endpoint="/v1/score/internal/rl")
        score = self._internal_result_score(result)
        if result.get("error"):
            return 0.0, result
        return score, result

    def score_batch_internal_rl(
        self,
        trajectories: List[List[List[float]]],
        scene_token: str,
        log_name: str,
        dataset: str,
        config_overrides: Optional[Dict[str, float]] = None,
        trajectory_frame: Literal["nous", "nuplan"] = "nous",
    ) -> List[dict[str, Any]]:
        """Score a batch of trajectories with the internal hard key-action metric."""
        payload: dict[str, Any] = {
            "trajectories": [
                self._normalize_trajectory_input(t, trajectory_frame)
                for t in trajectories
            ],
            "scene_token": scene_token,
            "log_name": log_name,
            "dataset": dataset,
        }
        if config_overrides:
            payload["config_overrides"] = config_overrides

        try:
            response = self._request_json(
                method="POST",
                path="/v1/score/internal/rl/batch",
                payload=payload,
            )
        except SimEngineClientError as exc:
            return [self._internal_error_result(str(exc)) for _ in trajectories]

        if not isinstance(response, dict):
            return [
                self._internal_error_result(
                    "Invalid JSON payload from /v1/score/internal/rl/batch: expected object"
                )
                for _ in trajectories
            ]

        results = response.get("results")
        if not isinstance(results, list) or len(results) != len(trajectories):
            return [
                self._internal_error_result(
                    "Invalid JSON payload from /v1/score/internal/rl/batch: malformed results"
                )
                for _ in trajectories
            ]

        return [
            self._normalize_internal_result(item, endpoint="/v1/score/internal/rl/batch")
            for item in results
        ]

    # ── Dataset Management ──────────────────────────────────────────────

    def list_datasets(self) -> Dict[str, str]:
        """List all registered datasets on the server."""
        try:
            response = self._request_json(method="GET", path="/v1/datasets")
        except SimEngineClientError as exc:
            return {"error": str(exc)}
        if isinstance(response, dict):
            return response.get("datasets", response)
        return {"error": "Invalid response"}

    def register_dataset(self, name: str, path: str) -> dict:
        """Register a new dataset on the server."""
        return self._request_json(
            method="POST",
            path="/v1/datasets",
            payload={"name": name, "path": path},
        )

    def unregister_dataset(self, name: str) -> dict:
        """Unregister a dataset from the server."""
        return self._request_json(method="DELETE", path=f"/v1/datasets/{name}")

    # ── Health ──────────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Return health metadata or an error payload when the service is unreachable."""
        try:
            response = self._request_json(method="GET", path="/v1/health")
        except SimEngineClientError as exc:
            return {"status": "error", "error": str(exc)}

        if isinstance(response, dict):
            return response
        return {"status": "error", "error": "Invalid JSON payload from /v1/health: expected object"}

    # ── Internal ────────────────────────────────────────────────────────

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(
            url=f"{self._base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )

        try:
            with urlopen(request, timeout=self._timeout) as response:
                raw_body = response.read()
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").strip()
            detail = f": {body}" if body else ""
            raise SimEngineClientError(f"HTTP {exc.code} from {path}{detail}") from exc
        except URLError as exc:
            reason = exc.reason
            if isinstance(reason, socket.timeout):
                raise SimEngineClientError(f"Request to {path} timed out after {self._timeout}s") from exc
            raise SimEngineClientError(f"Request to {path} failed: {reason}") from exc
        except socket.timeout as exc:
            raise SimEngineClientError(f"Request to {path} timed out after {self._timeout}s") from exc
        except OSError as exc:
            raise SimEngineClientError(f"Request to {path} failed: {exc}") from exc

        try:
            return json.loads(raw_body.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise SimEngineClientError(f"Response from {path} is not valid UTF-8") from exc
        except json.JSONDecodeError as exc:
            raise SimEngineClientError(f"Response from {path} is not valid JSON: {exc.msg}") from exc

    def _normalize_trajectory_input(
        self, trajectory: List[List[float]], trajectory_frame: Literal["nous", "nuplan"]
    ) -> List[List[float]]:
        if trajectory_frame == "nuplan":
            return trajectory
        arr = np.asarray(trajectory, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise SimEngineClientError(f"Invalid trajectory shape for conversion: {arr.shape}")
        xy = CoordinateConverter.nous_to_nuplan_batch(arr[:, :2])
        if arr.shape[1] == 2:
            return xy.tolist()
        return np.column_stack([xy, arr[:, 2:]]).tolist()

    def _normalize_output(self, result: dict[str, Any]) -> dict[str, Any]:
        pts = result.get("local_centerline_points")
        if isinstance(pts, list):
            converted = []
            for item in pts:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                x_nous, y_nous = CoordinateConverter.nuplan_to_nous(float(item[0]), float(item[1]))
                converted.append([x_nous, y_nous])
            result = dict(result)
            result["local_centerline_points"] = converted
        return result

    def _normalize_result(self, response: Any, endpoint: str) -> dict[str, Any]:
        if not isinstance(response, dict):
            return self._error_result(f"Invalid JSON payload from {endpoint}: expected object")

        result = self._error_result("")
        result.update(response)
        return result

    def _normalize_rl_result(self, response: Any, endpoint: str) -> dict[str, Any]:
        if not isinstance(response, dict):
            return self._rl_error_result(f"Invalid JSON payload from {endpoint}: expected object")

        result = self._rl_error_result("")
        result.update(response)
        return result

    def _normalize_internal_result(self, response: Any, endpoint: str) -> dict[str, Any]:
        if not isinstance(response, dict):
            return self._internal_error_result(f"Invalid JSON payload from {endpoint}: expected object")

        result = self._internal_error_result("")
        result.update(response)
        return result

    def _result_score(self, result: dict[str, Any]) -> float:
        try:
            return float(result.get("pdm_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _rl_result_score(self, result: dict[str, Any]) -> float:
        try:
            return float(result.get("rl_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _internal_result_score(self, result: dict[str, Any]) -> float:
        try:
            return float(result.get("internal_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _error_result(self, message: str) -> dict[str, Any]:
        return {
            "pdm_score": 0.0,
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "driving_direction_compliance": 1.0,
            "traffic_light_compliance": 1.0,
            "ego_progress": 0.0,
            "time_to_collision": 1.0,
            "lane_keeping": 1.0,
            "history_comfort": 1.0,
            "error": message or None,
        }

    def _rl_error_result(self, message: str) -> dict[str, Any]:
        return {
            "rl_score": 0.0,
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "driving_direction_compliance": 1.0,
            "traffic_light_compliance": 1.0,
            "ego_progress": 0.0,
            "time_to_collision": 1.0,
            "lane_keeping": 1.0,
            "history_comfort": 1.0,
            "pdms_score": 0.0,
            "pdms_no_at_fault_collisions": 1.0,
            "pdms_drivable_area_compliance": 1.0,
            "pdms_driving_direction_compliance": 1.0,
            "pdms_traffic_light_compliance": 1.0,
            "pdms_ego_progress": 0.0,
            "pdms_time_to_collision": 1.0,
            "pdms_lane_keeping": 1.0,
            "pdms_history_comfort": 1.0,
            "sub_rewards": {},
            "error": message or None,
        }

    def _internal_error_result(self, message: str) -> dict[str, Any]:
        return {
            "internal_score": 0.0,
            "safety_score": 0.0,
            "comfort_score": 0.0,
            "key_action_score": 0.0,
            "progress_score": 0.0,
            "no_at_fault_collisions": 1.0,
            "drivable_area_compliance": 1.0,
            "sample_valid": False,
            "invalid_reason": message or None,
            "first_no_nudge_upper_bound": None,
            "overrun_no_nudge_gate": False,
            "num_relevant_labeled": 0,
            "num_key_actions": 0,
            "num_key_actions_passed": 0,
            "ego_front_max": 0.0,
            "ego_rear_max": 0.0,
            "raw_progress": 0.0,
            "progress_norm": 0.0,
            "progress_norm_source": "none",
            "error": message or None,
        }


__all__ = ["SimEngineClient", "SimEngineClientError"]
