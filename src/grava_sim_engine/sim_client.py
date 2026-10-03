"""HTTP scoring in nuPlan coordinates with public PDMS, CDS and RL metric names."""

import json
from urllib.request import Request, urlopen

from grava_sim_engine.utils.coordinates import to_nuplan

SCORE_KEYS = {"pdms": "pdm_score", "cds": "cds", "continuous": "rl_score", "discrete": "rl_score"}


class SimEngineClient:
    """Score decoded trajectories; HTTP, JSON and service errors propagate."""

    def __init__(self, base_url: str = "http://localhost:8100", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    @staticmethod
    def _read_score(response: dict, scoring_mode: str) -> dict:
        if response.get("error"):
            raise RuntimeError(f"sim-engine {scoring_mode}: {response['error']}")
        result = dict(response)
        key = SCORE_KEYS[scoring_mode]
        result[key] = float(result[key])
        return result

    def score(
        self,
        trajectory: list[list[float]],
        scene_token: str,
        log_name: str,
        dataset: str,
        trajectory_frame: str = "nuplan",
        scoring_mode: str = "pdms",
        *,
        config_overrides: dict[str, float] | None = None,
        include_details: bool = False,
    ) -> tuple[float, dict]:
        payload = {
            "trajectory": to_nuplan(trajectory, trajectory_frame).tolist(),
            "scene_token": scene_token,
            "log_name": log_name,
            "dataset": dataset,
            "scoring_mode": scoring_mode,
            "config_overrides": config_overrides or {},
            "include_details": include_details,
        }
        result = self._read_score(self._post("/v1/score", payload), scoring_mode)
        return result[SCORE_KEYS[scoring_mode]], result

    def score_batch(
        self,
        trajectories: list[list[list[float]]],
        scene_token: str,
        log_name: str,
        dataset: str,
        trajectory_frame: str = "nuplan",
        scoring_mode: str = "pdms",
        *,
        config_overrides: dict[str, float] | None = None,
        include_details: bool = False,
    ) -> list[dict]:
        payload = {
            "trajectories": [to_nuplan(t, trajectory_frame).tolist() for t in trajectories],
            "scene_token": scene_token,
            "log_name": log_name,
            "dataset": dataset,
            "scoring_mode": scoring_mode,
            "config_overrides": config_overrides or {},
            "include_details": include_details,
        }
        response = self._post("/v1/score/batch", payload)
        return [
            self._read_score(result, scoring_mode)
            for _, result in zip(trajectories, response["results"], strict=True)
        ]

    def score_control(
        self,
        control_signals: list[list[float]],
        scene_token: str,
        log_name: str,
        dataset: str,
    ) -> tuple[float, dict]:
        result = self._read_score(
            self._post(
                "/v1/score/control",
                {
                    "control_signals": control_signals,
                    "scene_token": scene_token,
                    "log_name": log_name,
                    "dataset": dataset,
                },
            ),
            "pdms",
        )
        return result["pdm_score"], result

    def score_control_batch(
        self,
        control_signals_batch: list[list[list[float]]],
        scene_token: str,
        log_name: str,
        dataset: str,
    ) -> list[dict]:
        response = self._post(
            "/v1/score/control/batch",
            {
                "control_signals_batch": control_signals_batch,
                "scene_token": scene_token,
                "log_name": log_name,
                "dataset": dataset,
            },
        )
        return [
            self._read_score(result, "pdms")
            for _, result in zip(control_signals_batch, response["results"], strict=True)
        ]

    def health(self) -> dict:
        with urlopen(self.base_url + "/v1/health", timeout=self.timeout) as response:
            return json.load(response)

    def datasets(self) -> dict:
        with urlopen(self.base_url + "/v1/datasets", timeout=self.timeout) as response:
            return json.load(response)

    def warmup(self) -> dict:
        with urlopen(self.base_url + "/v1/warmup", timeout=self.timeout) as response:
            return json.load(response)
