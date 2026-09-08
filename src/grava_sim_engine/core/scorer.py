"""Backward-compatible facade for the split scorer modules.

All public APIs (PDMScorer, PDMScorerConfig, RLScorerConfig) remain importable
from this module. Internally they delegate to the new split implementations.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .scoring.base import PDMScorerConfig, RLScorerConfig, _GTSimResult
from .scoring.base import ScorerBase
from .scoring.scorer_v1 import PDMScorerV1
from .scoring.scorer_v2 import PDMScorerV2
from .scoring.scorer_rl import RLScorer as _RLScorerImpl
from .types import SceneContext, ScoringResult, RLScoringResult


class PDMScorer:
    """Unified facade that delegates to PDMScorerV1 or PDMScorerV2 based on config.

    Maintains backward compatibility with code that imports PDMScorer from
    grava_sim_engine.core.scorer.
    """

    def __init__(self, config: PDMScorerConfig | None = None):
        self._config = config or PDMScorerConfig()

    def _get_scorer(self) -> PDMScorerV1 | PDMScorerV2:
        if self._config.scoring_version == "v2":
            return PDMScorerV2(config=self._config)
        return PDMScorerV1()

    def score(self, waypoints_xy: np.ndarray, scene: SceneContext) -> ScoringResult:
        return self._get_scorer().score(waypoints_xy, scene)

    def score_batch(
        self, trajectories_xy: np.ndarray, scene: SceneContext,
    ) -> List[ScoringResult]:
        return self._get_scorer().score_batch(trajectories_xy, scene)

    def score_for_rl(
        self,
        waypoints_xy: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig | None = None,
    ) -> RLScoringResult:
        return self.score_batch_for_rl(waypoints_xy[None, ...], scene, rl_config)[0]

    def score_batch_for_rl(
        self,
        trajectories_xy: np.ndarray,
        scene: SceneContext,
        rl_config: RLScorerConfig | None = None,
    ) -> List[RLScoringResult]:
        scorer = _RLScorerImpl()
        return scorer.score_batch(trajectories_xy, scene, rl_config)

    def _simulate_and_score_gt(self, scene: SceneContext) -> _GTSimResult | None:
        """Backward-compatible GT simulation."""
        base = ScorerBase()
        return base._simulate_and_score_gt(scene)


# Re-export for backward compatibility
__all__ = [
    "PDMScorer",
    "PDMScorerConfig",
    "RLScorerConfig",
    "PDMScorerV1",
    "PDMScorerV2",
]
