from collections import defaultdict
from collections.abc import Sequence
from datetime import date

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from llmops.traces import TraceRow


class ScoreDriftConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    window_days: int = Field(default=7, ge=1)
    max_drop: float = Field(default=0.05, gt=0)


class ScoreDriftResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    alert: bool
    window_mean: float | None
    drop: float | None


def daily_mean_scores(rows: Sequence[TraceRow], task: str) -> list[float]:
    """Mean judge score per day for one task, oldest day first."""
    by_day: dict[date, list[float]] = defaultdict(list)
    for row in rows:
        if row.task == task and row.judge_score is not None:
            by_day[row.day].append(row.judge_score)
    return [float(np.mean(by_day[day])) for day in sorted(by_day)]


def check_score_drift(
    baseline: float, daily_scores: Sequence[float], config: ScoreDriftConfig
) -> ScoreDriftResult:
    """Alert when the rolling mean sits more than max_drop below the release baseline."""
    if len(daily_scores) < config.window_days:
        return ScoreDriftResult(alert=False, window_mean=None, drop=None)
    window_mean = float(np.mean(daily_scores[-config.window_days:]))
    drop = baseline - window_mean
    # rounding keeps a drop of exactly max_drop from tripping on float noise
    alert = round(drop, 9) > config.max_drop
    return ScoreDriftResult(alert=alert, window_mean=window_mean, drop=drop)


class EmbeddingDriftResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    alert: bool
    distance: float


def centroid(vectors: np.ndarray) -> np.ndarray:
    return vectors.mean(axis=0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        raise ValueError('cosine distance is undefined for a zero vector')
    return 1.0 - float(np.dot(a, b)) / denominator


def check_embedding_drift(
    baseline_centroid: np.ndarray, recent_vectors: np.ndarray, max_distance: float
) -> EmbeddingDriftResult:
    """Compare the centroid of recent input vectors to the centroid saved at release."""
    distance = cosine_distance(baseline_centroid, centroid(recent_vectors))
    return EmbeddingDriftResult(alert=distance > max_distance, distance=distance)
