import numpy as np
import pytest

from llmops.monitors.drift import (
    ScoreDriftConfig,
    check_embedding_drift,
    check_score_drift,
    cosine_distance,
    daily_mean_scores,
)
from llmops.traces import load_traces

CONFIG = ScoreDriftConfig(window_days=7, max_drop=0.05)


def test_decay_is_detected() -> None:
    series = [0.92, 0.92, 0.91, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.79]
    result = check_score_drift(0.92, series, CONFIG)
    assert result.alert is True
    assert result.drop == pytest.approx(0.92 - np.mean(series[-7:]))


def test_stable_series_stays_quiet() -> None:
    series = [0.92, 0.91, 0.93, 0.92, 0.91, 0.92, 0.93, 0.92]
    assert check_score_drift(0.92, series, CONFIG).alert is False


def test_short_series_never_alerts() -> None:
    result = check_score_drift(0.92, [0.5, 0.5, 0.5], CONFIG)
    assert result.alert is False
    assert result.window_mean is None


def test_drop_equal_to_delta_does_not_alert() -> None:
    assert check_score_drift(0.90, [0.85] * 7, CONFIG).alert is False


def test_centroid_drift_fires_on_shifted_inputs() -> None:
    rng = np.random.default_rng(7)
    baseline = rng.normal(size=(200, 16)) + np.array([5.0] + [0.0] * 15)
    same = rng.normal(size=(200, 16)) + np.array([5.0] + [0.0] * 15)
    shifted = rng.normal(size=(200, 16)) + np.array([0.0, 5.0] + [0.0] * 14)
    centre = baseline.mean(axis=0)
    assert check_embedding_drift(centre, same, max_distance=0.1).alert is False
    assert check_embedding_drift(centre, shifted, max_distance=0.1).alert is True


def test_cosine_distance_basics() -> None:
    assert cosine_distance(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cosine_distance(np.zeros(2), np.array([1.0, 0.0]))


def test_daily_scores_from_fixture(root) -> None:
    rows = load_traces(root / 'data' / 'traces_fixture.jsonl')
    scores = daily_mean_scores(rows, 'tweet_classifier')
    assert scores == pytest.approx([0.93, 0.905])
