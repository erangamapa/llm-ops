import pytest

from llmops.rollout.controller import MetricBreach, RolloutController, RolloutError, Stage
from llmops.rollout.shadow import ComparisonRecord, ShadowRunner


def breach() -> MetricBreach:
    return MetricBreach(metric='judge_score', value=0.7, threshold=0.85)


def test_legal_path_walks_to_full() -> None:
    controller = RolloutController('tweet_classifier', 'qwen-r2')
    controller.start_canary(5)
    controller.raise_canary(25)
    controller.promote_full()
    assert controller.stage is Stage.FULL
    assert controller.percent == 100
    assert [t.to_stage for t in controller.history] == [Stage.CANARY, Stage.CANARY, Stage.FULL]


def test_shadow_cannot_jump_to_full() -> None:
    controller = RolloutController('tweet_classifier', 'qwen-r2')
    with pytest.raises(RolloutError):
        controller.promote_full()
    assert controller.stage is Stage.SHADOW


def test_canary_cannot_shrink_or_leave_range() -> None:
    controller = RolloutController('tweet_classifier', 'qwen-r2')
    with pytest.raises(RolloutError):
        controller.raise_canary(10)
    with pytest.raises(RolloutError):
        controller.start_canary(100)
    controller.start_canary(20)
    with pytest.raises(RolloutError):
        controller.raise_canary(20)
    with pytest.raises(RolloutError):
        controller.start_canary(30)


def test_breach_during_canary_rolls_back() -> None:
    controller = RolloutController('tweet_classifier', 'qwen-r2')
    controller.start_canary(10)
    assert controller.on_breach(breach()) is True
    assert controller.stage is Stage.ROLLED_BACK
    assert controller.percent == 0
    assert controller.history[-1].reason == 'judge_score 0.7 past 0.85'


def test_breach_after_rollback_is_ignored_and_rollback_is_final() -> None:
    controller = RolloutController('tweet_classifier', 'qwen-r2')
    controller.on_breach(breach())
    assert controller.on_breach(breach()) is False
    with pytest.raises(RolloutError):
        controller.start_canary(5)
    with pytest.raises(RolloutError):
        controller.promote_full()


def test_shadow_runner_returns_live_and_records_both() -> None:
    records: list[ComparisonRecord] = []
    ticks = iter([0.0, 0.010, 0.010, 0.030])
    runner = ShadowRunner(
        'tweet_classifier',
        live=lambda prompt: 'live',
        candidate=lambda prompt: 'candidate',
        sink=records.append,
        clock=lambda: next(ticks),
    )
    assert runner.run('hello') == 'live'
    record = records[0]
    assert record.live_output == 'live'
    assert record.candidate_output == 'candidate'
    assert record.matches is False
    assert record.live_latency_ms == pytest.approx(10.0)
    assert record.candidate_latency_ms == pytest.approx(20.0)


def test_shadow_runner_survives_a_broken_candidate() -> None:
    records: list[ComparisonRecord] = []

    def broken(prompt: str) -> str:
        raise TimeoutError('adapter not loaded')

    runner = ShadowRunner('tweet_classifier', lambda p: 'live', broken, records.append)
    assert runner.run('hello') == 'live'
    assert records[0].candidate_output is None
    assert records[0].candidate_error == 'TimeoutError: adapter not loaded'
