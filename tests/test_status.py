from datetime import date
from pathlib import Path

from llmops.monitors.cost import daily_rollup, load_prices
from llmops.registry.models import load_pins
from llmops.reports.status import EvalScore, render_status
from llmops.rollout.controller import RolloutController
from llmops.traces import load_traces


def build(root: Path):
    pins = load_pins(root / 'registry' / 'pins.yaml')
    prices = load_prices(root / 'data' / 'prices.yaml')
    costs = daily_rollup(load_traces(root / 'data' / 'traces_fixture.jsonl'), prices)
    canary = RolloutController('tweet_classifier', 'qwen-r2')
    canary.start_canary(25)
    evals = {'tweet_classifier': EvalScore(score=0.91, decision='pass')}
    return pins, render_status(pins, {'tweet_classifier': canary}, evals, costs, date(2025, 3, 2))


def test_every_pinned_task_is_on_the_page(root: Path) -> None:
    pins, page = build(root)
    for task in pins.tasks:
        assert task in page.markdown
        assert task in page.html
    assert len(page.rows) == len(pins.tasks)


def test_page_shows_stage_eval_and_cost(root: Path) -> None:
    _, page = build(root)
    row = next(r for r in page.rows if r.task == 'tweet_classifier')
    assert row.stage == 'canary 25%'
    assert row.last_eval == '0.91 pass'
    assert row.cost_7d == '$0.0251'
    other = next(r for r in page.rows if r.task == 'injection_guard')
    assert other.stage == 'full'
    assert other.last_eval == 'none'
    assert other.cost_7d == '$0.0000'


def test_html_escapes_values(root: Path) -> None:
    pins, _ = build(root)
    tricky = pins.tasks['intent_router'].model_copy(update={'model': '<script>x</script>'})
    pins = pins.model_copy(update={'tasks': {**pins.tasks, 'intent_router': tricky}})
    page = render_status(pins, {}, {}, [], date(2025, 3, 2))
    assert '<script>x</script>' not in page.html
    assert '&lt;script&gt;' in page.html
