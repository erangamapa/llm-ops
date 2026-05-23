from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from llmops.monitors.cost import (
    UnpricedModelError,
    daily_rollup,
    load_prices,
    migration_table,
    render_migration_table,
)
from llmops.traces import TraceRow, load_traces


@pytest.fixture
def rows(root: Path) -> list[TraceRow]:
    return load_traces(root / 'data' / 'traces_fixture.jsonl')


def test_rollup_math_is_exact(root: Path, rows: list[TraceRow]) -> None:
    rollup = daily_rollup(rows, load_prices(root / 'data' / 'prices.yaml'))
    by_key = {(line.task, line.day): line for line in rollup}

    first = by_key[('tweet_classifier', date(2025, 3, 1))]
    assert (first.requests, first.input_tokens, first.output_tokens) == (4, 4000, 800)
    assert first.dollars == Decimal('0.024')

    second = by_key[('tweet_classifier', date(2025, 3, 2))]
    assert second.dollars == Decimal('0.00112')

    assert by_key[('bet_drafter', date(2025, 3, 1))].dollars == Decimal('0.0036')
    assert by_key[('intent_router', date(2025, 3, 2))].dollars == Decimal('0.00006')
    assert len(rollup) == 6


def test_migration_table_shows_open_cheaper(root: Path, rows: list[TraceRow]) -> None:
    lines = migration_table(rows, load_prices(root / 'data' / 'prices.yaml'))
    assert [line.task for line in lines] == ['tweet_classifier']
    line = lines[0]
    assert line.paid_per_request == Decimal('0.006')
    assert line.open_per_request == Decimal('0.00028')
    assert line.open_requests == 4
    assert line.saved == Decimal('0.02288')
    assert line.open_per_request < line.paid_per_request


def test_rendered_table_has_the_savings(root: Path, rows: list[TraceRow]) -> None:
    prices = load_prices(root / 'data' / 'prices.yaml')
    table = render_migration_table(migration_table(rows, prices))
    assert '| tweet_classifier | $0.006000 | $0.000280 | 4 | $0.022880 |' in table


def test_unpriced_model_is_refused(root: Path) -> None:
    stray = TraceRow(
        traceId='x', task='tweet_classifier', day=date(2025, 3, 1), model='mystery',
        inputTokens=1, outputTokens=1,
    )
    with pytest.raises(UnpricedModelError):
        daily_rollup([stray], load_prices(root / 'data' / 'prices.yaml'))
