from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from llmops.traces import TraceRow


class UnpricedModelError(Exception):
    """A trace names a model with no entry in the price table."""


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    kind: Literal['paid', 'open']
    input_per_1k: Decimal
    output_per_1k: Decimal


class PriceTable(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    models: dict[str, ModelPrice]

    def price(self, model: str) -> ModelPrice:
        if model not in self.models:
            raise UnpricedModelError(f'no price for model {model!r}')
        return self.models[model]


def load_prices(path: Path) -> PriceTable:
    return PriceTable.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))


def row_cost(row: TraceRow, prices: PriceTable) -> Decimal:
    price = prices.price(row.model)
    return (
        Decimal(row.input_tokens) / 1000 * price.input_per_1k
        + Decimal(row.output_tokens) / 1000 * price.output_per_1k
    )


class DailyCost(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: str
    day: date
    requests: int
    input_tokens: int
    output_tokens: int
    dollars: Decimal


def daily_rollup(rows: Sequence[TraceRow], prices: PriceTable) -> list[DailyCost]:
    buckets: dict[tuple[str, date], list[TraceRow]] = defaultdict(list)
    for row in rows:
        buckets[(row.task, row.day)].append(row)
    return [
        DailyCost(
            task=task,
            day=day,
            requests=len(group),
            input_tokens=sum(r.input_tokens for r in group),
            output_tokens=sum(r.output_tokens for r in group),
            dollars=sum((row_cost(r, prices) for r in group), Decimal(0)),
        )
        for (task, day), group in sorted(buckets.items())
    ]


class MigrationLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: str
    paid_per_request: Decimal
    open_per_request: Decimal
    open_requests: int
    saved: Decimal


def migration_table(rows: Sequence[TraceRow], prices: PriceTable) -> list[MigrationLine]:
    """For tasks seen on both paid and open models, what the open requests saved."""
    totals: dict[str, dict[str, tuple[int, Decimal]]] = defaultdict(dict)
    for row in rows:
        kind = prices.price(row.model).kind
        count, dollars = totals[row.task].get(kind, (0, Decimal(0)))
        totals[row.task][kind] = (count + 1, dollars + row_cost(row, prices))

    lines: list[MigrationLine] = []
    for task in sorted(totals):
        if 'paid' not in totals[task] or 'open' not in totals[task]:
            continue
        paid_count, paid_dollars = totals[task]['paid']
        open_count, open_dollars = totals[task]['open']
        paid_avg, open_avg = paid_dollars / paid_count, open_dollars / open_count
        lines.append(MigrationLine(
            task=task,
            paid_per_request=paid_avg,
            open_per_request=open_avg,
            open_requests=open_count,
            saved=(paid_avg - open_avg) * open_count,
        ))
    return lines


def render_migration_table(lines: Sequence[MigrationLine]) -> str:
    header = (
        '| task | paid per request | open per request | open requests | saved |\n'
        '| --- | --- | --- | --- | --- |'
    )
    body = [
        f'| {line.task} | ${line.paid_per_request:.6f} | ${line.open_per_request:.6f} '
        f'| {line.open_requests} | ${line.saved:.6f} |'
        for line in lines
    ]
    return '\n'.join([header, *body])
