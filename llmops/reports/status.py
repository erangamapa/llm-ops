import json
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from jinja2 import Environment, select_autoescape
from pydantic import BaseModel, ConfigDict, Field

from llmops.monitors.cost import DailyCost
from llmops.registry.models import PinSet
from llmops.rollout.controller import RolloutController, Stage


class EvalScore(BaseModel):
    """Latest eval result for a task, as a simple json file written by the eval gate."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    score: float = Field(ge=0, le=1)
    decision: str


def load_eval_scores(path: Path) -> dict[str, EvalScore]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    return {task: EvalScore.model_validate(value) for task, value in raw.items()}


class StatusRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: str
    tier: str
    model: str
    revision: str
    adapter: str
    prompt_version: str
    stage: str
    last_eval: str
    cost_7d: str


class StatusPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: list[StatusRow]
    markdown: str
    html: str


_HTML = Environment(autoescape=select_autoescape(default=True)).from_string(
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fleet status</title>
<style>
body { font-family: system-ui, sans-serif; margin: 2rem; }
table { border-collapse: collapse; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }
th { background: #f3f3f3; }
</style>
</head>
<body>
<h1>Fleet status</h1>
<p>As of {{ as_of }}. Cost covers the last 7 days.</p>
<table>
<thead>
<tr><th>Task</th><th>Tier</th><th>Model</th><th>Revision</th><th>Adapter</th>\
<th>Prompt</th><th>Rollout</th><th>Last eval</th><th>7 day cost</th></tr>
</thead>
<tbody>
{% for r in rows %}<tr><td>{{ r.task }}</td><td>{{ r.tier }}</td><td>{{ r.model }}</td>\
<td>{{ r.revision }}</td><td>{{ r.adapter }}</td><td>{{ r.prompt_version }}</td>\
<td>{{ r.stage }}</td><td>{{ r.last_eval }}</td><td>{{ r.cost_7d }}</td></tr>
{% endfor %}</tbody>
</table>
</body>
</html>
"""
)


def _stage_label(controller: RolloutController | None) -> str:
    if controller is None:
        return Stage.FULL.value
    if controller.stage is Stage.CANARY:
        return f'canary {controller.percent}%'
    return controller.stage.value


def render_status(
    pins: PinSet,
    rollouts: Mapping[str, RolloutController],
    evals: Mapping[str, EvalScore],
    daily_costs: Sequence[DailyCost],
    as_of: date,
) -> StatusPage:
    """One row per pinned task. A task with no rollout in flight is serving at full."""
    since = as_of - timedelta(days=6)
    rows: list[StatusRow] = []
    for task, pin in sorted(pins.tasks.items()):
        spend = sum(
            (c.dollars for c in daily_costs if c.task == task and since <= c.day <= as_of),
            Decimal(0),
        )
        evaluation = evals.get(task)
        rows.append(StatusRow(
            task=task,
            tier=pin.tier,
            model=pin.model,
            revision=pin.revision,
            adapter=pin.adapter or 'none',
            prompt_version=pin.prompt_version or 'none',
            stage=_stage_label(rollouts.get(task)),
            last_eval=f'{evaluation.score:.2f} {evaluation.decision}' if evaluation else 'none',
            cost_7d=f'${spend:.4f}',
        ))

    header = (
        '| task | tier | model | revision | adapter | prompt | rollout | last eval | 7 day cost |\n'
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |'
    )
    lines = [
        f'| {r.task} | {r.tier} | {r.model} | {r.revision} | {r.adapter} | {r.prompt_version} '
        f'| {r.stage} | {r.last_eval} | {r.cost_7d} |'
        for r in rows
    ]
    markdown = '\n'.join([f'# Fleet status as of {as_of}', '', header, *lines, ''])
    html = _HTML.render(rows=rows, as_of=as_of.isoformat())
    return StatusPage(rows=rows, markdown=markdown, html=html)
