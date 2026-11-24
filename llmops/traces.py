import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class TraceRow(BaseModel):
    """One row of the trace export. Field names follow the camelCase export."""

    model_config = ConfigDict(extra='ignore', populate_by_name=True, frozen=True)

    trace_id: str = Field(alias='traceId', min_length=1)
    task: str = Field(min_length=1)
    day: date
    model: str = Field(min_length=1)
    prompt_version: str | None = Field(default=None, alias='promptVersion')
    input_tokens: int = Field(alias='inputTokens', ge=0)
    output_tokens: int = Field(alias='outputTokens', ge=0)
    judge_score: float | None = Field(default=None, alias='judgeScore', ge=0, le=1)
    text: str = ''
    source: str = ''


def load_traces(path: Path) -> list[TraceRow]:
    with path.open(encoding='utf-8') as handle:
        return [TraceRow.model_validate(json.loads(line)) for line in handle if line.strip()]
