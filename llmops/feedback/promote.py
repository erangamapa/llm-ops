import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

EventType = Literal['injury', 'lineup', 'trade', 'suspension', 'result', 'rumor', 'other']
Certainty = Literal['confirmed', 'likely', 'rumor']
MarketDirection = Literal['up', 'down', 'neutral']


class PromotionError(Exception):
    """Base class for incident promotion failures."""


class DuplicateIncidentError(PromotionError):
    pass


class InvalidLabelError(PromotionError):
    pass


class GoldenCase(BaseModel):
    """Same row shape as the golden tweets in model-foundry, so staged rows merge as is."""

    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=4000)
    source: str = Field(min_length=1)
    event_type: EventType
    entity: str = Field(min_length=1)
    certainty: Certainty
    market_direction: MarketDirection
    label_source: Literal['human']


def input_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()


def manifest_path(path: Path) -> Path:
    return path.with_name(path.name + '.manifest.json')


def _staged_texts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding='utf-8') as handle:
        return {input_hash(json.loads(line)['text']) for line in handle if line.strip()}


def write_manifest(path: Path) -> dict[str, Any]:
    """Recompute the manifest in the format model-foundry golden.py reads."""
    raw = path.read_bytes()
    rows = sum(1 for line in raw.decode('utf-8').splitlines() if line.strip())
    manifest = {'sha256': hashlib.sha256(raw).hexdigest(), 'rows': rows}
    manifest_path(path).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def promote(
    trace_row: Mapping[str, Any], corrected_label: Mapping[str, str], staging_path: Path
) -> GoldenCase:
    """Turn a wrong production answer into a staged golden case with the human label."""
    text = trace_row.get('text')
    source = trace_row.get('source')
    trace_id = trace_row.get('traceId')
    if not text or not source or not trace_id:
        raise PromotionError('trace row needs traceId, text and source')

    digest = input_hash(text)
    if digest in _staged_texts(staging_path):
        raise DuplicateIncidentError(f'input {digest[:12]} is already staged')

    try:
        case = GoldenCase(
            id=f'incident-{digest[:10]}',
            text=text,
            source=source,
            label_source='human',
            **corrected_label,
        )
    except (ValidationError, TypeError) as error:
        raise InvalidLabelError(f'corrected label rejected for {trace_id}, {error}') from error

    staging_path.parent.mkdir(parents=True, exist_ok=True)
    with staging_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(case.model_dump(), ensure_ascii=False) + '\n')
    write_manifest(staging_path)
    return case
