import hashlib
import json
from pathlib import Path

import pytest

from llmops.feedback.promote import (
    DuplicateIncidentError,
    InvalidLabelError,
    PromotionError,
    manifest_path,
    promote,
)

LABEL = {
    'event_type': 'injury',
    'entity': 'Emil Dravic',
    'certainty': 'likely',
    'market_direction': 'down',
}


def trace(text: str = 'Emil Dravic limped off, doubtful for Saturday.') -> dict[str, str]:
    return {'traceId': 't-001', 'text': text, 'source': 'beat_reporter'}


def test_promote_appends_and_updates_manifest(tmp_path: Path) -> None:
    staging = tmp_path / 'golden_staging.jsonl'
    promote(trace(), LABEL, staging)
    second = trace('Idris Calloway is back in the eleven.')
    promote(second, {**LABEL, 'event_type': 'lineup'}, staging)

    lines = staging.read_text(encoding='utf-8').splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])['label_source'] == 'human'

    manifest = json.loads(manifest_path(staging).read_text(encoding='utf-8'))
    assert manifest == {'sha256': hashlib.sha256(staging.read_bytes()).hexdigest(), 'rows': 2}


def test_duplicate_input_is_refused(tmp_path: Path) -> None:
    staging = tmp_path / 'golden_staging.jsonl'
    promote(trace(), LABEL, staging)
    with pytest.raises(DuplicateIncidentError):
        promote(trace('  Emil Dravic limped off, doubtful for Saturday.  '), LABEL, staging)
    assert len(staging.read_text(encoding='utf-8').splitlines()) == 1


@pytest.mark.parametrize(
    'label',
    [
        {**LABEL, 'event_type': 'earthquake'},
        {**LABEL, 'entity': ''},
        {'event_type': 'injury'},
        {**LABEL, 'surprise': 'field'},
    ],
)
def test_bad_label_is_rejected_and_nothing_is_written(
    tmp_path: Path, label: dict[str, str]
) -> None:
    staging = tmp_path / 'golden_staging.jsonl'
    with pytest.raises(InvalidLabelError):
        promote(trace(), label, staging)
    assert not staging.exists()


def test_trace_without_text_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PromotionError):
        promote({'traceId': 't-9'}, LABEL, tmp_path / 'staging.jsonl')
