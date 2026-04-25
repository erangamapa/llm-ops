from pathlib import Path

import pytest

from llmops.registry.models import (
    KNOWN_TASKS,
    PinSet,
    UnknownTaskError,
    diff,
    load_pins,
)


def test_shipped_pins_cover_every_known_task(root: Path) -> None:
    pins = load_pins(root / 'registry' / 'pins.yaml')
    assert set(pins.tasks) == KNOWN_TASKS


def test_diff_names_each_changed_field(root: Path) -> None:
    old = load_pins(root / 'registry' / 'pins.yaml')
    classifier = old.get('tweet_classifier')
    new = PinSet(tasks={
        **old.tasks,
        'tweet_classifier': classifier.model_copy(
            update={'revision': 'aa11bb2', 'prompt_version': 'v4'}
        ),
    })
    assert diff(old, new).splitlines() == [
        'tweet_classifier: revision 8f3c2a1 -> aa11bb2',
        'tweet_classifier: prompt_version v3 -> v4',
    ]


def test_diff_reports_added_and_removed_tasks(root: Path) -> None:
    old = load_pins(root / 'registry' / 'pins.yaml')
    smaller = PinSet(tasks={k: v for k, v in old.tasks.items() if k != 'bet_drafter'})
    removed = 'bet_drafter: removed, was meta-llama/Llama-3.1-8B-Instruct@5d91b07'
    assert diff(old, smaller) == removed
    assert diff(smaller, old).startswith('bet_drafter: added on')


def test_identical_pins_have_no_changes(root: Path) -> None:
    pins = load_pins(root / 'registry' / 'pins.yaml')
    assert diff(pins, pins) == 'no changes'


def test_unknown_task_is_refused(root: Path, tmp_path: Path) -> None:
    pins = load_pins(root / 'registry' / 'pins.yaml')
    with pytest.raises(UnknownTaskError):
        pins.get('moderation')

    bad = tmp_path / 'pins.yaml'
    bad.write_text(
        'tasks:\n  moderation:\n    tier: frontier\n    provider: x\n'
        '    model: m\n    revision: r\n',
        encoding='utf-8',
    )
    with pytest.raises(UnknownTaskError):
        load_pins(bad)
