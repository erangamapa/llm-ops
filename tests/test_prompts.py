from pathlib import Path

import pytest

from llmops.registry.prompts import (
    DuplicateVersionError,
    Prompt,
    PromptNotFoundError,
    PromptRegistry,
)


def test_load_parses_front_matter(root: Path) -> None:
    prompt = PromptRegistry(root / 'registry' / 'prompts').load('tweet_classifier', 'v3')
    assert prompt.task == 'tweet_classifier'
    assert prompt.version == 'v3'
    assert prompt.owner == 'eranga'
    assert prompt.changelog.startswith('Added the source field')
    assert '{{ text }}' in prompt.body


def test_latest_picks_highest_version(root: Path) -> None:
    registry = PromptRegistry(root / 'registry' / 'prompts')
    assert registry.latest('tweet_classifier').version == 'v3'
    assert registry.versions('tweet_classifier') == ['v2', 'v3']


def test_latest_orders_numerically(tmp_path: Path) -> None:
    registry = PromptRegistry(tmp_path)
    for number in (2, 10):
        registry.register(Prompt(
            task='demo', version=f'v{number}', owner='me', changelog='x', body='body\n'
        ))
    assert registry.latest('demo').version == 'v10'


def test_unknown_version_and_task_are_refused(root: Path) -> None:
    registry = PromptRegistry(root / 'registry' / 'prompts')
    with pytest.raises(PromptNotFoundError):
        registry.load('tweet_classifier', 'v9')
    with pytest.raises(PromptNotFoundError):
        registry.latest('nope')


def test_register_refuses_existing_version(tmp_path: Path) -> None:
    registry = PromptRegistry(tmp_path)
    prompt = Prompt(task='demo', version='v1', owner='me', changelog='first', body='hello\n')
    registry.register(prompt)
    with pytest.raises(DuplicateVersionError):
        registry.register(prompt)
    assert registry.load('demo', 'v1').body == 'hello\n'


def test_two_files_declaring_one_version_are_refused(tmp_path: Path) -> None:
    directory = tmp_path / 'demo'
    directory.mkdir()
    header = '---\ntask: demo\nversion: v1\nowner: me\nchangelog: x\n---\nbody\n'
    (directory / 'v1.md').write_text(header, encoding='utf-8')
    (directory / 'copy.md').write_text(header, encoding='utf-8')
    with pytest.raises(DuplicateVersionError):
        PromptRegistry(tmp_path).load('demo', 'v1')
