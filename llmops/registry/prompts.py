import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

_FRONT_MATTER = re.compile(r'\A---\n(.*?)\n---\n(.*)\Z', re.DOTALL)
_VERSION = re.compile(r'\Av(\d+)\Z')


class PromptError(Exception):
    """Base class for prompt registry failures."""


class PromptNotFoundError(PromptError):
    pass


class DuplicateVersionError(PromptError):
    pass


class MalformedPromptError(PromptError):
    pass


class Prompt(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    task: str = Field(min_length=1)
    version: str = Field(pattern=r'^v\d+$')
    owner: str = Field(min_length=1)
    changelog: str = Field(min_length=1)
    body: str


def _version_number(version: str) -> int:
    match = _VERSION.match(version)
    if match is None:
        raise MalformedPromptError(f'version {version!r} must look like v1, v2')
    return int(match.group(1))


def parse_prompt(path: Path) -> Prompt:
    match = _FRONT_MATTER.match(path.read_text(encoding='utf-8'))
    if match is None:
        raise MalformedPromptError(f'{path} has no front matter header')
    try:
        header = yaml.safe_load(match.group(1)) or {}
        return Prompt(**header, body=match.group(2).strip() + '\n')
    except (yaml.YAMLError, TypeError, ValidationError) as error:
        raise MalformedPromptError(f'{path} has a bad header, {error}') from error


class PromptRegistry:
    """Prompts live at <root>/<task>/<version>.md. A service pins task plus version."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _scan(self, task: str) -> dict[str, Prompt]:
        directory = self._root / task
        if not directory.is_dir():
            raise PromptNotFoundError(f'no prompts for task {task!r}')
        found: dict[str, Prompt] = {}
        paths: dict[str, Path] = {}
        for path in sorted(directory.glob('*.md')):
            prompt = parse_prompt(path)
            if prompt.task != task:
                raise MalformedPromptError(f'{path} declares task {prompt.task!r}')
            if prompt.version in found:
                raise DuplicateVersionError(f'{task} {prompt.version} is defined twice')
            found[prompt.version] = prompt
            paths[prompt.version] = path
        for version, path in paths.items():
            if path.stem != version:
                raise MalformedPromptError(f'{path.name} declares version {version}')
        return found

    def load(self, task: str, version: str) -> Prompt:
        prompts = self._scan(task)
        if version not in prompts:
            raise PromptNotFoundError(f'{task} has no version {version}')
        return prompts[version]

    def latest(self, task: str) -> Prompt:
        prompts = self._scan(task)
        if not prompts:
            raise PromptNotFoundError(f'no prompts for task {task!r}')
        return prompts[max(prompts, key=_version_number)]

    def versions(self, task: str) -> list[str]:
        return sorted(self._scan(task), key=_version_number)

    def register(self, prompt: Prompt) -> Path:
        directory = self._root / prompt.task
        target = directory / f'{prompt.version}.md'
        if target.exists():
            raise DuplicateVersionError(f'{prompt.task} {prompt.version} already exists')
        directory.mkdir(parents=True, exist_ok=True)
        header = yaml.safe_dump(
            {
                'task': prompt.task,
                'version': prompt.version,
                'owner': prompt.owner,
                'changelog': prompt.changelog,
            },
            sort_keys=False,
        )
        target.write_text(f'---\n{header}---\n{prompt.body}', encoding='utf-8')
        return target
