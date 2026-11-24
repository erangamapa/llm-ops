from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Tier = Literal['rules', 'classical_ml', 'small_encoder', 'fine_tuned_small_llm', 'frontier']

KNOWN_TASKS = frozenset(
    {'tweet_classifier', 'bet_drafter', 'intent_router', 'injection_guard', 'tweet_impact'}
)


class PinError(Exception):
    """Base class for model pin failures."""


class UnknownTaskError(PinError):
    pass


class Pin(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    tier: Tier
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    adapter: str | None = None
    prompt_version: str | None = Field(default=None, pattern=r'^v\d+$')


class PinSet(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    tasks: dict[str, Pin]

    def get(self, task: str) -> Pin:
        if task not in self.tasks:
            raise UnknownTaskError(f'no pin for task {task!r}')
        return self.tasks[task]


def load_pins(path: Path) -> PinSet:
    pins = PinSet.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))
    unknown = sorted(set(pins.tasks) - KNOWN_TASKS)
    if unknown:
        raise UnknownTaskError(f'pins name unknown tasks {unknown}')
    return pins


def diff(old: PinSet, new: PinSet) -> str:
    """Describe what a deploy changes, one line per field."""
    lines: list[str] = []
    for task in sorted(set(old.tasks) | set(new.tasks)):
        before, after = old.tasks.get(task), new.tasks.get(task)
        if before is None and after is not None:
            lines.append(f'{task}: added on {after.model}@{after.revision}')
        elif after is None and before is not None:
            lines.append(f'{task}: removed, was {before.model}@{before.revision}')
        elif before is not None and after is not None:
            for field in Pin.model_fields:
                old_value, new_value = getattr(before, field), getattr(after, field)
                if old_value != new_value:
                    lines.append(f'{task}: {field} {old_value} -> {new_value}')
    return '\n'.join(lines) if lines else 'no changes'
