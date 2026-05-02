from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Stage(StrEnum):
    SHADOW = 'shadow'
    CANARY = 'canary'
    FULL = 'full'
    ROLLED_BACK = 'rolled_back'


class RolloutError(Exception):
    """An illegal transition was requested."""


class MetricBreach(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    metric: str = Field(min_length=1)
    value: float
    threshold: float


class Transition(BaseModel):
    model_config = ConfigDict(frozen=True)

    from_stage: Stage
    to_stage: Stage
    percent: int
    reason: str


class RolloutController:
    """One candidate model on its way to serving a task. Shadow, then canary, then full."""

    def __init__(self, task: str, candidate: str) -> None:
        self.task = task
        self.candidate = candidate
        self.stage = Stage.SHADOW
        self.percent = 0
        self.history: list[Transition] = []

    def _move(self, to_stage: Stage, percent: int, reason: str) -> None:
        self.history.append(Transition(
            from_stage=self.stage, to_stage=to_stage, percent=percent, reason=reason
        ))
        self.stage = to_stage
        self.percent = percent

    def start_canary(self, percent: int) -> None:
        if self.stage is not Stage.SHADOW:
            raise RolloutError(f'canary can only start from shadow, not {self.stage.value}')
        self._check_percent(percent)
        self._move(Stage.CANARY, percent, 'canary started')

    def raise_canary(self, percent: int) -> None:
        if self.stage is not Stage.CANARY:
            raise RolloutError(f'canary can only be raised from canary, not {self.stage.value}')
        self._check_percent(percent)
        if percent <= self.percent:
            raise RolloutError(f'canary must grow, {self.percent} to {percent}')
        self._move(Stage.CANARY, percent, 'canary raised')

    def promote_full(self) -> None:
        if self.stage is not Stage.CANARY:
            raise RolloutError(f'full needs a canary first, not {self.stage.value}')
        self._move(Stage.FULL, 100, 'promoted to full')

    def on_breach(self, breach: MetricBreach) -> bool:
        """Roll back on a metric breach. Returns True when this event caused the rollback."""
        if self.stage is Stage.ROLLED_BACK:
            return False
        reason = f'{breach.metric} {breach.value} past {breach.threshold}'
        self._move(Stage.ROLLED_BACK, 0, reason)
        return True

    @staticmethod
    def _check_percent(percent: int) -> None:
        if not 1 <= percent <= 99:
            raise RolloutError(f'canary percent must be 1 to 99, got {percent}')
