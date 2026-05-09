import hashlib
import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

ModelCall = Callable[[str], str]
RecordSink = Callable[['ComparisonRecord'], None]


class ComparisonRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    task: str
    input_sha256: str
    live_output: str
    candidate_output: str | None
    candidate_error: str | None
    matches: bool
    live_latency_ms: float
    candidate_latency_ms: float


class ShadowRunner:
    """Calls the candidate beside the live model, records both, and only ever returns live."""

    def __init__(
        self,
        task: str,
        live: ModelCall,
        candidate: ModelCall,
        sink: RecordSink,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._task = task
        self._live = live
        self._candidate = candidate
        self._sink = sink
        self._clock = clock

    def run(self, prompt: str) -> str:
        started = self._clock()
        live_output = self._live(prompt)
        live_ms = (self._clock() - started) * 1000

        candidate_output: str | None = None
        candidate_error: str | None = None
        started = self._clock()
        try:
            candidate_output = self._candidate(prompt)
        except Exception as error:  # a broken candidate must never touch serving
            candidate_error = f'{type(error).__name__}: {error}'
        candidate_ms = (self._clock() - started) * 1000

        self._sink(ComparisonRecord(
            task=self._task,
            input_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
            live_output=live_output,
            candidate_output=candidate_output,
            candidate_error=candidate_error,
            matches=candidate_output == live_output,
            live_latency_ms=live_ms,
            candidate_latency_ms=candidate_ms,
        ))
        return live_output
