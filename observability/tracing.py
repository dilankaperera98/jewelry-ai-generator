import logging
import time
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, computed_field

from observability.logging import log_event

logger = logging.getLogger(__name__)


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ToolCallTrace(BaseModel):
    tool_name: str
    step: int
    status: Literal["completed", "failed"]
    duration_ms: float


class ExecutionSummary(BaseModel):
    run_id: str
    workflow: Literal["generation", "agent"]
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    llm_calls: int
    tool_calls: int
    tools_used: list[str]
    tool_successes: int
    tool_failures: int
    step_count: int
    termination_reason: str
    validation_failures: int
    repair_attempts: int
    token_usage: TokenUsage | None = None


class RunTrace:
    """Mutable in-process recorder that exposes an immutable safe summary."""

    def __init__(self, run_id: str, workflow: Literal["generation", "agent"]):
        self.run_id = run_id
        self.workflow = workflow
        self.started_at = datetime.now(timezone.utc)
        self._started_perf = time.perf_counter()
        self.ended_at: datetime | None = None
        self.duration_ms: float | None = None
        self.llm_calls = 0
        self.tool_calls = 0
        self.tools_used: list[str] = []
        self.tool_events: list[ToolCallTrace] = []
        self.step_count = 0
        self.termination_reason = "running"
        self.validation_failures = 0
        self.repair_attempts = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._has_token_usage = False
        log_event(
            logger,
            logging.INFO,
            "run_started",
            run_id=run_id,
            workflow=workflow,
        )

    def llm_call_started(self, step: int) -> float:
        self.llm_calls += 1
        log_event(
            logger,
            logging.INFO,
            "llm_call_started",
            run_id=self.run_id,
            workflow=self.workflow,
            step=step,
            llm_call=self.llm_calls,
        )
        return time.perf_counter()

    def llm_call_completed(
        self, started_perf: float, step: int, usage: TokenUsage | None = None
    ) -> None:
        if usage is not None:
            self._has_token_usage = True
            self._input_tokens += usage.input_tokens
            self._output_tokens += usage.output_tokens
        log_event(
            logger,
            logging.INFO,
            "llm_call_completed",
            run_id=self.run_id,
            step=step,
            duration_ms=round((time.perf_counter() - started_perf) * 1000, 3),
            token_usage_available=usage is not None,
        )

    def llm_call_failed(self, started_perf: float, step: int) -> None:
        log_event(
            logger,
            logging.WARNING,
            "llm_call_failed",
            run_id=self.run_id,
            step=step,
            duration_ms=round((time.perf_counter() - started_perf) * 1000, 3),
        )

    def validation_failed(self, step: int) -> None:
        self.validation_failures += 1
        log_event(
            logger,
            logging.WARNING,
            "validation_failed",
            run_id=self.run_id,
            step=step,
            validation_failures=self.validation_failures,
        )

    def repair_attempted(self, step: int) -> None:
        self.repair_attempts += 1
        log_event(
            logger,
            logging.INFO,
            "repair_attempted",
            run_id=self.run_id,
            step=step,
            repair_attempts=self.repair_attempts,
        )

    def tool_requested(self, tool_name: str, step: int) -> float:
        self.tool_calls += 1
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        log_event(
            logger,
            logging.INFO,
            "tool_requested",
            run_id=self.run_id,
            tool_name=tool_name,
            step=step,
            tool_call=self.tool_calls,
        )
        return time.perf_counter()

    def tool_completed(self, tool_name: str, step: int, started_perf: float) -> None:
        duration_ms = round((time.perf_counter() - started_perf) * 1000, 3)
        self.tool_events.append(
            ToolCallTrace(
                tool_name=tool_name,
                step=step,
                status="completed",
                duration_ms=duration_ms,
            )
        )
        log_event(
            logger,
            logging.INFO,
            "tool_completed",
            run_id=self.run_id,
            tool_name=tool_name,
            step=step,
            duration_ms=duration_ms,
        )

    def tool_failed(self, tool_name: str, step: int, started_perf: float) -> None:
        duration_ms = round((time.perf_counter() - started_perf) * 1000, 3)
        self.tool_events.append(
            ToolCallTrace(
                tool_name=tool_name,
                step=step,
                status="failed",
                duration_ms=duration_ms,
            )
        )
        log_event(
            logger,
            logging.WARNING,
            "tool_failed",
            run_id=self.run_id,
            tool_name=tool_name,
            step=step,
            duration_ms=duration_ms,
        )

    def retrieval_completed(self, step: int, result_count: int) -> None:
        log_event(
            logger,
            logging.INFO,
            "retrieval_completed",
            run_id=self.run_id,
            step=step,
            result_count=result_count,
        )

    def finish(self, termination_reason: str, step_count: int, failed: bool = False) -> None:
        if self.ended_at is not None:
            return
        self.ended_at = datetime.now(timezone.utc)
        self.duration_ms = round((time.perf_counter() - self._started_perf) * 1000, 3)
        self.step_count = step_count
        self.termination_reason = termination_reason
        log_event(
            logger,
            logging.WARNING if failed else logging.INFO,
            "run_failed" if failed else "run_completed",
            run_id=self.run_id,
            workflow=self.workflow,
            duration_ms=self.duration_ms,
            llm_calls=self.llm_calls,
            tool_calls=self.tool_calls,
            step_count=step_count,
            termination_reason=termination_reason,
        )

    def summary(self) -> ExecutionSummary:
        if self.ended_at is None or self.duration_ms is None:
            raise RuntimeError("Run trace is not complete")
        token_usage = None
        if self._has_token_usage:
            token_usage = TokenUsage(
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
            )
        return ExecutionSummary(
            run_id=self.run_id,
            workflow=self.workflow,
            started_at=self.started_at,
            ended_at=self.ended_at,
            duration_ms=self.duration_ms,
            llm_calls=self.llm_calls,
            tool_calls=self.tool_calls,
            tools_used=self.tools_used,
            tool_successes=sum(event.status == "completed" for event in self.tool_events),
            tool_failures=sum(event.status == "failed" for event in self.tool_events),
            step_count=self.step_count,
            termination_reason=self.termination_reason,
            validation_failures=self.validation_failures,
            repair_attempts=self.repair_attempts,
            token_usage=token_usage,
        )
