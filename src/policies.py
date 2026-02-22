"""Guardrail policies for the QA Automator."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Ensures we do not brute-force flaky tests beyond the agreed limit."""

    max_attempts: int = 10

    def assert_within_budget(self, attempts: int, test_id: str) -> None:
        if attempts > self.max_attempts:
            raise ValueError(
                f"Test {test_id} exceeded the maximum allowed attempts ({self.max_attempts})."
            )


@dataclass(frozen=True)
class ScreenshotPolicy:
    """Controls Maestro screenshot usage to keep runs deterministic."""

    max_screenshots_per_test: int = 3

    def allow_capture(self, taken: int) -> bool:
        return taken < self.max_screenshots_per_test
