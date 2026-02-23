"""State tracking utility for automation progress."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class StateTrackerInput(BaseModel):
    """Supported inputs for state_tracker tool calls."""

    action: str = Field(description="Action: record_attempt | mark_problematic | summary")
    test_id: str | None = Field(default=None, description="Test identifier for updates.")
    attempt: int = Field(default=1, ge=1, description="Attempt count.")
    status: str = Field(default="failed", description="Current run status.")
    artifacts: List[str] = Field(default_factory=list, description="Artifact file paths.")
    reason: str = Field(default="Max attempts exhausted", description="Problem reason.")


class AutomationStateTrackerTool(BaseTool):
    name: str = "state_tracker"
    description: str = "Persist automation attempts and flag problematic tests."
    args_schema: type[BaseModel] = StateTrackerInput

    artifacts_dir: Path
    _report_path: Path = PrivateAttr()
    _state: Dict[str, Any] = PrivateAttr(default_factory=lambda: {"tests": [], "scenarios": []})

    def model_post_init(self, __context: Any) -> None:
        self._report_path = self.artifacts_dir / "automation_report.json"
        if self._report_path.exists():
            self._state = json.loads(self._report_path.read_text(encoding="utf-8"))
        self._normalize_state()

    def _run(
        self,
        action: str,
        test_id: str | None = None,
        attempt: int = 1,
        status: str = "failed",
        artifacts: List[str] | None = None,
        reason: str = "Max attempts exhausted",
    ) -> Dict[str, Any]:
        payload = {
            "action": action,
            "test_id": test_id,
            "attempt": attempt,
            "status": status,
            "artifacts": artifacts or [],
            "reason": reason,
        }
        if action == "record_attempt":
            return self.record_attempt(payload)
        if action == "mark_problematic":
            return self.mark_problematic(payload)
        if action == "summary":
            return self._state
        raise ValueError(f"Unknown action {action}")

    # API --------------------------------------------------------------
    def record_attempt(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        test_id = payload["test_id"]
        attempt = payload.get("attempt", 1)
        status = payload.get("status", "failed")
        artifacts: List[str] = payload.get("artifacts", [])

        entry = self._get_or_create(test_id)
        entry["attempts"] = attempt
        entry.setdefault("history", []).append({"attempt": attempt, "status": status})
        entry.setdefault("artifacts", []).extend(artifacts)
        entry["status"] = status
        self._write()
        return entry

    def mark_problematic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        test_id = payload["test_id"]
        reason = payload.get("reason", "Max attempts exhausted")
        entry = self._get_or_create(test_id)
        entry["status"] = "problematic"
        entry["reason"] = reason
        self._write()
        return entry

    # Helpers ----------------------------------------------------------
    def _get_or_create(self, test_id: str) -> Dict[str, Any]:
        bucket_key = "scenarios" if self._is_scenario_id(test_id) else "tests"
        for test in self._state[bucket_key]:
            if test["id"] == test_id:
                return test
        entry = {"id": test_id, "attempts": 0, "status": "pending", "artifacts": []}
        self._state[bucket_key].append(entry)
        return entry

    def _is_scenario_id(self, value: str | None) -> bool:
        return str(value or "").strip().startswith("scenario_")

    def _normalize_state(self) -> None:
        tests = self._state.get("tests", [])
        scenarios = self._state.get("scenarios", [])
        if not isinstance(tests, list):
            tests = []
        if not isinstance(scenarios, list):
            scenarios = []

        migrated_tests: List[Dict[str, Any]] = []
        migrated_scenarios: List[Dict[str, Any]] = list(scenarios)
        for item in tests:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "")).strip()
            if self._is_scenario_id(item_id):
                migrated_scenarios.append(item)
            else:
                migrated_tests.append(item)

        self._state = {
            "tests": migrated_tests,
            "scenarios": migrated_scenarios,
        }

    def _write(self) -> None:
        self._report_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
