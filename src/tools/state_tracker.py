"""State tracking utility for automation progress."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class AutomationStateTrackerTool:
    name = "state_tracker"
    description = "Persist automation attempts and flag problematic tests."

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.report_path = self.artifacts_dir / "automation_report.json"
        self._state: Dict[str, Any] = {"tests": []}
        if self.report_path.exists():
            self._state = json.loads(self.report_path.read_text(encoding="utf-8"))

    def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        action = payload.get("action")
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
        for test in self._state["tests"]:
            if test["id"] == test_id:
                return test
        entry = {"id": test_id, "attempts": 0, "status": "pending", "artifacts": []}
        self._state["tests"].append(entry)
        return entry

    def _write(self) -> None:
        self.report_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
