"""Tool for interpreting Qase-exported JSON test cases."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class QaseTestParserTool:
    """Parses input files and surfaces pending cases to the agent."""

    name = "qase_parser"
    description = (
        "Load Qase-exported JSON and previously tested metadata; return structured plans "
        "for the pending cases."
    )

    def __init__(self, test_cases: Path, tested_cases: Path) -> None:
        self.test_cases_path = Path(test_cases)
        self.tested_cases_path = Path(tested_cases)
        self._cases = self._load_cases()
        self._tested = self._load_tested()

    def __call__(self, query: str | None = None) -> Dict[str, Any]:
        """Return a prioritized backlog for the agent."""
        pending = [case for case in self._cases if case["id"] not in self._tested]
        return {
            "total": len(self._cases),
            "pending": len(pending),
            "cases": pending,
        }

    # Internal helpers -------------------------------------------------
    def _load_cases(self) -> List[Dict[str, Any]]:
        raw = json.loads(self.test_cases_path.read_text(encoding="utf-8"))
        cases: List[Dict[str, Any]] = []
        for item in raw:
            case_id = str(item.get("id") or item.get("case_id") or item.get("public_id"))
            steps = item.get("steps") or []
            cases.append(
                {
                    "id": case_id,
                    "title": item.get("title", ""),
                    "priority": item.get("priority", "medium"),
                    "preconditions": item.get("preconditions", ""),
                    "postconditions": item.get("postconditions", ""),
                    "steps": steps,
                }
            )
        return cases

    def _load_tested(self) -> List[str]:
        if not self.tested_cases_path.exists():
            return []
        data = json.loads(self.tested_cases_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [str(k) for k, v in data.items() if v]
        if isinstance(data, list):
            return [str(item) for item in data]
        return []
