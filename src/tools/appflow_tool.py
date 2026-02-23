"""Tool for persistent app screen-flow understanding across test runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class AppFlowInput(BaseModel):
    """Supported inputs for app_flow_memory tool calls."""

    action: str = Field(
        description="Action: suggest_context | record_observation | summary"
    )
    test_id: str | None = Field(default=None, description="Test case identifier.")
    scenario_id: str | None = Field(default=None, description="Scenario identifier.")
    title: str | None = Field(default=None, description="Test case title.")
    preconditions: str | None = Field(default=None, description="Case preconditions.")
    steps_text: str | None = Field(default=None, description="Compact steps text.")
    status: str | None = Field(default=None, description="Attempt status.")
    attempt: int | None = Field(default=None, ge=1, description="Attempt number.")
    location_hint: str | None = Field(
        default=None,
        description="Known app location/screen where case starts or fails.",
    )
    failure_cause: str | None = Field(default=None, description="Failure cause string.")
    notes: str | None = Field(default=None, description="Any extra inference notes.")


class AppFlowMemoryTool(BaseTool):
    """Stores and returns app-flow hints so agents can start tests from proper context."""

    name: str = "app_flow_memory"
    description: str = (
        "Persistent memory of app screen flow. Use suggest_context before writing a test "
        "and record_observation after each attempt to learn where flows start/fail."
    )
    args_schema: type[BaseModel] = AppFlowInput

    artifacts_dir: Path
    _knowledge_path: Path = PrivateAttr()
    _state: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self._knowledge_path = self.artifacts_dir / "app_flow_knowledge.json"
        if self._knowledge_path.exists():
            try:
                loaded = json.loads(self._knowledge_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._state = loaded
            except json.JSONDecodeError:
                self._state = {}
        if not self._state:
            self._state = {
                "version": 1,
                "updated_at": "",
                "cases": {},
                "scenario_hints": {},
                "global_hints": [],
            }

    def _run(
        self,
        action: str,
        test_id: str | None = None,
        scenario_id: str | None = None,
        title: str | None = None,
        preconditions: str | None = None,
        steps_text: str | None = None,
        status: str | None = None,
        attempt: int | None = None,
        location_hint: str | None = None,
        failure_cause: str | None = None,
        notes: str | None = None,
    ) -> Dict[str, Any]:
        if action == "suggest_context":
            return self._suggest_context(
                test_id=test_id,
                scenario_id=scenario_id,
                title=title,
                preconditions=preconditions,
                steps_text=steps_text,
            )
        if action == "record_observation":
            return self._record_observation(
                test_id=test_id,
                scenario_id=scenario_id,
                title=title,
                status=status,
                attempt=attempt,
                location_hint=location_hint,
                failure_cause=failure_cause,
                notes=notes,
            )
        if action == "summary":
            return self._summary()
        raise ValueError(f"Unknown action {action}")

    def _suggest_context(
        self,
        test_id: str | None,
        scenario_id: str | None,
        title: str | None,
        preconditions: str | None,
        steps_text: str | None,
    ) -> Dict[str, Any]:
        case_entry = self._case_entry(test_id) if test_id else None
        scenario_hint = (
            self._state.get("scenario_hints", {}).get(scenario_id, {}) if scenario_id else {}
        )

        recommendation = "unknown"
        confidence = "low"
        rationale: List[str] = []

        if case_entry and case_entry.get("preferred_start"):
            recommendation = str(case_entry["preferred_start"])
            confidence = "high"
            rationale.append("Exact test case has historical preferred start screen.")
        elif scenario_hint and scenario_hint.get("preferred_start"):
            recommendation = str(scenario_hint["preferred_start"])
            confidence = "medium"
            rationale.append("Scenario-level hint is available from previous attempts.")

        inferred = self._infer_from_text(title=title, preconditions=preconditions, steps_text=steps_text)
        if recommendation == "unknown" and inferred:
            recommendation = inferred
            confidence = "low"
            rationale.append("Heuristic inference from case text/preconditions.")

        if case_entry and case_entry.get("common_failure_causes"):
            rationale.append(
                f"Common failures: {', '.join(case_entry['common_failure_causes'][:3])}"
            )

        return {
            "test_id": test_id,
            "scenario_id": scenario_id,
            "recommended_start": recommendation,
            "confidence": confidence,
            "rationale": rationale,
            "known_case_observations": len(case_entry.get("observations", [])) if case_entry else 0,
        }

    def _record_observation(
        self,
        test_id: str | None,
        scenario_id: str | None,
        title: str | None,
        status: str | None,
        attempt: int | None,
        location_hint: str | None,
        failure_cause: str | None,
        notes: str | None,
    ) -> Dict[str, Any]:
        if not test_id:
            raise ValueError("record_observation requires test_id")
        now = datetime.now(timezone.utc).isoformat()
        case = self._state.setdefault("cases", {}).setdefault(
            test_id,
            {
                "title": title or "",
                "preferred_start": "",
                "common_failure_causes": [],
                "observations": [],
            },
        )
        if title and not case.get("title"):
            case["title"] = title
        if location_hint:
            case["preferred_start"] = location_hint
        if failure_cause:
            causes: List[str] = case.setdefault("common_failure_causes", [])
            if failure_cause not in causes:
                causes.append(failure_cause)
            case["common_failure_causes"] = causes[-5:]

        case.setdefault("observations", []).append(
            {
                "time": now,
                "status": status or "unknown",
                "attempt": attempt or 1,
                "location_hint": location_hint or "",
                "failure_cause": failure_cause or "",
                "notes": notes or "",
            }
        )
        case["observations"] = case["observations"][-20:]

        if scenario_id:
            scenario = self._state.setdefault("scenario_hints", {}).setdefault(
                scenario_id,
                {"preferred_start": "", "last_seen_case_ids": []},
            )
            if location_hint:
                scenario["preferred_start"] = location_hint
            ids: List[str] = scenario.setdefault("last_seen_case_ids", [])
            if test_id not in ids:
                ids.append(test_id)
            scenario["last_seen_case_ids"] = ids[-15:]

        self._state["updated_at"] = now
        self._write()
        return {
            "ok": True,
            "test_id": test_id,
            "scenario_id": scenario_id,
            "stored_observations": len(case.get("observations", [])),
            "preferred_start": case.get("preferred_start", ""),
        }

    def _summary(self) -> Dict[str, Any]:
        cases = self._state.get("cases", {})
        scenario_hints = self._state.get("scenario_hints", {})
        top_case_hints = []
        for case_id, payload in list(cases.items())[:10]:
            top_case_hints.append(
                {
                    "test_id": case_id,
                    "preferred_start": payload.get("preferred_start", ""),
                    "observations": len(payload.get("observations", [])),
                }
            )
        return {
            "updated_at": self._state.get("updated_at", ""),
            "known_cases": len(cases),
            "known_scenarios": len(scenario_hints),
            "top_case_hints": top_case_hints,
            "knowledge_path": str(self._knowledge_path),
        }

    def _case_entry(self, test_id: str) -> Dict[str, Any] | None:
        return self._state.get("cases", {}).get(test_id)

    def _infer_from_text(
        self,
        title: str | None,
        preconditions: str | None,
        steps_text: str | None,
    ) -> str | None:
        haystack = " ".join(
            part for part in [title or "", preconditions or "", steps_text or ""] if part
        ).lower()
        if not haystack:
            return None
        if "onboarding" in haystack:
            return "onboarding"
        if "login" in haystack or "sign in" in haystack:
            return "auth/login"
        if "profile" in haystack:
            return "profile"
        if "settings" in haystack:
            return "settings"
        return None

    def _write(self) -> None:
        self._knowledge_path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
