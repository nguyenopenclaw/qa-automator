"""Tool for persistent app screen-flow understanding across test runs."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class AppFlowInput(BaseModel):
    """Supported inputs for app_flow_memory tool calls."""

    action: str = Field(
        description="Action: suggest_context | record_plan | record_observation | summary"
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
    recommended_start: str | None = Field(
        default=None,
        description="Hypothesized start context derived during planning.",
    )
    confidence: str | None = Field(default=None, description="Confidence label for plan.")
    failure_cause: str | None = Field(default=None, description="Failure cause string.")
    notes: str | None = Field(default=None, description="Any extra inference notes.")


class AppFlowMemoryTool(BaseTool):
    """Stores and returns app-flow hints so agents can start tests from proper context."""

    name: str = "app_flow_memory"
    description: str = (
        "Persistent memory of app screen flow. Use suggest_context before writing a test, "
        "record_plan to capture each hypothesis, and record_observation after attempts so the "
        "map improves every run."
    )
    args_schema: type[BaseModel] = AppFlowInput

    artifacts_dir: Path
    _knowledge_path: Path = PrivateAttr()
    _memory_dir: Path = PrivateAttr()
    _checkpoint_dir: Path = PrivateAttr()
    _detail_dir: Path = PrivateAttr()
    _detail_catalog_path: Path = PrivateAttr()
    _state: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _detail_catalog: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _case_limit: int = PrivateAttr(default=300)
    _scenario_limit: int = PrivateAttr(default=200)
    _score_decay: float = PrivateAttr(default=0.92)
    _max_score_entries: int = PrivateAttr(default=8)
    _max_failure_entries: int = PrivateAttr(default=8)
    _detail_file_limit: int = PrivateAttr(default=800)
    _detail_events_limit: int = PrivateAttr(default=80)

    def model_post_init(self, __context: Any) -> None:
        self._memory_dir = self.artifacts_dir / "app_flow_memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._knowledge_path = self._memory_dir / "state.json"
        self._checkpoint_dir = self._memory_dir / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._detail_dir = self._memory_dir / "details"
        self._detail_dir.mkdir(parents=True, exist_ok=True)
        self._detail_catalog_path = self._memory_dir / "detail_catalog.json"
        legacy_path = self.artifacts_dir / "app_flow_knowledge.json"
        if legacy_path.exists() and not self._knowledge_path.exists():
            legacy_path.replace(self._knowledge_path)
        if self._knowledge_path.exists():
            try:
                loaded = json.loads(self._knowledge_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._state = loaded
            except json.JSONDecodeError:
                self._state = {}
        if not self._state:
            self._state = {
                "version": 2,
                "updated_at": "",
                "cases": {},
                "scenario_hints": {},
                "global_hints": [],
            }
        self._ensure_schema()
        self._load_detail_catalog()

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
        recommended_start: str | None = None,
        confidence: str | None = None,
    ) -> Dict[str, Any]:
        if action == "suggest_context":
            return self._suggest_context(
                test_id=test_id,
                scenario_id=scenario_id,
                title=title,
                preconditions=preconditions,
                steps_text=steps_text,
            )
        if action == "record_plan":
            return self._record_plan(
                test_id=test_id,
                scenario_id=scenario_id,
                title=title,
                recommended_start=recommended_start or location_hint,
                confidence=confidence,
                notes=notes,
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
        elif case_entry and case_entry.get("start_score_map"):
            top_starts = self._sorted_map_keys(case_entry["start_score_map"])
            if top_starts:
                recommendation = top_starts[0]
                confidence = "medium"
                rationale.append("Case-level weighted start scores from previous runs.")
        elif scenario_hint and scenario_hint.get("preferred_start"):
            recommendation = str(scenario_hint["preferred_start"])
            confidence = "medium"
            rationale.append("Scenario-level hint is available from previous attempts.")
        elif scenario_hint and scenario_hint.get("start_score_map"):
            top_starts = self._sorted_map_keys(scenario_hint["start_score_map"])
            if top_starts:
                recommendation = top_starts[0]
                confidence = "low"
                rationale.append("Scenario-level weighted start scores suggest this entry.")

        inferred = self._infer_from_text(title=title, preconditions=preconditions, steps_text=steps_text)
        if recommendation == "unknown" and inferred:
            recommendation = inferred
            confidence = "low"
            rationale.append("Heuristic inference from case text/preconditions.")

        common_failures = self._common_failure_causes(case_entry or {})
        if common_failures:
            rationale.append(
                f"Common failures: {', '.join(common_failures[:3])}"
            )
        detail_hints = self._collect_detail_hints(test_id=test_id, scenario_id=scenario_id)
        if detail_hints.get("best_start") and recommendation == "unknown":
            recommendation = str(detail_hints["best_start"])
            confidence = "low"
            rationale.append("Detailed segment memory suggests this start context.")
        if detail_hints.get("top_failures"):
            rationale.append(f"Detailed failures: {', '.join(detail_hints['top_failures'][:3])}")

        return {
            "test_id": test_id,
            "scenario_id": scenario_id,
            "recommended_start": recommendation,
            "confidence": confidence,
            "rationale": rationale,
            "known_case_observations": len(case_entry.get("observations", [])) if case_entry else 0,
        }

    def _record_plan(
        self,
        test_id: str | None,
        scenario_id: str | None,
        title: str | None,
        recommended_start: str | None,
        confidence: str | None,
        notes: str | None,
    ) -> Dict[str, Any]:
        if not test_id:
            raise ValueError("record_plan requires test_id")
        now = datetime.now(timezone.utc).isoformat()
        case = self._state.setdefault("cases", {}).setdefault(
            test_id,
            {
                "title": title or "",
                "preferred_start": "",
                "common_failure_causes": [],
                "observations": [],
                "plans": [],
                "start_score_map": {},
                "failure_cause_count": {},
                "status_count": {},
                "last_seen_at": "",
            },
        )
        self._normalize_case_entry(case)
        if "plans" not in case:
            case["plans"] = []
        plan_entry = {
            "time": now,
            "scenario_id": scenario_id or "",
            "recommended_start": recommended_start or "unknown",
            "confidence": (confidence or "low").lower(),
            "notes": notes or "",
        }
        case["plans"].append(plan_entry)
        case["plans"] = case["plans"][-20:]
        if title and not case.get("title"):
            case["title"] = title
        if recommended_start and not case.get("preferred_start"):
            case["preferred_start"] = recommended_start
        case["last_seen_at"] = now
        if recommended_start:
            self._decay_map(case["start_score_map"])
            self._bump_map(case["start_score_map"], recommended_start, 0.75)
            case["preferred_start"] = self._best_map_key(case["start_score_map"]) or case["preferred_start"]

        if scenario_id:
            scenario = self._state.setdefault("scenario_hints", {}).setdefault(
                scenario_id,
                {
                    "preferred_start": "",
                    "last_seen_case_ids": [],
                    "start_score_map": {},
                    "last_seen_at": "",
                },
            )
            self._normalize_scenario_entry(scenario)
            if recommended_start and not scenario.get("preferred_start"):
                scenario["preferred_start"] = recommended_start
            if recommended_start:
                self._decay_map(scenario["start_score_map"])
                self._bump_map(scenario["start_score_map"], recommended_start, 0.5)
                scenario["preferred_start"] = (
                    self._best_map_key(scenario["start_score_map"]) or scenario["preferred_start"]
                )
            ids: List[str] = scenario.setdefault("last_seen_case_ids", [])
            if test_id not in ids:
                ids.append(test_id)
            scenario["last_seen_case_ids"] = ids[-15:]
            scenario["last_seen_at"] = now
        self._append_detail_events(
            test_id=test_id,
            scenario_id=scenario_id,
            event_time=now,
            event={
                "kind": "plan",
                "title": title or "",
                "recommended_start": recommended_start or "unknown",
                "confidence": (confidence or "low").lower(),
                "notes": notes or "",
            },
            start_context=recommended_start,
            failure_cause=None,
            status=None,
            attempt=None,
        )

        self._state["updated_at"] = now
        self._write()
        return {
            "ok": True,
            "test_id": test_id,
            "scenario_id": scenario_id,
            "plans_tracked": len(case.get("plans", [])),
            "recommended_start": plan_entry["recommended_start"],
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
                "plans": [],
                "start_score_map": {},
                "failure_cause_count": {},
                "status_count": {},
                "last_seen_at": "",
            },
        )
        self._normalize_case_entry(case)
        if title and not case.get("title"):
            case["title"] = title
        if location_hint:
            case["preferred_start"] = location_hint
            self._decay_map(case["start_score_map"])
            status_lower = (status or "unknown").strip().lower()
            # Passed runs should influence start recommendation stronger than failed runs.
            weight = 2.5 if status_lower == "passed" else 1.0
            self._bump_map(case["start_score_map"], location_hint, weight)
            case["preferred_start"] = self._best_map_key(case["start_score_map"]) or case["preferred_start"]
        status_lower = (status or "unknown").strip().lower()
        self._bump_map(case["status_count"], status_lower, 1.0)
        if failure_cause:
            causes: List[str] = case.setdefault("common_failure_causes", [])
            if failure_cause not in causes:
                causes.append(failure_cause)
            case["common_failure_causes"] = causes[-5:]
            self._bump_map(case["failure_cause_count"], failure_cause.strip(), 1.0)

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
        case["last_seen_at"] = now
        case["common_failure_causes"] = self._common_failure_causes(case)[:5]

        if scenario_id:
            scenario = self._state.setdefault("scenario_hints", {}).setdefault(
                scenario_id,
                {
                    "preferred_start": "",
                    "last_seen_case_ids": [],
                    "start_score_map": {},
                    "last_seen_at": "",
                },
            )
            self._normalize_scenario_entry(scenario)
            if location_hint:
                self._decay_map(scenario["start_score_map"])
                self._bump_map(
                    scenario["start_score_map"],
                    location_hint,
                    1.5 if status_lower == "passed" else 0.75,
                )
                scenario["preferred_start"] = (
                    self._best_map_key(scenario["start_score_map"]) or location_hint
                )
            ids: List[str] = scenario.setdefault("last_seen_case_ids", [])
            if test_id not in ids:
                ids.append(test_id)
            scenario["last_seen_case_ids"] = ids[-15:]
            scenario["last_seen_at"] = now
        self._append_detail_events(
            test_id=test_id,
            scenario_id=scenario_id,
            event_time=now,
            event={
                "kind": "observation",
                "title": title or "",
                "status": status_lower,
                "attempt": attempt or 1,
                "location_hint": location_hint or "",
                "failure_cause": failure_cause or "",
                "notes": notes or "",
            },
            start_context=location_hint,
            failure_cause=failure_cause,
            status=status_lower,
            attempt=attempt,
        )

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
                    "plans": len(payload.get("plans", [])),
                }
            )
        checkpoint_files = sorted(self._checkpoint_dir.glob("*.json"))
        return {
            "updated_at": self._state.get("updated_at", ""),
            "known_cases": len(cases),
            "known_scenarios": len(scenario_hints),
            "top_case_hints": top_case_hints,
            "knowledge_path": str(self._knowledge_path),
            "checkpoint_count": len(checkpoint_files),
            "checkpoint_dir": str(self._checkpoint_dir),
            "detail_catalog_path": str(self._detail_catalog_path),
            "detail_segments": len(self._detail_catalog.get("segments", {})),
            "detail_dir": str(self._detail_dir),
            "memory_limits": {
                "max_cases": self._case_limit,
                "max_scenarios": self._scenario_limit,
                "max_checkpoints": 20,
                "max_detail_segments": self._detail_file_limit,
                "max_events_per_segment": self._detail_events_limit,
            },
        }

    def _case_entry(self, test_id: str) -> Dict[str, Any] | None:
        return self._state.get("cases", {}).get(test_id)

    def _ensure_schema(self) -> None:
        self._state["version"] = max(int(self._state.get("version", 1)), 2)
        self._state.setdefault("updated_at", "")
        self._state.setdefault("cases", {})
        self._state.setdefault("scenario_hints", {})
        self._state.setdefault("global_hints", [])
        for payload in self._state["cases"].values():
            if isinstance(payload, dict):
                self._normalize_case_entry(payload)
        for payload in self._state["scenario_hints"].values():
            if isinstance(payload, dict):
                self._normalize_scenario_entry(payload)

    def _normalize_case_entry(self, case: Dict[str, Any]) -> None:
        case.setdefault("title", "")
        case.setdefault("preferred_start", "")
        case.setdefault("common_failure_causes", [])
        case.setdefault("observations", [])
        case.setdefault("plans", [])
        case["start_score_map"] = self._sanitize_numeric_map(case.get("start_score_map"), as_float=True)
        case["failure_cause_count"] = self._sanitize_numeric_map(
            case.get("failure_cause_count"), as_float=True
        )
        case["status_count"] = self._sanitize_numeric_map(case.get("status_count"), as_float=True)
        case.setdefault("last_seen_at", "")
        case["start_score_map"] = self._pruned_numeric_map(case["start_score_map"], self._max_score_entries)
        case["failure_cause_count"] = self._pruned_numeric_map(
            case["failure_cause_count"], self._max_failure_entries
        )

    def _normalize_scenario_entry(self, scenario: Dict[str, Any]) -> None:
        scenario.setdefault("preferred_start", "")
        scenario.setdefault("last_seen_case_ids", [])
        scenario["start_score_map"] = self._sanitize_numeric_map(scenario.get("start_score_map"), as_float=True)
        scenario.setdefault("last_seen_at", "")
        ids = [str(item) for item in scenario.get("last_seen_case_ids", []) if str(item).strip()]
        scenario["last_seen_case_ids"] = ids[-15:]
        scenario["start_score_map"] = self._pruned_numeric_map(
            scenario["start_score_map"], self._max_score_entries
        )

    def _sanitize_numeric_map(self, raw: Any, as_float: bool) -> Dict[str, float]:
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, float] = {}
        for key, value in raw.items():
            norm_key = str(key).strip()
            if not norm_key:
                continue
            try:
                norm_val = float(value)
            except (TypeError, ValueError):
                continue
            if norm_val <= 0:
                continue
            out[norm_key] = float(norm_val) if as_float else int(norm_val)
        return out

    def _sorted_map_keys(self, mapping: Dict[str, float]) -> List[str]:
        return [key for key, _ in sorted(mapping.items(), key=lambda item: item[1], reverse=True)]

    def _best_map_key(self, mapping: Dict[str, float]) -> str | None:
        if not mapping:
            return None
        return self._sorted_map_keys(mapping)[0]

    def _pruned_numeric_map(self, mapping: Dict[str, float], max_entries: int) -> Dict[str, float]:
        if len(mapping) <= max_entries:
            return mapping
        top_keys = set(self._sorted_map_keys(mapping)[:max_entries])
        return {key: mapping[key] for key in mapping if key in top_keys}

    def _decay_map(self, mapping: Dict[str, float]) -> None:
        for key in list(mapping.keys()):
            decayed = float(mapping[key]) * self._score_decay
            if decayed < 0.05:
                mapping.pop(key, None)
            else:
                mapping[key] = round(decayed, 6)

    def _bump_map(self, mapping: Dict[str, float], key: str, amount: float) -> None:
        norm_key = str(key).strip()
        if not norm_key:
            return
        mapping[norm_key] = float(mapping.get(norm_key, 0.0)) + float(amount)
        trimmed = self._pruned_numeric_map(mapping, max(self._max_score_entries, self._max_failure_entries))
        mapping.clear()
        mapping.update(trimmed)

    def _common_failure_causes(self, case: Dict[str, Any]) -> List[str]:
        count_map = case.get("failure_cause_count")
        if isinstance(count_map, dict) and count_map:
            return self._sorted_map_keys(count_map)
        legacy = case.get("common_failure_causes", [])
        if not isinstance(legacy, list):
            return []
        return [str(item) for item in legacy if str(item).strip()]

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

    def _load_detail_catalog(self) -> None:
        self._detail_catalog = {
            "version": 1,
            "updated_at": "",
            "segments": {},
        }
        if not self._detail_catalog_path.exists():
            return
        try:
            raw = json.loads(self._detail_catalog_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(raw, dict):
            return
        segments = raw.get("segments")
        if not isinstance(segments, dict):
            segments = {}
        self._detail_catalog = {
            "version": 1,
            "updated_at": str(raw.get("updated_at", "")),
            "segments": {
                str(key): value
                for key, value in segments.items()
                if isinstance(value, dict)
            },
        }

    def _collect_detail_hints(self, test_id: str | None, scenario_id: str | None) -> Dict[str, Any]:
        aggregated_starts: Dict[str, float] = {}
        aggregated_failures: Dict[str, float] = {}
        segment_ids: List[str] = []
        if test_id:
            segment_ids.append(self._segment_id("case", test_id))
        if scenario_id:
            segment_ids.append(self._segment_id("scenario", scenario_id))
        for seg_id in segment_ids:
            payload = self._read_segment(seg_id)
            stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
            start_map = stats.get("start_score_map", {}) if isinstance(stats, dict) else {}
            fail_map = stats.get("failure_cause_count", {}) if isinstance(stats, dict) else {}
            if isinstance(start_map, dict):
                for key, value in start_map.items():
                    try:
                        aggregated_starts[str(key)] = aggregated_starts.get(str(key), 0.0) + float(value)
                    except (TypeError, ValueError):
                        continue
            if isinstance(fail_map, dict):
                for key, value in fail_map.items():
                    try:
                        aggregated_failures[str(key)] = aggregated_failures.get(str(key), 0.0) + float(value)
                    except (TypeError, ValueError):
                        continue
        return {
            "best_start": self._best_map_key(aggregated_starts),
            "top_failures": self._sorted_map_keys(aggregated_failures)[:5],
        }

    def _append_detail_events(
        self,
        test_id: str,
        scenario_id: str | None,
        event_time: str,
        event: Dict[str, Any],
        start_context: str | None,
        failure_cause: str | None,
        status: str | None,
        attempt: int | None,
    ) -> None:
        targets = [("case", test_id)]
        if scenario_id:
            targets.append(("scenario", scenario_id))
        if start_context:
            targets.append(("start", start_context))
        if failure_cause:
            targets.append(("failure", failure_cause))
        for segment_type, segment_key in targets:
            self._append_segment_event(
                segment_type=segment_type,
                segment_key=segment_key,
                event_time=event_time,
                event=event,
                start_context=start_context,
                failure_cause=failure_cause,
                status=status,
                attempt=attempt,
                test_id=test_id,
                scenario_id=scenario_id,
            )
        self._write_detail_catalog()

    def _append_segment_event(
        self,
        segment_type: str,
        segment_key: str,
        event_time: str,
        event: Dict[str, Any],
        start_context: str | None,
        failure_cause: str | None,
        status: str | None,
        attempt: int | None,
        test_id: str,
        scenario_id: str | None,
    ) -> None:
        seg_id = self._segment_id(segment_type, segment_key)
        payload = self._read_segment(seg_id)
        if not payload:
            payload = {
                "segment_id": seg_id,
                "segment_type": segment_type,
                "segment_key": str(segment_key),
                "updated_at": "",
                "events": [],
                "stats": {
                    "start_score_map": {},
                    "failure_cause_count": {},
                    "status_count": {},
                    "attempt_max": 0,
                },
            }
        events = payload.setdefault("events", [])
        if not isinstance(events, list):
            events = []
            payload["events"] = events
        event_record = dict(event)
        event_record["time"] = event_time
        event_record["test_id"] = test_id
        event_record["scenario_id"] = scenario_id or ""
        events.append(event_record)
        payload["events"] = events[-self._detail_events_limit :]
        payload["updated_at"] = event_time

        stats = payload.setdefault("stats", {})
        if not isinstance(stats, dict):
            stats = {}
            payload["stats"] = stats
        start_map = self._sanitize_numeric_map(stats.get("start_score_map"), as_float=True)
        failure_map = self._sanitize_numeric_map(stats.get("failure_cause_count"), as_float=True)
        status_map = self._sanitize_numeric_map(stats.get("status_count"), as_float=True)
        if start_context:
            self._decay_map(start_map)
            boost = 2.0 if (status or "").lower() == "passed" else 0.8
            self._bump_map(start_map, start_context, boost)
        if failure_cause:
            self._bump_map(failure_map, failure_cause, 1.0)
        if status:
            self._bump_map(status_map, status, 1.0)
        stats["start_score_map"] = self._pruned_numeric_map(start_map, self._max_score_entries)
        stats["failure_cause_count"] = self._pruned_numeric_map(failure_map, self._max_failure_entries)
        stats["status_count"] = self._pruned_numeric_map(status_map, self._max_failure_entries)
        current_attempt_max = int(stats.get("attempt_max", 0) or 0)
        if attempt and attempt > current_attempt_max:
            stats["attempt_max"] = int(attempt)

        self._write_segment(seg_id, payload)
        self._touch_catalog(
            segment_id=seg_id,
            segment_type=segment_type,
            segment_key=segment_key,
            event_time=event_time,
            entries=len(payload["events"]),
        )

    def _segment_id(self, segment_type: str, segment_key: str) -> str:
        key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(segment_key).strip()).strip("._")
        if not key:
            key = "unknown"
        return f"{segment_type}__{key[:96]}"

    def _segment_path(self, segment_id: str) -> Path:
        return self._detail_dir / f"{segment_id}.json"

    def _read_segment(self, segment_id: str) -> Dict[str, Any]:
        path = self._segment_path(segment_id)
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return raw if isinstance(raw, dict) else {}

    def _write_segment(self, segment_id: str, payload: Dict[str, Any]) -> None:
        path = self._segment_path(segment_id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _touch_catalog(
        self,
        segment_id: str,
        segment_type: str,
        segment_key: str,
        event_time: str,
        entries: int,
    ) -> None:
        segments = self._detail_catalog.setdefault("segments", {})
        if not isinstance(segments, dict):
            segments = {}
            self._detail_catalog["segments"] = segments
        segments[segment_id] = {
            "segment_id": segment_id,
            "segment_type": segment_type,
            "segment_key": str(segment_key),
            "path": str(self._segment_path(segment_id)),
            "entries": int(entries),
            "last_seen_at": event_time,
        }
        self._detail_catalog["updated_at"] = event_time

    def _write_detail_catalog(self) -> None:
        self._prune_detail_catalog()
        payload = json.dumps(self._detail_catalog, ensure_ascii=False, indent=2)
        self._detail_catalog_path.write_text(payload, encoding="utf-8")

    def _prune_detail_catalog(self) -> None:
        segments = self._detail_catalog.get("segments", {})
        if not isinstance(segments, dict):
            self._detail_catalog["segments"] = {}
            return
        if len(segments) <= self._detail_file_limit:
            return
        ranked = sorted(
            segments.items(),
            key=lambda item: str(item[1].get("last_seen_at", "")) if isinstance(item[1], dict) else "",
            reverse=True,
        )
        keep = dict(ranked[: self._detail_file_limit])
        stale_ids = [seg_id for seg_id, _ in ranked[self._detail_file_limit :]]
        self._detail_catalog["segments"] = keep
        for seg_id in stale_ids:
            try:
                self._segment_path(seg_id).unlink()
            except OSError:
                pass

    def _write(self) -> None:
        self._prune_state()
        payload = json.dumps(self._state, ensure_ascii=False, indent=2)
        self._knowledge_path.write_text(payload, encoding="utf-8")
        self._write_checkpoint(payload)
        self._write_detail_catalog()

    def _prune_state(self) -> None:
        self._ensure_schema()
        cases = self._state.get("cases", {})
        scenario_hints = self._state.get("scenario_hints", {})
        if isinstance(cases, dict) and len(cases) > self._case_limit:
            ranked = sorted(
                cases.items(),
                key=lambda item: self._entry_rank(item[1]),
                reverse=True,
            )
            self._state["cases"] = dict(ranked[: self._case_limit])
        if isinstance(scenario_hints, dict) and len(scenario_hints) > self._scenario_limit:
            ranked = sorted(
                scenario_hints.items(),
                key=lambda item: self._entry_rank(item[1]),
                reverse=True,
            )
            self._state["scenario_hints"] = dict(ranked[: self._scenario_limit])

    def _entry_rank(self, payload: Any) -> tuple[str, int]:
        if not isinstance(payload, dict):
            return ("", 0)
        last_seen = str(payload.get("last_seen_at", ""))
        activity = 0
        activity += len(payload.get("observations", [])) if isinstance(payload.get("observations"), list) else 0
        activity += len(payload.get("plans", [])) if isinstance(payload.get("plans"), list) else 0
        activity += len(payload.get("start_score_map", {})) if isinstance(payload.get("start_score_map"), dict) else 0
        return (last_seen, activity)

    def _write_checkpoint(self, payload: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        checkpoint_path = self._checkpoint_dir / f"state-{timestamp}.json"
        checkpoint_path.write_text(payload, encoding="utf-8")
        self._prune_checkpoints()

    def _prune_checkpoints(self, max_count: int = 20) -> None:
        checkpoints = sorted(self._checkpoint_dir.glob("state-*.json"))
        if len(checkpoints) <= max_count:
            return
        for stale in checkpoints[:-max_count]:
            try:
                stale.unlink()
            except OSError:
                pass
