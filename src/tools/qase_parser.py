"""Tool for interpreting Qase-exported JSON test cases."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class QaseParserInput(BaseModel):
    """Supported inputs for qase_parser tool calls."""

    query: str | None = Field(default=None, description="Optional parser query.")


class QaseTestParserTool(BaseTool):
    """Parses Qase input, groups pending cases into e2e scenarios."""

    name: str = "qase_parser"
    description: str = (
        "Load Qase-exported JSON, group pending test cases into end-to-end scenarios, "
        "persist scenarios to scenarios.json, and return compact scenario payloads."
    )
    args_schema: type[BaseModel] = QaseParserInput

    test_cases_path: Path
    tested_cases_path: Path
    artifacts_dir: Path | None = None
    custom_scenario_path: Path | None = None
    max_cases_per_scenario: int = 5
    target_root_suite: str = "Regression"
    _cases: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _tested: List[str] = PrivateAttr(default_factory=list)
    _scenarios: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _scenarios_path: Path = PrivateAttr()
    _current_scenario_path: Path | None = PrivateAttr(default=None)
    _source_hash: str = PrivateAttr(default="")

    def model_post_init(self, __context: Any) -> None:
        self._scenarios_path = self.test_cases_path.parent / "scenarios.json"
        self._source_hash = self._compute_source_hash()
        if self.artifacts_dir is None:
            self.artifacts_dir = self.test_cases_path.parent / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._current_scenario_path = self.artifacts_dir / "current_scenario.json"
        self._cases = self._load_cases()
        self._tested = self._load_tested()
        custom_scenario = self._load_custom_scenario()
        if custom_scenario:
            self._inject_custom_cases(custom_scenario)
        cached = self._load_cached_scenarios()
        if cached is not None:
            self._scenarios = cached
            if custom_scenario:
                self._upsert_custom_scenario(custom_scenario)
            self._write_current_scenario(None)
            return
        self._scenarios = self._build_scenarios()
        if custom_scenario:
            self._upsert_custom_scenario(custom_scenario)
        self._persist_scenarios()
        self._write_current_scenario(None)

    def _run(self, query: str | None = None) -> Dict[str, Any]:
        """Return scenarios in a context-safe, one-by-one format."""
        pending = [case for case in self._cases if case["id"] not in self._tested]
        pending_ids = {case["id"] for case in pending}

        if query and query.startswith("case_id:"):
            requested_id = query.split(":", 1)[1].strip()
            for case in self._cases:
                if case["id"] == requested_id:
                    return {"found": True, "case": case}
            return {"found": False, "case_id": requested_id}

        if query and query.startswith("scenario_id:"):
            requested_id = query.split(":", 1)[1].strip()
            scenario = self._get_scenario_by_id(requested_id, pending_ids)
            if not scenario:
                return {"found": False, "scenario_id": requested_id}
            self._write_current_scenario(scenario)
            return {
                "found": True,
                "mode": "single_scenario",
                "scenario": scenario,
                "scenarios_path": str(self._scenarios_path),
            }

        next_scenario = self._get_next_pending_scenario(pending_ids)
        compact_scenarios = [
            self._scenario_summary(item, pending_ids)
            for item in self._scenarios
            if self._pending_count(item, pending_ids) > 0
        ]

        self._write_current_scenario(next_scenario)
        return {
            "mode": "single_scenario",
            "total": len(self._cases),
            "pending": len(pending),
            "scenarios_total": len(self._scenarios),
            "scenarios_pending": len(compact_scenarios),
            "next_scenario": next_scenario,
            "scenarios": compact_scenarios,
            "scenarios_path": str(self._scenarios_path),
        }

    # Internal helpers -------------------------------------------------
    def _load_cases(self) -> List[Dict[str, Any]]:
        raw = json.loads(self.test_cases_path.read_text(encoding="utf-8"))
        items = self._extract_case_items(raw)
        cases: List[Dict[str, Any]] = []
        for item in items:
            case_id = str(item.get("id") or item.get("case_id") or item.get("public_id"))
            steps = item.get("steps") or []
            tags = item.get("tags") or []
            tag_strings = []
            for tag in tags:
                if isinstance(tag, str):
                    tag_strings.append(tag.lower())
                elif isinstance(tag, dict) and tag.get("title"):
                    tag_strings.append(str(tag["title"]).lower())
            is_onboarding = "onboarding" in tag_strings or "onboarding" in (item.get("title", "").lower())
            suite_path = str(item.get("_suite_path", "")).strip()
            cases.append(
                {
                    "id": case_id,
                    "title": item.get("title", ""),
                    "priority": item.get("priority", "medium"),
                    "preconditions": item.get("preconditions", ""),
                    "postconditions": item.get("postconditions", ""),
                    "steps": steps,
                    "tags": tag_strings,
                    "is_onboarding": is_onboarding,
                    "suite_path": suite_path,
                }
            )
        return cases

    def _load_tested(self) -> List[str]:
        if not self.tested_cases_path.exists():
            return []
        content = self.tested_cases_path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        data = json.loads(content)
        if isinstance(data, dict):
            return [str(k) for k, v in data.items() if v]
        if isinstance(data, list):
            return [str(item) for item in data]
        return []

    def _extract_case_items(self, raw: Any) -> List[Dict[str, Any]]:
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if isinstance(raw, dict):
            collected: List[Dict[str, Any]] = []

            direct_cases = raw.get("cases")
            if isinstance(direct_cases, list):
                collected.extend(item for item in direct_cases if isinstance(item, dict))

            suites = raw.get("suites")
            if isinstance(suites, list):
                for suite in suites:
                    self._collect_cases_from_suite_tree(suite, [], collected)

            # Fallback: treat this object as a case when it looks like one.
            if not collected and any(key in raw for key in ("id", "title", "steps")):
                collected.append(raw)

            return collected
        return []

    def _load_custom_scenario(self) -> Dict[str, Any] | None:
        if not self.custom_scenario_path:
            return None
        if not self.custom_scenario_path.exists():
            return None
        raw_payload = json.loads(self.custom_scenario_path.read_text(encoding="utf-8"))
        scenario = (
            raw_payload.get("scenario")
            if isinstance(raw_payload, dict) and isinstance(raw_payload.get("scenario"), dict)
            else raw_payload
        )
        if not isinstance(scenario, dict):
            raise ValueError("Custom scenario payload must be a JSON object.")
        scenario_id = str(scenario.get("id", "")).strip()
        if not scenario_id:
            raise ValueError("Custom scenario requires non-empty `id`.")
        cases_raw = scenario.get("cases")
        if not isinstance(cases_raw, list) or not cases_raw:
            raise ValueError("Custom scenario requires non-empty `cases` list.")
        normalized_cases: List[Dict[str, Any]] = []
        for index, case in enumerate(cases_raw, start=1):
            if not isinstance(case, dict):
                continue
            case_id = str(case.get("id") or f"{scenario_id}_case_{index:03d}")
            steps = case.get("steps") if isinstance(case.get("steps"), list) else []
            normalized_cases.append(
                {
                    "id": case_id,
                    "title": str(case.get("title", f"Custom case {index}")),
                    "priority": str(case.get("priority", scenario.get("priority", "high"))),
                    "preconditions": case.get("preconditions", ""),
                    "postconditions": case.get("postconditions", ""),
                    "steps": steps,
                    "tags": case.get("tags") if isinstance(case.get("tags"), list) else [],
                    "is_onboarding": bool(case.get("is_onboarding", scenario.get("is_onboarding", False))),
                    "suite_path": str(case.get("suite_path", "Custom / User Scenario")),
                }
            )
        if not normalized_cases:
            raise ValueError("Custom scenario has no valid case objects.")
        return {
            "id": scenario_id,
            "title": str(scenario.get("title", "Custom user scenario")),
            "priority": str(scenario.get("priority", "high")),
            "is_onboarding": bool(scenario.get("is_onboarding", False)),
            "cases": normalized_cases,
        }

    def _inject_custom_cases(self, scenario: Dict[str, Any]) -> None:
        existing_case_ids = {case["id"] for case in self._cases}
        for case in scenario["cases"]:
            if case["id"] not in existing_case_ids:
                self._cases.append(case)

    def _upsert_custom_scenario(self, scenario: Dict[str, Any]) -> None:
        case_ids = [case["id"] for case in scenario["cases"]]
        custom_summary = {
            "id": scenario["id"],
            "title": scenario["title"],
            "priority": scenario["priority"],
            "is_onboarding": scenario["is_onboarding"],
            "cases_count": len(scenario["cases"]),
            "total_steps": sum(len(case.get("steps", [])) for case in scenario["cases"]),
            "case_ids": case_ids,
        }
        self._scenarios = [item for item in self._scenarios if item.get("id") != scenario["id"]]
        self._scenarios.insert(0, custom_summary)

    def _collect_cases_from_suite_tree(
        self,
        node: Any,
        parents: List[str],
        collected: List[Dict[str, Any]],
    ) -> None:
        if not isinstance(node, dict):
            return
        title = str(node.get("title") or "").strip()
        suite_path_parts = [*parents, title] if title else list(parents)

        cases = node.get("cases")
        if isinstance(cases, list):
            for case in cases:
                if isinstance(case, dict):
                    enriched = dict(case)
                    enriched["_suite_path"] = " / ".join(suite_path_parts)
                    collected.append(enriched)
        suites = node.get("suites")
        if isinstance(suites, list):
            for suite in suites:
                self._collect_cases_from_suite_tree(suite, suite_path_parts, collected)

    def _build_scenarios(self) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for case in self._cases:
            suite_path = case.get("suite_path") or "Ungrouped"
            root_suite = suite_path.split(" / ")[0] if suite_path else "Ungrouped"
            if root_suite.strip().lower() != self.target_root_suite.strip().lower():
                continue
            onboarding_marker = "onboarding" if case.get("is_onboarding") else "regular"
            key = f"{root_suite}::{onboarding_marker}"
            grouped.setdefault(key, []).append(case)

        scenarios: List[Dict[str, Any]] = []
        scenario_index = 1
        for key in sorted(grouped):
            cases = sorted(grouped[key], key=self._case_sort_key)
            for offset in range(0, len(cases), self.max_cases_per_scenario):
                chunk = cases[offset : offset + self.max_cases_per_scenario]
                case_ids = [item["id"] for item in chunk]
                title = self._scenario_title_from_chunk(chunk, scenario_index)
                scenarios.append(
                    {
                        "id": f"scenario_{scenario_index:04d}",
                        "title": title,
                        "priority": self._scenario_priority(chunk),
                        "is_onboarding": any(item.get("is_onboarding") for item in chunk),
                        "cases_count": len(chunk),
                        "total_steps": sum(len(item.get("steps", [])) for item in chunk),
                        "case_ids": case_ids,
                    }
                )
                scenario_index += 1
        return scenarios

    def _case_sort_key(self, case: Dict[str, Any]) -> tuple[int, str]:
        priority = str(case.get("priority", "medium")).lower()
        rank_map = {"high": 0, "medium": 1, "low": 2}
        rank = rank_map.get(priority, 3)
        return (rank, str(case.get("id", "")))

    def _scenario_priority(self, chunk: List[Dict[str, Any]]) -> str:
        priorities = {str(case.get("priority", "medium")).lower() for case in chunk}
        if "high" in priorities:
            return "high"
        if "medium" in priorities:
            return "medium"
        if "low" in priorities:
            return "low"
        return "undefined"

    def _scenario_title_from_chunk(self, chunk: List[Dict[str, Any]], index: int) -> str:
        first = chunk[0] if chunk else {}
        suite_path = str(first.get("suite_path") or "Ungrouped")
        return f"{suite_path} - batch {index}"

    def _persist_scenarios(self) -> None:
        payload = {
            "source_hash": self._source_hash,
            "source_file": str(self.test_cases_path),
            "total_cases": len(self._cases),
            "tested_cases": len(self._tested),
            "scenarios_total": len(self._scenarios),
            "scenarios": self._scenarios,
        }
        self._scenarios_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_scenario_by_id(
        self,
        scenario_id: str,
        pending_ids: set[str],
    ) -> Dict[str, Any] | None:
        scenario = next((item for item in self._scenarios if item["id"] == scenario_id), None)
        if not scenario:
            return None
        return self._expand_scenario(scenario, pending_ids)

    def _get_next_pending_scenario(self, pending_ids: set[str]) -> Dict[str, Any] | None:
        for scenario in self._scenarios:
            if self._pending_count(scenario, pending_ids) > 0:
                return self._expand_scenario(scenario, pending_ids)
        return None

    def _expand_scenario(self, scenario: Dict[str, Any], pending_ids: set[str]) -> Dict[str, Any]:
        ids = set(scenario.get("case_ids", []))
        cases = [
            case
            for case in self._cases
            if case["id"] in ids and case["id"] in pending_ids
        ]
        return {
            **scenario,
            "pending_cases_count": self._pending_count(scenario, pending_ids),
            "cases": cases,
        }

    def _scenario_summary(self, scenario: Dict[str, Any], pending_ids: set[str]) -> Dict[str, Any]:
        return {
            "id": scenario["id"],
            "title": scenario["title"],
            "cases_count": scenario["cases_count"],
            "pending_cases_count": self._pending_count(scenario, pending_ids),
            "total_steps": scenario["total_steps"],
            "is_onboarding": scenario["is_onboarding"],
            "priority": scenario["priority"],
        }

    def _write_current_scenario(self, scenario: Dict[str, Any] | None) -> None:
        if not self._current_scenario_path:
            return
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario,
            "scenarios_path": str(self._scenarios_path),
        }
        self._current_scenario_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _pending_count(self, scenario: Dict[str, Any], pending_ids: set[str]) -> int:
        case_ids = scenario.get("case_ids", [])
        return sum(1 for case_id in case_ids if case_id in pending_ids)

    def _compute_source_hash(self) -> str:
        content = self.test_cases_path.read_bytes()
        digest = hashlib.sha256()
        digest.update(content)
        digest.update(str(self.max_cases_per_scenario).encode("utf-8"))
        digest.update(self.target_root_suite.encode("utf-8"))
        if self.custom_scenario_path and self.custom_scenario_path.exists():
            digest.update(str(self.custom_scenario_path).encode("utf-8"))
            digest.update(self.custom_scenario_path.read_bytes())
        return digest.hexdigest()

    def _load_cached_scenarios(self) -> List[Dict[str, Any]] | None:
        if not self._scenarios_path.exists():
            return None
        try:
            payload = json.loads(self._scenarios_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("source_hash") != self._source_hash:
            return None
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, list):
            return None
        return [item for item in scenarios if isinstance(item, dict)]
