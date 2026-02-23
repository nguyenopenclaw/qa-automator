"""Tool for interpreting Qase-exported JSON test cases."""
from __future__ import annotations

import json
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
    max_cases_per_scenario: int = 5
    target_root_suite: str = "Regression"
    _cases: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _tested: List[str] = PrivateAttr(default_factory=list)
    _scenarios: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _scenarios_path: Path = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._cases = self._load_cases()
        self._tested = self._load_tested()
        self._scenarios = self._build_scenarios()
        self._scenarios_path = self.test_cases_path.parent / "scenarios.json"
        self._persist_scenarios()

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
            return {
                "found": True,
                "mode": "single_scenario",
                "scenario": scenario,
                "scenarios_path": str(self._scenarios_path),
            }

        next_scenario = self._get_next_pending_scenario(pending_ids)
        compact_scenarios = [
            {
                "id": item["id"],
                "title": item["title"],
                "cases_count": item["cases_count"],
                "total_steps": item["total_steps"],
                "is_onboarding": item["is_onboarding"],
                "priority": item["priority"],
            }
            for item in self._scenarios
            if item["pending_cases_count"] > 0
        ]

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
                pending_count = len([case_id for case_id in case_ids if case_id not in self._tested])
                title = self._scenario_title_from_chunk(chunk, scenario_index)
                scenarios.append(
                    {
                        "id": f"scenario_{scenario_index:04d}",
                        "title": title,
                        "priority": self._scenario_priority(chunk),
                        "is_onboarding": any(item.get("is_onboarding") for item in chunk),
                        "cases_count": len(chunk),
                        "pending_cases_count": pending_count,
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
            if scenario["pending_cases_count"] > 0:
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
            "cases": cases,
        }
