"""Tool for inspecting stored screenshots and UI hierarchies."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class ScreenInspectorInput(BaseModel):
    """Supported inputs for screen_inspector tool calls."""

    action: str = Field(description="Action: list_attempts | inspect")
    test_id: str = Field(description="Test identifier to inspect.")
    attempt: int | None = Field(
        default=None,
        description="Attempt number. If omitted, use most recent available attempt.",
    )
    include_hierarchy: bool = Field(
        default=False,
        description="Return entire hierarchy JSON when True (can be large).",
    )


class ScreenInspectorTool(BaseTool):
    """Expose screenshots and hierarchy summaries for AppFlow planning."""

    name: str = "screen_inspector"
    description: str = (
        "Inspect stored Maestro screenshots and hierarchy snapshots to understand the "
        "current UI beyond textual failure excerpts."
    )
    args_schema: type[BaseModel] = ScreenInspectorInput

    artifacts_dir: Path
    max_ui_texts: int = 40
    _screens_dir: Path = PrivateAttr()
    _debug_dir: Path = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._screens_dir = self.artifacts_dir / "screenshots"
        self._debug_dir = self.artifacts_dir / "debug_snapshots"

    def _run(
        self,
        action: str,
        test_id: str,
        attempt: int | None = None,
        include_hierarchy: bool = False,
    ) -> Dict[str, Any]:
        if action == "list_attempts":
            return self._list_attempts(test_id)
        if action == "inspect":
            return self._inspect(test_id, attempt, include_hierarchy)
        raise ValueError(f"Unknown action {action}")

    # Helpers -----------------------------------------------------------------
    def _list_attempts(self, test_id: str) -> Dict[str, Any]:
        attempts = {}
        shots_path = self._screens_dir / test_id
        if shots_path.exists():
            for file in shots_path.glob("attempt-*.png"):
                match = re.search(r"attempt-(\d+)\.png", file.name)
                if not match:
                    continue
                attempts.setdefault(int(match.group(1)), {})["screenshot"] = str(file)

        debug_path = self._debug_dir / test_id
        if debug_path.exists():
            for folder in debug_path.glob("attempt-*"):
                match = re.search(r"attempt-(\d+)", folder.name)
                if not match or not folder.is_dir():
                    continue
                data = attempts.setdefault(int(match.group(1)), {})
                data["debug_snapshot"] = str(folder)
                data["hierarchy"] = str(folder / "hierarchy.json")

        if not attempts:
            return {"test_id": test_id, "attempts": []}

        summarized = [
            {
                "attempt": idx,
                "screenshot": info.get("screenshot"),
                "debug_snapshot": info.get("debug_snapshot"),
            }
            for idx, info in sorted(attempts.items())
        ]
        return {"test_id": test_id, "attempts": summarized}

    def _inspect(
        self,
        test_id: str,
        attempt: int | None,
        include_hierarchy: bool,
    ) -> Dict[str, Any]:
        inventory = self._list_attempts(test_id)["attempts"]
        if not inventory:
            return {"test_id": test_id, "found": False, "error": "no_attempts_recorded"}

        selected = None
        if attempt is not None:
            for entry in inventory:
                if entry["attempt"] == attempt:
                    selected = entry
                    break
        if selected is None:
            selected = inventory[-1]

        response: Dict[str, Any] = {
            "test_id": test_id,
            "attempt": selected["attempt"],
            "found": True,
            "screenshot": selected.get("screenshot"),
            "debug_snapshot": selected.get("debug_snapshot"),
        }

        hierarchy_path = None
        if selected.get("debug_snapshot"):
            candidate = Path(selected["debug_snapshot"]) / "hierarchy.json"
            if candidate.exists():
                hierarchy_path = candidate

        if hierarchy_path and hierarchy_path.exists():
            try:
                hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                hierarchy = None
            if hierarchy:
                texts = self._extract_ui_text_from_hierarchy(hierarchy)
                response["ui_text_candidates"] = texts[: self.max_ui_texts]
                response["hierarchy_path"] = str(hierarchy_path)
                if include_hierarchy:
                    response["hierarchy"] = hierarchy
        return response

    def _extract_ui_text_from_hierarchy(self, hierarchy: Dict[str, Any]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []

        def walk(node: Any) -> None:
            if not isinstance(node, dict):
                return
            attrs = node.get("attributes")
            if isinstance(attrs, dict):
                for key in ("accessibilityText", "text", "label", "value"):
                    raw = attrs.get(key)
                    value = str(raw or "").strip()
                    if not value:
                        continue
                    low = value.lower()
                    if len(value) > 90 or len(low.split()) > 10:
                        continue
                    if low not in seen:
                        seen.add(low)
                        ordered.append(value)
            children = node.get("children")
            if isinstance(children, list):
                for child in children:
                    walk(child)

        walk(hierarchy)
        return ordered
