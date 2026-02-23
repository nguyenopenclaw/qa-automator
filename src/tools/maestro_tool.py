"""Tool that proxies Maestro CLI interactions."""
from __future__ import annotations

import json
import os
import re
import hashlib
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr


class MaestroToolInput(BaseModel):
    """Supported inputs for maestro_cli tool calls."""

    payload: str = Field(
        default="{}",
        description=(
            "JSON payload with fields: test_case, attempt, screenshot, is_onboarding, "
            "scenario_id, optional flow_scope ('test_case'|'scenario'), optional "
            "flow_clear_state, optional flow_yaml."
        ),
    )


class MaestroAutomationTool(BaseTool):
    """Generates Maestro flows and executes them per case or whole scenario."""

    name: str = "maestro_cli"
    description: str = (
        "Generate temporary Maestro flows for the provided payload, "
        "execute them against the supplied app binary, and capture screenshots/logs."
    )
    args_schema: type[BaseModel] = MaestroToolInput

    app_path: Path
    artifacts_dir: Path
    generated_flows_dir: Path = Path("samples/automated")
    maestro_bin: str = "maestro"
    device: str | None = None
    app_id: str = "default"
    skip_onboarding_deeplink: str | None = None
    app_install_tool: str = "maestro"
    ios_simulator_target: str = "booted"
    install_app_before_test: bool = True
    install_app_once: bool = True
    reinstall_app_per_scenario: bool = True
    flow_clear_state_default: bool = True
    command_timeout_seconds: int = 120
    screenshot_max_side_px: int = 1440
    screenshot_jpeg_quality: int = 75
    failure_excerpt_max_chars: int = 4000
    _app_install_done: bool = PrivateAttr(default=False)
    _last_scenario_id: str | None = PrivateAttr(default=None)
    _note_dir: Path = PrivateAttr(default=None)
    _debug_snapshots_dir: Path = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._note_dir = self.artifacts_dir / "test_screenshots"
        self._note_dir.mkdir(parents=True, exist_ok=True)
        self._debug_snapshots_dir = self.artifacts_dir / "debug_snapshots"
        self._debug_snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _run(
        self,
        payload: str = "{}",
    ) -> Dict[str, Any]:
        try:
            resolved_payload: Dict[str, Any] = json.loads(payload) if payload else {}
        except json.JSONDecodeError as exc:
            # Agent occasionally emits malformed JSON payload strings.
            # Return a structured failure with a log path instead of crashing the tool.
            log_dir = self.artifacts_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_file = log_dir / f"payload-parse-error-{ts}.log"
            log_file.write_text(
                (
                    "Failed to parse maestro_cli payload as JSON.\n"
                    f"error: {exc}\n"
                    f"payload: {payload}\n"
                ),
                encoding="utf-8",
            )
            return {
                "test_id": "unknown",
                "status": "failed",
                "attempt": 1,
                "error": f"invalid_payload_json: {exc}",
                "artifacts": [str(log_file)],
                "failure_context": {
                    "cause": "invalid_payload_json",
                    "recommendation": "Send strict JSON payload with fields test_case and attempt.",
                    "log_excerpt": log_file.read_text(encoding="utf-8"),
                },
            }
        test_case = resolved_payload.get("test_case", {})
        attempt = int(resolved_payload.get("attempt", 1))
        screenshot = bool(resolved_payload.get("screenshot", False))
        is_onboarding = resolved_payload.get("is_onboarding")
        scenario_id = resolved_payload.get("scenario_id")
        flow_scope = str(resolved_payload.get("flow_scope", "test_case") or "test_case").strip().lower()
        if flow_scope not in {"test_case", "scenario"}:
            flow_scope = "test_case"
        flow_yaml = resolved_payload.get("flow_yaml")
        if flow_yaml is not None:
            flow_yaml = str(flow_yaml)
        flow_clear_state = resolved_payload.get("flow_clear_state")
        if isinstance(flow_clear_state, str):
            flow_clear_state = flow_clear_state.strip().lower() in {"1", "true", "yes", "on"}
        elif not isinstance(flow_clear_state, bool):
            flow_clear_state = None
        return self.run_test_case(
            test_case,
            attempt,
            screenshot,
            is_onboarding,
            scenario_id,
            flow_scope,
            flow_clear_state,
            flow_yaml,
        )

    # Core execution ---------------------------------------------------
    def run_test_case(
        self,
        test_case: Dict[str, Any],
        attempt: int,
        request_screenshot: bool = False,
        is_onboarding: bool | None = None,
        scenario_id: str | None = None,
        flow_scope: str = "test_case",
        flow_clear_state: bool | None = None,
        flow_yaml: str | None = None,
    ) -> Dict[str, Any]:
        test_id = str(test_case.get("id", f"anon-{attempt}"))
        execution_id = scenario_id if flow_scope == "scenario" and scenario_id else test_id
        install_failure = self._ensure_app_installed(execution_id, scenario_id)
        if install_failure:
            return install_failure
        if not is_onboarding:
            self._skip_onboarding_if_possible(execution_id)
        self._note_screenshots_dir().mkdir(parents=True, exist_ok=True)
        flow_path = self._write_flow(
            test_case=test_case,
            attempt=attempt,
            scenario_id=scenario_id,
            flow_scope=flow_scope,
            flow_clear_state=flow_clear_state,
            flow_yaml=flow_yaml,
        )
        if not self._flow_contains_assertions(flow_path):
            log_path = self.artifacts_dir / "logs"
            log_path.mkdir(parents=True, exist_ok=True)
            log_file = log_path / f"{execution_id}-attempt-{attempt}.log"
            log_file.write_text(
                (
                    "Generated Maestro flow has no assertVisible/assertNotVisible commands.\n"
                    "Flow is debug-only and cannot be marked as passed.\n"
                    f"flow_path={flow_path}\n"
                ),
                encoding="utf-8",
            )
            return {
                "test_id": execution_id,
                "status": "failed",
                "attempt": attempt,
                "artifacts": [str(log_file), str(flow_path)],
                "error": "missing_assertions",
                "failure_context": {
                    "cause": "missing_assertions",
                    "recommendation": (
                        "Add explicit assertVisible/assertNotVisible checks that validate "
                        "expected results from the testcase, then retry."
                    ),
                    "log_excerpt": log_file.read_text(encoding="utf-8"),
                    "log_path": str(log_file),
                },
            }
        cmd = [self.maestro_bin]
        if self.device:
            cmd.extend(["--device", self.device])
        cmd.extend(["test", str(flow_path)])

        log_path = self.artifacts_dir / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"{execution_id}-attempt-{attempt}.log"

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.command_timeout_seconds,
            )
            log_file.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        except FileNotFoundError:
            log_file.write_text(
                f"maestro binary not found: {self.maestro_bin}",
                encoding="utf-8",
            )
            return {
                "test_id": execution_id,
                "status": "failed",
                "attempt": attempt,
                "artifacts": [str(log_file)],
                "error": "maestro_binary_not_found",
                "failure_context": self._build_failure_context(
                    stdout="",
                    stderr=f"maestro binary not found: {self.maestro_bin}",
                    log_file=log_file,
                ),
            }
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            log_file.write_text(
                f"maestro command timed out after {self.command_timeout_seconds}s\n{stdout}\n{stderr}",
                encoding="utf-8",
            )
            return {
                "test_id": execution_id,
                "status": "failed",
                "attempt": attempt,
                "artifacts": [str(log_file)],
                "error": "maestro_timeout",
                "failure_context": self._build_failure_context(
                    stdout=stdout,
                    stderr=stderr or f"maestro command timed out after {self.command_timeout_seconds}s",
                    log_file=log_file,
                ),
            }

        artifacts: List[str] = [str(log_file)]
        if request_screenshot:
            shot = self._capture_screenshot(execution_id, attempt)
            if shot:
                artifacts.append(shot)

        status = "passed" if result.returncode == 0 else "failed"
        response: Dict[str, Any] = {
            "test_id": execution_id,
            "status": status,
            "attempt": attempt,
            "artifacts": artifacts,
        }
        if status == "failed" and not request_screenshot:
            shot = self._capture_screenshot(execution_id, attempt)
            if shot:
                artifacts.append(shot)
        if status == "failed":
            failure_context = self._build_failure_context(
                stdout=result.stdout,
                stderr=result.stderr,
                log_file=log_file,
            )
            debug_dir = self._extract_maestro_debug_dir(
                "\n".join(
                    [
                        result.stdout or "",
                        result.stderr or "",
                        log_file.read_text(encoding="utf-8"),
                    ]
                )
            )
            snapshot_dir: str | None = None
            if debug_dir:
                debug_context, snapshot_dir = self._collect_debug_context(debug_dir, test_id, attempt)
                target = snapshot_dir or debug_dir
                failure_context["debug_artifacts_dir"] = target
                if target not in artifacts:
                    artifacts.append(target)
                if debug_context:
                    failure_context["debug_context"] = debug_context
            navigation_context = self._build_navigation_context(
                flow_path=flow_path,
                failed_step_index=failure_context.get("failed_step_index"),
                last_successful_step_index=failure_context.get("last_successful_step_index"),
                debug_context=failure_context.get("debug_context"),
            )
            failure_context["navigation_context"] = navigation_context
            response["navigation_context"] = navigation_context
            response["failure_context"] = failure_context
            response["error"] = failure_context.get("cause")
        else:
            response["navigation_context"] = self._build_navigation_context(
                flow_path=flow_path,
                failed_step_index=None,
                last_successful_step_index=None,
                debug_context=None,
            )

        return response

    # Helpers ----------------------------------------------------------
    def _build_navigation_context(
        self,
        flow_path: Path,
        failed_step_index: int | None,
        last_successful_step_index: int | None,
        debug_context: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        commands = self._parse_flow_commands(flow_path)
        cursor = (
            int(failed_step_index)
            if isinstance(failed_step_index, int)
            else int(last_successful_step_index)
            if isinstance(last_successful_step_index, int)
            else len(commands)
        )
        cursor = max(1, min(cursor, len(commands) or 1))

        screen_asserts = [
            item
            for item in commands
            if item.get("command") in {"assertVisible", "assertNotVisible"}
            and str(item.get("value") or "").strip()
            and not self._is_placeholder_assertion(str(item.get("value") or ""))
        ]
        screen_chain: List[str] = []
        seen_screens: set[str] = set()
        for item in screen_asserts:
            value = str(item.get("value") or "").strip()
            low = value.lower()
            if low in seen_screens:
                continue
            seen_screens.add(low)
            screen_chain.append(value)

        before_cursor = [item for item in screen_asserts if int(item.get("index", 0) or 0) <= cursor]
        after_cursor = [item for item in screen_asserts if int(item.get("index", 0) or 0) > cursor]
        current_screen = str(before_cursor[-1].get("value")) if before_cursor else ""
        from_screen = str(before_cursor[-2].get("value")) if len(before_cursor) >= 2 else ""
        next_screen = str(after_cursor[0].get("value")) if after_cursor else ""

        action_hint = ""
        for item in reversed(commands):
            idx = int(item.get("index", 0) or 0)
            if idx > cursor:
                continue
            if item.get("command") in {"tapOn", "scrollUntilVisible", "runFlow"}:
                value = str(item.get("value") or "").strip()
                action_hint = f"{item['command']}:{value}" if value else str(item["command"])
                break

        elements: List[str] = []
        if isinstance(debug_context, dict):
            raw = debug_context.get("ui_text_candidates")
            if isinstance(raw, list):
                elements.extend(str(item or "").strip() for item in raw)
            failed_selector = str(debug_context.get("failed_selector") or "").strip()
            if failed_selector:
                elements.append(f"failed_selector:{failed_selector}")

        dedup_elements: List[str] = []
        seen_elements: set[str] = set()
        for item in elements:
            text = str(item or "").strip()
            if not text:
                continue
            low = text.lower()
            if low in seen_elements:
                continue
            seen_elements.add(low)
            dedup_elements.append(text)

        return {
            "flow_path": str(flow_path),
            "step_cursor": cursor,
            "from_screen": from_screen,
            "current_screen": current_screen,
            "next_screen": next_screen,
            "action_hint": action_hint,
            "screen_chain": screen_chain[:25],
            "elements": dedup_elements[:25],
        }

    def _parse_flow_commands(self, flow_path: Path) -> List[Dict[str, Any]]:
        try:
            content = flow_path.read_text(encoding="utf-8")
        except OSError:
            return []
        commands: List[Dict[str, Any]] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            body = stripped[2:].strip()
            if not body:
                continue
            command, has_colon, raw_value = body.partition(":")
            cmd = command.strip()
            if not cmd:
                continue
            value = self._decode_flow_scalar(raw_value) if has_colon else ""
            commands.append(
                {
                    "index": len(commands) + 1,
                    "command": cmd,
                    "value": value,
                }
            )
        return commands

    def _decode_flow_scalar(self, raw_value: str) -> str:
        value = str(raw_value or "").strip()
        if not value:
            return ""
        if value.startswith('"') or value.startswith("'"):
            try:
                parsed = json.loads(value)
                return str(parsed)
            except (json.JSONDecodeError, TypeError, ValueError):
                return value.strip("\"'")
        return value

    def _ensure_app_installed(self, test_id: str, scenario_id: str | None = None) -> Dict[str, Any] | None:
        if not self.install_app_before_test:
            return None
        if self.reinstall_app_per_scenario:
            boundary_id = (scenario_id or f"test:{test_id}").strip()
            if boundary_id != self._last_scenario_id:
                reinstall_failure = self._reinstall_app_for_boundary(test_id, boundary_id)
                if reinstall_failure:
                    return reinstall_failure
                self._last_scenario_id = boundary_id
                self._app_install_done = True
            return None

        if self.install_app_once and self._app_install_done:
            return None

        log_dir = self.artifacts_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_name = "app-install.log" if self.install_app_once else f"{test_id}-app-install.log"
        log_file = log_dir / log_name

        cmd = self._build_install_cmd()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.command_timeout_seconds,
            )
            output = (result.stdout or "") + ("\n" if result.stdout or result.stderr else "") + (result.stderr or "")
            log_file.write_text(output, encoding="utf-8")
            if result.returncode != 0:
                return {
                    "test_id": test_id,
                    "status": "failed",
                    "attempt": 1,
                    "artifacts": [str(log_file)],
                    "error": "app_install_failed",
                    "failure_context": {
                        "cause": "app_install_failed",
                        "recommendation": (
                            f"Verify installer backend '{self._normalized_install_tool()}', app_path, "
                            "and connected simulator/device."
                        ),
                        "log_excerpt": self._trim_excerpt(output),
                        "log_path": str(log_file),
                    },
                }
        except FileNotFoundError:
            log_file.write_text(
                f"install backend executable not found for tool '{self._normalized_install_tool()}'",
                encoding="utf-8",
            )
            return {
                "test_id": test_id,
                "status": "failed",
                "attempt": 1,
                "artifacts": [str(log_file)],
                "error": "install_backend_not_found",
                "failure_context": {
                    "cause": "install_backend_not_found",
                    "recommendation": (
                        "Install required CLI (xcrun/maestro) or adjust MAESTRO_APP_INSTALL_TOOL."
                    ),
                    "log_excerpt": log_file.read_text(encoding="utf-8"),
                    "log_path": str(log_file),
                },
            }
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            output = f"maestro install timed out after {self.command_timeout_seconds}s\n{stdout}\n{stderr}"
            log_file.write_text(output, encoding="utf-8")
            return {
                "test_id": test_id,
                "status": "failed",
                "attempt": 1,
                "artifacts": [str(log_file)],
                "error": "app_install_timeout",
                "failure_context": {
                    "cause": "app_install_timeout",
                    "recommendation": "Ensure simulator/device is booted and app artifact is accessible, then retry.",
                    "log_excerpt": self._trim_excerpt(output),
                    "log_path": str(log_file),
                },
            }

        self._app_install_done = True
        return None

    def _reinstall_app_for_boundary(self, test_id: str, boundary_id: str) -> Dict[str, Any] | None:
        """Clean reinstall app on scenario boundary to reset onboarding state."""
        log_dir = self.artifacts_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_boundary = re.sub(r"[^a-zA-Z0-9_.-]+", "_", boundary_id)
        log_file = log_dir / f"{safe_boundary}-app-reinstall.log"

        uninstall_cmd = self._build_uninstall_cmd()

        install_cmd = self._build_install_cmd()

        uninstall_result = None
        uninstall_output = ""
        try:
            uninstall_result = subprocess.run(
                uninstall_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.command_timeout_seconds,
            )
            uninstall_output = (uninstall_result.stdout or "") + "\n" + (uninstall_result.stderr or "")
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            uninstall_output = f"uninstall step failed: {exc}"

        try:
            install_result = subprocess.run(
                install_cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.command_timeout_seconds,
            )
        except FileNotFoundError:
            log_file.write_text(
                "maestro binary not found during reinstall\n" + uninstall_output,
                encoding="utf-8",
            )
            return {
                "test_id": test_id,
                "status": "failed",
                "attempt": 1,
                "artifacts": [str(log_file)],
                "error": "install_backend_not_found",
                "failure_context": {
                    "cause": "install_backend_not_found",
                    "recommendation": (
                        "Install required CLI (xcrun/maestro) or adjust MAESTRO_APP_INSTALL_TOOL."
                    ),
                    "log_excerpt": self._trim_excerpt(log_file.read_text(encoding="utf-8")),
                    "log_path": str(log_file),
                },
            }
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
            body = (
                f"install step timed out after {self.command_timeout_seconds}s\n"
                f"{uninstall_output}\n{stdout}\n{stderr}"
            )
            log_file.write_text(body, encoding="utf-8")
            return {
                "test_id": test_id,
                "status": "failed",
                "attempt": 1,
                "artifacts": [str(log_file)],
                "error": "app_install_timeout",
                "failure_context": {
                    "cause": "app_install_timeout",
                    "recommendation": "Ensure simulator/device is booted and app artifact is accessible, then retry.",
                    "log_excerpt": self._trim_excerpt(body),
                    "log_path": str(log_file),
                },
            }

        install_output = (install_result.stdout or "") + "\n" + (install_result.stderr or "")
        body = (
            f"scenario_boundary={boundary_id}\n"
            f"uninstall_exit={getattr(uninstall_result, 'returncode', 'n/a')}\n"
            f"{uninstall_output}\n"
            f"install_exit={install_result.returncode}\n"
            f"{install_output}"
        )
        log_file.write_text(body, encoding="utf-8")
        if install_result.returncode != 0:
            return {
                "test_id": test_id,
                "status": "failed",
                "attempt": 1,
                "artifacts": [str(log_file)],
                "error": "app_install_failed",
                "failure_context": {
                    "cause": "app_install_failed",
                    "recommendation": "Verify app_path and connected simulator/device, then retry scenario.",
                    "log_excerpt": self._trim_excerpt(body),
                    "log_path": str(log_file),
                },
            }
        return None

    def _normalized_install_tool(self) -> str:
        tool = (self.app_install_tool or "maestro").strip().lower()
        return tool if tool in {"maestro", "xcrun"} else "maestro"

    def _build_install_cmd(self) -> List[str]:
        tool = self._normalized_install_tool()
        if tool == "xcrun":
            target = (self.ios_simulator_target or "booted").strip() or "booted"
            return ["xcrun", "simctl", "install", target, str(self.app_path)]
        cmd = [self.maestro_bin]
        if self.device:
            cmd.extend(["--device", self.device])
        cmd.extend(["install", str(self.app_path)])
        return cmd

    def _build_uninstall_cmd(self) -> List[str]:
        tool = self._normalized_install_tool()
        if tool == "xcrun":
            target = (self.ios_simulator_target or "booted").strip() or "booted"
            return ["xcrun", "simctl", "uninstall", target, self.app_id]
        cmd = [self.maestro_bin]
        if self.device:
            cmd.extend(["--device", self.device])
        cmd.extend(["uninstall", self.app_id])
        return cmd

    def _write_flow(
        self,
        test_case: Dict[str, Any],
        attempt: int,
        scenario_id: str | None = None,
        flow_scope: str = "test_case",
        flow_clear_state: bool | None = None,
        flow_yaml: str | None = None,
    ) -> Path:
        flow_dir = self.generated_flows_dir
        flow_dir.mkdir(parents=True, exist_ok=True)
        if flow_scope == "scenario" and scenario_id:
            file_stem = str(scenario_id).strip() or "scenario"
        else:
            file_stem = str(test_case.get("id", "anon")).strip() or "anon"
        flow_path = flow_dir / f"{file_stem}.yaml"
        app_id = (self.app_id or "default").strip() or "default"
        resolved_clear_state = self._resolve_flow_clear_state(flow_clear_state, attempt)
        if flow_yaml and flow_yaml.strip():
            flow_content = self._normalize_flow_yaml(
                flow_yaml=flow_yaml,
                app_id=app_id,
                clear_state=resolved_clear_state,
            )
        else:
            steps_yaml = self._steps_to_yaml(
                steps=test_case.get("steps", []),
                clear_state=resolved_clear_state,
            )
            flow_content = f"appId: {app_id}\n---\n" + steps_yaml
        flow_path.write_text(flow_content, encoding="utf-8")
        return flow_path

    def _normalize_flow_yaml(self, flow_yaml: str, app_id: str, clear_state: bool) -> str:
        raw = flow_yaml.strip()
        if not raw:
            return f"appId: {app_id}\n---\n" + "\n".join(
                self._default_launch_app_lines(clear_state=clear_state)
            )

        body = raw
        if raw.startswith("appId:"):
            _, _, tail = raw.partition("---")
            body = tail.strip() if tail else ""
        elif raw.startswith("---"):
            body = raw.removeprefix("---").strip()

        normalized_body = self._ensure_launch_app_block(body=body, clear_state=clear_state)
        return f"appId: {app_id}\n---\n{normalized_body}"

    def _steps_to_yaml(self, steps: List[Dict[str, Any]], clear_state: bool) -> str:
        lines: List[str] = self._default_launch_app_lines(clear_state=clear_state)
        for step in steps:
            lines.extend(self._normalize_step_to_commands(step))
        return "\n".join(lines)

    def _ensure_launch_app_block(self, body: str, clear_state: bool) -> str:
        lines = body.splitlines() if body else []
        launch_pattern = re.compile(r"^(\s*)-\s*launchApp\s*:?\s*$")
        launch_idx: int | None = None
        launch_indent_len = 0

        for idx, line in enumerate(lines):
            match = launch_pattern.match(line)
            if match:
                launch_idx = idx
                launch_indent_len = len(match.group(1))
                break

        default_lines = self._default_launch_app_lines(clear_state=clear_state)
        if launch_idx is None:
            return "\n".join(default_lines + lines).strip()

        prefix = " " * launch_indent_len
        indented_defaults = [f"{prefix}{default_lines[0]}"] + [
            f"{prefix}{line}" for line in default_lines[1:]
        ]
        normalized: List[str] = lines[:launch_idx] + indented_defaults

        idx = launch_idx + 1
        while idx < len(lines):
            current = lines[idx]
            current_strip = current.strip()
            if not current_strip:
                idx += 1
                continue

            current_indent = len(current) - len(current.lstrip(" "))
            is_next_list_item = current.lstrip(" ").startswith("- ")
            if is_next_list_item and current_indent <= launch_indent_len:
                break
            idx += 1

        normalized.extend(lines[idx:])
        return "\n".join(normalized).strip()

    def _default_launch_app_lines(self, clear_state: bool) -> List[str]:
        """Standard app start config aligned with project onboarding flow."""
        return [
            "- launchApp:",
            f"    clearState: {'true' if clear_state else 'false'}",
            "    clearKeychain: false",
            "    stopApp: true",
            "    permissions: { all: allow }",
        ]

    def _resolve_flow_clear_state(self, explicit: bool | None, attempt: int) -> bool:
        if explicit is not None:
            return explicit
        # Deterministic runs: always start from clean app state by default.
        return bool(self.flow_clear_state_default)

    def _normalize_step_to_commands(self, step: Dict[str, Any]) -> List[str]:
        """
        Convert verbose Qase prose into valid Maestro commands.

        Never emit unknown command names; fallback is a safe screenshot marker.
        """
        raw_action = str(step.get("action") or step.get("type") or "").strip()
        raw_payload = step.get("payload") or step.get("value") or step.get("text")
        expected_result = str(step.get("expected_result") or "").strip()
        raw_action_lower = raw_action.lower()
        cmds: List[str] = []

        supported = {
            "launchApp",
            "tapOn",
            "inputText",
            "scrollUntilVisible",
            "assertVisible",
            "assertNotVisible",
            "waitForAnimationToEnd",
            "extendedWaitUntil",
            "takeScreenshot",
            "runFlow",
        }

        # If upstream already provided a known Maestro command, keep it.
        if raw_action in supported:
            if raw_action in {"assertVisible", "assertNotVisible"}:
                payload_text = str(raw_payload or "").strip()
                if self._is_placeholder_assertion(payload_text):
                    return [self._render_comment(f"placeholder assertion skipped: {payload_text}")]
            rendered = self._render_command(raw_action, raw_payload)
            if rendered:
                cmds.append(rendered)
            return cmds or [self._render_comment("empty-known-command")]

        # Heuristic mapping from prose -> valid Maestro commands.
        quoted_bits = self._extract_quoted_text(raw_action)
        candidate = quoted_bits[0] if quoted_bits else None
        if not candidate:
            candidate = self._extract_parenthesized_text(raw_action)

        if any(token in raw_action_lower for token in ("тап", "tap", "нажм", "клик")):
            target = candidate or self._infer_common_target(raw_action_lower)
            if target:
                cmds.append(self._render_command("tapOn", target))
            else:
                cmds.append(self._render_comment(f"tap action not mapped: {raw_action}"))
        elif any(token in raw_action_lower for token in ("введ", "input", "enter", "заполн")):
            text_value = raw_payload or candidate
            if text_value:
                cmds.append(self._render_command("inputText", text_value))
            else:
                cmds.append(self._render_comment(f"input action not mapped: {raw_action}"))
        elif any(token in raw_action_lower for token in ("пролист", "scroll", "swipe", "свайп")):
            target = candidate or self._first_non_empty_line(expected_result)
            if target:
                cmds.append(self._render_command("scrollUntilVisible", target))
            else:
                cmds.append("- waitForAnimationToEnd")
        elif any(token in raw_action_lower for token in ("открыт", "экран", "переход")):
            # State-like prose usually describes expected screen; make it explicit.
            if candidate:
                if not self._is_placeholder_assertion(candidate):
                    cmds.append(self._render_command("assertVisible", candidate))
                else:
                    cmds.append(self._render_comment(f"screen assertion unresolved: {candidate}"))
            else:
                first_line = self._first_non_empty_line(raw_action)
                if first_line:
                    cmds.append(self._render_comment(first_line))
        else:
            if raw_action:
                cmds.append(self._render_comment(raw_action))

        # Promote compact expected_result to assertion when practical.
        expected_line = self._first_non_empty_line(expected_result)
        expected_quoted = self._extract_quoted_text(expected_result)
        expected_candidate = expected_quoted[0] if expected_quoted else self._extract_parenthesized_text(
            expected_result
        )
        if expected_candidate and not self._is_placeholder_assertion(expected_candidate):
            cmds.append(self._render_command("assertVisible", expected_candidate))
        elif (
            expected_line
            and len(expected_line) <= 120
            and len(expected_line.split()) <= 8
            and not self._is_placeholder_assertion(expected_line)
        ):
            cmds.append(self._render_command("assertVisible", expected_line))
        elif expected_line:
            cmds.append(self._render_comment(f"expected: {expected_line[:200]}"))

        if not cmds:
            cmds.append(self._render_comment("Unmapped step"))
        return cmds

    def _render_command(self, command: str, payload: Any = None) -> str | None:
        no_payload_cmds = {"launchApp", "waitForAnimationToEnd"}
        if command in no_payload_cmds:
            return f"- {command}"
        if payload is None or payload == "":
            return None
        return f"- {command}: {json.dumps(str(payload), ensure_ascii=False)}"

    def _render_comment(self, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        # Some Maestro versions do not support `comment`; use deterministic evidence capture instead.
        return f"- takeScreenshot: \"{self._note_screenshot_name(digest)}\""

    def _note_screenshots_dir(self) -> Path:
        return self._note_dir

    def _note_screenshot_name(self, digest: str) -> str:
        # Maestro appends .png automatically; pass path without extension.
        return str(self._note_screenshots_dir() / f"note-{digest}")

    def _is_placeholder_assertion(self, text: str) -> bool:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return True

        exact_placeholders = {
            "app launched",
            "screenshot captured",
            "onboarding quiz visible",
            "quiz q2 visible",
            "quiz q3 visible",
            "screen visible",
        }
        if normalized in exact_placeholders:
            return True

        prefix_placeholders = (
            "открыт ",
            "открыта ",
            "открыто ",
            "переключение на экран",
            "отображается экран",
            "пройти онбординг",
            "тап на ",
            "tap ",
            "launch ",
        )
        if any(normalized.startswith(prefix) for prefix in prefix_placeholders):
            return True

        # Treat long prose-like checks as unstable selectors.
        if "\n" in normalized or len(normalized.split()) > 10:
            return True

        return False

    def _extract_quoted_text(self, source: str) -> List[str]:
        if not source:
            return []
        matches = re.findall(r"[\"'«](.+?)[\"'»]", source)
        return [item.strip() for item in matches if item and item.strip()]

    def _extract_parenthesized_text(self, source: str) -> str | None:
        if not source:
            return None
        match = re.search(r"\(([^)]+)\)", source)
        if not match:
            return None
        text = match.group(1).strip()
        return text or None

    def _first_non_empty_line(self, source: str) -> str | None:
        if not source:
            return None
        for line in source.splitlines():
            text = line.strip()
            if text:
                return text
        return None

    def _infer_common_target(self, action_lower: str) -> str | None:
        if "продолж" in action_lower:
            return "Continue"
        if "назад" in action_lower:
            return "Back"
        return None

    def _build_failure_context(self, stdout: str, stderr: str, log_file: Path) -> Dict[str, Any]:
        """Return compact diagnostics so the agent can fix flow on next retry."""
        combined = "\n".join(part for part in [stdout, stderr] if part).strip()
        lower = combined.lower()
        cause = "unknown_maestro_failure"
        recommendation = (
            "Inspect failed command in log excerpt, add synchronization, and prefer stable selectors."
        )

        if "timed out" in lower or "timeout" in lower:
            cause = "timeout"
            recommendation = (
                "Add waitForAnimationToEnd/extendedWaitUntil before next assertion or tap, "
                "then retry."
            )
        elif "element" in lower and "not found" in lower:
            cause = "element_not_found"
            recommendation = (
                "Update selector text/id to match current screen and use scrollUntilVisible "
                "before interacting. If testcase text is in another language, trust screenshot "
                "labels (often English UI) as source of truth for selector names."
            )
        elif "assert" in lower and "failed" in lower:
            cause = "assertion_failed"
            recommendation = (
                "Verify expected state transition; add sync step before assertion and adjust "
                "assertVisible/assertNotVisible."
            )
        elif "yaml" in lower and ("parse" in lower or "invalid" in lower):
            cause = "invalid_yaml"
            recommendation = (
                "Rewrite step into valid Maestro command syntax; do not use raw prose as YAML keys."
            )
        elif "binary not found" in lower:
            cause = "maestro_binary_not_found"
            recommendation = "Install Maestro CLI or set MAESTRO_BIN to a valid executable path."

        return {
            "cause": cause,
            "recommendation": recommendation,
            "log_excerpt": self._trim_excerpt(combined),
            "log_path": str(log_file),
            # Step indexes are filled from Maestro debug command logs when available.
            "failed_step_index": None,
            "last_successful_step_index": None,
            "retry_from_step_index": None,
        }

    def _trim_excerpt(self, content: str) -> str:
        if not content:
            return ""
        max_chars = max(256, int(self.failure_excerpt_max_chars))
        if len(content) <= max_chars:
            return content
        return content[-max_chars:]

    def _extract_maestro_debug_dir(self, content: str) -> str | None:
        if not content:
            return None
        match = re.search(r"(/[^\s]*\.maestro/tests/[^\s]+)", content)
        if not match:
            return None
        return match.group(1)

    def _flow_contains_assertions(self, flow_path: Path) -> bool:
        try:
            content = flow_path.read_text(encoding="utf-8")
        except OSError:
            return False
        return bool(re.search(r"^\s*-\s*(assertVisible|assertNotVisible):\s+", content, flags=re.MULTILINE))

    def _collect_debug_context(
        self,
        debug_dir: str,
        test_id: str,
        attempt: int,
    ) -> tuple[Dict[str, Any], str | None]:
        """Inline selector hints from Maestro debug files and persist snapshot."""
        root = Path(debug_dir)
        if not root.exists() or not root.is_dir():
            return {}, None

        command_logs = sorted(root.glob("commands-*.json"))
        if not command_logs:
            return {}, None

        latest = command_logs[-1]
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}, None
        if not isinstance(payload, list):
            return {}, None

        failed_selector = ""
        failed_step_index: int | None = None
        last_successful_step_index = 0
        failed_command_name = ""
        hierarchy_root: Dict[str, Any] | None = None
        for idx, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata")
            if not isinstance(metadata, dict):
                continue

            status = str(metadata.get("status") or "").strip().upper()
            if status in {"PASSED", "SUCCESS", "COMPLETED", "OK"}:
                last_successful_step_index = idx
                continue
            if status != "FAILED":
                continue
            if failed_step_index is None:
                failed_step_index = idx

            command = item.get("command")
            if isinstance(command, dict):
                command_names = [str(name).strip() for name in command.keys() if str(name).strip()]
                if command_names and not failed_command_name:
                    failed_command_name = command_names[0]
                tap = command.get("tapOnElement")
                if isinstance(tap, dict):
                    selector = tap.get("selector")
                    if isinstance(selector, dict):
                        failed_selector = str(
                            selector.get("textRegex")
                            or selector.get("text")
                            or selector.get("idRegex")
                            or selector.get("id")
                            or ""
                        ).strip()

            error = metadata.get("error")
            if isinstance(error, dict):
                root_candidate = error.get("hierarchyRoot")
                if isinstance(root_candidate, dict):
                    hierarchy_root = root_candidate
                    break

        if not hierarchy_root:
            context: Dict[str, Any] = {}
            if failed_selector:
                context["failed_selector"] = failed_selector
            if failed_command_name:
                context["failed_command"] = failed_command_name
            if failed_step_index is not None:
                safe_last_success = min(last_successful_step_index, max(failed_step_index - 1, 0))
                context["failed_step_index"] = failed_step_index
                context["last_successful_step_index"] = safe_last_success
                context["retry_from_step_index"] = safe_last_success + 1
            return context, None

        texts = self._extract_ui_text_candidates(hierarchy_root)
        result: Dict[str, Any] = {
            "source": str(latest),
            "ui_text_candidates": texts[:20],
        }
        if failed_selector:
            result["failed_selector"] = failed_selector
        if failed_command_name:
            result["failed_command"] = failed_command_name
        if failed_step_index is not None:
            safe_last_success = min(last_successful_step_index, max(failed_step_index - 1, 0))
            result["failed_step_index"] = failed_step_index
            result["last_successful_step_index"] = safe_last_success
            result["retry_from_step_index"] = safe_last_success + 1
        if texts:
            result["hint"] = (
                "Use ui_text_candidates as real on-screen labels for tap/assert "
                "instead of testcase prose."
            )
        snapshot_dir = self._write_debug_snapshot(
            test_id=test_id,
            attempt=attempt,
            hierarchy_root=hierarchy_root,
            debug_context=result,
            source_log=latest,
            maestro_debug_dir=debug_dir,
        )
        return result, snapshot_dir

    def _extract_ui_text_candidates(self, root: Dict[str, Any]) -> List[str]:
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

        walk(root)
        return ordered

    def _write_debug_snapshot(
        self,
        test_id: str,
        attempt: int,
        hierarchy_root: Dict[str, Any],
        debug_context: Dict[str, Any],
        source_log: Path,
        maestro_debug_dir: str,
    ) -> str:
        base = self._debug_snapshots_dir / test_id / f"attempt-{attempt}"
        shutil.rmtree(base, ignore_errors=True)
        base.mkdir(parents=True, exist_ok=True)
        (base / "hierarchy.json").write_text(
            json.dumps(hierarchy_root, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        context_payload = dict(debug_context)
        context_payload["captured_at"] = datetime.now(timezone.utc).isoformat()
        (base / "context.json").write_text(
            json.dumps(context_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        meta = {
            "source_log": str(source_log),
            "maestro_debug_dir": maestro_debug_dir,
        }
        (base / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(base)

    def _capture_screenshot(self, test_id: str, attempt: int) -> str | None:
        shots_dir = self.artifacts_dir / "screenshots" / test_id
        shots_dir.mkdir(parents=True, exist_ok=True)
        shot_path_png = shots_dir / f"attempt-{attempt}.png"
        shot_path_jpg = shots_dir / f"attempt-{attempt}.jpg"
        cmd = [self.maestro_bin]
        if self.device:
            cmd.extend(["--device", self.device])
        cmd.extend(["screenshot", str(shot_path_png)])
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=self.command_timeout_seconds,
            )
            prepared_path = self._optimize_screenshot_for_model(shot_path_png, shot_path_jpg)
            return str(prepared_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Fallback for iOS simulator when Maestro screenshot command is unavailable/flaky.
        target = (self.ios_simulator_target or "booted").strip() or "booted"
        fallback_cmd = ["xcrun", "simctl", "io", target, "screenshot", str(shot_path_png)]
        try:
            subprocess.run(
                fallback_cmd,
                check=True,
                capture_output=True,
                timeout=self.command_timeout_seconds,
            )
            prepared_path = self._optimize_screenshot_for_model(shot_path_png, shot_path_jpg)
            return str(prepared_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _optimize_screenshot_for_model(self, source_png_path: Path, target_jpg_path: Path) -> Path:
        """Resize screenshot and convert to JPG to reduce payload size."""
        try:
            max_side = int(os.getenv("MAESTRO_SCREENSHOT_MAX_SIDE_PX", self.screenshot_max_side_px))
        except ValueError:
            max_side = self.screenshot_max_side_px

        sips_bin = shutil.which("sips")
        if not sips_bin:
            return source_png_path

        try:
            quality = int(os.getenv("MAESTRO_SCREENSHOT_JPEG_QUALITY", self.screenshot_jpeg_quality))
        except ValueError:
            quality = self.screenshot_jpeg_quality
        quality = max(1, min(100, quality))

        try:
            if max_side > 0:
                subprocess.run(
                    [sips_bin, "-Z", str(max_side), str(source_png_path)],
                    check=True,
                    capture_output=True,
                    timeout=20,
                )
            subprocess.run(
                [
                    sips_bin,
                    "--setProperty",
                    "format",
                    "jpeg",
                    "--setProperty",
                    "formatOptions",
                    str(quality),
                    str(source_png_path),
                    "--out",
                    str(target_jpg_path),
                ],
                check=True,
                capture_output=True,
                timeout=20,
            )
            if source_png_path.exists():
                source_png_path.unlink(missing_ok=True)
            return target_jpg_path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            # Best-effort optimization: keep original screenshot if conversion failed.
            return source_png_path

    def _skip_onboarding_if_possible(self, test_id: str) -> None:
        if not self.skip_onboarding_deeplink:
            return
        cmd = [self.maestro_bin]
        if self.device:
            cmd.extend(["--device", self.device])
        cmd.extend(["open", "--url", self.skip_onboarding_deeplink])
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=self.command_timeout_seconds,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            log_dir = self.artifacts_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            payload = getattr(exc, "stdout", None)
            body = payload.decode() if isinstance(payload, bytes) else str(payload)
            (log_dir / f"{test_id}-skip-onboarding.log").write_text(
                body or "failed to trigger deeplink",
                encoding="utf-8",
            )
