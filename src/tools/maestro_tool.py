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
        description="JSON payload with fields: test_case, attempt, screenshot, is_onboarding.",
    )


class MaestroAutomationTool(BaseTool):
    """Generates Maestro flows and executes them for a single test case."""

    name: str = "maestro_cli"
    description: str = (
        "Generate temporary Maestro flows for the provided test case payload, "
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
    command_timeout_seconds: int = 120
    screenshot_max_side_px: int = 1440
    failure_excerpt_max_chars: int = 4000
    _app_install_done: bool = PrivateAttr(default=False)
    _last_scenario_id: str | None = PrivateAttr(default=None)

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
        return self.run_test_case(test_case, attempt, screenshot, is_onboarding, scenario_id)

    # Core execution ---------------------------------------------------
    def run_test_case(
        self,
        test_case: Dict[str, Any],
        attempt: int,
        request_screenshot: bool = False,
        is_onboarding: bool | None = None,
        scenario_id: str | None = None,
    ) -> Dict[str, Any]:
        test_id = test_case.get("id", f"anon-{attempt}")
        install_failure = self._ensure_app_installed(test_id, scenario_id)
        if install_failure:
            return install_failure
        if not is_onboarding:
            self._skip_onboarding_if_possible(test_id)
        flow_path = self._write_flow(test_case)
        cmd = [self.maestro_bin]
        if self.device:
            cmd.extend(["--device", self.device])
        cmd.extend(["test", str(flow_path)])

        log_path = self.artifacts_dir / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"{test_id}-attempt-{attempt}.log"

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
                "test_id": test_id,
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
                "test_id": test_id,
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
            shot = self._capture_screenshot(test_id, attempt)
            if shot:
                artifacts.append(shot)

        status = "passed" if result.returncode == 0 else "failed"
        response: Dict[str, Any] = {
            "test_id": test_id,
            "status": status,
            "attempt": attempt,
            "artifacts": artifacts,
        }
        if status == "failed" and not request_screenshot:
            shot = self._capture_screenshot(test_id, attempt)
            if shot:
                artifacts.append(shot)
        if status == "failed":
            response["failure_context"] = self._build_failure_context(
                stdout=result.stdout,
                stderr=result.stderr,
                log_file=log_file,
            )
            response["error"] = response["failure_context"].get("cause")

        return response

    # Helpers ----------------------------------------------------------
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

    def _write_flow(self, test_case: Dict[str, Any]) -> Path:
        flow_dir = self.generated_flows_dir
        flow_dir.mkdir(parents=True, exist_ok=True)
        flow_path = flow_dir / f"{test_case.get('id', 'anon')}.yaml"
        steps_yaml = self._steps_to_yaml(test_case.get("steps", []))
        app_id = (self.app_id or "default").strip() or "default"
        flow_content = f"appId: {app_id}\n---\n" + steps_yaml
        flow_path.write_text(flow_content, encoding="utf-8")
        return flow_path

    def _steps_to_yaml(self, steps: List[Dict[str, Any]]) -> str:
        lines: List[str] = self._default_launch_app_lines()
        for step in steps:
            lines.extend(self._normalize_step_to_commands(step))
        return "\n".join(lines)

    def _default_launch_app_lines(self) -> List[str]:
        """Standard app start config aligned with project onboarding flow."""
        return [
            "- launchApp:",
            "    clearState: true",
            "    clearKeychain: false",
            "    stopApp: false",
            "    permissions: { all: deny }",
        ]

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
                cmds.append(self._render_command("assertVisible", candidate))
            else:
                first_line = self._first_non_empty_line(raw_action)
                if first_line:
                    cmds.append(self._render_comment(first_line))
        else:
            if raw_action:
                cmds.append(self._render_comment(raw_action))

        # Promote compact expected_result to assertion when practical.
        expected_line = self._first_non_empty_line(expected_result)
        if expected_line and len(expected_line) <= 120:
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
        return f"- takeScreenshot: \"note-{digest}\""

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

    def _build_failure_context(self, stdout: str, stderr: str, log_file: Path) -> Dict[str, str]:
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
        }

    def _trim_excerpt(self, content: str) -> str:
        if not content:
            return ""
        max_chars = max(256, int(self.failure_excerpt_max_chars))
        if len(content) <= max_chars:
            return content
        return content[-max_chars:]

    def _capture_screenshot(self, test_id: str, attempt: int) -> str | None:
        shots_dir = self.artifacts_dir / "screenshots" / test_id
        shots_dir.mkdir(parents=True, exist_ok=True)
        shot_path = shots_dir / f"attempt-{attempt}.png"
        cmd = [self.maestro_bin]
        if self.device:
            cmd.extend(["--device", self.device])
        cmd.extend(["screenshot", str(shot_path)])
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=self.command_timeout_seconds,
            )
            self._shrink_screenshot_for_model(shot_path)
            return str(shot_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None

    def _shrink_screenshot_for_model(self, shot_path: Path) -> None:
        """Reduce image dimensions before it is attached to model context."""
        try:
            max_side = int(os.getenv("MAESTRO_SCREENSHOT_MAX_SIDE_PX", self.screenshot_max_side_px))
        except ValueError:
            max_side = self.screenshot_max_side_px
        if max_side <= 0:
            return

        sips_bin = shutil.which("sips")
        if not sips_bin:
            return

        try:
            subprocess.run(
                [sips_bin, "-Z", str(max_side), str(shot_path)],
                check=True,
                capture_output=True,
                timeout=20,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            # Best-effort optimization: keep original screenshot if resize failed.
            return

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
