"""Tool that proxies Maestro CLI interactions."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


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
    skip_onboarding_deeplink: str | None = None
    command_timeout_seconds: int = 120
    screenshot_max_side_px: int = 1440
    failure_excerpt_max_chars: int = 4000

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
        return self.run_test_case(test_case, attempt, screenshot, is_onboarding)

    # Core execution ---------------------------------------------------
    def run_test_case(
        self,
        test_case: Dict[str, Any],
        attempt: int,
        request_screenshot: bool = False,
        is_onboarding: bool | None = None,
    ) -> Dict[str, Any]:
        test_id = test_case.get("id", f"anon-{attempt}")
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
    def _write_flow(self, test_case: Dict[str, Any]) -> Path:
        flow_dir = self.generated_flows_dir
        flow_dir.mkdir(parents=True, exist_ok=True)
        flow_path = flow_dir / f"{test_case.get('id', 'anon')}.yaml"
        steps_yaml = self._steps_to_yaml(test_case.get("steps", []))
        flow_content = "appId: default\n---\n" + steps_yaml
        flow_path.write_text(flow_content, encoding="utf-8")
        return flow_path

    def _steps_to_yaml(self, steps: List[Dict[str, Any]]) -> str:
        lines = []
        for step in steps:
            action = step.get("action") or step.get("type") or "tapOn"
            payload = step.get("payload") or step.get("value") or step.get("text") or ""
            lines.append(f"- {action}: {json.dumps(payload)}")
        return "\n".join(lines) if lines else "- launchApp"

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
                "before interacting."
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
