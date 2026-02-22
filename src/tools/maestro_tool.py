"""Tool that proxies Maestro CLI interactions."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List


class MaestroAutomationTool:
    """Generates Maestro flows and executes them for a single test case."""

    name = "maestro_cli"
    description = (
        "Generate temporary Maestro flows for the provided test case payload, "
        "execute them against the supplied app binary, and capture screenshots/logs."
    )

    def __init__(
        self,
        app_path: Path,
        artifacts_dir: Path,
        maestro_bin: str | None = None,
        device: str | None = None,
        skip_onboarding_deeplink: str | None = None,
    ) -> None:
        self.app_path = Path(app_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.maestro_bin = maestro_bin or "maestro"
        self.device = device
        self.skip_onboarding_deeplink = skip_onboarding_deeplink

    def __call__(self, payload: str | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(payload, str):
            payload = json.loads(payload)
        test_case = payload.get("test_case", {})
        attempt = payload.get("attempt", 1)
        request_screenshot = payload.get("screenshot", False)
        return self.run_test_case(
            test_case, attempt, request_screenshot, payload.get("is_onboarding")
        )

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
        cmd = [self.maestro_bin, "test", str(flow_path), "-a", str(self.app_path)]
        if self.device:
            cmd.extend(["-d", self.device])

        log_path = self.artifacts_dir / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"{test_id}-attempt-{attempt}.log"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        log_file.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")

        artifacts: List[str] = [str(log_file)]
        if request_screenshot:
            shot = self._capture_screenshot(test_id, attempt)
            if shot:
                artifacts.append(shot)

        status = "passed" if result.returncode == 0 else "failed"
        return {
            "test_id": test_id,
            "status": status,
            "attempt": attempt,
            "artifacts": artifacts,
        }

    # Helpers ----------------------------------------------------------
    def _write_flow(self, test_case: Dict[str, Any]) -> Path:
        flow_dir = self.artifacts_dir / "flows"
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

    def _capture_screenshot(self, test_id: str, attempt: int) -> str | None:
        shots_dir = self.artifacts_dir / "screenshots" / test_id
        shots_dir.mkdir(parents=True, exist_ok=True)
        shot_path = shots_dir / f"attempt-{attempt}.png"
        cmd = [self.maestro_bin, "screenshot", str(shot_path)]
        if self.device:
            cmd.extend(["-d", self.device])
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return str(shot_path)
        except subprocess.CalledProcessError:
            return None

def _skip_onboarding_if_possible(self, test_id: str) -> None:
        if not self.skip_onboarding_deeplink:
            return
        cmd = [self.maestro_bin, "open", "--url", self.skip_onboarding_deeplink]
        if self.device:
            cmd.extend(["-d", self.device])
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            log_dir = self.artifacts_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            payload = getattr(exc, "stdout", None)
            body = payload.decode() if isinstance(payload, bytes) else str(payload)
            (log_dir / f"{test_id}-skip-onboarding.log").write_text(
                body or "failed to trigger deeplink",
                encoding="utf-8",
            )
