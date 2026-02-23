"""Entry-point CLI for the QA Automator crew."""
from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from crewai import Crew, Process
from rich.console import Console
from rich.table import Table

from agents import appflow_specialist_agent, maestro_senior_agent, qa_manager_agent
from tasks import automate_tests_task, map_appflow_task, parse_inputs_task, summarize_results_task
from tools.appflow_tool import AppFlowMemoryTool
from tools.maestro_tool import MaestroAutomationTool
from tools.qase_parser import QaseTestParserTool
from tools.screen_inspector_tool import ScreenInspectorTool
from tools.state_tracker import AutomationStateTrackerTool

console = Console()
app = typer.Typer(help="Run the QA Automator crew against a set of Qase test cases")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@app.command()
def run(
    test_cases: Path = typer.Option(..., exists=True, help="Qase JSON export"),
    tested: Path = typer.Option(..., exists=True, help="JSON list of already automated cases"),
    app_path: Path = typer.Option(..., exists=True, help="Path to APK/IPA under test"),
    output: Path = typer.Option(Path("artifacts"), help="Directory for artifacts"),
    automated_dir: Path = typer.Option(
        Path("samples/automated"),
        help="Directory where generated Maestro tests (.yaml) are stored",
    ),
    max_attempts: int = typer.Option(10, min=1, max=10, help="Attempts per test"),
):
    """Execute the crew end-to-end."""
    _ensure_dir(output)
    _ensure_dir(output / "screenshots")
    _ensure_dir(automated_dir)

    maestro_tool = MaestroAutomationTool(
        app_path=app_path,
        artifacts_dir=output,
        generated_flows_dir=automated_dir,
        maestro_bin=os.getenv("MAESTRO_BIN"),
        device=os.getenv("MAESTRO_DEVICE"),
        app_id=os.getenv("MAESTRO_APP_ID", "default"),
        skip_onboarding_deeplink=os.getenv("APP_SKIP_ONBOARDING_DEEPLINK"),
        app_install_tool=os.getenv("MAESTRO_APP_INSTALL_TOOL", "xcrun"),
        ios_simulator_target=os.getenv("IOS_SIMULATOR_TARGET", "booted"),
        install_app_before_test=_env_bool("MAESTRO_INSTALL_APP_BEFORE_TEST", True),
        install_app_once=_env_bool("MAESTRO_INSTALL_APP_ONCE", True),
        reinstall_app_per_scenario=_env_bool("MAESTRO_REINSTALL_APP_PER_SCENARIO", True),
        flow_clear_state_default=_env_bool("MAESTRO_FLOW_CLEAR_STATE_DEFAULT", True),
    )
    qase_tool = QaseTestParserTool(
        test_cases_path=test_cases,
        tested_cases_path=tested,
        artifacts_dir=output,
    )
    state_tool = AutomationStateTrackerTool(artifacts_dir=output)
    appflow_tool = AppFlowMemoryTool(artifacts_dir=output)
    screen_tool = ScreenInspectorTool(artifacts_dir=output)

    manager = qa_manager_agent(maestro_tool, qase_tool, state_tool, appflow_tool)
    appflow = appflow_specialist_agent(appflow_tool, qase_tool, screen_tool)
    maestro_senior = maestro_senior_agent()

    crew = Crew(
        agents=[manager, appflow, maestro_senior],
        tasks=[
            parse_inputs_task(manager, str(test_cases), str(tested)),
            map_appflow_task(appflow, str(output)),
            automate_tests_task(manager, str(app_path), str(output), max_attempts),
            summarize_results_task(manager),
        ],
        process=Process.sequential,
        verbose=True,
    )

    console.log("Starting QA automation crew...")
    result = crew.kickoff()

    report_path = output / "automation_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        _print_summary(report)

    console.print("\n[bold green]Crew run complete[/bold green]")
    console.print(result)


def _print_summary(report: dict) -> None:
    table = Table(title="Automation summary")
    table.add_column("Test ID")
    table.add_column("Status")
    table.add_column("Attempts", justify="right")
    table.add_column("Artifacts")

    for case in report.get("tests", []):
        table.add_row(
            case.get("id", "?"),
            case.get("status", "unknown"),
            str(case.get("attempts", 0)),
            ", ".join(case.get("artifacts", [])),
        )

    console.print(table)


if __name__ == "__main__":
    app()
