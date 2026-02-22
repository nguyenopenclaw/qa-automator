"""Entry-point CLI for the QA Automator crew."""
from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from crewai import Crew, Process
from rich.console import Console
from rich.table import Table

from agents import qa_manager_agent
from tasks import automate_tests_task, parse_inputs_task, summarize_results_task
from tools.maestro_tool import MaestroAutomationTool
from tools.qase_parser import QaseTestParserTool
from tools.state_tracker import AutomationStateTrackerTool

console = Console()
app = typer.Typer(help="Run the QA Automator crew against a set of Qase test cases")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@app.command()
def run(
    test_cases: Path = typer.Option(..., exists=True, help="Qase JSON export"),
    tested: Path = typer.Option(..., exists=True, help="JSON list of already automated cases"),
    app_path: Path = typer.Option(..., exists=True, help="Path to APK/IPA under test"),
    output: Path = typer.Option(Path("artifacts"), help="Directory for artifacts"),
    max_attempts: int = typer.Option(10, min=1, max=10, help="Attempts per test"),
):
    """Execute the crew end-to-end."""
    _ensure_dir(output)
    _ensure_dir(output / "flows")
    _ensure_dir(output / "screenshots")

    maestro_tool = MaestroAutomationTool(
        app_path=app_path,
        artifacts_dir=output,
        maestro_bin=os.getenv("MAESTRO_BIN"),
        device=os.getenv("MAESTRO_DEVICE"),
        skip_onboarding_deeplink=os.getenv("APP_SKIP_ONBOARDING_DEEPLINK"),
    )
    qase_tool = QaseTestParserTool(test_cases=test_cases, tested_cases=tested)
    state_tool = AutomationStateTrackerTool(artifacts_dir=output)

    manager = qa_manager_agent(maestro_tool, qase_tool, state_tool)

    crew = Crew(
        agents=[manager],
        tasks=[
            parse_inputs_task(manager, str(test_cases), str(tested)),
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
