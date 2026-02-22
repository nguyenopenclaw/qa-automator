"""Task definitions for the QA Automator project."""
from __future__ import annotations

from crewai import Task


def parse_inputs_task(agent, test_cases_path: str, tested_info_path: str) -> Task:
    description = f"""
    Load the Qase-exported test cases from `{test_cases_path}` and the already-tested
    metadata from `{tested_info_path}`. Produce a prioritized execution plan that lists
    each test identifier, title, remaining manual notes, and whether it still needs automation.
    """

    expected_output = (
        "A markdown table summarizing the pending test cases with columns id, title, "
        "status, priority, and rationale."
    )

    return Task(
        name="Parse Qase inputs",
        description=description,
        expected_output=expected_output,
        agent=agent,
    )


def automate_tests_task(agent, app_path: str, artifacts_dir: str, max_attempts: int) -> Task:
    description = f"""
    Iterate through the prioritized backlog and, for each pending case, synthesize a Maestro
    flow, run it against `{app_path}`, and capture artifacts under `{artifacts_dir}`.
    Respect the attempt limit of {max_attempts} tries per test. For each attempt log inputs,
    Maestro stdout/stderr, and whether a screenshot was requested. Mark unresolved cases as
    problematic.
    """

    expected_output = (
        "JSON summary saved to artifacts/automation_report.json that lists every test id, "
        "attempt count, final status (passed, failed, problematic), and artifact pointers."
    )

    return Task(
        name="Automate tests with Maestro",
        description=description,
        expected_output=expected_output,
        agent=agent,
    )


def summarize_results_task(agent) -> Task:
    description = (
        "Produce a final report that highlights: total tests, automated successfully, flagged as "
        "problematic, and actionable insights for manual QA. Include links to screenshots/logs."
    )

    expected_output = "Markdown report stored at artifacts/summary.md"

    return Task(
        name="Summarize automation run",
        description=description,
        expected_output=expected_output,
        agent=agent,
    )
