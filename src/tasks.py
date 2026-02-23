"""Task definitions for the QA Automator project."""
from __future__ import annotations

from crewai import Task


def parse_inputs_task(agent, test_cases_path: str, tested_info_path: str) -> Task:
    description = f"""
    Load the Qase-exported test cases from `{test_cases_path}` and the already-tested
    metadata from `{tested_info_path}`.
    Build end-to-end scenarios by combining related test cases and save them into
    `scenarios.json`. Return only a compact summary plus the next single scenario
    to automate so the context stays small.
    """

    expected_output = (
        "A compact report with total/pending counts, path to scenarios.json, and one "
        "next scenario ready for automation."
    )

    return Task(
        name="Parse Qase inputs",
        description=description,
        expected_output=expected_output,
        agent=agent,
    )


def automate_tests_task(agent, app_path: str, artifacts_dir: str, max_attempts: int) -> Task:
    description = f"""
    Read exactly one pending scenario from qase_parser and automate only that scenario in
    this run (do not process multiple scenarios). For each case in this chosen scenario,
    synthesize a naive Maestro flow, run it against `{app_path}`, and capture artifacts under
    `{artifacts_dir}`. Always grab a screenshot whenever a run fails so you can inspect
    the UI state before the next iteration. You MUST use guidance from
    `skills/maestro-test-writing/SKILL.md` while writing/fixing Maestro flow files.
    For every failed run, inspect returned `failure_context` (log excerpt, cause,
    recommendation) and adjust steps before retrying. Respect the attempt limit of
    {max_attempts} tries per case. For each attempt log inputs, Maestro stdout/stderr,
    and whether additional screenshots were requested. Mark unresolved cases as problematic.
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
