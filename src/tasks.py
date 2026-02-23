"""Task definitions for the QA Automator project."""
from __future__ import annotations

from crewai import Task


def parse_inputs_task(agent, test_cases_path: str, tested_info_path: str) -> Task:
    description = f"""
    Load Qase-exported test cases from `{test_cases_path}` and already-tested metadata
    from `{tested_info_path}`.
    Build grouped end-to-end scenarios and save them to `scenarios.json`.
    Keep context compact: return only high-signal summary and one next scenario candidate.

    Output contract:
    - total test count and pending test count,
    - path to `scenarios.json`,
    - exactly one next scenario id selected for downstream planning.
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


def map_appflow_task(agent, artifacts_dir: str) -> Task:
    description = f"""
    Open `{artifacts_dir}/manager_plan.json`, read `selected_scenario_id_for_this_run`, then
    fetch that exact scenario via qase_parser query `scenario_id:<id>`.
    Do not rely on implicit "next scenario" selection.

    Planning protocol (mandatory):
    0) If `selected_scenario_id_for_this_run` is missing, stop and report a blocking error.
    1) For each case in the scenario, call `app_flow_memory.suggest_context`.
    2) Persist each hypothesis immediately via `app_flow_memory.record_plan`.
    3) If memory has no data, infer from title/preconditions/steps and mark low-confidence.
    4) Do not call `screen_inspector` in this pre-run planning task.

    Persist structured AppFlow outputs:
    - screen graph files in `{artifacts_dir}/app_flow_memory/screens/screen_*.json`
      (elements + transitions + screenshot evidence per screen),
    - flow files in `{artifacts_dir}/app_flow_memory/flows/flow_*.json`
      (screen chain references + short descriptions, strictly 1 flow per scenario),
    - consolidated scenario plan in `{artifacts_dir}/appflow_plan_<scenario_id>.json`.
    Ensure every case has explicit entry hypothesis and is linked to known/expected screens.
    """

    expected_output = (
        "JSON plan saved to artifacts/appflow_plan_<scenario_id>.json and updated "
        "screen_*.json/flow_*.json graph artifacts for the selected scenario."
    )

    return Task(
        name="Draft AppFlow plan",
        description=description,
        expected_output=expected_output,
        agent=agent,
    )


def plan_automation_sequence_task(agent, artifacts_dir: str) -> Task:
    description = f"""
    Build the global execution plan before YAML implementation starts.

    Planning protocol (mandatory):
    1) Read pending scenarios from qase_parser.
    2) Use state_tracker and app_flow_memory to summarize blockers/flaky hotspots.
    3) Enforce dependency gating: foundational/start flows first, deep flows later.
    4) Select exactly one scenario for this run and justify risk reduction.

    Save plan to `{artifacts_dir}/manager_plan.json` with:
    - prioritized_scenarios (ordered ids with rationale),
    - selected_scenario_id_for_this_run,
    - dependencies_or_known_blockers,
    - handoff_notes_for_appflow_and_automator.
    Keep output compact, explicit, and immediately executable by downstream agents.
    """

    expected_output = (
        "JSON plan saved to artifacts/manager_plan.json with scenario order, selected scenario, "
        "and handoff notes for execution agents."
    )

    return Task(
        name="Plan automation sequence",
        description=description,
        expected_output=expected_output,
        agent=agent,
    )


def automate_tests_task(agent, app_path: str, artifacts_dir: str, max_attempts: int) -> Task:
    description = f"""
    Automate exactly one pending scenario from qase_parser in this run.
    You MUST include all scenario items from that scenario in one consolidated YAML flow
    (do not split into multiple YAML files). Build flow in scenario-item order, run against `{app_path}`,
    and store artifacts under `{artifacts_dir}`.

    Execution protocol (mandatory):
    1) Read manager priorities from `{artifacts_dir}/manager_plan.json`.
    2) Read latest AppFlow plan in `{artifacts_dir}`; if item guidance is missing, request
       `recommended_start` from AppFlow specialist.
    3) Draft/update one scenario-level flow and call maestro_cli with
       `flow_scope: "scenario"` and `scenario_id`.
    4) After every draft/edit iteration, send full YAML to MaestroSenior and apply corrections
       before maestro_cli run.
    5) On failure, inspect `failure_context` (cause, recommendation, log excerpt,
       debug_context.ui_text_candidates, failed_selector, failed_step_index,
       last_successful_step_index, retry_from_step_index, navigation_context), share inline evidence with AppFlow,
       rewrite YAML from the failure point forward, and keep already validated prefix steps.
    6) After each attempt, record AppFlow observation: item id (`test_id`), scenario id, status,
       location_hint, failure_cause, screenshot_path, and confirmed flag. Use
       `navigation_context.current_screen` as location_hint (fallback `from_screen`), include full
       navigation JSON + artifacts in notes, and set confirmed=true only for passed attempts with
       existing screenshot evidence.

    Quality constraints:
    - Always capture screenshot on failed runs.
    - Never reuse unchanged selectors/commands that already failed.
    - If parsed step text language differs from app UI, trust visible UI labels/ids and
      ui_text_candidates over parser text.
    - Use guidance from `skills/maestro-test-writing/SKILL.md`.

    Retry loop runs until scenario pass or attempt limit reaches {max_attempts}.
    Mark unresolved scenario items as problematic.
    """

    expected_output = (
        "JSON summary saved to artifacts/automation_report.json that lists every scenario item id "
        "(test_id), "
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
        "problematic, and actionable insights for manual QA. Include links to screenshots/logs. "
        "Keep it brief and decision-oriented."
    )

    expected_output = "Markdown report stored at artifacts/summary.md"

    return Task(
        name="Summarize automation run",
        description=description,
        expected_output=expected_output,
        agent=agent,
    )
