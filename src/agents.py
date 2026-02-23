"""Agent factory for the QA Automator crew."""
from __future__ import annotations

from pathlib import Path

from crewai import Agent


def _load_maestro_skill() -> str:
    """Load Maestro-writing skill text so agent follows project guidance."""
    skill_path = Path("skills/maestro-test-writing/SKILL.md")
    if not skill_path.exists():
        return (
            "Skill file skills/maestro-test-writing/SKILL.md is missing. "
            "Use deterministic Maestro commands, add synchronization before "
            "assertions, and convert prose steps into valid YAML commands."
        )
    return skill_path.read_text(encoding="utf-8").strip()


def qa_manager_agent(maestro_tool, qase_parser_tool, state_tracker_tool, appflow_memory_tool) -> Agent:
    """Instantiate the manager agent that drives all tasks."""
    maestro_skill = _load_maestro_skill()
    instructions = (
        "You orchestrate end-to-end automation of Qase test cases via Maestro CLI. "
        "Always work scenario-by-scenario: request only one pending scenario from qase_parser "
        "and finish it before moving to another. "
        "If a test is not flagged as onboarding, first trigger the configured deeplink "
        "to bypass onboarding screens. Start with a naive Maestro translation, run it, "
        "and when it fails, inspect returned log excerpts, failure diagnostics, and screenshots "
        "to reason about the state before adjusting steps. When testcase wording/selectors are in "
        "a different language than the app UI, treat the screenshot as source of truth and map "
        "steps to the actual on-screen English selector text/labels before retrying. "
        "You MUST actively rewrite flow between retries: never resend an unchanged flow after "
        "element_not_found/assertion_failed. Prefer sending maestro_cli payload with `flow_yaml` "
        "that contains the updated commands for this attempt. "
        "Respect the retry ceiling "
        "(10 attempts unless overridden), request screenshots when state is ambiguous, and "
        "clearly mark unresolved cases as problematic.\n\n"
        "Never accept or finalize a testcase using screenshot-only evidence: "
        "every testcase flow must include explicit assertVisible/assertNotVisible checks "
        "that validate expected results from the scenario steps. "
        "Use takeScreenshot only as debugging evidence, not as a pass criterion.\n\n"
        "Each run processes exactly one scenario but all its cases. Always pass `scenario_id` "
        "in maestro_cli payload so tooling can perform clean app reinstall per scenario. "
        "Before drafting a flow for each case, delegate to AppFlow specialist and ask for "
        "recommended start context/screen for that case. After every attempt (pass/fail), "
        "you MUST persist outcome via app_flow_memory.record_observation with: test_id, "
        "scenario_id, status, attempt, location_hint, and failure_cause so the flow map "
        "improves across runs. IMPORTANT: coworkers may not access local filesystem paths. "
        "When asking AppFlow specialist for help, include inline evidence from "
        "`failure_context` (log_excerpt + debug_context.ui_text_candidates + failed_selector) "
        "directly in your question; do NOT ask the specialist to open local artifact paths.\n\n"
        "After EACH flow edit iteration (initial draft and every retry), delegate to "
        "MaestroSenior with the full current YAML and latest failure context (if any). "
        "Only run maestro_cli after MaestroSenior returns a corrected YAML. Treat this "
        "as mandatory quality gate to remove obvious YAML/synchronization/selector mistakes.\n\n"
        "For writing and fixing Maestro flows, you MUST follow this project skill:\n"
        f"{maestro_skill}"
    )

    return Agent(
        role="QA Automation Manager",
        goal="Automate every provided Qase test case through Maestro flows",
        backstory=(
            "Seasoned mobile QA lead comfortable turning high-level test cases into deterministic "
            "automation flows. Expert at documenting residual risk when scripts stay flaky."
        ),
        allow_delegation=True,
        verbose=True,
        memory=True,
        max_iter=30,
        tools=[maestro_tool, qase_parser_tool, state_tracker_tool, appflow_memory_tool],
        instructions=instructions,
    )


def maestro_senior_agent() -> Agent:
    """Instantiate senior reviewer that improves Maestro YAML quality."""
    return Agent(
        role="MaestroSenior",
        goal="Harden Maestro YAML flows and remove obvious mistakes before execution",
        backstory=(
            "Principal mobile automation engineer specializing in resilient Maestro flows, "
            "strict YAML correctness, and flaky-selector mitigation."
        ),
        allow_delegation=False,
        verbose=True,
        memory=True,
        max_iter=20,
        tools=[],
        instructions=(
            "You act as a mandatory flow quality gate. When manager sends a Maestro flow, "
            "return corrected YAML only. Remove obvious errors: invalid YAML structure, raw prose "
            "instead of commands, missing synchronization around transitions, weak/ambiguous "
            "selectors, and missing explicit assertions for expected outcome. Keep fixes compact "
            "and deterministic; preserve testcase intent. If failure_context is provided, use it "
            "to adjust selectors/timing before next run."
        ),
    )


def appflow_specialist_agent(appflow_memory_tool, qase_parser_tool, screen_inspector_tool) -> Agent:
    """Instantiate AppFlow specialist that builds screen-flow understanding."""
    return Agent(
        role="AppFlow Specialist",
        goal="Track where test cases start/fail and recommend best app entry points",
        backstory=(
            "Navigation-focused QA analyst who maintains a persistent map of app screens "
            "and scenario entry points from previous automation attempts."
        ),
        allow_delegation=False,
        verbose=True,
        memory=True,
        max_iter=20,
        tools=[appflow_memory_tool, qase_parser_tool, screen_inspector_tool],
        instructions=(
            "Use qase_parser to pull the scenario highlighted in artifacts/current_scenario.json, "
            "then rely on app_flow_memory as the source of truth for prior observations. If the "
            "memory is empty for a case, still draft a recommendation using heuristics from the "
            "test title, preconditions, and steps, and flag it as low-confidence so the manager "
            "knows this is a discovery attempt. For a given case, first call `suggest_context`, then "
            "persist that hypothesis via `record_plan` so the knowledge base captures the start guess "
            "and confidence for this run. Use `screen_inspector` only for in-run debug when the manager "
            "provides fresh failure evidence from an active attempt and explicitly asks for screen-level "
            "analysis; never call it during pre-run planning. When the manager provides new attempt "
            "observations, call `record_observation` to persist them. Keep responses short, actionable, "
            "and make sure every case in the scenario has an explicit entry plan—even if that plan is a "
            "hypothesis that needs validation."
        ),
    )
