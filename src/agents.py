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


def qa_manager_agent(maestro_tool, qase_parser_tool, state_tracker_tool) -> Agent:
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
        "Respect the retry ceiling "
        "(10 attempts unless overridden), request screenshots when state is ambiguous, and "
        "clearly mark unresolved cases as problematic.\n\n"
        "Each run processes exactly one scenario but all its cases. Always pass `scenario_id` "
        "in maestro_cli payload so tooling can perform clean app reinstall per scenario. "
        "Before drafting a flow for each case, delegate to AppFlow specialist and ask for "
        "recommended start context/screen for that case. After each attempt (pass/fail), "
        "send the observed location and failure cause back to AppFlow so its flow map "
        "improves across runs.\n\n"
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
        tools=[maestro_tool, qase_parser_tool, state_tracker_tool],
        instructions=instructions,
    )


def appflow_specialist_agent(appflow_memory_tool) -> Agent:
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
        tools=[appflow_memory_tool],
        instructions=(
            "Use app_flow_memory as source of truth. When asked for a case, first call "
            "`suggest_context` and return a concise recommendation: starting screen, confidence, "
            "and rationale. When manager provides new attempt observations, call "
            "`record_observation` to persist them. Keep responses short and actionable."
        ),
    )
