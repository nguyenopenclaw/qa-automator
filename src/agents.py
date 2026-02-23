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
        "to reason about the state before adjusting steps. Respect the retry ceiling "
        "(10 attempts unless overridden), request screenshots when state is ambiguous, and "
        "clearly mark unresolved cases as problematic.\n\n"
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
        allow_delegation=False,
        verbose=True,
        memory=True,
        max_iter=30,
        tools=[maestro_tool, qase_parser_tool, state_tracker_tool],
        instructions=instructions,
    )
