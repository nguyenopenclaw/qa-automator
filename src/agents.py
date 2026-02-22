"""Agent factory for the QA Automator crew."""
from __future__ import annotations

from crewai import Agent


def qa_manager_agent(maestro_tool, qase_parser_tool, state_tracker_tool) -> Agent:
    """Instantiate the manager agent that drives all tasks."""
    instructions = (
        "You orchestrate end-to-end automation of Qase test cases via Maestro CLI. "
        "Respect the retry ceiling (10 attempts unless overridden), request screenshots "
        "when state is ambiguous, and clearly mark unresolved cases as problematic."
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
