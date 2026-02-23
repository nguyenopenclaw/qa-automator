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


def qa_manager_agent(qase_parser_tool, state_tracker_tool, appflow_memory_tool) -> Agent:
    """Instantiate strategic manager that plans automation order."""
    instructions = (
        "You are the global planner for QA automation. Build the scenario queue, define "
        "execution order, and keep the run focused on one scenario at a time. "
        "Use qase_parser as source of truth for pending scenarios and choose the next best "
        "scenario by balancing risk, dependencies, and blocked history from state_tracker. "
        "Produce a concise execution plan that explains why this scenario is next and what "
        "signals indicate readiness to proceed. "
        "Do not write Maestro YAML and do not run maestro_cli directly: delegate implementation "
        "to Automator. Ensure AppFlow and Automator receive clear intent, scenario priority, "
        "and acceptance expectations. "
        "When there are repeated blockers, mark them explicitly with remediation hints so "
        "Automator can either retry with targeted probes or escalate as problematic."
    )

    return Agent(
        role="QA Automation Manager",
        goal="Automate every provided Qase test case through Maestro flows",
        backstory=(
            "Seasoned QA lead who thinks in execution strategy, scenario dependencies, "
            "and delivery sequencing across large regression sets."
        ),
        allow_delegation=True,
        verbose=True,
        memory=True,
        max_iter=30,
        tools=[qase_parser_tool, state_tracker_tool, appflow_memory_tool],
        instructions=instructions,
    )


def automator_agent(maestro_tool, qase_parser_tool, state_tracker_tool, appflow_memory_tool) -> Agent:
    """Instantiate execution specialist that writes and fixes Maestro YAML."""
    maestro_skill = _load_maestro_skill()
    instructions = (
        "You own YAML implementation for Maestro tests. For each selected scenario and case, "
        "write flows, execute them, inspect failures, and iteratively fix commands/selectors. "
        "If a test is not flagged as onboarding, first trigger the configured deeplink to bypass "
        "onboarding screens. Start with a naive Maestro translation, run it, and when it fails, "
        "inspect returned log excerpts, failure diagnostics, and screenshots to reason about state "
        "before adjusting steps. When testcase wording/selectors are in a different language than "
        "the app UI, treat screenshot and ui_text_candidates as source of truth and map steps to "
        "actual on-screen English selectors before retrying. "
        "You MUST actively rewrite flow between retries: never resend unchanged YAML after "
        "element_not_found/assertion_failed. Prefer maestro_cli payloads with updated `flow_yaml`.\n\n"
        "Never finalize a testcase using screenshot-only evidence: every flow must include explicit "
        "assertVisible/assertNotVisible checks validating expected results. Use takeScreenshot only "
        "as debugging evidence.\n\n"
        "Each run processes exactly one scenario but all its cases. Always pass `scenario_id` in "
        "maestro_cli payload so tooling can perform clean app reinstall per scenario. Before drafting "
        "flow for each case, delegate to AppFlow specialist for recommended start context/screen. "
        "After every attempt (pass/fail), persist outcome via app_flow_memory.record_observation "
        "with: test_id, scenario_id, status, attempt, location_hint, and failure_cause. IMPORTANT: "
        "coworkers may not access local filesystem paths. When asking AppFlow specialist for help, "
        "include inline evidence from failure_context (log_excerpt + "
        "debug_context.ui_text_candidates + failed_selector), not local artifact paths.\n\n"
        "After EACH flow edit iteration (initial draft and every retry), delegate to MaestroSenior "
        "with full current YAML and latest failure context (if any). Only run maestro_cli after "
        "MaestroSenior returns corrected YAML. This review is a mandatory quality gate.\n\n"
        "For writing and fixing Maestro flows, you MUST follow this project skill:\n"
        f"{maestro_skill}"
    )

    return Agent(
        role="Automator",
        goal="Implement and stabilize Maestro YAML flows for selected scenarios",
        backstory=(
            "Hands-on mobile automation engineer focused on turning scenario intent into "
            "reliable Maestro YAML, with rapid failure-driven iteration."
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
