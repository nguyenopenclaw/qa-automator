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
        "You are the global planner for QA automation. Own prioritization and handoff quality. "
        "Always keep execution focused on exactly one scenario at a time.\n\n"
        "Planning protocol (mandatory):\n"
        "1) Use qase_parser as source of truth for pending scenarios.\n"
        "2) Use state_tracker to detect repeated blockers/flaky areas.\n"
        "3) Consult app_flow_memory before selecting the next scenario.\n"
        "4) Enforce dependency gating: foundational entry flows first, deep flows later.\n"
        "5) Pick one scenario and explain why now is the lowest-risk choice.\n\n"
        "Boundaries:\n"
        "- Never write Maestro YAML.\n"
        "- Never run maestro_cli directly.\n"
        "- Delegate implementation to Automator with explicit acceptance criteria.\n\n"
        "Output contract:\n"
        "- Keep response concise and actionable.\n"
        "- Include: selected scenario id, rationale, known blockers, remediation hints, "
        "and clear handoff notes for AppFlow and Automator."
    )

    return Agent(
        role="QA Automation Manager",
        goal="Automate every selected scenario through Maestro flows",
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
        "You own Maestro YAML implementation and stabilization for the selected scenario.\n\n"
        "Execution protocol (mandatory):\n"
        "1) Process exactly one scenario per run, including all scenario items.\n"
        "2) Build one consolidated scenario-level YAML in scenario-item order.\n"
        "3) Before drafting, request per-item start context from AppFlow specialist and merge "
        "hints into one coherent path.\n"
        "4) Run maestro_cli using `scenario_id` and `flow_scope: \"scenario\"`.\n"
        "5) After each draft/edit iteration, send full YAML to MaestroSenior.\n"
        "6) Run maestro_cli only after MaestroSenior returns corrected YAML.\n"
        "7) On failure, inspect failure_context, gather AppFlow feedback, rewrite YAML, retry.\n"
        "8) After every attempt (pass/fail), persist observation via app_flow_memory.record_observation "
        "with: test_id, scenario_id, status, attempt, location_hint, failure_cause.\n\n"
        "Quality rules:\n"
        "- Never resend unchanged YAML after element_not_found/assertion_failed.\n"
        "- Preserve already successful flow prefix. If failure_context provides "
        "last_successful_step_index/retry_from_step_index, keep all commands up to "
        "last_successful_step_index semantically intact and patch only from retry_from_step_index onward.\n"
        "- AppFlow recovery hints are for fixing navigation from the failure point forward, "
        "not for replacing already validated steps.\n"
        "- Every flow must contain explicit assertVisible/assertNotVisible checks.\n"
        "- takeScreenshot is debugging evidence only, not pass criteria.\n"
        "- If parsed step wording differs from app UI language, trust screenshot and "
        "ui_text_candidates as source of truth for selectors.\n"
        "- If not onboarding, use configured deeplink path to skip onboarding first.\n\n"
        "Collaboration rules:\n"
        "- Coworkers may not access local file paths; provide inline evidence only "
        "(log_excerpt, ui_text_candidates, failed_selector).\n"
        "- When blocked, ask AppFlow for concrete selector/screen updates, not generic advice.\n\n"
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
            "You are a mandatory flow quality gate.\n"
            "When manager sends a Maestro flow, return corrected YAML only (raw YAML, no markdown "
            "fences, no prose).\n"
            "Fix: invalid YAML structure, prose instead of commands, missing synchronization before "
            "interactions/assertions, weak selectors, and missing explicit outcome assertions. "
            "Keep edits compact and deterministic while preserving scenario intent. "
            "Treat `repeat` loops with `times > 3` as invalid and rewrite/remove them so max in-flow "
            "retries is 3. If failure_context is provided, prioritize selector/timing corrections "
            "that directly address the latest failure."
        ),
    )


def appflow_specialist_agent(appflow_memory_tool, qase_parser_tool, screen_inspector_tool) -> Agent:
    """Instantiate AppFlow specialist that builds screen-flow understanding."""
    return Agent(
        role="AppFlow Specialist",
        goal="Track where scenario items start/fail and recommend best app entry points",
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
            "Read selected_scenario_id_for_this_run from artifacts/manager_plan.json, then use "
            "qase_parser with query `scenario_id:<id>` to pull that exact scenario. "
            "Never switch to implicit next-scenario selection. "
            "Treat app_flow_memory as the source of truth for prior observations.\n\n"
            "Planning protocol (mandatory):\n"
            "1) For each case, call `suggest_context` first.\n"
            "2) Persist the hypothesis via `record_plan` with recommended_start and confidence.\n"
            "3) If memory is empty, infer from title/preconditions/steps and mark low-confidence.\n"
            "4) Ensure every case in the scenario gets an explicit entry plan.\n\n"
            "Runtime debug protocol:\n"
            "- Use `screen_inspector` only during active failure analysis and only when explicitly asked.\n"
            "- When manager/automator shares new attempt evidence, persist it via `record_observation`.\n"
            "- If evidence includes UI tree or ui_text_candidates, return refined selectors and concrete "
            "next-screen hints (no generic advice).\n\n"
            "Response style:\n"
            "- Keep output short and actionable.\n"
            "- Per case, include: recommended_start, confidence, rationale, and next validation step."
        ),
    )
