# QA Automator

Automate mobile test cases defined in Qase by orchestrating [Maestro CLI](https://maestro.mobile.dev/) runs. The **qa-automator** CrewAI project ingests exported test-case JSON, previous execution state, and an application binary path, then iteratively converts each case into Maestro flows. Every test is given up to 10 automation attempts; unsuccessful cases are flagged for human follow-up.

## Features
- **4-agent CrewAI orchestration**: `Manager` (strategy), `AppFlow` (navigation context), `Automator` (YAML implementation), `MaestroSenior` (quality gate).
- **Dedicated AppFlow specialist** that maps scenario entry points using persistent memory (or heuristics when cold-starting) before automation runs.
- **Dedicated Automator agent** that writes/fixes Maestro YAML and runs retries based on failure diagnostics.
- **Mandatory MaestroSenior review** before every Maestro run to enforce YAML quality and deterministic assertions.
- **Screen inspector tool** that lets the AppFlow agent review stored screenshots plus UI hierarchies for each attempt.
- **Dynamic Maestro CLI runner** that builds per-test flows, executes them, and captures screenshots for state inspection.
- **State tracker** that records up to 10 attempts per test and marks problematic cases.
- **Artifact-aware**: Stores logs, flow files, and screenshots under `artifacts/` for debugging.

## Prerequisites
- Python 3.11+
- [Maestro CLI](https://maestro.mobile.dev/) installed and accessible via `$MAESTRO_BIN`.
- Android emulator or device connected (set `MAESTRO_DEVICE` when multiple targets exist).
- Qase export of test cases in JSON format.
- (Optional) Deep-link that bypasses onboarding, provided via `APP_SKIP_ONBOARDING_DEEPLINK`.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# Copy environment template
cp .env.example .env
# Fill in Maestro settings and optional tuning knobs.
# Load variables into the current shell session.
set -a
source .env
set +a

# Run the crew
PYTHONPATH=src python src/main.py \
  --test-cases ./samples/testcases.json \
  --tested ./samples/tested.json \
  --app-path ./apps/MyApp.apk
```

## Inputs
| Argument | Description |
| --- | --- |
| `--test-cases` | Path to JSON exported from Qase containing test steps. |
| `--tested` | JSON file describing already automated / executed cases. |
| `--app-path` | Path to the mobile application file (APK/IPA) under test. |
| `--output` | (Optional) Directory for logs and results. Defaults to `./artifacts`. |
| `--automated-dir` | (Optional) Directory for generated Maestro flows. Defaults to `./samples/automated`. |
| `--max-attempts` | Attempts per test before marking as problematic (default 10). |

## Environment variables
| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `MAESTRO_BIN` | No | `maestro` | Path or command name for Maestro CLI executable. |
| `MAESTRO_DEVICE` | No | _(empty)_ | Target device ID when multiple emulators/devices are connected. |
| `MAESTRO_APP_ID` | No | `default` | App ID inserted into generated Maestro YAML flows. |
| `APP_SKIP_ONBOARDING_DEEPLINK` | No | _(empty)_ | Deep-link opened before non-onboarding tests to skip onboarding. |
| `MAESTRO_SCREENSHOT_MAX_SIDE_PX` | No | `1440` | Max image side for captured screenshots before attaching to model context. |

### Where to set `scenarios.json` path

You do **not** pass a separate CLI argument for `scenarios.json`.

The file is generated automatically by `qase_parser` as:
- `<directory of --test-cases>/scenarios.json`

Example:
- if `--test-cases ./samples/testcases.json`, then scenarios are saved to `./samples/scenarios.json`.


### Onboarding awareness

Qase cases that include the tag **`onboarding`** (case-insensitive) are treated as onboarding\nflows. Every other case is preceded by a Maestro `open --url <DEEPLINK>` call using\n`APP_SKIP_ONBOARDING_DEEPLINK` so automation starts past onboarding screens. If a test\nstill needs onboarding but lacks the tag, add it in Qase or adjust `qase_parser` logic.

## Output
- `artifacts/current_scenario.json`: Snapshot of the next scenario all agents must focus on.
- `artifacts/manager_plan.json`: Global plan from `Manager` with scenario priority and handoff notes.
- `artifacts/appflow_plan_<scenario_id>.md`: AppFlow specialist's plan with per-case entry points.
- `artifacts/app_flow_memory/state.json`: Persistent AppFlow knowledge base (auto-snapshotted per run).
- `artifacts/debug_snapshots/<test_id>/attempt-<n>/`: Copies of Maestro hierarchy + context per attempt (consumed by screen_inspector).
- `artifacts/automation_report.json`: Execution summary with pass/fail/problem flags.
- `samples/automated/<test_id>.yaml`: Generated Maestro flows (or custom `--automated-dir` path).
- `artifacts/screenshots/<test_id>/attempt-*.png`: Captured screens when requested.
- `artifacts/logs/<test_id>-attempt-<n>.log`: Maestro stdout/stderr for each attempt.
- `<test-cases-dir>/scenarios.json`: Auto-generated end-to-end scenarios grouped from Qase cases.

## Automation workflow

1. **Parse inputs** – the QA Manager calls `qase_parser` which groups cases into scenarios and writes both `scenarios.json` and `artifacts/current_scenario.json`.
2. **Plan sequence** – `Manager` prioritizes scenarios, captures blockers/dependencies, and saves `artifacts/manager_plan.json` for downstream handoff.
3. **Plan navigation** – `AppFlow` reads the current scenario snapshot, queries `qase_parser`/`app_flow_memory`, inspects stored screenshots + hierarchies via `screen_inspector`, and emits `artifacts/appflow_plan_<scenario_id>.md`. Every hypothesis is persisted via `app_flow_memory.record_plan`, so `artifacts/app_flow_memory/state.json` grows on each run—even when memory starts empty.
4. **Automate with Maestro** – `Automator` uses manager + appflow plans, converts each case to Maestro YAML, delegates review to `MaestroSenior`, executes, and logs screenshots + stdout/stderr for each attempt.
5. **Inspect & retry** – on failures `Automator` studies `failure_context`, AppFlow hints, and screenshots, then rewrites YAML and re-runs (up to 10 attempts).
6. **Report** – `automation_report.json` and `summary.md` document final status plus artifact pointers for manual QA follow-up.

## Project structure
```
qa-automator/
├── .env.example
├── .gitignore
├── README.md
├── PROJECT.json
├── config.yaml
├── pyproject.toml
├── src/
│   ├── agents.py
│   ├── tasks.py
│   ├── policies.py
│   ├── main.py
│   └── tools/
│       ├── __init__.py
│       ├── maestro_tool.py
│       ├── qase_parser.py
│       └── state_tracker.py
└── tests/
    └── sample_inputs.md
```

## Development scripts
- `PYTHONPATH=src python -m compileall src` – syntax check.
- `pytest` (optional) – add tests under `tests/`.

## Limitations / TODOs
- Flow synthesis currently assumes Qase steps map 1:1 to Maestro commands; adapt the `MaestroFlowBuilder` for custom vocabularies.
- Extend `state_tracker` to persist history across sessions (e.g., SQLite).
- Add richer screenshot heuristics to detect app state transitions.

## License
MIT (update if different).
