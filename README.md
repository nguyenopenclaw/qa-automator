# QA Automator

Automate mobile test cases defined in Qase by orchestrating [Maestro CLI](https://maestro.mobile.dev/) runs. The **qa-automator** CrewAI project ingests exported test-case JSON, previous execution state, and an application binary path, then iteratively converts each case into Maestro flows. Every test is given up to 10 automation attempts; unsuccessful cases are flagged for human follow-up.

## Features
- **CrewAI orchestration** with a manager agent coordinating preparation, execution, and reporting tasks.
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
- `artifacts/automation_report.json`: Execution summary with pass/fail/problem flags.
- `samples/automated/<test_id>.yaml`: Generated Maestro flows (or custom `--automated-dir` path).
- `artifacts/screenshots/<test_id>/attempt-*.png`: Captured screens when requested.
- `artifacts/logs/<test_id>-attempt-<n>.log`: Maestro stdout/stderr for each attempt.
- `<test-cases-dir>/scenarios.json`: Auto-generated end-to-end scenarios grouped from Qase cases.

## Automation workflow

1. The agent performs a naive transformation from each Qase case into a Maestro flow.
2. It executes the flow immediately via Maestro CLI.
3. On any failure it captures a screenshot (and logs stdout/stderr) to inspect the current UI state before making adjustments or flagging the case.
4. Up to 10 attempts are made per case; unresolved items are surfaced as problematic with pointers to the collected artifacts.

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
