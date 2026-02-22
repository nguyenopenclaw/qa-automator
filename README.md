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

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# Copy environment template
cp .env.example .env
# Fill in Qase project info, Maestro settings, and optional tuning knobs.

# Run the crew
PYTHONPATH=src python src/main.py \
  --test-cases ./samples/testcases.json \
  --tested ./samples/tested.json \
  --app ./apps/MyApp.apk
```

## Inputs
| Argument | Description |
| --- | --- |
| `--test-cases` | Path to JSON exported from Qase containing test steps. |
| `--tested` | JSON file describing already automated / executed cases. |
| `--app` | Path to the mobile application file (APK/IPA) under test. |
| `--output` | (Optional) Directory for logs and results. Defaults to `./artifacts`. |

## Output
- `artifacts/automation_report.json`: Execution summary with pass/fail/problem flags.
- `artifacts/flows/<test_id>.yaml`: Generated Maestro flows.
- `artifacts/screenshots/<test_id>/attempt-*.png`: Captured screens when requested.

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
