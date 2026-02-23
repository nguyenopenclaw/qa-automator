---
name: maestro-test-writing
description: Writes compact, robust Maestro YAML flows from Qase-style test steps. Use when generating or fixing mobile UI automation flows, especially onboarding, personalization, slider/graph transitions, and Apple registration scenarios.
---

# Maestro Test Writing

## Purpose

Generate short, deterministic Maestro flows from natural-language test cases used in this repo.

## Use When

- User asks to create/repair Maestro YAML.
- Qase step text is verbose or multiline and must be converted into valid commands.
- Flow fails due to flaky waits, ambiguous selectors, or invalid YAML structure.

## Minimal Workflow

1. Convert each human step into explicit Maestro commands.
2. Add synchronization before assertions (`assertVisible`, optional `extendedWaitUntil`).
3. Keep selectors stable: prefer IDs/text that are unlikely to change.
4. For uncertain UI state, capture evidence with `takeScreenshot`.
5. Keep flow small; split complex parts with `runFlow` if needed.

## Command Set (Default)

Prefer this small subset first:

- `launchApp`
- `tapOn`
- `inputText`
- `scrollUntilVisible`
- `assertVisible`
- `assertNotVisible`
- `waitForAnimationToEnd`
- `extendedWaitUntil`
- `takeScreenshot`
- `runFlow`

Use advanced/system commands only when required by scenario.

## Translation Rules (Qase -> Maestro)

- Never copy raw prose lines directly as YAML keys.
- A natural-language action must become one or more valid commands.
- Multiline Qase text: split into atomic actions.
- Expected result becomes assertions (`assertVisible` / `assertNotVisible`).
- If precondition says "after onboarding", include explicit navigation/setup commands.

## Stability Rules

- Start each flow with app setup:
  - `launchApp` (use `clearState` only if scenario requires clean install behavior).
- Add one sync point before first key assertion:
  - `waitForAnimationToEnd` or `extendedWaitUntil`.
- For lists/carousels:
  - use `scrollUntilVisible` instead of repeated blind swipes.
- For flaky steps:
  - wrap block with `retry`.

## Scenario Patterns

### 1) Onboarding screen transition

```yaml
appId: com.example.app
---
- launchApp
- tapOn: "Continue"
- waitForAnimationToEnd
- assertVisible: "Where do you publish content?"
```

### 2) Slider -> graph transition

```yaml
appId: com.example.app
---
- assertVisible: "How much do you earn from your content?"
- tapOn: "Continue"
- extendedWaitUntil:
    visible: "Graph"
    timeout: 8000
- takeScreenshot: "graph-screen"
```

### 3) Apple registration sheet (system UI sensitive)

```yaml
appId: com.example.app
---
- tapOn: "Sign up with Apple"
- extendedWaitUntil:
    visible: "Continue with Apple"
    timeout: 8000
- takeScreenshot: "apple-sheet-opened"
```

If system sheet is not stable in CI/device, mark as environment-limited and keep assertions minimal but explicit.

## Definition of Done

- YAML is valid Maestro syntax.
- No raw Qase prose left as command keys.
- Each expected result has at least one assertion.
- At least one synchronization point exists around navigation-heavy transitions.
- Failure path captures screenshot evidence.

## Additional Reference

- Full Maestro command reference: https://docs.maestro.dev/reference/commands-available

