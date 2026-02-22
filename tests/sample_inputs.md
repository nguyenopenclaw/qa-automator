# Sample inputs

Create placeholder files to experiment with the CLI:

```jsonc
// samples/testcases.json
[
  {
    "id": "TC-1",
    "title": "Login succeeds",
    "priority": "high",
    "steps": [
      {"action": "tapOn", "payload": "#username"},
      {"action": "inputText", "payload": "demo@example.com"},
      {"action": "tapOn", "payload": "#password"},
      {"action": "inputText", "payload": "hunter2"},
      {"action": "tapOn", "payload": "Login"}
    ]
  }
]
```

```json
// samples/tested.json
["TC-0"]
```
