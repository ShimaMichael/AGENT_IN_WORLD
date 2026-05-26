# Cognitive Gridworld

Cognitive Gridworld is an experimental AI agent framework that places an autonomous agent inside a partially observable grid environment. The agent must perceive only nearby tiles, maintain memory, form hypotheses, plan toward long-horizon goals, reflect on failures, and act through a constrained action interface.

The environment is intentionally simple. The interesting behavior comes from limited perception, imperfect information, persistent memory, hierarchical planning, and dynamic world events.

## Features

- **Partial observability**: the agent receives only local observations, not the full map.
- **Internal world model**: memory tracks known tiles, discovered objects, explored regions, failed attempts, hypotheses, and reflections.
- **Hierarchical planning**: goals are decomposed into subgoals such as finding the keycard, unlocking the security door, and reaching the exit.
- **Reflection loop**: repeated failures, loops, and inefficient patterns are detected and stored as strategy notes.
- **Dynamic environment**: environmental messages and changing door state create a non-static task setting.
- **Structured decisions**: every decision uses a JSON-like shape with `current_goal`, `reasoning`, and `action`.
- **LLM-ready interface**: policy code can be swapped for an external model provider while preserving the same structured action contract.

## Scenario

The default map is a small research facility:

```text
#############
#A....#....E#
#..##.#.##..#
#..K..D.....#
#############
```

Legend:

- `A`: agent start
- `K`: keycard
- `D`: locked security door
- `E`: exit
- `#`: wall

The agent must explore, locate the keycard, unlock the door, and escape through the exit.

## Run

```bash
python3 main.py
```

Useful options:

```bash
python3 main.py --no-clear --delay 0
python3 main.py --debug-full-map
python3 main.py --max-turns 80
```

## Project Structure

- `main.py`: terminal runner, trace rendering, decision execution loop
- `gridworld.py`: environment, partial observations, action execution, dynamic events
- `memory.py`: agent memory, beliefs, hypotheses, and action history
- `planner.py`: planning, reflection, policy interface, and LLM integration placeholder
- `project_dataclasses.py`: shared action, observation, result, and plan types

## Decision Interface

The agent emits structured decisions:

```python
{
    "current_goal": "Find the keycard",
    "reasoning": "Eastern corridor remains unexplored.",
    "action": "MOVE_EAST",
}
```

Valid actions:

```text
MOVE_NORTH
MOVE_SOUTH
MOVE_EAST
MOVE_WEST
PICK_UP
OPEN_DOOR
SCAN_AREA
WAIT
```

This keeps the action space interpretable and makes it straightforward to connect an LLM provider later.
