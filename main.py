import argparse
import os

import memory
import planner
import dataclasses
import memory

def clear_terminal(enabled: bool) -> None:
    if enabled:
        os.system("cls" if os.name == "nt" else "clear")


def summarize_observation(observation: memory.Observation) -> list[str]:
    interesting = [
        tile
        for tile in observation.visible_tiles
        if tile.tile_type not in {"empty", "agent"}
    ]
    if not interesting:
        return ["No salient objects nearby."]
    return [
        f"{tile.tile_type} at rel={tile.relative_position} abs={tile.absolute_position}: {tile.description}"
        for tile in interesting
    ]

def format_memory(memory: memory.Memory) -> list[str]:
    beliefs = ", ".join(f"{name}@{pos}" for name, pos in sorted(memory.object_beliefs.items()))
    if not beliefs:
        beliefs = "none yet"
    return [
        f"Known tiles: {len(memory.known_world)} | Explored positions: {len(memory.explored)}",
        f"Object beliefs: {beliefs}",
    ]


def render_turn(
    world: memory.GridWorld,
    memory: memory.Memory,
    observation: memory.Observation,
    plan: memory.Plan,
    decision: dict[str, str],
    result: memory.ActionResult,
    reflection: str | None,
    debug_full_map: bool,
) -> str:
    sections = [
        "Cognitive Gridworld",
        "=" * 72,
        f"Turn {observation.turn:02d} | Position {observation.position} | Energy {observation.energy} | Inventory {sorted(observation.inventory) or ['empty']}",
        "",
        "Known Map" + (" (debug full map)" if debug_full_map else " (agent memory only)"),
        world.render(memory, debug_full_map=debug_full_map),
        "",
        "Observation",
        *[f"- {line}" for line in summarize_observation(observation)],
        "",
        "Memory",
        *[f"- {line}" for line in format_memory(memory)],
        "",
        "Hypotheses",
        *[f"- {item}" for item in (memory.hypotheses or ["No hypotheses formed yet."])],
        "",
        "Plan",
        f"- Goal: {plan.current_goal}",
        f"- Subgoal: {plan.subgoal}",
        f"- Reason: {plan.reason}",
        "",
        "Structured Decision",
        f"- current_goal: {decision['current_goal']}",
        f"- reasoning: {decision['reasoning']}",
        f"- action: {decision['action']}",
        "",
        "Action Result",
        f"- {'success' if result.success else 'failure'}: {result.message}",
    ]
    if reflection:
        sections.extend(["", "Reflection", f"- {reflection}"])
    if observation.messages:
        sections.extend(["", "Environment Events"])
        sections.extend(f"- {message}" for message in observation.messages)
    return "\n".join(sections)


def execute_decision(world: memory.GridWorld, decision: dict[str, str]) -> tuple[memory.Action | None, memory.ActionResult]:
    try:
        action = memory.Action(decision["action"])
    except (KeyError, ValueError):
        return None, memory.ActionResult(False, f"Invalid action emitted: {decision.get('action')!r}")
    return action, world.step(action)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Cognitive Gridworld terminal demo.")
    parser.add_argument("--max-turns", type=int, default=60, help="Maximum number of turns to run.")
    parser.add_argument("--delay", type=float, default=0.15, help="Delay between turns in seconds.")
    parser.add_argument("--debug-full-map", action="store_true", help="Render the true full map.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal each turn.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    success = run_demo(
        max_turns=args.max_turns,
        delay=args.delay,
        debug_full_map=args.debug_full_map,
        no_clear=args.no_clear,
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
