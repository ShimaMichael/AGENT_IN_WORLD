import argparse
import os
import time

from gridworld import GridWorld
from memory import Memory
from planner import LLMPolicy, LocalPolicy, Planner, ReflectionEngine
from project_dataclasses import Action, ActionResult, Observation, Plan

def clear_terminal(enabled: bool) -> None:
    if enabled:
        os.system("cls" if os.name == "nt" else "clear")


def summarize_observation(observation: Observation) -> list[str]:
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

def format_memory(agent_memory: Memory) -> list[str]:
    beliefs = ", ".join(f"{name}@{pos}" for name, pos in sorted(agent_memory.object_beliefs.items()))
    if not beliefs:
        beliefs = "none yet"
    return [
        f"Known tiles: {len(agent_memory.known_world)} | Explored positions: {len(agent_memory.explored)}",
        f"Object beliefs: {beliefs}",
    ]


def render_turn(
    world: GridWorld,
    memory: Memory,
    observation: Observation,
    plan: Plan,
    decision: dict[str, str],
    result: ActionResult,
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


def execute_decision(world: GridWorld, decision: dict[str, str]) -> tuple[Action | None, ActionResult]:
    try:
        action = Action(decision["action"])
    except (KeyError, ValueError):
        return None, ActionResult(False, f"Invalid action emitted: {decision.get('action')!r}")
    return action, world.step(action)


def run_demo(
    max_turns: int,
    delay: float,
    debug_full_map: bool,
    no_clear: bool,
    policy_name: str,
    gemini_model: str | None,
) -> bool:
    world = GridWorld()
    agent_memory = Memory()
    agent_planner = Planner()
    policy = LLMPolicy(model=gemini_model) if policy_name == "gemini" else LocalPolicy()
    reflection_engine = ReflectionEngine()

    final_result = ActionResult(False, "Maximum turns reached.")
    for _ in range(max_turns):
        observation = world.observe()
        agent_memory.update_from_observation(observation)
        plan = agent_planner.make_plan(observation, agent_memory)
        decision = policy.decide(observation, agent_memory, plan)
        action, result = execute_decision(world, decision)
        if action is not None:
            agent_memory.record_action(action, result)
        reflection = reflection_engine.reflect(agent_memory, result)
        if reflection:
            agent_memory.add_reflection(reflection)

        clear_terminal(not no_clear)
        print(
            render_turn(
                world=world,
                memory=agent_memory,
                observation=observation,
                plan=plan,
                decision=decision,
                result=result,
                reflection=reflection,
                debug_full_map=debug_full_map,
            )
        )
        final_result = result
        if result.done:
            break
        if delay > 0:
            time.sleep(delay)

    print("\nFinal reflections:")
    for item in agent_memory.reflections or ["No major strategic corrections were needed."]:
        print(f"- {item}")
    print(f"\nOutcome: {final_result.message}")
    return final_result.success and final_result.done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Cognitive Gridworld terminal demo.")
    parser.add_argument("--max-turns", type=int, default=60, help="Maximum number of turns to run.")
    parser.add_argument("--delay", type=float, default=0.15, help="Delay between turns in seconds.")
    parser.add_argument("--debug-full-map", action="store_true", help="Render the true full map.")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the terminal each turn.")
    parser.add_argument(
        "--policy",
        choices=["gemini", "local"],
        default="gemini",
        help="Decision policy to use. Gemini requires GEMINI_API_KEY.",
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help="Gemini model for --policy gemini. Defaults to GEMINI_MODEL or gemini-3.5-flash.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.policy == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY is required for --policy gemini. Use --policy local for offline runs.")
    success = run_demo(
        max_turns=args.max_turns,
        delay=args.delay,
        debug_full_map=args.debug_full_map,
        no_clear=args.no_clear,
        policy_name=args.policy,
        gemini_model=args.gemini_model,
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
