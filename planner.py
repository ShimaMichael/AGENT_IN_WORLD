from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Iterable

from memory import Memory
from project_dataclasses import Action, ActionResult, MOVE_DELTAS, Observation, Plan, Position


class Planner:
    def make_plan(self, observation: Observation, memory: Memory) -> Plan:
        if "keycard" not in observation.inventory:
            target = memory.object_beliefs.get("keycard") or memory.object_beliefs.get("possible_keycard")
            if target:
                return Plan(
                    current_goal="Retrieve keycard",
                    subgoal=f"Navigate to suspected keycard at {target}",
                    target=target,
                    reason="A metallic object/keycard has been observed and is required for the locked door.",
                )
            return Plan(
                current_goal="Find the keycard",
                subgoal="Explore unknown reachable space",
                target=None,
                reason="No confirmed keycard location is known yet.",
            )

        door = memory.object_beliefs.get("locked_door")
        if door and self._distance(observation.position, door) <= 1:
            return Plan(
                current_goal="Unlock security door",
                subgoal="Use keycard on adjacent locked door",
                target=door,
                reason="The agent has the keycard and is next to the locked door.",
            )

        exit_position = memory.object_beliefs.get("exit")
        if exit_position:
            return Plan(
                current_goal="Reach extraction point",
                subgoal=f"Navigate to exit at {exit_position}",
                target=exit_position,
                reason="The keycard is secured and the exit has been discovered.",
            )

        if door:
            return Plan(
                current_goal="Reach and unlock security door",
                subgoal=f"Navigate back to locked door at {door}",
                target=door,
                reason="The door is known and must be opened before escape.",
            )

        return Plan(
            current_goal="Locate extraction route",
            subgoal="Explore beyond known corridors",
            target=None,
            reason="The keycard is secured, but the door or exit location is not fully known.",
        )

    def _distance(self, left: Position, right: Position) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])


class ReflectionEngine:
    def reflect(self, memory: Memory, result: ActionResult) -> str | None:
        if not result.success:
            if "wall" in result.message:
                return "A blocked movement updated the map; avoid treating walls as routes."
            if "locked door" in result.message.lower() and "keycard" not in result.message.lower():
                return "The locked door remains a dependency; prioritize finding authorization."
            if "No useful object" in result.message:
                return "Picking up without a confirmed object wastes time; inspect object locations first."

        if len(memory.recent_positions) == memory.recent_positions.maxlen:
            counts = Counter(memory.recent_positions)
            position, visits = counts.most_common(1)[0]
            if visits >= 4:
                return f"Repeated visits around {position} suggest a loop; prefer frontier exploration."

        if len(memory.recent_actions) == memory.recent_actions.maxlen:
            action, repeats = Counter(memory.recent_actions).most_common(1)[0]
            if repeats >= 5:
                return f"Action pattern {action.value} is repetitive; choose a different route if possible."

        return None


class LocalPolicy:
    def decide(self, observation: Observation, memory: Memory, plan: Plan) -> dict[str, str]:
        if plan.current_goal == "Unlock security door":
            action = Action.OPEN_DOOR
            reasoning = "The keycard is available and the door is adjacent."
        elif self._standing_on_keycard(observation, memory):
            action = Action.PICK_UP
            reasoning = "The keycard is at the current position."
        elif plan.target:
            action = self._next_step_toward(observation.position, plan.target, memory)
            reasoning = f"Move along known passable tiles toward {plan.target}."
        else:
            action = self._explore_frontier(observation.position, memory)
            reasoning = "Choose a reachable frontier adjacent to unknown space."

        return {
            "current_goal": plan.current_goal,
            "reasoning": reasoning,
            "action": action.value,
        }

    def _standing_on_keycard(self, observation: Observation, memory: Memory) -> bool:
        if "keycard" in observation.inventory:
            return False
        if memory.object_beliefs.get("keycard") == observation.position:
            return True
        return any(
            tile.relative_position == (0, 0) and tile.tile_type == "keycard"
            for tile in observation.visible_tiles
        )

    def _next_step_toward(self, start: Position, target: Position, memory: Memory) -> Action:
        path = self._find_path(start, target, memory)
        if path:
            return self._action_between(start, path[0])
        return self._explore_frontier(start, memory)

    def _explore_frontier(self, start: Position, memory: Memory) -> Action:
        frontier_targets = self._frontier_targets(memory)
        best_path: list[Position] | None = None
        for target in sorted(frontier_targets, key=lambda pos: self._distance(start, pos)):
            path = self._find_path(start, target, memory)
            if path and (best_path is None or len(path) < len(best_path)):
                best_path = path

        if best_path:
            return self._action_between(start, best_path[0])

        for action, delta in MOVE_DELTAS.items():
            neighbor = (start[0] + delta[0], start[1] + delta[1])
            if self._is_passable(memory.known_world.get(neighbor)):
                return action
        return Action.SCAN_AREA

    def _frontier_targets(self, memory: Memory) -> list[Position]:
        targets: list[Position] = []
        for position, tile_type in memory.known_world.items():
            if not self._is_passable(tile_type):
                continue
            if any(neighbor not in memory.known_world for neighbor in self._neighbors(position)):
                targets.append(position)
        return targets

    def _find_path(self, start: Position, target: Position, memory: Memory) -> list[Position] | None:
        if start == target:
            return []
        queue: Deque[tuple[Position, list[Position]]] = deque([(start, [])])
        seen = {start}

        while queue:
            position, path = queue.popleft()
            for neighbor in self._neighbors(position):
                if neighbor in seen:
                    continue
                tile_type = memory.known_world.get(neighbor)
                if neighbor != target and not self._is_passable(tile_type):
                    continue
                if neighbor == target and tile_type == "locked_door":
                    return path
                if neighbor == target and self._is_known_or_goal(tile_type):
                    return path + [neighbor]
                if self._is_passable(tile_type):
                    seen.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def _is_known_or_goal(self, tile_type: str | None) -> bool:
        return tile_type in {"empty", "agent", "keycard", "ambiguous_object", "exit", "open_door", "locked_door"}

    def _is_passable(self, tile_type: str | None) -> bool:
        return tile_type in {"empty", "agent", "keycard", "ambiguous_object", "exit", "open_door"}

    def _neighbors(self, position: Position) -> Iterable[Position]:
        x, y = position
        yield (x, y - 1)
        yield (x + 1, y)
        yield (x, y + 1)
        yield (x - 1, y)

    def _action_between(self, start: Position, next_position: Position) -> Action:
        dx = next_position[0] - start[0]
        dy = next_position[1] - start[1]
        for action, delta in MOVE_DELTAS.items():
            if delta == (dx, dy):
                return action
        return Action.SCAN_AREA

    def _distance(self, left: Position, right: Position) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])


class LLMPolicy:

    def decide(self, observation: Observation, memory: Memory, plan: Plan) -> dict[str, str]:
        raise NotImplementedError(
            "LLMPolicy is intentionally a stub. Implement provider calls here while preserving "
            "the structured decision shape: current_goal, reasoning, action."
        )
