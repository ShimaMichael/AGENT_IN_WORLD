from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections import Counter, deque
from dataclasses import asdict
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
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 20.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required when using the Gemini LLM policy.")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
        self.timeout = timeout

    def decide(self, observation: Observation, memory: Memory, plan: Plan) -> dict[str, str]:
        prompt = self._build_prompt(observation, memory, plan)
        response = self._generate(prompt)
        decision = self._parse_decision(response)
        action = self._validate_action(decision.get("action"))
        return {
            "current_goal": str(decision.get("current_goal") or plan.current_goal),
            "reasoning": str(decision.get("reasoning") or "Gemini selected a valid action."),
            "action": action.value,
        }

    def _build_prompt(self, observation: Observation, memory: Memory, plan: Plan) -> str:
        payload = {
            "task": (
                "Choose exactly one next action for a partially observable gridworld agent. "
                "Return JSON only, with keys current_goal, reasoning, and action."
            ),
            "rules": {
                "valid_actions": [action.value for action in Action],
                "objective": "Find the keycard, unlock the locked door, and reach the exit.",
                "movement": "Walls and locked doors block movement. OPEN_DOOR only works next to the door with the keycard.",
            },
            "plan": asdict(plan),
            "observation": {
                "turn": observation.turn,
                "position": observation.position,
                "inventory": sorted(observation.inventory),
                "energy": observation.energy,
                "visible_tiles": [asdict(tile) for tile in observation.visible_tiles],
                "messages": observation.messages,
            },
            "memory": {
                "known_world": self._stringify_positions(memory.known_world),
                "object_beliefs": self._stringify_positions(memory.object_beliefs),
                "hypotheses": memory.hypotheses[-5:],
                "reflections": memory.reflections[-5:],
                "failed_attempts": memory.failed_attempts[-6:],
                "recent_positions": list(memory.recent_positions),
                "recent_actions": [action.value for action in memory.recent_actions],
            },
            "json_schema": {
                "current_goal": "short description of the current goal",
                "reasoning": "one concise sentence explaining the action choice",
                "action": "one valid action string",
            },
        }
        return json.dumps(payload, separators=(",", ":"))

    def _generate(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            message = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API request failed with HTTP {error.code}: {message}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Gemini API request failed: {error.reason}") from error

        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini API returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [part.get("text", "") for part in parts if "text" in part]
        response_text = "".join(text_parts).strip()
        if not response_text:
            raise RuntimeError(f"Gemini API returned no text content: {data}")
        return response_text

    def _parse_decision(self, response_text: str) -> dict[str, str]:
        try:
            decision = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Gemini returned invalid JSON: {response_text}") from error
        if not isinstance(decision, dict):
            raise RuntimeError(f"Gemini decision must be a JSON object: {response_text}")
        return decision

    def _validate_action(self, action: object) -> Action:
        try:
            return Action(str(action))
        except ValueError as error:
            valid = ", ".join(action.value for action in Action)
            raise RuntimeError(f"Gemini emitted invalid action {action!r}. Valid actions: {valid}") from error

    def _stringify_positions(self, values: dict[Position, str] | dict[str, Position]) -> dict[str, object]:
        return {str(key): value for key, value in values.items()}
