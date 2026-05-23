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