from dataclasses import dataclass, field

@dataclass
class Memory:
    known_world: dict[Position, str] = field(default_factory=dict)
    object_beliefs: dict[str, Position] = field(default_factory=dict)
    hypotheses: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    failed_attempts: list[str] = field(default_factory=list)
    explored: set[Position] = field(default_factory=set)
    recent_positions: Deque[Position] = field(default_factory=lambda: deque(maxlen=8))
    recent_actions: Deque[Action] = field(default_factory=lambda: deque(maxlen=8))

    def update_from_observation(self, observation: Observation) -> None:
        self.explored.add(observation.position)
        self.recent_positions.append(observation.position)

        for tile in observation.visible_tiles:
            tile_type = "empty" if tile.tile_type == "agent" else tile.tile_type
            self.known_world[tile.absolute_position] = tile_type
            if tile.tile_type == "keycard":
                self.object_beliefs["keycard"] = tile.absolute_position
                self._remember_hypothesis("The metallic object is confirmed to be the keycard.")
            elif "metallic" in tile.description and "keycard" not in self.object_beliefs:
                self.object_beliefs["possible_keycard"] = tile.absolute_position
                self._remember_hypothesis("A faint metallic object may be the keycard.")
            elif tile.tile_type == "locked_door":
                self.object_beliefs["locked_door"] = tile.absolute_position
                self._remember_hypothesis("The locked security door likely requires the keycard.")
            elif tile.tile_type == "open_door":
                self.object_beliefs.pop("locked_door", None)
                self.object_beliefs["open_door"] = tile.absolute_position
            elif tile.tile_type == "exit":
                self.object_beliefs["exit"] = tile.absolute_position

    def record_action(self, action: Action, result: ActionResult) -> None:
        self.recent_actions.append(action)
        if not result.success:
            self.failed_attempts.append(f"{action.value}: {result.message}")
            self.failed_attempts = self.failed_attempts[-6:]

        if "picked up the keycard" in result.message.lower():
            self.object_beliefs.pop("keycard", None)
            self.object_beliefs.pop("possible_keycard", None)

    def add_reflection(self, reflection: str) -> None:
        if reflection and reflection not in self.reflections:
            self.reflections.append(reflection)
            self.reflections = self.reflections[-5:]

    def _remember_hypothesis(self, hypothesis: str) -> None:
        if hypothesis not in self.hypotheses:
            self.hypotheses.append(hypothesis)
            self.hypotheses = self.hypotheses[-5:]