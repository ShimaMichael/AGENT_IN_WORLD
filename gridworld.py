class GridWorld:
    """Partially observable research-facility gridworld."""

    def __init__(self, observation_radius: int = 2, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.observation_radius = observation_radius
        self.raw_map = [
            "#############",
            "#A....#....E#",
            "#..##.#.##..#",
            "#..K..D.....#",
            "#############",
        ]
        self.height = len(self.raw_map)
        self.width = len(self.raw_map[0])
        self.agent_position = self._find("A")
        self.key_position = self._find("K")
        self.door_position = self._find("D")
        self.exit_position = self._find("E")
        self.inventory: set[str] = set()
        self.turn = 0
        self.energy = 80
        self.door_locked = True
        self.done = False
        self.event_log: list[str] = ["Boot sequence complete. Objective: retrieve keycard and escape."]

    def observe(self) -> Observation:
        visible_tiles: list[VisibleTile] = []
        ax, ay = self.agent_position
        radius = self.observation_radius

        for y in range(ay - radius, ay + radius + 1):
            for x in range(ax - radius, ax + radius + 1):
                if abs(x - ax) + abs(y - ay) > radius:
                    continue
                if not self._in_bounds((x, y)):
                    continue
                absolute = (x, y)
                relative = (x - ax, y - ay)
                tile_type = self.tile_type(absolute)
                visible_tiles.append(
                    VisibleTile(
                        relative_position=relative,
                        absolute_position=absolute,
                        tile_type=tile_type,
                        description=self.describe_tile(absolute, tile_type),
                    )
                )

        messages = self.event_log[-3:]
        return Observation(
            turn=self.turn,
            position=self.agent_position,
            visible_tiles=visible_tiles,
            inventory=set(self.inventory),
            energy=self.energy,
            messages=messages,
        )

    def step(self, action: Action) -> ActionResult:
        if self.done:
            return ActionResult(False, "Simulation has already ended.", True)

        self.turn += 1
        self.energy -= 1
        self._dynamic_events()

        if action in MOVE_DELTAS:
            result = self._move(action)
        elif action == Action.PICK_UP:
            result = self._pick_up()
        elif action == Action.OPEN_DOOR:
            result = self._open_door()
        elif action == Action.SCAN_AREA:
            self.energy -= 1
            result = ActionResult(True, "The agent scans carefully, sharpening local observations.")
        elif action == Action.WAIT:
            result = ActionResult(True, "The agent waits and listens for environmental changes.")
        else:
            result = ActionResult(False, f"Invalid action: {action}")

        if self.energy <= 0 and not result.done:
            self.done = True
            return ActionResult(False, "Energy depleted before escape.", True)

        return result

    def tile_type(self, position: Position) -> str:
        if position == self.agent_position:
            return "agent"
        if position == self.key_position and "keycard" not in self.inventory:
            return "keycard" if self.turn >= 2 else "ambiguous_object"
        if position == self.door_position:
            return "locked_door" if self.door_locked else "open_door"
        if position == self.exit_position:
            return "exit"

        char = self._char_at(position)
        if char == "#":
            return "wall"
        return "empty"

    def describe_tile(self, position: Position, tile_type: str) -> str:
        descriptions = {
            "agent": "Current agent position.",
            "wall": "A solid wall blocks passage.",
            "empty": "Empty floor.",
            "locked_door": "A locked security door blocks the corridor.",
            "open_door": "An unlocked security door stands open.",
            "exit": "A marked extraction point is visible.",
            "keycard": "A keycard lies on the floor.",
            "ambiguous_object": "You notice a faint metallic object in the distance.",
        }
        if tile_type == "empty" and self.rng.random() < 0.04:
            return "Empty floor, though the shadows make it hard to read."
        return descriptions[tile_type]

    def render(self, memory: Memory | None = None, debug_full_map: bool = False) -> str:
        rows: list[str] = []
        for y in range(self.height):
            cells: list[str] = []
            for x in range(self.width):
                position = (x, y)
                if debug_full_map:
                    cells.append(self._render_true_cell(position))
                elif memory is None or position not in memory.known_world:
                    cells.append("?")
                elif position == self.agent_position:
                    cells.append("A")
                else:
                    cells.append(self._render_known_cell(memory.known_world[position]))
            rows.append("".join(cells))
        return "\n".join(rows)

    def _move(self, action: Action) -> ActionResult:
        dx, dy = MOVE_DELTAS[action]
        target = (self.agent_position[0] + dx, self.agent_position[1] + dy)
        tile = self.tile_type(target)

        if tile == "wall":
            return ActionResult(False, f"Blocked by wall at {target}.")
        if tile == "locked_door":
            return ActionResult(False, f"Locked door blocks movement at {target}.")
        if tile == "exit":
            if "keycard" in self.inventory:
                self.agent_position = target
                self.done = True
                return ActionResult(True, "Exit reached with keycard. Mission complete.", True)
            return ActionResult(False, "Exit requires the keycard authorization.")

        self.agent_position = target
        return ActionResult(True, f"Moved to {target}.")

    def _pick_up(self) -> ActionResult:
        if self.agent_position == self.key_position and "keycard" not in self.inventory:
            self.inventory.add("keycard")
            return ActionResult(True, "Picked up the keycard.")
        return ActionResult(False, "No useful object is present here.")

    def _open_door(self) -> ActionResult:
        if self._manhattan(self.agent_position, self.door_position) != 1:
            return ActionResult(False, "No door is adjacent.")
        if not self.door_locked:
            return ActionResult(True, "The security door is already open.")
        if "keycard" not in self.inventory:
            return ActionResult(False, "The locked door rejects access without a keycard.")
        self.door_locked = False
        return ActionResult(True, "Keycard accepted. The security door unlocks.")

    def _dynamic_events(self) -> None:
        if self.turn == 8 and self.door_locked:
            self.event_log.append("A magnetic lock hums somewhere deeper in the facility.")
        if self.turn == 14 and self.door_locked:
            self.event_log.append("The locked door briefly flickers but remains sealed.")
        if self.turn % 11 == 0:
            self.event_log.append("Ventilation noise masks distant details for a moment.")

    def _render_true_cell(self, position: Position) -> str:
        if position == self.agent_position:
            return "A"
        if position == self.key_position and "keycard" not in self.inventory:
            return "K"
        if position == self.door_position:
            return "D" if self.door_locked else "/"
        if position == self.exit_position:
            return "E"
        return self._char_at(position)

    def _render_known_cell(self, tile_type: str) -> str:
        return {
            "agent": "A",
            "wall": "#",
            "empty": ".",
            "locked_door": "D",
            "open_door": "/",
            "exit": "E",
            "keycard": "K",
            "ambiguous_object": "?",
        }.get(tile_type, "?")

    def _find(self, char: str) -> Position:
        for y, row in enumerate(self.raw_map):
            x = row.find(char)
            if x >= 0:
                return (x, y)
        raise ValueError(f"Map does not contain {char!r}")

    def _char_at(self, position: Position) -> str:
        x, y = position
        return self.raw_map[y][x]

    def _in_bounds(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def _manhattan(self, left: Position, right: Position) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])