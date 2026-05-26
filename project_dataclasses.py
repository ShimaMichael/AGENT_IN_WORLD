from __future__ import annotations

from enum import Enum
from dataclasses import dataclass

Position = tuple[int, int]


class Action(str, Enum):
    MOVE_NORTH = "MOVE_NORTH"
    MOVE_SOUTH = "MOVE_SOUTH"
    MOVE_EAST = "MOVE_EAST"
    MOVE_WEST = "MOVE_WEST"
    PICK_UP = "PICK_UP"
    OPEN_DOOR = "OPEN_DOOR"
    SCAN_AREA = "SCAN_AREA"
    WAIT = "WAIT"


MOVE_DELTAS: dict[Action, Position] = {
    Action.MOVE_NORTH: (0, -1),
    Action.MOVE_SOUTH: (0, 1),
    Action.MOVE_EAST: (1, 0),
    Action.MOVE_WEST: (-1, 0),
}


PASSABLE_TILES = {".", "A", "K", "E"}


@dataclass
class VisibleTile:
    relative_position: Position
    absolute_position: Position
    tile_type: str
    description: str


@dataclass
class Observation:
    turn: int
    position: Position
    visible_tiles: list[VisibleTile]
    inventory: set[str]
    energy: int
    messages: list[str]

        
@dataclass
class ActionResult:
    success: bool
    message: str
    done: bool = False


@dataclass
class Plan:
    current_goal: str
    subgoal: str
    target: Position | None
    reason: str
