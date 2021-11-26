""" Tests Grid Object behavior and properties """
import unittest
from typing import Type

import pytest

from gym_gridverse.agent import Agent
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import (
    Beacon,
    Box,
    Color,
    Door,
    Exit,
    Floor,
    GridObject,
    Hidden,
    Key,
    MovingObstacle,
    NoneGridObject,
    Telepod,
    Wall,
    grid_object_registry,
)
from gym_gridverse.state import State


class DummyNonRegisteredObject(GridObject, register=False):
    """Some dummy grid object that is _not_ registered"""


@pytest.mark.parametrize(
    'object_type,expected',
    [
        (DummyNonRegisteredObject, False),
        (NoneGridObject, True),
        (Hidden, True),
        (Floor, True),
        (Wall, True),
        (Exit, True),
        (Door, True),
        (Key, True),
        (MovingObstacle, True),
        (Box, True),
        (Telepod, True),
        (Beacon, True),
    ],
)
def test_registration(object_type: Type[GridObject], expected: bool):
    assert (object_type in grid_object_registry) == expected


def test_none_grid_object_registration():
    """Tests the registration as a Grid Object"""
    assert NoneGridObject in grid_object_registry


def test_hidden_registration():
    """Tests the registration as a Grid Object"""
    assert Hidden in grid_object_registry


def test_grid_object_registration():
    """Test registration of type indices"""

    assert len(grid_object_registry) == 11
    unittest.TestCase().assertCountEqual(
        [
            NoneGridObject.type_index(),
            Hidden.type_index(),
            Floor.type_index(),
            Wall.type_index(),
            Exit.type_index(),
            Door.type_index(),
            Key.type_index(),
            MovingObstacle.type_index(),
            Box.type_index(),
            Telepod.type_index(),
            Beacon.type_index(),
        ],
        range(len(grid_object_registry)),
    )

    for obj_cls in [
        NoneGridObject,
        Hidden,
        Floor,
        Wall,
        Exit,
        Door,
        Key,
        MovingObstacle,
        Box,
        Telepod,
        Beacon,
    ]:
        assert grid_object_registry[obj_cls.type_index()] is obj_cls


def simple_state_without_object() -> State:
    """Returns a 2x2 (empty) grid with an agent without an item"""
    # TODO this should raise error, which means it is not being used anywhere
    return State(
        Grid(height=2, width=2),
        Agent(Position(0, 0), Orientation.F, Floor()),
    )


def test_none_grid_object_properties():
    """Basic stupid tests for none grid object"""

    none = NoneGridObject()

    assert none.color == Color.NONE
    assert none.state_index == 0

    assert none.can_be_represented_in_state()
    assert none.num_states() == 1


def test_hidden_properties():
    """Basic stupid tests for hidden grid object"""

    hidden = Hidden()

    assert not hidden.transparent
    assert hidden.color == Color.NONE
    assert hidden.state_index == 0

    assert not hidden.can_be_represented_in_state()
    assert hidden.num_states() == 1


def test_floor_properties():
    """Basic stupid tests for floor grid object"""

    floor = Floor()

    assert floor.transparent
    assert not floor.blocks_movement
    assert floor.color == Color.NONE
    assert not floor.can_be_picked_up
    assert floor.state_index == 0

    assert floor.can_be_represented_in_state()
    assert floor.num_states() == 1


def test_wall_properties():
    """Basic property tests"""

    wall = Wall()

    assert not wall.transparent
    assert wall.blocks_movement
    assert wall.color == Color.NONE
    assert not wall.can_be_picked_up
    assert wall.state_index == 0

    assert wall.can_be_represented_in_state()
    assert wall.num_states() == 1


def test_exit_properties():
    """Basic property tests"""

    exit_ = Exit()

    assert exit_.transparent
    assert not exit_.blocks_movement
    assert exit_.color == Color.NONE
    assert not exit_.can_be_picked_up
    assert exit_.state_index == 0

    assert exit_.can_be_represented_in_state()
    assert exit_.num_states() == 1


def test_door_open_door_properties():
    """Basic property tests"""

    color = Color.GREEN
    open_door = Door(Door.Status.OPEN, color)

    assert open_door.transparent
    assert open_door.color == color
    assert not open_door.can_be_picked_up
    assert open_door.state_index == Door.Status.OPEN.value
    assert open_door.is_open
    assert not open_door.locked
    assert not open_door.blocks_movement

    assert open_door.can_be_represented_in_state()
    assert open_door.num_states() == 3


def test_door_closed_door_properties():
    """Basic property tests"""

    color = Color.NONE
    closed_door = Door(Door.Status.CLOSED, color)

    assert not closed_door.transparent
    assert closed_door.color == color
    assert not closed_door.can_be_picked_up
    assert closed_door.state_index == Door.Status.CLOSED.value
    assert not closed_door.is_open
    assert not closed_door.locked
    assert closed_door.blocks_movement

    assert closed_door.can_be_represented_in_state()


def test_door_locked_door_properties():
    """Basic property tests"""

    color = Color.NONE
    locked_door = Door(Door.Status.LOCKED, color)

    assert not locked_door.transparent
    assert locked_door.color == color
    assert not locked_door.can_be_picked_up
    assert locked_door.state_index == Door.Status.LOCKED.value
    assert not locked_door.is_open
    assert locked_door.locked
    assert locked_door.blocks_movement

    assert locked_door.can_be_represented_in_state()


def test_key_properties():
    """Basic property tests"""

    color = Color.YELLOW
    key = Key(color)

    assert key.transparent
    assert not key.blocks_movement
    assert key.color == color
    assert key.can_be_picked_up
    assert key.state_index == 0

    assert key.can_be_represented_in_state()
    assert key.num_states() == 1


def test_moving_obstacle_basic_properties():
    """Tests basic properties of the moving obstacle"""

    obstacle = MovingObstacle()

    assert obstacle.transparent
    assert not obstacle.blocks_movement
    assert obstacle.color == Color.NONE
    assert not obstacle.can_be_picked_up
    assert obstacle.state_index == 0

    assert obstacle.can_be_represented_in_state()
    assert obstacle.num_states() == 1


def test_box_basic_properties():
    """Tests basic properties of box"""

    box = Box(Floor())

    assert box.transparent
    assert box.blocks_movement
    assert box.color == Color.NONE
    assert not box.can_be_picked_up
    assert box.state_index == 0

    assert not box.can_be_represented_in_state()
    assert box.num_states() == 1


def test_telepod_properties():
    """Basic property tests of telepod"""

    color = Color.YELLOW
    telepod = Telepod(color)

    assert telepod.transparent
    assert not telepod.blocks_movement
    assert telepod.color == color
    assert not telepod.can_be_picked_up
    assert telepod.state_index == 0

    assert telepod.can_be_represented_in_state()
    assert telepod.num_states() == 1


def test_beacon_properties():
    """Basic property tests of beacon"""

    color = Color.YELLOW
    beacon = Beacon(color)

    assert beacon.transparent
    assert not beacon.blocks_movement
    assert beacon.color == color
    assert not beacon.can_be_picked_up
    assert beacon.state_index == 0

    assert beacon.can_be_represented_in_state()
    assert beacon.num_states() == 1


def test_custom_object():
    """Basic property tests of (newly defined) custom objects"""

    class ColoredFloor(GridObject):
        """Most basic _colored_ object in the grid, represents empty cell"""

        def __init__(self, color: Color = Color.NONE):
            self.state_index = 0
            self.color = color
            self.transparent = True
            self.can_be_picked_up = False
            self.blocks_movement = False

        @classmethod
        def can_be_represented_in_state(cls) -> bool:
            return True

        @classmethod
        def num_states(cls) -> int:
            return 1

        def __repr__(self):
            return f'{self.__class__.__name__}({self.color})'

    colored_floor = ColoredFloor(Color.YELLOW)

    assert colored_floor.transparent
    assert not colored_floor.blocks_movement
    assert colored_floor.color == Color.YELLOW
    assert not colored_floor.can_be_picked_up
    assert colored_floor.state_index == 0

    assert colored_floor.can_be_represented_in_state()
    assert colored_floor.num_states() == 1

    assert colored_floor.type_index() == len(grid_object_registry) - 1
    assert ColoredFloor.type_index() == len(grid_object_registry) - 1
    assert type(colored_floor) in grid_object_registry
