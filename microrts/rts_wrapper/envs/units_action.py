from .units_name import *

from enum import Enum

ACTION_TYPE_NONE = 0
ACTION_TYPE_MOVE = 1
ACTION_TYPE_HARVEST = 2
ACTION_TYPE_RETURN = 3
ACTION_TYPE_PRODUCE = 4
ACTION_TYPE_ATTACK_LOCATION = 5

ACTION_PARAMETER_DIRECTION_NONE = -1
ACTION_PARAMETER_DIRECTION_UP = 0
ACTION_PARAMETER_DIRECTION_RIGHT = 1
ACTION_PARAMETER_DIRECTION_DOWN = 2
ACTION_PARAMETER_DIRECTION_LEFT = 3
ACTION_PARAMETER_VALID_DIRECTION_NUMS = 4


class BaseAction(Enum):
    __type_name__ = UNIT_TYPE_NAME_BASE

    DO_NONE = 0
    DO_LAY_WORKER = 1


class BarracksAction(Enum):
    __type_name__ = UNIT_TYPE_NAME_BARRACKS

    DO_NONE = 0
    DO_LAY_LIGHT = 1
    DO_LAY_HEAVY = 2
    DO_LAY_RANGED = 3

    @staticmethod
    def get_index(action):
        return list(BarracksAction).index(action)


class WorkerAction(Enum):
    __type_name__ = UNIT_TYPE_NAME_WORKER

    DO_NONE = -1

    # 0, 1, 2, 3 must agree with up, right, down, left
    # need to convert to valid actions using unit_valid_action according to the specific condition.
    # example:DO_UP_PROBE: walk up when no obstacles, but attack when enemy
    DO_UP_PROBE = 0
    DO_RIGHT_PROBE = 1
    DO_DOWN_PROBE = 2
    DO_LEFT_PROBE = 3

    # produce: randomly pick directions
    DO_LAY_BASE = 4  # type4 unitType:base
    DO_LAY_BARRACKS = 5  # type4 unitType:barracks


class LightAction(Enum):
    __type_name__ = UNIT_TYPE_NAME_LIGHT
    DO_NONE = -1

    DO_UP_PROBE = 0
    DO_RIGHT_PROBE = 1
    DO_DOWN_PROBE = 2
    DO_LEFT_PROBE = 3


class HeavyAction(Enum):
    __type_name__ = UNIT_TYPE_NAME_HEAVY
    DO_NONE = -1

    DO_UP_PROBE = 0
    DO_RIGHT_PROBE = 1
    DO_DOWN_PROBE = 2
    DO_LEFT_PROBE = 3


class RangedAction(Enum):
    __type_name__ = UNIT_TYPE_NAME_RANGED

    DO_NONE = -1

    DO_UP_PROBE = 0
    DO_RIGHT_PROBE = 1
    DO_DOWN_PROBE = 2
    DO_LEFT_PROBE = 3

    DO_ATTACK_NEAREST = 4  # need java coding
    DO_ATTACK_WEAKEST = 5


action_collection = [BaseAction, BarracksAction, WorkerAction, LightAction, HeavyAction, RangedAction]

AGENT_ACTIONS_MAP = {}
for _action in action_collection:
    AGENT_ACTIONS_MAP[_action.__type_name__] = _action