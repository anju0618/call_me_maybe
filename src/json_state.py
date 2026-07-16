# src/json_state.py

from enum import Enum, auto


class JsonState(Enum):
    FUNCTION_NAME = auto()
    PARAM_VALUE = auto()
