# src/json_state.py

from enum import Enum, auto


class JsonState(Enum):
    START = auto()
    PROMPT_VALUE = auto()
    NAME_KEY = auto()
    FUNCTION_NAME = auto()
    PARAMS_START = auto()
    PARAM_KEY = auto()
    PARAM_VALUE = auto()
