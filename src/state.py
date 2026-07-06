# src/state.py

from enum import Enum, auto


class state(Enum):
    """
    現在の進行度合いを管理する列挙型クラス
    STARTで冒頭の{"prompt":
    PROMPT_VALUEで"質問",
    """
    START = auto()
    PRO_VALUE = auto()
    FUN_KEY = auto()
    FUN_VALUE = auto()
    PARA_KEY = auto()
    PARA_VALUE = auto()

