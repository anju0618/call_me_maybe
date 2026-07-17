# src/json_state.py

from enum import Enum, auto


class JsonState(Enum):
    """
    LLMが現在、JSONのどの部分を推論しているかを表す状態（ステート）。
    推論が不要な固定文字列（ボイラープレート）はシステム側で
    自動入力してスキップ（強制ワープ）する仕組みにしたため、
    LLMが実際に考えるべき状態は以下の2つだけになります。
    """

    # 1. 関数名（例: "fn_add_numbers" や "fn_greet"）を推論している状態
    FUNCTION_NAME = auto()

    # 2. パラメータの具体的な値（例: 12.0 や "hello"）を推論している状態
    PARAM_VALUE = auto()