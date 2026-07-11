# src/json_state.py

from enum import Enum, auto


class JsonState(Enum):
    """
    JSON生成の現在地（ステート）を管理するEnum。
    AIが今JSONのどの部分を生成しているかを追跡するために使用する。
    """
    START = auto()          # 1. 処理開始直後。冒頭の `{"prompt": "` を目指す
    PROMPT_VALUE = auto()   # 2. ユーザー入力（プロンプト）の文字列をそのまま出力中
    NAME_KEY = auto()       # 3. `", "name": "` のキー部分を出力中
    FUNCTION_NAME = auto()  # 4. 候補の関数名（例: fn_add_numbers）を出力中
    PARAMS_START = auto()   # 5. `", "parameters": {` のキー部分を出力中
    PARAM_KEY = auto()      # 6. 引数名（例: "a": や "name":）を出力中
    PARAM_VALUE = auto()    # 7. 引数の値（数値や文字列）を出力中（最難関フェーズ）
