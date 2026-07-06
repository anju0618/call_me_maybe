# src/json_state.py

from enum import Enum, auto


class JsonState(Enum):
    START = auto()          # 1. 冒頭の `{"prompt": "` を出力する
    PROMPT_VALUE = auto()   # 2. 入力されたプロンプト文字列をそのまま出力する
    NAME_KEY = auto()       # 3. `", "name": "` を出力する
    FUNCTION_NAME = auto()  # 4. 候補の関数名（fn_add_numbers など）を選ばせる
    PARAMS_START = auto()   # 5. `", "parameters": {` を出力する
    PARAM_KEY = auto()      # 6. 引数名（"a": や "name":）を出力する
    PARAM_VALUE = auto()    # 7. 引数の値をAIにパースさせるフェーズ
