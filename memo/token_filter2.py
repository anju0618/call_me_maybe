# src/token_filter.py

import json
from typing import Any, Dict, List, Set
from pydantic import BaseModel, Field


class TokenFilter(BaseModel):
    """
    文脈に合わせて「次に出力していいトークン」だけを抽出する絶対防壁クラス。
    """

    # vocab.json（単語帳）のファイルパス
    vocab_path: str
    # IDからトークン文字列を引くための逆引き辞書。例: {1: '"', 2: 'a'}
    id_to_token: Dict[int, str] = Field(default_factory=dict)
    # 単語帳に存在するすべてのトークンIDのセット（許可リストの初期値などに使う）
    all_token_ids: Set[int] = Field(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        # Pydanticの初期化直後に自動で呼ばれる。vocab.jsonを読み込む。
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab_dict: Dict[str, int] = json.load(f)

        # {"a": 2} を {2: "a"} に反転して保存
        mapping = {v: k for k, v in vocab_dict.items()}
        self.id_to_token.update(mapping)
        self.all_token_ids.update(self.id_to_token.keys())

    def filter_by_prefix(
        self, current_text: str, full_target: str
    ) -> List[int]:
        """
        目標の文字列(full_target)に向かって、次に出力すべきトークンIDだけを返す。
        """
        # 現在の文字列が目標から既に逸脱している場合は、許可できるトークンは無い
        if not full_target.startswith(current_text):
            return []

        # remainder = 目標文字列から現在地を引いた「残りの文字列」
        remainder = full_target[len(current_text):]
        if not remainder:
            return []

        allowed_ids: List[int] = []

        # 全てのボキャブラリ（数万件）を走査してチェック
        for t_id, t_str in self.id_to_token.items():
            # AI特有の空白文字（Ġなど）を人間のスペースに置換して掃除
            clean_str = t_str.replace("Ġ", " ").replace(" ", " ")
            if not clean_str:
                continue

            # このトークンが「残りの文字列」の先頭と一致するなら許可リストへ
            if remainder.startswith(clean_str):
                allowed_ids.append(t_id)

        return allowed_ids

    def filter_numeric_tokens(self, is_start: bool = False) -> List[int]:
        """
        引数が「数値(number)」の時に、数字として有効なトークンのみを許可する。
        """
        allowed_ids: List[int] = []

        # is_start=True（数字の1文字目）なら、マイナス(-)やJSON仕様の空白( )も許可
        if is_start:
            valid_chars = set("0123456789.- ")
        # 2文字目以降は純粋な数字と小数点のみ（途中に空白が入ると数字が途切れるため）
        else:
            valid_chars = set("0123456789.")

        for t_id, t_str in self.id_to_token.items():
            clean_str = t_str.replace("Ġ", " ").replace(" ", " ")

            if not clean_str:
                continue

            # トークンのすべての文字が valid_chars に含まれているかチェック
            # 例: "123" -> OK,  "12a" -> 'a'が含まれるのでNG
            if all(c in valid_chars for c in clean_str):
                allowed_ids.append(t_id)

        return allowed_ids
