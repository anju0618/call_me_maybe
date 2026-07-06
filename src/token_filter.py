# src/token_filter.py

import json
from typing import Any, Dict, List, Set
from pydantic import BaseModel, Field


class TokenFilter(BaseModel):
    """特定の文脈で許可するトークンIDを抽出するクラス"""

    vocab_path: str

    id_to_token: Dict[int, str] = Field(default_factory=dict)
    all_token_ids: Set[int] = Field(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        """Pydantic_val後に自動で実行される初期化処理"""
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab_dict: Dict[str, int] = json.load(f)

        mapping = {v: k for k, v in vocab_dict.items()}
        self.id_to_token.update(mapping)
        self.all_token_ids.update(self.id_to_token.keys())

    def filter_by_prefix(
        self, current_prefix: str, target_string: str
    ) -> List[int]:
        """ターゲットの文字列の、次に繋がり得るトークンのみ許可"""
        remainder = target_string[len(current_prefix):]
        if not remainder:
            return []

        allowed_ids: List[int] = []
        for t_id, t_str in self.id_to_token.items():
            clean_str = t_str.replace("Ġ", " ").replace(" ", " ")
            if clean_str and remainder.startswith(clean_str):
                allowed_ids.append(t_id)

        return allowed_ids

    def filter_numeric_tokens(self, is_start: bool = False) -> List[int]:
        """数値として有効なトークンのみ許可"""
        allowed_ids: List[int] = []

        if is_start:
            valid_chars = set("0123456789.- ")
        else:
            valid_chars = set("0123456789.")

        for t_id, t_str in self.id_to_token.items():
            clean_str = t_str.replace("Ġ", " ").replace(" ", " ")

            if not clean_str:
                continue

            if all(c in valid_chars for c in clean_str):
                allowed_ids.append(t_id)

        return allowed_ids
