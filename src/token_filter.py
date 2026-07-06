# src/token_filter.py

import json
from typing import Dict, List, Set


class TokenFilter:
    """LLMのボキャブラリを解析し、特定の文脈で許可するトークンIDを抽出するクラ。"""

    def __init__(self, vocab_path: str):
        """vocab.jsonを読み込み、逆引き辞書"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self._vocab: Dict[str, int] = json.load(f)
        # IDがkey,strがitemに変換
        self.id_to_token: Dict[int, str] = {
            v: k for k, v in self._vocab.items()
        }
        # 全トークンIDの集合（マスクの初期化用）
        self.all_token_ids: Set[int] = set(self.id_to_token.keys())

    def filter_by_prefix(
            self,
            current_prefix: str,
            target_string: str
            ) -> List[int]:
        """
        ターゲット文字列に対して、現在の出力状況から次に繋がり得るトークンのみを許可
        例: target_string が "fn_add" で、現在すでに "fn_" まで出力されている際
            "a", "ad", "add" などのトークンを許可
        """
        remainder = target_string[len(current_prefix):]
        if not remainder:
            return []

        allowed_ids: List[int] = []
        for t_id, t_str in self.id_to_token.items():
            clean_str = t_str.replace("Ġ", " ").replace(" ", " ")

            if remainder.startswith(clean_str) or clean_str.startswith(remainder):
                allowed_ids.append(t_id)

        return allowed_ids

    def filter_numeric_tokens(self, is_start: bool = False) -> List[int]:
        """有効な数値トークンのみを許可"""
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
