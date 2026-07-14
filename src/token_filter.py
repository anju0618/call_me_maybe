# src/token_filter.py

import json
import functools
from typing import Any, Dict, List, Set, Tuple
from pydantic import BaseModel, Field


class TokenFilter(BaseModel):
    # vocab.jsonのパス
    vocab_path: str
    # idからトークンわかる辞書.例: {1: '"', 2: 'a'}
    id_to_token: Dict[int, str] = Field(default_factory=dict)
    # 全トークンIDのセット
    all_token_ids: Set[int] = Field(default_factory=set)
    # 自作エンコーダ用：クリーンな文字列とIDのペアを文字数順（降順）にしたリスト
    sorted_clean_tokens: List[Tuple[str, int]] = Field(default_factory=list)

    def __hash__(self) -> int:
        # lru_cacheを使うためにselfをハッシュ可能にするおまじない
        return hash(self.vocab_path)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TokenFilter):
            return False
        return self.vocab_path == other.vocab_path

    def model_post_init(self, __context: Any) -> None:
        # Pydantic初期化時に実行
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab_dict: Dict[str, int] = json.load(f)

        # keyとvalueを反転させる
        mapping = {v: k for k, v in vocab_dict.items()}
        self.id_to_token.update(mapping)
        self.all_token_ids.update(self.id_to_token.keys())

        # 【ボーナス要件】自作エンコーダの準備
        # 最長一致（Greedy）で検索するため、文字数が長い順にソートしてキャッシュ
        clean_list = []
        for t_id, t_str in self.id_to_token.items():
            cl_str = t_str.replace("Ġ", " ").replace(" ", " ")
            if cl_str:
                clean_list.append((cl_str, t_id))
        
        clean_list.sort(key=lambda x: len(x[0]), reverse=True)
        self.sorted_clean_tokens = clean_list

    # 【ボーナス要件】自作エンコーダー（文字列 -> トークンID配列）
    def encode(self, text: str) -> List[int]:
        token_ids = []
        i = 0
        text_len = len(text)
        while i < text_len:
            match_found = False
            for t_str, t_id in self.sorted_clean_tokens:
                if text.startswith(t_str, i):
                    token_ids.append(t_id)
                    i += len(t_str)
                    match_found = True
                    break
            if not match_found:
                # 未知の文字はスキップ（無限ループ防止）
                i += 1
        return token_ids

    # 【ボーナス要件】自作デコーダー（トークンID配列 -> 文字列）
    def decode(self, token_ids: List[int]) -> str:
        result = ""
        for t_id in token_ids:
            if t_id in self.id_to_token:
                t_str = self.id_to_token[t_id]
                result += t_str.replace("Ġ", " ").replace(" ", " ")
        return result

    @functools.lru_cache(maxsize=10000)
    def filter_by_prefix(
        self, current_text: str, full_target: str
    ) -> List[int]:
        # ターゲットの続きになるトークンIDだけ返す
        if not full_target.startswith(current_text):
            return []

        remainder = full_target[len(current_text):]
        if not remainder:
            return []

        allowed_ids: List[int] = []

        # 全てのボキャブラリをチェック
        for t_id, t_str in self.id_to_token.items():
            clean_str = t_str.replace("Ġ", " ").replace(" ", " ")
            if not clean_str:
                continue

            if remainder.startswith(clean_str):
                allowed_ids.append(t_id)

        return allowed_ids

    @functools.lru_cache(maxsize=10)
    def filter_numeric_tokens(
        self, is_start: bool = False, is_integer: bool = False
    ) -> List[int]:
        # 数値トークンだけを許可する用（integer対応版）
        allowed_ids: List[int] = []

        if is_integer:
            # integerの場合は小数点(.)を許容しない
            if is_start:
                valid_chars = set("0123456789- ")
            else:
                valid_chars = set("0123456789-")
        else:
            if is_start:
                valid_chars = set("0123456789.- ")
            else:
                valid_chars = set("0123456789.-")

        for t_id, t_str in self.id_to_token.items():
            clean_str = t_str.replace("Ġ", " ").replace(" ", " ")

            if not clean_str:
                continue

            # 全部チェック "123" ok "123a" ko
            if all(c in valid_chars for c in clean_str):
                allowed_ids.append(t_id)

        return allowed_ids
