# src/token_filter.py

import json
import functools
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
from pydantic import BaseModel, Field


class TokenFilter(BaseModel):
    # vocab.jsonのパス
    vocab_path: str
    id_to_token: Dict[int, str] = Field(default_factory=dict)
    all_token_ids: Set[int] = Field(default_factory=set)
    sorted_clean_tokens: List[Tuple[str, int]] = Field(default_factory=list)

    # --- 【爆速化】事前計算キャッシュ ---
    string_start_tokens: Dict[str, Set[int]] = Field(default_factory=dict)
    valid_string_tokens: Set[int] = Field(default_factory=set)

    # 【究極の爆速化】1文字目で検索範囲を劇的に絞り込む辞書
    char_to_tokens: Dict[
        str, List[Tuple[str, int]]
    ] = Field(default_factory=lambda: defaultdict(list))

    def __hash__(self) -> int:
        return hash(self.vocab_path)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TokenFilter):
            return False
        return self.vocab_path == other.vocab_path

    def model_post_init(self, __context: Any) -> None:
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab_dict: Dict[str, int] = json.load(f)

        mapping = {v: k for k, v in vocab_dict.items()}
        self.id_to_token.update(mapping)
        self.all_token_ids.update(self.id_to_token.keys())

        self.string_start_tokens = {
            '"': set(),
            '"/': set(),
            '"C': set(),
            '"C:\\': set(),
            '"{': set()
        }
        inv_chars = set("\n\rĊ")
        valid_str_toks = set()
        char_dict: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        clean_list = []
        for t_id, t_str in self.id_to_token.items():
            # 重いreplace処理は起動時の1回だけ！
            cl_str = t_str.replace("Ġ", " ").replace(" ", " ")
            if cl_str:
                clean_list.append((cl_str, t_id))
                # 最初の1文字目をキーにした辞書を作る
                char_dict[cl_str[0]].append((cl_str, t_id))

            if cl_str in self.string_start_tokens:
                self.string_start_tokens[cl_str].add(t_id)

            if not any(c in inv_chars for c in cl_str):
                valid_str_toks.add(t_id)

        clean_list.sort(key=lambda x: len(x[0]), reverse=True)
        self.sorted_clean_tokens = clean_list
        self.valid_string_tokens = valid_str_toks

        # 1文字辞書の中身も文字数順にソートしておく
        for char in char_dict:
            char_dict[char].sort(key=lambda x: len(x[0]), reverse=True)
        self.char_to_tokens = dict(char_dict)

    def encode(self, text: str) -> List[int]:
        token_ids = []
        i = 0
        text_len = len(text)
        while i < text_len:
            match_found = False
            first_char = text[i]
            # 【爆速化】1文字目だけで検索対象を絞る
            if first_char in self.char_to_tokens:
                for t_str, t_id in self.char_to_tokens[first_char]:
                    if text.startswith(t_str, i):
                        token_ids.append(t_id)
                        i += len(t_str)
                        match_found = True
                        break
            if not match_found:
                i += 1
        return token_ids

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
        if not full_target.startswith(current_text):
            return []

        remainder = full_target[len(current_text):]
        if not remainder:
            return []

        allowed_ids: List[int] = []
        first_char = remainder[0]

        # 【超爆速化】15万件のループを廃止！
        # 残りの文字列の最初の文字が一致するトークンだけを検索！
        if first_char in self.char_to_tokens:
            for clean_str, t_id in self.char_to_tokens[first_char]:
                if remainder.startswith(clean_str):
                    allowed_ids.append(t_id)

        return allowed_ids

    @functools.lru_cache(maxsize=10)
    def filter_numeric_tokens(
        self, is_start: bool = False, is_integer: bool = False
    ) -> List[int]:
        allowed_ids: List[int] = []

        if is_integer:
            if is_start:
                valid_chars = set("0123456789- ")
            else:
                valid_chars = set("0123456789-")
        else:
            if is_start:
                valid_chars = set("0123456789.- ")
            else:
                valid_chars = set("0123456789.-")

        # 起動時に作っておいたクリーンリストを使う
        for clean_str, t_id in self.sorted_clean_tokens:
            if all(c in valid_chars for c in clean_str):
                allowed_ids.append(t_id)

        return allowed_ids
