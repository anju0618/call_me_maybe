# src/token_filter.py

import json
import functools
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
from pydantic import BaseModel, Field


class TokenFilter(BaseModel):
    """
    LLMが「出力してはいけない文字」を選ばないようにするためのフィルター。
    vocab.json を読み込み、トークンIDと文字列の変換や、
    次に許可されるトークンの絞り込み（Constrained Decoding）を行います。
    """
    vocab_path: str
    id_to_token: Dict[int, str] = Field(default_factory=dict)
    all_token_ids: Set[int] = Field(default_factory=set)

    # トークン文字列を長さ順（長い順）に並べたリスト
    sorted_clean_tokens: List[Tuple[str, int]] = Field(default_factory=list)

    # 文字列の推論中に許可される安全なトークン群
    valid_string_tokens: Set[int] = Field(default_factory=set)

    # 【究極の爆速化】1文字目で検索範囲を劇的に絞り込むための辞書キャッシュ
    # 例: char_to_tokens["a"] -> [("apple", 123), ("and", 456), ...]
    char_to_tokens: Dict[str, List[Tuple[str, int]]] = Field(
        default_factory=dict
    )

    def __hash__(self) -> int:
        return hash(self.vocab_path)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TokenFilter):
            return False
        return self.vocab_path == other.vocab_path

    def model_post_init(self, __context: Any) -> None:
        # ボキャブラリファイル（単語帳）を読み込む
        with open(self.vocab_path, "r", encoding="utf-8") as f:
            vocab_dict: Dict[str, int] = json.load(f)

        mapping = {v: k for k, v in vocab_dict.items()}
        self.id_to_token.update(mapping)
        self.all_token_ids.update(self.id_to_token.keys())

        # 許可してはいけない文字（改行など）
        inv_chars = set("\n\rĊ")
        valid_str_toks = set()
        c_dict: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        clean_list = []
        for t_id, t_str in self.id_to_token.items():
            # LLM特有の空白文字（Ġ）を通常の半角スペースに変換
            cl_str = t_str.replace("Ġ", " ").replace(" ", " ")
            if cl_str:
                clean_list.append((cl_str, t_id))
                c_dict[cl_str[0]].append((cl_str, t_id))

            # 改行などが含まれていないトークンを安全とみなす
            if not any(c in inv_chars for c in cl_str):
                # 【重要】状態遷移（終了判定）をすり抜けてしまう危険な
                # 合体トークン（例: '",' や '"}' ）を事前ブロックする
                if '",' in cl_str or '"}' in cl_str or '":' in cl_str:
                    continue
                valid_str_toks.add(t_id)

        # 最長一致で検索できるよう、文字数が多い順にソートしておく
        clean_list.sort(key=lambda x: len(x[0]), reverse=True)
        self.sorted_clean_tokens = clean_list
        self.valid_string_tokens = valid_str_toks

        for char in c_dict:
            c_dict[char].sort(key=lambda x: len(x[0]), reverse=True)
        self.char_to_tokens = dict(c_dict)

    def encode(self, text: str) -> List[int]:
        """文字列をトークンIDのリストに変換する"""
        token_ids = []
        i = 0
        text_len = len(text)
        while i < text_len:
            match_found = False
            first_char = text[i]
            # 1文字目で検索対象を絞り込むことで爆速化
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
        """トークンIDのリストを文字列に復元する"""
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
        """目標の文字列（full_target）に向かうための正しいトークンを抽出"""
        if not full_target.startswith(current_text):
            return []

        remainder = full_target[len(current_text):]
        if not remainder:
            return []

        allowed_ids: List[int] = []
        first_char = remainder[0]

        # 残りの文字列の先頭文字に一致するトークンのみを検索（爆速化）
        if first_char in self.char_to_tokens:
            for clean_str, t_id in self.char_to_tokens[first_char]:
                if remainder.startswith(clean_str):
                    allowed_ids.append(t_id)

        return allowed_ids

    @functools.lru_cache(maxsize=10)
    def filter_numeric_tokens(
        self, is_start: bool = False, is_integer: bool = False
    ) -> List[int]:
        """数値として許可されるトークン（0-9, -, .）のみを抽出"""
        allowed_ids: List[int] = []

        if is_integer:
            valid_chars = set("0123456789-")
        else:
            valid_chars = set("0123456789.-")

        for clean_str, t_id in self.sorted_clean_tokens:
            if all(c in valid_chars for c in clean_str):
                allowed_ids.append(t_id)

        return allowed_ids
