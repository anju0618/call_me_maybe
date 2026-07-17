# tests/test_tokenizer.py

import json
import os
import tempfile
import pytest
from typing import Generator
from src.token_filter import TokenFilter


@pytest.fixture
def dummy_vocab_path() -> Generator[str, None, None]:
    """テスト用のダミーボキャブラリを作成するフィクスチャ"""
    vocab_data = {
        "He": 1,
        "llo": 2,
        ",": 3,
        "Ġworld": 4, # LLM特有の空白付きトークン
        "!": 5,
        "world": 6  # 最長一致のテスト用 (Ġなし)
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f)
    yield path
    os.remove(path)


def test_custom_tokenizer_encode(dummy_vocab_path: str) -> None:
    """文字列が正しくトークンIDの配列にエンコード（最長一致）されるか"""
    tokenizer = TokenFilter(vocab_path=dummy_vocab_path)
    # "Hello, world!" をエンコードする
    # 半角スペースが Ġ に変換されたものとして "Ġworld" (ID: 4) が選ばれるべき
    text = "Hello, world!"
    encoded = tokenizer.encode(text)
    assert encoded == [1, 2, 3, 4, 5]


def test_custom_tokenizer_decode(dummy_vocab_path: str) -> None:
    """トークンIDの配列が正しい文字列にデコードされるか"""
    tokenizer = TokenFilter(vocab_path=dummy_vocab_path)
    token_ids = [1, 2, 3, 4, 5]
    decoded = tokenizer.decode(token_ids)
    # デコード時に LLM特有の Ġ が半角スペースに変換されて復元されるはず
    assert decoded == "Hello, world!"
