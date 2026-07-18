# tests/test_token_filter.py

import json
import os
import tempfile
import pytest
from typing import Generator
from src.token_filter import TokenFilter


@pytest.fixture
def dummy_vocab_path() -> Generator[str, None, None]:
    """テスト用のダミーボキャブラリ（単語帳）を作成するフィクスチャ"""
    vocab_data = {
        "\"": 1,
        "a": 2,
        "1": 3,
        "2": 4,
        ".": 5,
        ",": 6,
        " ": 7,
        " \", \"": 8,
        "name": 9,
        "fn_add": 10,
        "Ċ": 11,
        "}": 12,
        "}}Ċ": 13,
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f)

    yield path

    os.remove(path)


def test_token_filter_initialization(dummy_vocab_path: str) -> None:
    """初期化とボキャブラリの読み込みが正しく行われるか"""
    tf = TokenFilter(vocab_path=dummy_vocab_path)
    assert tf.id_to_token[1] == "\""
    assert tf.id_to_token[11] == "Ċ"
    assert 13 in tf.all_token_ids


def test_filter_by_prefix_strict_match(dummy_vocab_path: str) -> None:
    """厳密前方一致が正しく機能するか"""
    tf = TokenFilter(vocab_path=dummy_vocab_path)
    allowed = tf.filter_by_prefix('{"prompt": ', '{"prompt": "')

    assert 1 in allowed  # '"' (id: 1)
    assert 2 not in allowed  # 'a' (id: 2)


def test_filter_by_prefix_overshoot_prevention(dummy_vocab_path: str) -> None:
    """目標を飛び越えるような合体トークンを正しくブロックするか"""
    tf = TokenFilter(vocab_path=dummy_vocab_path)

    current_text = '{"a": 2'
    target = '{"a": 2}'

    allowed = tf.filter_by_prefix(current_text, target)

    assert 12 in allowed  # '}' (id: 12)
    # 合体トークン '}}Ċ' (id: 13) は飛び越え（オーバーシュート）になるためブロックされるべき
    assert 13 not in allowed


def test_filter_numeric_tokens(dummy_vocab_path: str) -> None:
    """数値トークンのみを正確に抽出できるか"""
    tf = TokenFilter(vocab_path=dummy_vocab_path)
    allowed_start = tf.filter_numeric_tokens(is_start=True)
    assert 3 in allowed_start  # "1"
    assert 5 in allowed_start  # "."
    assert 7 in allowed_start  # " "
    assert 2 not in allowed_start  # "a" (文字は弾かれる)
    assert 11 not in allowed_start  # "Ċ" (改行は弾かれる)

    # is_start=False の場合、空白(" ")は許可されない
    allowed_mid = tf.filter_numeric_tokens(is_start=False)
    assert 3 in allowed_mid  # "1"
    assert 5 in allowed_mid  # "."
    assert 7 not in allowed_mid  # " " (途中の空白は弾かれる)
