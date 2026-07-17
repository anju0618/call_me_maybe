# tests/test_json_generator.py

import json
import os
import tempfile
import typing
import pytest
from typing import Generator
from src.json_generator import JsonGenerator
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

# mypyの strict モードと通常モードの判定差異による
# unused-ignore エラーを根本から解決するための分岐処理
if typing.TYPE_CHECKING:
    # 型チェック時：Anyの継承を避けるため、ただの独立クラスに見せかける
    class DummyModel:
        def __init__(self) -> None:
            pass
else:
    # 実行時：Pydanticの型チェックを通すため、本物を継承する
    class DummyModel(Small_LLM_Model):
        def __init__(self) -> None:
            pass


@pytest.fixture
def dummy_setup() -> Generator[str, None, None]:
    """最低限のvocab.jsonを一時ファイルとして作成するフィクスチャ"""
    vocab_data = {"\"": 1, "{": 2, "}": 3, "a": 4}
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f)
    # テストの実行中は path を渡し、終わったら削除する（yield）
    yield path
    os.remove(path)


def test_json_generator_initialization(dummy_setup: str) -> None:
    """JsonGeneratorが正しく初期化され、トークンフィルターがセットアップされるか"""
    # 型チェッカー上は Small_LLM_Model として扱い、
    # 実行時はPydanticの型チェック(isinstance)を通過させる
    mock_model = typing.cast(Small_LLM_Model, DummyModel())

    # テスト用の簡単な関数定義
    dummy_functions = [
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers.",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            }
        }
    ]

    generator = JsonGenerator(
        model=mock_model,
        vocab_path=dummy_setup,
        functions=dummy_functions,
        debug=True
    )

    # 初期化の確認。各種プロパティが正しくセットされているかアサート
    assert generator.vocab_path == dummy_setup
    assert generator.debug is True
    assert generator.token_filter is not None
    # フィルターにボキャブラリ（4つ）が正しく読み込まれているか
    assert len(generator.token_filter.id_to_token) == 4
    assert generator.token_filter.id_to_token[2] == "{"