# tests/test_json_generator.py

import json
import os
import tempfile
import pytest
from typing import Generator
from src.json_generator import JsonGenerator
from llm_sdk import Small_LLM_Model  # 追加

# Pydanticの厳格な型チェックを通過させるためのダミーモデル
class DummyModel(Small_LLM_Model):
    def __init__(self) -> None:
        # 本物の重いAIモデルの読み込み処理をスキップ（何もしない）
        pass


@pytest.fixture
def dummy_setup() -> Generator[str, None, None]:
    """最低限のvocab.jsonを作成する"""
    vocab_data = {"\"": 1, "{": 2, "}": 3, "a": 4}
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f)
        
    yield path
    os.remove(path)


def test_json_generator_initialization(dummy_setup: str) -> None:
    """JsonGeneratorが正しく初期化され、トークンフィルターがセットアップされるか"""
    # MagicMockの代わりに、型チェックを通過するダミーモデルを使う
    mock_model = DummyModel()
    
    # テスト用の関数定義
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

    # 初期化の確認
    assert generator.vocab_path == dummy_setup
    assert generator.debug is True
    assert generator.token_filter is not None
    
    # フィルターにボキャブラリが正しく読み込まれているか
    assert len(generator.token_filter.id_to_token) == 4
    assert generator.token_filter.id_to_token[2] == "{"
