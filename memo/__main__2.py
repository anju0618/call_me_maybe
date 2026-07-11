# src/__main__.py

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.json_generator import JsonGenerator


def parse_arguments() -> argparse.Namespace:
    # ターミナルから受け取る引数の設定。--debug でログ表示をONにできる
    parser = argparse.ArgumentParser(
        description="Function Calling CLI with LLM"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the output results JSON file."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable real-time state transition logging."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    func_def_path = Path(args.functions_definition)
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        # ファイルの存在確認
        if not func_def_path.exists():
            raise FileNotFoundError(
                f"Functions definition file not found: {func_def_path}"
            )
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # JSONの読み込み
        with open(func_def_path, 'r', encoding='utf-8') as f:
            functions_def = json.load(f)

        with open(input_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)

        print(
            f"Loaded {len(functions_def)} functions "
            f"and {len(prompts)} prompts."
        )

        print("\nInitializing LLM Model and JsonGenerator...")
        model = Small_LLM_Model()
        vocab_path = model.get_path_to_vocab_file()

        # 推論エンジンの組み立て
        generator = JsonGenerator(
            model=model,
            vocab_path=vocab_path,
            functions=functions_def,
            debug=args.debug
        )

        results: List[Dict[str, Any]] = []
        print("\n--- Starting Constrained Decoding ---")

        # テスト問題を1問ずつ処理するループ
        for i, item in enumerate(prompts):
            prompt_text = item.get("prompt", "")
            print(f"\n[{i + 1}/{len(prompts)}] Processing: '{prompt_text}'")

            # 万が一のパースエラーや無限ループ時の保険として3回までリトライさせる
            max_retries = 3
            for attempt in range(max_retries):
                json_str = generator.generate_function_call(prompt_text)

                try:
                    # 出来上がった文字列が本当にJSONとして正しいかテスト
                    parsed_result = json.loads(json_str)
                    results.append(parsed_result)
                    break # 成功したらリトライを抜ける
                except json.JSONDecodeError as je:
                    if attempt < max_retries - 1:
                        print(
                            f"  ⚠️ Retry {attempt + 1}/{max_retries}: "
                            f"Invalid JSON detected. Retrying..."
                        )
                    else:
                        # 3回失敗したら諦めて、原因特定のために生の出力を表示して落とす
                        print("\n❌ [CRITICAL] All retries failed!")
                        print("=== RAW LLM OUTPUT ===")
                        print(json_str)
                        print("=======================")
                        raise je

        # 結果をファイルに書き出す（フォルダが無ければ作成）
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"\nSuccessfully saved results to {output_path}")

    # クラッシュ時は標準エラー出力に内容を出して終了コード1を返す
    except json.JSONDecodeError as e:
        print(f"\nJSON Parsing Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
