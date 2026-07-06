# src/__main__.py

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数処理"""
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
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    func_def_path = Path(args.functions_definition)
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        if not func_def_path.exists():
            raise FileNotFoundError(
                f"Functions definition file not found: {func_def_path}"
            )
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(func_def_path, 'r', encoding='utf-8') as f:
            functions_def = json.load(f)

        with open(input_path, 'r', encoding='utf-8') as f:
            prompts = json.load(f)

        print(
            f"Loaded {len(functions_def)} functions "
            f"and {len(prompts)} prompts."
        )

        # ---------------------------------------------------
        # LLM SDK テスト
        print("\n--- LLM SDK Test ---")
        print("Loading LLM model (Qwen3-0.6B)...")
        model = Small_LLM_Model()

        test_text = "What is the sum of 2 and 3?"
        # encodeは2次元テンソルを返してくるので、
        # 中身のトークンIDリストを抽出して表示します
        input_tensor = model.encode(test_text)
        input_ids = input_tensor[0].tolist()

        print(f"Original Text: {test_text}")
        print(f"Token IDs: {input_ids}")

        vocab_path = model.get_path_to_vocab_file()
        print(f"Vocab file path: {vocab_path}")
        print("---------------------\n")
        # ---------------------------------------------------

        results: List[Dict[str, Any]] = []
        for item in prompts:
            prompt_text = item.get("prompt", "")
            dummy_result = {
                "prompt": prompt_text,
                "name": "dummy_function",
                "parameters": {}
            }
            results.append(dummy_result)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"Successfully saved results to {output_path}")

    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
