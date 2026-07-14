# src/__main__.py
import argparse
import json
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.json_generator import JsonGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrained Decoding")
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    func_path = Path(args.functions_definition)

    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        sys.exit(1)
    if not func_path.exists():
        print(f"Error: {func_path} not found.")
        sys.exit(1)

    with open(func_path, "r", encoding="utf-8") as f:
        functions = json.load(f)

    with open(input_path, "r", encoding="utf-8") as f:
        tests = json.load(f)

    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()

    generator = JsonGenerator(
        model=model,
        vocab_path=vocab_path,
        functions=functions,
        debug=args.debug
    )

    results = []
    for idx, t in enumerate(tests):
        prompt = t["prompt"]
        print(f"Processing {idx+1}/{len(tests)}: {prompt}")
        res_str = generator.generate_function_call(prompt)
        try:
            res_json = json.loads(res_str)
            if res_json.get("name") == "unknown":
                print("  ⚠️ Skipped: Unknown function.")
            else:
                results.append(res_json)
                print("  ✅ Success.")
        except json.JSONDecodeError:
            print(f"  ❌ Parse Error: {res_str}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
