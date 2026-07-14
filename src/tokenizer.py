# src/tokenizer.py

import json
import argparse
from typing import Dict, List, Tuple


class CustomTokenizer:
    """
    vocab.jsonのみを使用して
    文字列とトークンIDの変換を行うトークナイザー
    """
    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_dict: Dict[str, int] = json.load(f)

        # IDとトークンのマッピングを作成
        self.token_to_id = vocab_dict
        self.id_to_token = {v: k for k, v in vocab_dict.items()}

        # 最長一致（Greedy）検索用：文字数が長い順にソートして保持
        clean_list = []
        for t_id, t_str in self.id_to_token.items():
            cl_str = t_str.replace("Ġ", " ").replace(" ", " ")
            if cl_str:
                clean_list.append((cl_str, t_id))

        clean_list.sort(key=lambda x: len(x[0]), reverse=True)
        self.sorted_clean_tokens: List[Tuple[str, int]] = clean_list

    def encode(self, text: str) -> List[int]:
        """文字列をトークンIDのリストに変換（エンコードと同じ）"""
        token_ids = []
        i = 0
        text_len = len(text)
        while i < text_len:
            match_found = False
            # 長い文字列のトークンから順に前方一致を試す
            for t_str, t_id in self.sorted_clean_tokens:
                if text.startswith(t_str, i):
                    token_ids.append(t_id)
                    i += len(t_str)
                    match_found = True
                    break
            if not match_found:
                # 未知の文字（ボキャブラリにない文字）はスキップして無限ループを防止
                i += 1
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """トークンIDのリストを文字列に復元（デコード）"""
        result = ""
        for t_id in token_ids:
            if t_id in self.id_to_token:
                t_str = self.id_to_token[t_id]
                # AI特有のスペース記号を人間の半角スペースに変換
                result += t_str.replace("Ġ", " ").replace(" ", " ")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Tokenizer CLI")
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to the vocabulary JSON file (e.g., vocab.json)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a tokenizer test.",
        help="Text to encode and decode"
    )
    args = parser.parse_args()

    tokenizer = CustomTokenizer(args.vocab)
    
    print("=== 42 Call Me Maybe: Custom Tokenizer ===")
    print(f"Original Text : '{args.text}'")
    
    encoded_ids = tokenizer.encode(args.text)
    print(f"Encoded IDs   : {encoded_ids}")
    
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"Decoded Text  : '{decoded_text}'")
    
    if args.text == decoded_text:
        print("\n✅ Success: Decoded text perfectly matches the original!")
    else:
        print("\n⚠️ Warning: Decoded text differs (expected behavior for some special characters).")