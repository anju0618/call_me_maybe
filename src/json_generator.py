# src/json_generator.py

import json
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field
from src.json_state import JsonState
from src.token_filter import TokenFilter
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class JsonGenerator(BaseModel):
    """
    制約デコーディング（Constrained Decoding）を用いて、
    LLMに必ず有効な関数呼び出しのJSONを生成させるクラス。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    vocab_path: str
    functions: List[Dict[str, Any]]
    debug: bool = Field(default=False)

    token_filter: Optional[TokenFilter] = Field(default=None)
    all_ids: Set[int] = Field(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        tf = TokenFilter(vocab_path=self.vocab_path)
        self.token_filter = tf
        self.all_ids.update(tf.all_token_ids)

    def generate_function_call(self, prompt: str) -> str:
        if self.token_filter is None:
            raise RuntimeError("TokenFilter is not initialized.")

        # LLMに渡す関数の定義を整理（descriptionは残し、複雑な型は除去）
        clean_funcs = []
        for func in self.functions:
            c_func = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": {}
            }
            for pk, p_info in func.get("parameters", {}).items():
                c_func["parameters"][pk] = p_info.get("type", "string")
            clean_funcs.append(c_func)

        # 該当しない質問が来た時のために「unknown」という逃げ道を用意
        clean_funcs.append({"name": "unknown", "parameters": {}})
        f_str = json.dumps(clean_funcs, ensure_ascii=False)

        # 【プロンプト設計】LLMに抽出ルールを学習させる（1行79文字制限対応）
        context = (
            "System: Extract parameters EXACTLY from the User prompt.\n"
            "- Add '.0' to whole numbers (e.g., 3.0).\n"
            "- DO NOT add leading spaces to string values.\n"
            "- Preserve quotes inside strings.\n"
            "- Remove 'Format template: ' prefix.\n"
            "- Select 'unknown' for irrelevant prompts.\n\n"
            "Ex1:\nUser: What is the product of 12 and 4?\n"
            "JSON: {\"prompt\":\"What is the product of 12 and 4?\","
            "\"name\":\"fn_multiply_numbers\",\"parameters\":"
            "{\"a\":12.0,\"b\":4.0}}\n\n"
            "Ex2:\nUser: Format template: Say \"hi\" to {x}\n"
            "JSON: {\"prompt\":\"Format template: Say \\\"hi\\\" "
            "to {x}\",\"name\":\"fn_format_template\","
            "\"parameters\":{\"template\":\"Say \\\"hi\\\" to "
            "{x}\"}}\n\n"
            "Ex3:\nUser: Read file at /tmp/data.csv with ascii\n"
            "JSON: {\"prompt\":\"Read file at /tmp/data.csv with "
            "ascii\",\"name\":\"fn_read_file\",\"parameters\":{"
            "\"path\":\"/tmp/data.csv\",\"encoding\":\"ascii\"}}\n\n"
            "Ex4:\nUser: Greet shrek\n"
            "JSON: {\"prompt\":\"Greet shrek\",\"name\":\"fn_greet\","
            "\"parameters\":{\"name\":\"shrek\"}}\n\n"
            "Ex5:\nUser: Replace numbers in 'abc 12' with X\n"
            "JSON: {\"prompt\":\"Replace numbers in 'abc 12' with X\","
            "\"name\":\"fn_substitute_string_with_regex\","
            "\"parameters\":{\"source_string\":\"abc 12\",\"regex\":"
            "\"/\\\\d+/g\",\"replacement\":\"X\"}}\n\n"
            "Ex6:\nUser: Is 10 a prime number?\n"
            "JSON: {\"prompt\":\"Is 10 a prime number?\",\"name\":"
            "\"fn_is_prime\",\"parameters\":{\"n\":10}}\n\n"
            "Ex7:\nUser: what is the capital of Paris?\n"
            "JSON: {\"prompt\":\"what is the capital of Paris?\","
            "\"name\":\"unknown\",\"parameters\":{}}\n\n"
            f"Functions:{f_str}\nUser:{prompt}\nJSON:"
        )

        selected_function: Dict[str, Any] = {}
        current_param_index = 0
        param_keys: List[str] = []

        # 値の文字列がどこから始まったかを記録する変数
        # 例: '{"prompt":"...", "name":"fn", "parameters":{"a":' まで記録
        value_start_text = ""

        # ========================================================
        # 【強制ワープ（Fast-Forwarding）の準備】
        # ボイラープレート（固定文字列）をLLMに推論させると遅く、幻覚の
        # 原因になるため、最初からプロンプト転記部分は自力で入力しておく。
        # ========================================================
        p_json = json.dumps(prompt)
        # prefix: '{"prompt":"質問文","name":"' （LLMは関数名から推論開始）
        prefix = '{"prompt":' + p_json + ',"name":"'

        current_text = prefix
        current_state = JsonState.FUNCTION_NAME

        # LLMの履歴(input_ids)に「すでに自分がprefixまで喋った」と錯覚させる
        input_ids = self.token_filter.encode(context + current_text)

        # 1トークンずつ生成するループ（最大500トークン）
        for _ in range(500):
            if not input_ids:
                logits = [0.0] * len(self.token_filter.id_to_token)
            else:
                raw_logits = self.model.get_logits_from_input_ids(input_ids)
                if hasattr(raw_logits, "tolist"):
                    logits = raw_logits.tolist()
                else:
                    logits = list(raw_logits)

            allowed_tokens: Set[int] = set()

            # --------------------------------------------------------
            # 制約（Constrained Decoding）の設定
            # 現在のStateに応じて、選んでよいトークンIDをallowed_tokensに詰める
            # --------------------------------------------------------
            if current_state == JsonState.FUNCTION_NAME:
                al_names = [f["name"] for f in self.functions] + ["unknown"]
                for f_name in al_names:
                    # 例: '{"prompt":"...","name":"fn_add_numbers"' を目指す
                    ft = prefix + f_name + '"'
                    tokens = self.token_filter.filter_by_prefix(
                        current_text, ft
                    )
                    allowed_tokens.update(tokens)

            elif current_state == JsonState.PARAM_VALUE:
                p_key = param_keys[current_param_index]
                p_info = selected_function["parameters"][p_key]
                p_type = p_info.get("type", "string")

                if p_type in ["number", "integer"]:
                    # 数字として有効なトークンだけを許可
                    tokens_list = self.token_filter.filter_numeric_tokens(
                        is_start=(len(current_text) == len(value_start_text)),
                        is_integer=(p_type == "integer")
                    )
                    allowed_tokens = set(tokens_list)

                    # 現在抽出中の数字部分を取得（例: "12.0"）
                    num_part = current_text[len(value_start_text):]
                    c_len = 0
                    v_chars = (
                        "0123456789.-" if p_type == "number"
                        else "0123456789-"
                    )
                    for char in num_part:
                        if char in v_chars:
                            c_len += 1
                        else:
                            break
                    clean_num = num_part[:c_len]

                    # 数値の終了条件として「,（カンマ）」と「}（カッコ）」を許可
                    full_exit_comma = value_start_text + clean_num + ","
                    full_exit_brace = value_start_text + clean_num + "}"

                    allowed_tokens.update(
                        self.token_filter.filter_by_prefix(
                            current_text, full_exit_comma
                        )
                    )
                    allowed_tokens.update(
                        self.token_filter.filter_by_prefix(
                            current_text, full_exit_brace
                        )
                    )

                elif p_type == "boolean":
                    # true または false の文字列のみを許可
                    targets = [
                        value_start_text + "true,",
                        value_start_text + "false,",
                        value_start_text + "true}",
                        value_start_text + "false}"
                    ]
                    for t in targets:
                        allowed_tokens.update(
                            self.token_filter.filter_by_prefix(current_text, t)
                        )

                else:
                    # 文字列の推論: 改行などを含まないすべての文字を許可
                    t_filter = self.token_filter
                    allowed_tokens.update(t_filter.valid_string_tokens)

                    # 文字列の終了を示す「"」を許可
                    full_exit_q = current_text + '"'
                    allowed_tokens.update(
                        self.token_filter.filter_by_prefix(
                            current_text, full_exit_q
                        )
                    )

            if not allowed_tokens and self.debug:
                print(f"⚠️ [WARNING] No tokens at: {current_state.name}!")

            # Logits（確率）が最も高い「許可されたトークン」を選ぶ
            if allowed_tokens:
                valid_ids = {t for t in allowed_tokens if t < len(logits)}
                if valid_ids:
                    next_token_id = max(valid_ids, key=logits.__getitem__)
                else:
                    next_token_id = int(logits.index(max(logits)))
            else:
                next_token_id = int(logits.index(max(logits)))

            next_token_str = self.token_filter.id_to_token[next_token_id]
            clean_next_str = next_token_str.replace("Ġ", " ").replace(" ", " ")
            current_text += clean_next_str

            input_ids.append(next_token_id)

            if self.debug:
                print(
                    f"  [State: {current_state.name:13}] "
                    f"Appended: {repr(clean_next_str):12} -> "
                    f"Current: {repr(current_text)}"
                )

            # ========================================================
            # 状態遷移のチェック ＆ 強制ワープ実行
            # LLMが1つデータを出力し終えたら、Python側で次のキー名を自力で
            # 足してしまい、LLMには値の穴埋めだけをやらせます。
            # ========================================================
            while True:
                old_state = current_state

                # 【関数名の抽出完了チェック】
                if current_state == JsonState.FUNCTION_NAME:
                    # 末尾にダブルクォーテーションが出たら関数名が確定
                    if current_text.endswith('"'):
                        f_name = current_text[len(prefix):-1]

                        if f_name == "unknown":
                            return current_text + ',"parameters":{}}'

                        selected_func = None
                        for f in self.functions:
                            if f["name"] == f_name:
                                selected_func = f
                                break

                        if not selected_func:
                            return current_text + ',"parameters":{}}'

                        param_keys = list(
                            selected_func.get("parameters", {}).keys()
                        )

                        # 引数がない関数ならここで完成
                        if not param_keys:
                            return current_text + ',"parameters":{}}'

                        # 【爆速化ワープ】最初のパラメータのキー名と記号を足す
                        # 例: ',"parameters":{"a":' （型に応じて " の有無を調整）
                        p_type = selected_func["parameters"][
                            param_keys[0]
                        ].get("type", "string")

                        if p_type == "string":
                            ff_str = ',"parameters":{"' + param_keys[0] + '":"'
                        else:
                            ff_str = ',"parameters":{"' + param_keys[0] + '":'

                        current_text += ff_str
                        if self.token_filter is not None:
                            input_ids.extend(self.token_filter.encode(ff_str))

                        current_param_index = 0
                        current_state = JsonState.PARAM_VALUE
                        value_start_text = current_text
                        selected_function = selected_func

                # 【パラメータ値の抽出完了チェック】
                elif current_state == JsonState.PARAM_VALUE:
                    p_key = param_keys[current_param_index]
                    p_type = selected_function["parameters"][p_key].get(
                        "type", "string"
                    )

                    value_is_finished = False

                    if p_type in ["number", "integer", "boolean"]:
                        # 数値やboolは、末尾に「,」か「}」が出たら値の終了とみなす
                        if current_text.endswith(','):
                            current_text = current_text[:-1]
                            value_is_finished = True
                        elif current_text.endswith('}'):
                            current_text = current_text[:-1]
                            value_is_finished = True
                    else:
                        # 文字列は、エスケープされていない「"」が出たら終了とみなす
                        val_str = current_text[len(value_start_text):]
                        if len(val_str) > 0 and val_str.endswith('"'):
                            escape_count = 0
                            # 末尾の " の直前にある \（バックスラッシュ）の数を数える
                            for char in reversed(val_str[:-1]):
                                if char == '\\':
                                    escape_count += 1
                                else:
                                    break
                            # \ が偶数個なら、最後の " はエスケープされていない（文字列の終わり）
                            if escape_count % 2 == 0:
                                value_is_finished = True

                    # 1つの値が抽出完了した場合の処理
                    if value_is_finished:
                        current_param_index += 1

                        # まだ次の引数が残っている場合
                        if current_param_index < len(param_keys):
                            n_key = param_keys[current_param_index]
                            n_type = selected_function["parameters"][
                                n_key
                            ].get("type", "string")

                            # 【爆速化ワープ】次のパラメータのキー名を自力で足す
                            # 例: ',"b":'
                            if n_type == "string":
                                ff_str = ',"' + n_key + '":"'
                            else:
                                ff_str = ',"' + n_key + '":'

                            current_text += ff_str
                            if self.token_filter is not None:
                                input_ids.extend(
                                    self.token_filter.encode(ff_str)
                                )
                            value_start_text = current_text

                        # すべての引数が出力し終わった場合
                        else:
                            return current_text + '}}'

                if current_state == old_state:
                    break

        return current_text
