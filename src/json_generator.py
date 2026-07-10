# src/json_generator.py

import json
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field
from src.json_state import JsonState
from src.token_filter import TokenFilter
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class JsonGenerator(BaseModel):
    # Pydantic用
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    vocab_path: str
    functions: List[Dict[str, Any]]
    debug: bool = Field(default=False)

    token_filter: Optional[TokenFilter] = Field(default=None)
    all_ids: Set[int] = Field(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        # フィルターの準備
        tf = TokenFilter(vocab_path=self.vocab_path)
        self.token_filter = tf
        self.all_ids.update(tf.all_token_ids)

    def generate_function_call(self, prompt: str) -> str:
        if self.token_filter is None:
            raise RuntimeError("TokenFilter is not initialized.")

        # descriptionのコピペ対策。型だけ残す
        clean_funcs = []
        for func in self.functions:
            c_func = {"name": func["name"], "parameters": {}}
            for pk, p_info in func.get("parameters", {}).items():
                c_func["parameters"][pk] = p_info.get("type", "string")
            clean_funcs.append(c_func)

        funcs_str = json.dumps(clean_funcs, ensure_ascii=False)

        # Few-Shotを入れてハルシネーションを防ぐ
        context = (
            "System: You are an expert data extraction AI. "
            "Extract the exact substrings directly from the User prompt. "
            "Do NOT invent words. Do NOT use placeholders like "
            "'description'.\n\n"
            "Example 1:\n"
            "User: Greet alice\n"
            "JSON: {\"prompt\": \"Greet alice\", \"name\": \"fn_greet\", "
            "\"parameters\": {\"name\": \"alice\"}}\n\n"
            "Example 2:\n"
            "User: Substitute the word 'apple' with 'orange' in "
            "'I have an apple'\n"
            "JSON: {\"prompt\": \"Substitute the word 'apple' with "
            "'orange' in 'I have an apple'\", \"name\": "
            "\"fn_substitute_string_with_regex\", \"parameters\": "
            "{\"source_string\": \"I have an apple\", \"regex\": "
            "\"/apple/g\", \"replacement\": \"orange\"}}\n\n"
            f"Functions: {funcs_str}\n"
            f"User: {prompt}\n"
            "JSON:"
        )

        current_text = ""
        current_state = JsonState.START

        selected_function: Dict[str, Any] = {}
        current_param_index = 0
        param_keys: List[str] = []
        is_numeric_start = True
        param_base_text = ""
        value_start_text = ""

        # エスケープ対策
        prompt_json = json.dumps(prompt)

        for _ in range(500):
            full_prompt = context + current_text
            input_tensor = self.model.encode(full_prompt)

            # tensorかlistか判定
            if hasattr(input_tensor, "tolist"):
                input_ids = input_tensor[0].tolist()
            else:
                input_ids = input_tensor

            if not input_ids:
                logits = [0.0] * len(self.token_filter.id_to_token)
            else:
                raw_logits = self.model.get_logits_from_input_ids(
                    input_ids
                )
                if hasattr(raw_logits, "tolist"):
                    logits = raw_logits.tolist()
                else:
                    logits = list(raw_logits)

            allowed_tokens: Set[int] = set()

            # --- 1. 状態ごとに絶対ターゲットを作る ---
            if current_state == JsonState.START:
                full_target = '{"prompt": "'
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(
                        current_text, full_target
                    )
                )

            elif current_state == JsonState.PROMPT_VALUE:
                full_target = '{"prompt": ' + prompt_json
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(
                        current_text, full_target
                    )
                )

            elif current_state == JsonState.NAME_KEY:
                full_target = '{"prompt": ' + prompt_json + ', "name": "'
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(
                        current_text, full_target
                    )
                )

            elif current_state == JsonState.FUNCTION_NAME:
                # 全関数のパターンを許容する
                for func in self.functions:
                    full_target = (
                        '{"prompt": ' + prompt_json + ', "name": "'
                        + func["name"] + '"'
                    )
                    tokens = self.token_filter.filter_by_prefix(
                        current_text, full_target
                    )
                    allowed_tokens.update(tokens)

            elif current_state == JsonState.PARAMS_START:
                full_target = (
                    '{"prompt": ' + prompt_json + ', "name": "'
                    + selected_function["name"] + '", "parameters": {'
                )
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(
                        current_text, full_target
                    )
                )

            elif current_state == JsonState.PARAM_KEY:
                if current_param_index < len(param_keys):
                    p_key = param_keys[current_param_index]
                    full_target = param_base_text + f'"{p_key}": '
                    allowed_tokens = set(
                        self.token_filter.filter_by_prefix(
                            current_text, full_target
                        )
                    )
                else:
                    # 全部終わったらカッコを閉じる
                    if len(param_keys) == 0:
                        full_target = param_base_text + "}}"
                    else:
                        full_target = param_base_text + "}"

                    allowed_tokens = set(
                        self.token_filter.filter_by_prefix(
                            current_text, full_target
                        )
                    )

            elif current_state == JsonState.PARAM_VALUE:
                p_key = param_keys[current_param_index]
                p_info = selected_function["parameters"][p_key]
                p_type = p_info.get("type")

                if p_type == "number":
                    tokens_list = (
                        self.token_filter.filter_numeric_tokens(
                            is_start=is_numeric_start
                        )
                    )
                    allowed_tokens = set(tokens_list)
                    is_numeric_start = False

                    num_part = current_text[len(value_start_text):]
                    c_len = 0
                    for char in num_part:
                        if char in "0123456789.-":
                            c_len += 1
                        else:
                            break
                    clean_num = num_part[:c_len]

                    # 次の引数があるかでの分岐
                    if current_param_index + 1 < len(param_keys):
                        n_key = param_keys[current_param_index + 1]
                        full_exit_target = (
                            value_start_text + clean_num
                            + f', "{n_key}": '
                        )
                    else:
                        full_exit_target = (
                            value_start_text + clean_num + "}"
                        )

                    allowed_tokens.update(
                        self.token_filter.filter_by_prefix(
                            current_text, full_exit_target
                        )
                    )

                elif p_type == "string":
                    s_part = current_text[len(value_start_text):]
                    if not s_part.startswith('"'):
                        # 【究極の修正】最初の文字は必ず単独の `"` のみ許可
                        # 合体トークンのハイジャックを物理的に防ぐ
                        for t_id, t_str in (
                            self.token_filter.id_to_token.items()
                        ):
                            cl_str = t_str.replace("Ġ", " ").replace(" ", " ")
                            if cl_str == '"':
                                allowed_tokens.add(t_id)

                        # 出口のターゲット
                        if current_param_index + 1 < len(param_keys):
                            n_key = param_keys[current_param_index + 1]
                            full_exit = '"' + f', "{n_key}": '
                        else:
                            full_exit = '"}'

                        allowed_tokens.update(
                            self.token_filter.filter_by_prefix(
                                current_text, current_text + full_exit
                            )
                        )
                    else:
                        quote_idx = s_part.find('"', 1)
                        if quote_idx == -1:
                            # 文字列の中身を生成
                            inv_chars = set("\"\n\rĊ{}")
                            for t_id, t_str in (
                                self.token_filter.id_to_token.items()
                            ):
                                cl_str = t_str.replace(
                                    "Ġ", " "
                                ).replace(" ", " ")
                                if not any(c in inv_chars for c in cl_str):
                                    allowed_tokens.add(t_id)

                            if current_param_index + 1 < len(param_keys):
                                n_key = param_keys[current_param_index + 1]
                                full_exit = (
                                    current_text + '"' + f', "{n_key}": '
                                )
                            else:
                                full_exit = current_text + '"}'

                            allowed_tokens.update(
                                self.token_filter.filter_by_prefix(
                                    current_text, full_exit
                                )
                            )
                        else:
                            # 閉じた後の処理
                            cl_str_val = s_part[1:quote_idx]
                            if current_param_index + 1 < len(param_keys):
                                n_key = param_keys[current_param_index + 1]
                                full_exit = (
                                    value_start_text + '"' + cl_str_val + '"'
                                    + f', "{n_key}": '
                                )
                            else:
                                full_exit = (
                                    value_start_text + '"' + cl_str_val + '"}'
                                )

                            allowed_tokens.update(
                                self.token_filter.filter_by_prefix(
                                    current_text, full_exit
                                )
                            )

            # --- 2. ロジット・マスキング ---
            if not allowed_tokens and self.debug:
                print(
                    f"⚠️ [WARNING] No tokens allowed at: "
                    f"{current_state.name}!"
                )

            # 許可外のトークンを-infにしてマスクする
            for token_id in range(len(logits)):
                if token_id not in allowed_tokens:
                    logits[token_id] = float("-inf")

            # 最大スコアのトークンを選ぶ
            next_token_id = int(logits.index(max(logits)))
            next_token_str = self.token_filter.id_to_token[
                next_token_id
            ]
            clean_next_str = next_token_str.replace(
                "Ġ", " "
            ).replace(" ", " ")
            current_text += clean_next_str

            if self.debug:
                print(
                    f"  [State: {current_state.name:13}] "
                    f"Appended: {repr(clean_next_str):12} -> "
                    f"Current: {repr(current_text)}"
                )

            # --- 3. 状態の同期(Cascade) ---
            # 合体トークンで一気に進んだ時用に追いつくまで回す
            while True:
                old_state = current_state

                if current_state == JsonState.START:
                    if current_text.endswith('{"prompt": "'):
                        current_state = JsonState.PROMPT_VALUE

                elif current_state == JsonState.PROMPT_VALUE:
                    expected_pv = '{"prompt": ' + prompt_json
                    if current_text == expected_pv:
                        current_state = JsonState.NAME_KEY

                elif current_state == JsonState.NAME_KEY:
                    expected_nk = (
                        '{"prompt": ' + prompt_json + ', "name": "'
                    )
                    if current_text == expected_nk:
                        current_state = JsonState.FUNCTION_NAME

                elif current_state == JsonState.FUNCTION_NAME:
                    for func in self.functions:
                        full_match = (
                            '{"prompt": ' + prompt_json + ', "name": "'
                            + func["name"] + '"'
                        )
                        if current_text == full_match:
                            selected_function = func
                            param_keys = list(
                                func.get("parameters", {}).keys()
                            )
                            current_param_index = 0
                            current_state = JsonState.PARAMS_START
                            break

                elif current_state == JsonState.PARAMS_START:
                    target_ps = (
                        '{"prompt": ' + prompt_json + ', "name": "'
                        + selected_function["name"]
                        + '", "parameters": {'
                    )
                    if current_text == target_ps:
                        current_state = JsonState.PARAM_KEY
                        param_base_text = current_text

                elif current_state == JsonState.PARAM_KEY:
                    if current_param_index < len(param_keys):
                        p_key = param_keys[current_param_index]
                        expected_pk = param_base_text + f'"{p_key}": '
                        if current_text == expected_pk:
                            current_state = JsonState.PARAM_VALUE
                            is_numeric_start = True
                            value_start_text = current_text
                    else:
                        if len(param_keys) == 0:
                            if current_text == param_base_text + "}}":
                                return current_text
                        else:
                            if current_text == param_base_text + "}":
                                return current_text

                elif current_state == JsonState.PARAM_VALUE:
                    p_key = param_keys[current_param_index]
                    p_info = selected_function["parameters"][p_key]
                    p_type = p_info.get("type")

                    if p_type == "number":
                        num_part = current_text[len(value_start_text):]
                        c_len = 0
                        for char in num_part:
                            if char in "0123456789.-":
                                c_len += 1
                            else:
                                break
                        clean_num = num_part[:c_len]
                        suffix = num_part[c_len:]

                        if suffix:
                            if current_param_index + 1 < len(param_keys):
                                if suffix.startswith(", "):
                                    current_param_index += 1
                                    param_base_text = (
                                        value_start_text + clean_num
                                        + ", "
                                    )
                                    current_state = JsonState.PARAM_KEY
                            else:
                                if suffix.startswith("}"):
                                    current_param_index += 1
                                    param_base_text = (
                                        value_start_text + clean_num
                                        + "}"
                                    )
                                    current_state = JsonState.PARAM_KEY

                    elif p_type == "string":
                        s_part = current_text[len(value_start_text):]
                        if s_part.startswith('"') and len(s_part) > 1:
                            quote_idx = s_part.find('"', 1)
                            if quote_idx != -1:
                                cl_str_val = s_part[1:quote_idx]
                                suffix = s_part[quote_idx + 1:]

                                if suffix:
                                    if current_param_index + 1 < len(
                                        param_keys
                                    ):
                                        if suffix.startswith(", "):
                                            current_param_index += 1
                                            param_base_text = (
                                                value_start_text + '"'
                                                + cl_str_val + '", '
                                            )
                                            current_state = (
                                                JsonState.PARAM_KEY
                                            )
                                    else:
                                        if suffix.startswith("}"):
                                            current_param_index += 1
                                            param_base_text = (
                                                value_start_text + '"'
                                                + cl_str_val + '"}'
                                            )
                                            current_state = (
                                                JsonState.PARAM_KEY
                                            )

                if current_state == old_state:
                    break

        return current_text
