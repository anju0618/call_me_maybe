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

        # 該当しないプロンプトを処理するための「unknown」関数
        clean_funcs.append({
            "name": "unknown",
            "parameters": {}
        })

        funcs_str = json.dumps(clean_funcs, ensure_ascii=False)

        # 具体例でハルシネーションを防止
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
            "Example 3:\n"
            "User: Replace all numbers in 'Room 123' with 'X'\n"
            "JSON: {\"prompt\": \"Replace all numbers in 'Room 123' "
            "with 'X'\", \"name\": \"fn_substitute_string_with_regex\", "
            "\"parameters\": {\"source_string\": \"Room 123\", \"regex\": "
            "\"/[0-9]+/g\", \"replacement\": \"X\"}}\n\n"
            "Example 4:\n"
            "User: What is the capital of Tokyo?\n"
            "JSON: {\"prompt\": \"What is the capital of Tokyo?\", \"name\": "
            "\"unknown\", \"parameters\": {}}\n\n"
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

        # プロンプト内の記号をエスケープ
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
                # 全関数のパターン + unknown を許容する
                allowed_names = (
                    [f["name"] for f in self.functions] + ["unknown"]
                )
                for f_name in allowed_names:
                    full_target = (
                        '{"prompt": ' + prompt_json + ', "name": "'
                        + f_name + '"'
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
                p_type = p_info.get("type", "string")

                if p_type in ["number", "integer"]:
                    tokens_list = (
                        self.token_filter.filter_numeric_tokens(
                            is_start=is_numeric_start,
                            is_integer=(p_type == "integer")
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

                elif p_type == "boolean":
                    targets = []
                    if current_param_index + 1 < len(param_keys):
                        n_key = param_keys[current_param_index + 1]
                        targets.append(
                            value_start_text + "true" + f', "{n_key}": '
                        )
                        targets.append(
                            value_start_text + "false" + f', "{n_key}": '
                        )
                    else:
                        targets.append(value_start_text + "true}")
                        targets.append(value_start_text + "false}")

                    for t in targets:
                        allowed_tokens.update(
                            self.token_filter.filter_by_prefix(
                                current_text, t
                            )
                        )

                else:
                    # string またはその他のフォールバック
                    s_part = current_text[len(value_start_text):]
                    if not s_part.startswith('"'):
                        for t_id, t_str in (
                            self.token_filter.id_to_token.items()
                        ):
                            cl_str = t_str.replace("Ġ", " ").replace(" ", " ")
                            if cl_str == '"':
                                allowed_tokens.add(t_id)
                    else:
                        quote_idx = s_part.find('"', 1)
                        if quote_idx == -1:
                            inv_chars = set("\"\\\n\rĊ{}")
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

            for token_id in range(len(logits)):
                if token_id not in allowed_tokens:
                    logits[token_id] = float("-inf")

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
                    allowed_names = (
                        [f["name"] for f in self.functions] + ["unknown"]
                    )
                    for f_name in allowed_names:
                        full_match = (
                            '{"prompt": ' + prompt_json + ', "name": "'
                            + f_name + '"'
                        )
                        if current_text == full_match:
                            if f_name == "unknown":
                                selected_function = {
                                    "name": "unknown", "parameters": {}
                                }
                            else:
                                selected_function = next(
                                    f for f in self.functions 
                                    if f["name"] == f_name
                                )
                                
                            param_keys = list(
                                selected_function.get("parameters", {}).keys()
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
                    p_type = p_info.get("type", "string")

                    if p_type in ["number", "integer"]:
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

                    elif p_type == "boolean":
                        if current_param_index + 1 < len(param_keys):
                            n_key = param_keys[current_param_index + 1]
                            t_true = (
                                value_start_text + "true" + f', "{n_key}": '
                            )
                            t_false = (
                                value_start_text + "false" + f', "{n_key}": '
                            )
                            if (current_text == t_true or
                                    current_text == t_false):
                                current_param_index += 1
                                param_base_text = current_text
                                current_state = JsonState.PARAM_KEY
                        else:
                            t_true = value_start_text + "true}"
                            t_false = value_start_text + "false}"
                            if (current_text == t_true or
                                    current_text == t_false):
                                current_param_index += 1
                                param_base_text = current_text
                                current_state = JsonState.PARAM_KEY

                    else:
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
