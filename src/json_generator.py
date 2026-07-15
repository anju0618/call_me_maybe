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

        f_str = json.dumps(clean_funcs, ensure_ascii=False)

        # 【超爆速化・デッドロック防止】空白を排除し、LLMにコンパクトなJSONを学習させる
        context = (
            "System: You are an expert data extraction AI.\n"
            "- Extract paths EXACTLY, including the leading '/'.\n"
            "- Use JS regex (e.g., /pattern/g), escape backslashes (\\\\d+).\n"
            "- Select 'unknown' function for irrelevant prompts.\n\n"
            "Ex1:\nUser: Find sum of 265 and 345\n"
            "JSON: {\"prompt\":\"Find sum of 265 and 345\",\"name\":"
            "\"fn_add_numbers\",\"parameters\":{\"a\":265,\"b\":345}}\n\n"
            "Ex2:\nUser: Compute product of 3 and 5\n"
            "JSON: {\"prompt\":\"Compute product of 3 and 5\",\"name\":"
            "\"fn_multiply_numbers\",\"parameters\":{\"a\":3.0,\"b\":5.0}}\n\n"
            "Ex3:\nUser: Say hello to shrek\n"
            "JSON: {\"prompt\":\"Say hello to shrek\",\"name\":\"fn_greet\","
            "\"parameters\":{\"name\":\"shrek\"}}\n\n"
            "Ex4:\nUser: Replace numbers in 'abc 12' with X\n"
            "JSON: {\"prompt\":\"Replace numbers in 'abc 12' with X\","
            "\"name\":\"fn_substitute_string_with_regex\",\"parameters\":"
            "{\"source_string\":\"abc 12\",\"regex\":\"/\\\\d+/g\","
            "\"replacement\":\"X\"}}\n\n"
            "Ex5:\nUser: Read file at /var/log.txt with ascii\n"
            "JSON: {\"prompt\":\"Read file at /var/log.txt with ascii\","
            "\"name\":\"fn_read_file\",\"parameters\":{\"path\":"
            "\"/var/log.txt\",\"encoding\":\"ascii\"}}\n\n"
            "Ex6:\nUser: what is the capital of Paris?\n"
            "JSON: {\"prompt\":\"what is the capital of Paris?\",\"name\":\"unknown\","
            "\"parameters\":{}}\n\n"
            f"Functions:{f_str}\nUser:{prompt}\nJSON:"
        )

        current_text = ""
        current_state = JsonState.START

        selected_function: Dict[str, Any] = {}
        current_param_index = 0
        param_keys: List[str] = []
        is_numeric_start = True
        param_base_text = ""
        value_start_text = ""

        p_json = json.dumps(prompt)
        input_ids = self.token_filter.encode(context)

        for _ in range(500):
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

            if current_state == JsonState.START:
                # 【修正】空白を完全排除
                full_target = '{"prompt":"'
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(
                        current_text, full_target
                    )
                )

            elif current_state == JsonState.PROMPT_VALUE:
                full_target = '{"prompt":' + p_json
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(
                        current_text, full_target
                    )
                )

            elif current_state == JsonState.NAME_KEY:
                full_target = '{"prompt":' + p_json + ',"name":"'
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(
                        current_text, full_target
                    )
                )

            elif current_state == JsonState.FUNCTION_NAME:
                al_names = [f["name"] for f in self.functions] + ["unknown"]
                for f_name in al_names:
                    ft = '{"prompt":' + p_json + ',"name":"' + f_name + '"'
                    tokens = self.token_filter.filter_by_prefix(
                        current_text, ft
                    )
                    allowed_tokens.update(tokens)

            elif current_state == JsonState.PARAMS_START:
                ft = (
                    '{"prompt":' + p_json + ',"name":"'
                    + selected_function["name"] + '","parameters":{'
                )
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(
                        current_text, ft
                    )
                )

            elif current_state == JsonState.PARAM_KEY:
                if current_param_index < len(param_keys):
                    p_key = param_keys[current_param_index]
                    ft = param_base_text + f'"{p_key}":'
                else:
                    if len(param_keys) == 0:
                        ft = param_base_text + "}}"
                    else:
                        ft = param_base_text + "}"

                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(
                        current_text, ft
                    )
                )

            elif current_state == JsonState.PARAM_VALUE:
                p_key = param_keys[current_param_index]
                p_info = selected_function["parameters"][p_key]
                p_type = p_info.get("type", "string")

                if p_type in ["number", "integer"]:
                    tokens_list = self.token_filter.filter_numeric_tokens(
                        is_start=is_numeric_start,
                        is_integer=(p_type == "integer")
                    )
                    allowed_tokens = set(tokens_list)
                    is_numeric_start = False

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
                            if suffix.startswith(","):
                                current_param_index += 1
                                param_base_text = (
                                    value_start_text + clean_num + ","
                                )
                                current_state = JsonState.PARAM_KEY
                        else:
                            if suffix.startswith("}"):
                                current_param_index += 1
                                param_base_text = (
                                    value_start_text + clean_num + "}"
                                )
                                current_state = JsonState.PARAM_KEY

                    if current_param_index + 1 < len(param_keys):
                        n_key = param_keys[current_param_index + 1]
                        full_exit_target = (
                            value_start_text + clean_num
                            + f',"{n_key}":'
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
                    if current_param_index + 1 < len(param_keys):
                        n_key = param_keys[current_param_index + 1]
                        t_true = value_start_text + "true" + f',"{n_key}":'
                        t_false = value_start_text + "false" + f',"{n_key}":'
                        if current_text in (t_true, t_false):
                            current_param_index += 1
                            param_base_text = current_text
                            current_state = JsonState.PARAM_KEY
                    else:
                        t_true = value_start_text + "true}"
                        t_false = value_start_text + "false}"
                        if current_text in (t_true, t_false):
                            current_param_index += 1
                            param_base_text = current_text
                            current_state = JsonState.PARAM_KEY

                    if current_param_index + 1 < len(param_keys):
                        n_key = param_keys[current_param_index + 1]
                        targets = [
                            value_start_text + "true" + f',"{n_key}":',
                            value_start_text + "false" + f',"{n_key}":'
                        ]
                    else:
                        targets = [
                            value_start_text + "true}",
                            value_start_text + "false}"
                        ]

                    for t in targets:
                        allowed_tokens.update(
                            self.token_filter.filter_by_prefix(
                                current_text, t
                            )
                        )

                else:
                    s_part = current_text[len(value_start_text):]
                    if not s_part.startswith('"'):
                        t_filter = self.token_filter
                        allowed_tokens.update(
                            t_filter.string_start_tokens['"']
                        )
                        if p_key in ["path", "regex"]:
                            allowed_tokens.update(
                                t_filter.string_start_tokens['"/']
                            )
                            allowed_tokens.update(
                                t_filter.string_start_tokens['"C']
                            )
                            allowed_tokens.update(
                                t_filter.string_start_tokens['"C:\\']
                            )
                        elif p_key == "template":
                            allowed_tokens.update(
                                t_filter.string_start_tokens['"{']
                            )
                    else:
                        escape = False
                        quote_idx = -1
                        for idx in range(1, len(s_part)):
                            if escape:
                                escape = False
                            elif s_part[idx] == '\\':
                                escape = True
                            elif s_part[idx] == '"':
                                quote_idx = idx
                                break

                        if quote_idx == -1:
                            t_filter = self.token_filter
                            allowed_tokens.update(
                                t_filter.valid_string_tokens
                            )

                            if current_param_index + 1 < len(param_keys):
                                n_key = param_keys[current_param_index + 1]
                                full_exit = (
                                    current_text + '"' + f',"{n_key}":'
                                )
                            else:
                                full_exit = current_text + '"}'

                            allowed_tokens.update(
                                self.token_filter.filter_by_prefix(
                                    current_text, full_exit
                                )
                            )
                        else:
                            val_with_q = s_part[:quote_idx+1]
                            if current_param_index + 1 < len(param_keys):
                                n_key = param_keys[current_param_index + 1]
                                full_exit = (
                                    value_start_text + val_with_q
                                    + f',"{n_key}":'
                                )
                            else:
                                full_exit = (
                                    value_start_text + val_with_q + "}"
                                )

                            allowed_tokens.update(
                                self.token_filter.filter_by_prefix(
                                    current_text, full_exit
                                )
                            )

            if not allowed_tokens and self.debug:
                print(
                    f"⚠️ [WARNING] No tokens allowed at: "
                    f"{current_state.name}!"
                )

            # 【安全装置】IndexErrorを防ぐために有効なトークンのみを選択
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

            while True:
                old_state = current_state

                if current_state == JsonState.START:
                    if current_text.endswith('{"prompt":"'):
                        current_state = JsonState.PROMPT_VALUE

                elif current_state == JsonState.PROMPT_VALUE:
                    expected_pv = '{"prompt":' + p_json
                    if current_text == expected_pv:
                        current_state = JsonState.NAME_KEY

                elif current_state == JsonState.NAME_KEY:
                    expected_nk = '{"prompt":' + p_json + ',"name":"'
                    if current_text == expected_nk:
                        current_state = JsonState.FUNCTION_NAME

                elif current_state == JsonState.FUNCTION_NAME:
                    al_names = (
                        [f["name"] for f in self.functions] + ["unknown"]
                    )
                    for f_name in al_names:
                        fm = '{"prompt":' + p_json + ',"name":"' + f_name + '"'
                        if current_text == fm:
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
                                selected_function.get(
                                    "parameters", {}
                                ).keys()
                            )
                            current_param_index = 0
                            current_state = JsonState.PARAMS_START
                            break

                elif current_state == JsonState.PARAMS_START:
                    t_ps = (
                        '{"prompt":' + p_json + ',"name":"'
                        + selected_function["name"] + '","parameters":{'
                    )
                    if current_text == t_ps:
                        current_state = JsonState.PARAM_KEY
                        param_base_text = current_text

                elif current_state == JsonState.PARAM_KEY:
                    if current_param_index < len(param_keys):
                        p_key = param_keys[current_param_index]
                        expected_pk = param_base_text + f'"{p_key}":'
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
                                if suffix.startswith(","):
                                    current_param_index += 1
                                    param_base_text = (
                                        value_start_text + clean_num + ","
                                    )
                                    current_state = JsonState.PARAM_KEY
                            else:
                                if suffix.startswith("}"):
                                    current_param_index += 1
                                    param_base_text = (
                                        value_start_text + clean_num + "}"
                                    )
                                    current_state = JsonState.PARAM_KEY

                    elif p_type == "boolean":
                        if current_param_index + 1 < len(param_keys):
                            n_key = param_keys[current_param_index + 1]
                            t_true = value_start_text + "true" + f',"{n_key}":'
                            t_false = (
                                value_start_text + "false" + f',"{n_key}":'
                            )
                            if current_text in (t_true, t_false):
                                current_param_index += 1
                                param_base_text = current_text
                                current_state = JsonState.PARAM_KEY
                        else:
                            t_true = value_start_text + "true}"
                            t_false = value_start_text + "false}"
                            if current_text in (t_true, t_false):
                                current_param_index += 1
                                param_base_text = current_text
                                current_state = JsonState.PARAM_KEY

                    else:
                        s_part = current_text[len(value_start_text):]
                        if s_part.startswith('"') and len(s_part) > 1:
                            escape = False
                            quote_idx = -1
                            for idx in range(1, len(s_part)):
                                if escape:
                                    escape = False
                                elif s_part[idx] == '\\':
                                    escape = True
                                elif s_part[idx] == '"':
                                    quote_idx = idx
                                    break

                            if quote_idx != -1:
                                val_with_q = s_part[:quote_idx+1]
                                suffix = s_part[quote_idx + 1:]

                                if suffix:
                                    has_nxt = (
                                        current_param_index + 1
                                        < len(param_keys)
                                    )
                                    if has_nxt:
                                        if suffix.startswith(","):
                                            current_param_index += 1
                                            param_base_text = (
                                                value_start_text
                                                + val_with_q + ","
                                            )
                                            current_state = (
                                                JsonState.PARAM_KEY
                                            )
                                    else:
                                        if suffix.startswith("}"):
                                            current_param_index += 1
                                            param_base_text = (
                                                value_start_text
                                                + val_with_q + "}"
                                            )
                                            current_state = (
                                                JsonState.PARAM_KEY
                                            )

                if current_state == old_state:
                    break

        return current_text
