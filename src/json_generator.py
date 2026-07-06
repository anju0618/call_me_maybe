# src/json_generator.py

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field
from src.json_state import JsonState
from src.token_filter import TokenFilter

# mypy警告よけ
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class JsonGenerator(BaseModel):
    """ステートマシーンに基づき、正しいJSONのみをLLMに生成させるクラス。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    vocab_path: str
    functions: List[Dict[str, Any]]

    token_filter: Optional[TokenFilter] = Field(default=None)
    all_ids: Set[int] = Field(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        """初期化直後にTokenFilterを組み立てる。"""
        tf = TokenFilter(vocab_path=self.vocab_path)
        self.token_filter = tf
        self.all_ids.update(tf.all_token_ids)

    def generate_function_call(self, prompt: str) -> str:
        """制約付きデコードを行ってJSON文字列を生成します。"""
        if self.token_filter is None:
            raise RuntimeError("TokenFilter is not initialized.")

        current_text = ""
        current_state = JsonState.START

        selected_function: Dict[str, Any] = {}
        current_param_index = 0
        param_keys: List[str] = []
        is_numeric_start = True

        for _ in range(500):
            input_tensor = self.model.encode(current_text)

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
                    import torch

                    if isinstance(raw_logits, torch.Tensor):
                        if raw_logits.dim() == 3:
                            logits = raw_logits[0, -1].tolist()
                        elif raw_logits.dim() == 2:
                            if raw_logits.size(0) == 1:
                                logits = raw_logits[0].tolist()
                            else:
                                logits = raw_logits[-1].tolist()
                        else:
                            logits = raw_logits.tolist()
                    else:
                        logits = raw_logits.tolist()
                else:
                    logits = list(raw_logits)

            allowed_tokens: Set[int] = set()

            if current_state == JsonState.START:
                target = '{"prompt": "'
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(
                        current_text, target
                    )
                )

            elif current_state == JsonState.PROMPT_VALUE:
                target = '{"prompt": "' + prompt + '"'
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(
                        current_text, target
                    )
                )

            elif current_state == JsonState.NAME_KEY:
                target = '{"prompt": "' + prompt + '", "name": "'
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(
                        current_text, target
                    )
                )

            elif current_state == JsonState.FUNCTION_NAME:
                prefix_len = len(
                    '{"prompt": "' + prompt + '", "name": "'
                )
                current_func_prefix = current_text[prefix_len:]

                for func in self.functions:
                    tokens = self.token_filter.filter_by_prefix(
                        current_func_prefix, func["name"] + '"'
                    )
                    allowed_tokens.update(tokens)

            elif current_state == JsonState.PARAMS_START:
                target = (
                    '{"prompt": "' + prompt + '", "name": "'
                    + selected_function["name"] + '", "parameters": {'
                )
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(
                        current_text, target
                    )
                )

            elif current_state == JsonState.PARAM_KEY:
                if current_param_index < len(param_keys):
                    p_key = param_keys[current_param_index]
                    target = current_text + f'"{p_key}": '
                    allowed_tokens = set(
                        self.token_filter.filter_by_prefix(
                            current_text, target
                        )
                    )
                else:
                    target = current_text + "}"
                    allowed_tokens = set(
                        self.token_filter.filter_by_prefix(
                            current_text, target
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

                    if current_param_index + 1 < len(param_keys):
                        allowed_tokens.update(
                            self.token_filter.filter_by_prefix(
                                current_text, current_text + ", "
                            )
                        )
                    else:
                        allowed_tokens.update(
                            self.token_filter.filter_by_prefix(
                                current_text, current_text + "}"
                            )
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

            if current_state == JsonState.START:
                if current_text == '{"prompt": "':
                    current_state = JsonState.PROMPT_VALUE

            elif current_state == JsonState.PROMPT_VALUE:
                if current_text == '{"prompt": "' + prompt + '"':
                    current_state = JsonState.NAME_KEY

            elif current_state == JsonState.NAME_KEY:
                full_key = '{"prompt": "' + prompt + '", "name": "'
                if current_text == full_key:
                    current_state = JsonState.FUNCTION_NAME

            elif current_state == JsonState.FUNCTION_NAME:
                for func in self.functions:
                    full_match = (
                        '{"prompt": "' + prompt + '", "name": "'
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
                target = (
                    '{"prompt": "' + prompt + '", "name": "'
                    + selected_function["name"] + '", "parameters": {'
                )
                if current_text == target:
                    current_state = JsonState.PARAM_KEY

            elif current_state == JsonState.PARAM_KEY:
                if current_param_index < len(param_keys):
                    p_key = param_keys[current_param_index]
                    if current_text.endswith(f'"{p_key}": '):
                        current_state = JsonState.PARAM_VALUE
                        is_numeric_start = True
                else:
                    if current_text.endswith("}"):
                        return current_text

            elif current_state == JsonState.PARAM_VALUE:
                if current_text.endswith(", "):
                    current_param_index += 1
                    current_state = JsonState.PARAM_KEY
                elif current_text.endswith("}"):
                    current_param_index += 1
                    current_state = JsonState.PARAM_KEY

        return current_text
