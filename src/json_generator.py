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
    # 外部のLLMモデルクラスなどをそのままプロパティに持てるようにする
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    vocab_path: str
    functions: List[Dict[str, Any]]
    debug: bool = Field(default=False)

    token_filter: Optional[TokenFilter] = Field(default=None)
    all_ids: Set[int] = Field(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        # クラス初期化後にトークンフィルターを作成し、ボキャブラリを登録
        tf = TokenFilter(vocab_path=self.vocab_path)
        self.token_filter = tf
        self.all_ids.update(tf.all_token_ids)

    def generate_function_call(self, prompt: str) -> str:
        """プロンプトを受け取り、100%パース可能なJSON文字列を返す"""
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

        # 【プロンプト設計】LLMに抽出ルールを学習させる
        context = (
            "System: Extract parameters EXACTLY from the User prompt.\n"
            "- Add '.0' to whole numbers (e.g., 3.0).\n"
            "- DO NOT add spaces to the beginning of string values.\n"
            "- Regex: /[aeiou]/g for vowels, /cat/g for cat.\n"
            "- Remove 'Format template: ' prefix.\n"
            "- If 'Format template:', ALWAYS use fn_format_template.\n"
            "- Copy names perfectly letter-by-letter (e.g., shrek).\n"
            "- Select 'unknown' for irrelevant prompts.\n\n"
            "Ex1:\nUser: Format template: Welcome {user}'s page!\n"
            "JSON: {\"prompt\":\"Format template: Welcome {user}'s page!\","
            "\"name\":\"fn_format_template\",\"parameters\":{"
            "\"template\":\"Welcome {user}'s page!\"}}\n\n"
            "Ex2:\nUser: Read file at /tmp/data.csv with ascii\n"
            "JSON: {\"prompt\":\"Read file at /tmp/data.csv with "
            "ascii\",\"name\":\"fn_read_file\",\"parameters\":{"
            "\"path\":\"/tmp/data.csv\",\"encoding\":\"ascii\"}}\n\n"
            "Ex3:\nUser: Greet fiona\n"
            "JSON: {\"prompt\":\"Greet fiona\",\"name\":\"fn_greet\","
            "\"parameters\":{\"name\":\"fiona\"}}\n\n"
            "Ex4:\nUser: Replace all vowels in 'text' with *\n"
            "JSON: {\"prompt\":\"Replace all vowels in 'text' with *\","
            "\"name\":\"fn_substitute_string_with_regex\",\"parameters\":{"
            "\"source_string\":\"text\",\"regex\":\"/[aeiouAEIOU]/g\","
            "\"replacement\":\"*\"}}\n\n"
            "Ex5:\nUser: What is the product of 12 and 4?\n"
            "JSON: {\"prompt\":\"What is the product of 12 and 4?\","
            "\"name\":\"fn_multiply_numbers\",\"parameters\":"
            "{\"a\":12.0,\"b\":4.0}}\n\n"
            "Ex6:\nUser: Is 10 an even number?\n"
            "JSON: {\"prompt\":\"Is 10 an even number?\",\"name\":"
            "\"fn_is_even\",\"parameters\":{\"n\":10}}\n\n"
            "Ex7:\nUser: what is the capital of Paris?\n"
            "JSON: {\"prompt\":\"what is the capital of Paris?\","
            "\"name\":\"unknown\",\"parameters\":{}}\n\n"
            f"Functions:{f_str}\nUser:{prompt}\nJSON:"
        )

        selected_function: Dict[str, Any] = {}
        current_param_index = 0
        param_keys: List[str] = []

        # 値の文字列がどこから始まったかを記録する変数
        # （例: '{"prompt":"...", "name":"fn", "parameters":{"a":' まで記録）
        value_start_text = ""

        # 最初からプロンプト転記部分は自力で入力しておく。
        p_json = json.dumps(prompt)
        prefix = '{"prompt":' + p_json + ',"name":"'

        current_text = prefix
        current_state = JsonState.FUNCTION_NAME

        # LLMの履歴(input_ids)に「すでに自分がprefixまで喋った」と錯覚させる
        input_ids = self.token_filter.encode(context + current_text)

        # 1トークンずつ生成するループ
        for _ in range(500):
            if not input_ids:
                logits = [0.0] * len(self.token_filter.id_to_token)
            else:
                # LLMに現在の履歴を渡し、次に来る全トークンの確率を取得
                raw_logits = self.model.get_logits_from_input_ids(input_ids)
                if hasattr(raw_logits, "tolist"):
                    logits = raw_logits.tolist()
                else:
                    logits = list(raw_logits)

            # 出力してよいトークンID
            allowed_tokens: Set[int] = set()

            # 制約の設定
            if current_state == JsonState.FUNCTION_NAME:
                al_names = [f["name"] for f in self.functions] + ["unknown"]
                for f_name in al_names:
                    # 目標: prefix + "関数名" + '"' (例: '... "name":"fn_add"')
                    ft = prefix + f_name + '"'
                    tokens = self.token_filter.filter_by_prefix(
                        current_text, ft
                    )
                    allowed_tokens.update(tokens)

            elif current_state == JsonState.PARAM_VALUE:
                # 現在推論中のパラメータのキー名と型を取得
                p_key = param_keys[current_param_index]
                p_type = selected_function["parameters"][p_key].get(
                    "type", "string"
                )

                # 次の固定文字列を作成
                if current_param_index + 1 < len(param_keys):
                    n_key = param_keys[current_param_index + 1]
                    n_type = selected_function["parameters"][n_key].get(
                        "type", "string"
                    )
                    # 次が文字列型なら `"キー名":"`、数値なら `"キー名":`
                    if n_type == "string":
                        ff_next = ',"' + n_key + '":"'
                    else:
                        ff_next = ',"' + n_key + '":'
                else:
                    # 次の引数がない場合はかっことじ
                    ff_next = "}}"

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
                        "0123456789.-" if p_type == "number" else "0123456789-"
                    )
                    for char in num_part:
                        if char in v_chars:
                            c_len += 1
                        else:
                            break
                    clean_num = num_part[:c_len]

                    # LLMに「次に来るキー名」も目標として見せることで、
                    # LLMが0を連打せずスムーズに数値を閉じさせる
                    full_exit = value_start_text + clean_num + ff_next
                    allowed_tokens.update(
                        self.token_filter.filter_by_prefix(
                            current_text, full_exit
                        )
                    )

                elif p_type == "boolean":
                    # tかfだけで判別して、残りは固定をぶち込んだほうが
                    # はやいかも
                    targets = [
                        value_start_text + "true" + ff_next,
                        value_start_text + "false" + ff_next
                    ]
                    for t in targets:
                        allowed_tokens.update(
                            self.token_filter.filter_by_prefix(current_text, t)
                        )

                else:
                    # 文字列の推論: 改行などを除いた全ての文字を許可
                    t_filter = self.token_filter
                    allowed_tokens.update(t_filter.valid_string_tokens)

                    # 文字列の終了を示す「"」の直後に、次の文字列を繋げる
                    full_exit = current_text + '"' + ff_next
                    allowed_tokens.update(
                        self.token_filter.filter_by_prefix(
                            current_text, full_exit
                        )
                    )

            if not allowed_tokens and self.debug:
                print(f"⚠️ [WARNING] No tokens at: {current_state.name}!")

            # ロジットマスキング
            # -infを代入するんじゃねくて
            # allowdtokenから最もロジットの大きなものを選び出す
            if allowed_tokens:
                valid_ids = {t for t in allowed_tokens if t < len(logits)}
                if valid_ids:
                    # valid_idsの中から、logitsの値が最大のもの
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

            # 状態チェック
            while True:
                old_state = current_state

                if current_state == JsonState.FUNCTION_NAME:
                    # 末尾にダブルクォーテーションが出たら関数名が確定
                    if current_text.endswith('"'):
                        # prefixを切り飛ばして関数名だけを取得
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

                        # 引数ないと終了
                        if not param_keys:
                            return current_text + ',"parameters":{}}'

                        p_type = selected_func["parameters"][
                            param_keys[0]
                        ].get("type", "string")

                        # 最初の引数キー名を足して推論させない
                        # 例: ',"parameters":{"a":'
                        if p_type == "string":
                            ff_str = ',"parameters":{"' + param_keys[0] + '":"'
                        else:
                            ff_str = ',"parameters":{"' + param_keys[0] + '":'

                        current_text += ff_str
                        # 足した文字をLLMの履歴に同期
                        if self.token_filter is not None:
                            input_ids.extend(self.token_filter.encode(ff_str))

                        # 次の状態へ進む
                        current_param_index = 0
                        current_state = JsonState.PARAM_VALUE
                        value_start_text = current_text
                        selected_function = selected_func

                # パラメータの抽出完了チェック
                elif current_state == JsonState.PARAM_VALUE:
                    p_key = param_keys[current_param_index]
                    p_type = selected_function["parameters"][p_key].get(
                        "type", "string"
                    )

                    value_is_finished = False

                    if p_type in ["number", "integer"]:
                        num_part = current_text[len(value_start_text):]
                        # ▼ set() を外して、前半と同じただの文字列にする！
                        v_chars = (
                            "0123456789.-" if p_type == "number"
                            else "0123456789-"
                        )
                        c_len = 0
                        for char in num_part:
                            if char in v_chars:
                                c_len += 1
                            else:
                                break

                        # 数字以外の文字（LLMが生成した次のワープ文字など）が出力されたら終了
                        if len(num_part) > c_len:
                            # オーバーシュートした分を削り、綺麗な数値だけにする
                            current_text = value_start_text + num_part[:c_len]
                            value_is_finished = True

                    elif p_type == "boolean":
                        bool_part = current_text[len(value_start_text):]
                        is_t = bool_part.startswith("true")
                        if is_t and len(bool_part) > 4:
                            current_text = value_start_text + "true"
                            value_is_finished = True

                        is_f = bool_part.startswith("false")
                        if is_f and len(bool_part) > 5:
                            current_text = value_start_text + "false"
                            value_is_finished = True

                    else:
                        val_str = current_text[len(value_start_text):]
                        escape = False
                        quote_idx = -1
                        for i in range(len(val_str)):
                            if escape:
                                escape = False
                            elif val_str[i] == '\\':
                                escape = True
                            elif val_str[i] == '"':
                                quote_idx = i
                                break

                        # エスケープされていない " が見つかったら文字列の終了
                        if quote_idx != -1:
                            val_clean = val_str[:quote_idx + 1]
                            current_text = value_start_text + val_clean
                            value_is_finished = True

                    # 値が1つ完成した場合
                    if value_is_finished:
                        current_param_index += 1

                        # まだ次の引数が残っている場合
                        if current_param_index < len(param_keys):
                            n_key = param_keys[current_param_index]
                            n_type = selected_function["parameters"][
                                n_key
                            ].get("type", "string")

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
                        else:
                            # すべての引数が出力し終わったらカッコとじ
                            return current_text + '}}'

                if current_state == old_state:
                    break

        return current_text
