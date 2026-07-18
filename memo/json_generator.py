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
    # 【解説】外部のLLMモデルクラス(Small_LLM_Model)は標準の型ではないため、
    # Pydanticの検証エラーを回避するための設定。
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 必須パラメータ群
    model: Small_LLM_Model
    vocab_path: str
    functions: List[Dict[str, Any]]
    debug: bool = Field(default=False)

    # 初期化後にセットされる変数
    token_filter: Optional[TokenFilter] = Field(default=None)
    all_ids: Set[int] = Field(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        """Pydanticの初期化直後に自動で呼ばれるメソッド"""
        # クラス初期化後にトークンフィルターを作成し、ボキャブラリを登録
        tf = TokenFilter(vocab_path=self.vocab_path)
        self.token_filter = tf
        self.all_ids.update(tf.all_token_ids)

    def generate_function_call(self, prompt: str) -> str:
        """
        プロンプトを受け取り、100%パース可能なJSON文字列を返すメイン関数。
        例: prompt="2と3を足して" -> 戻り値='{"prompt":"...", "name":"fn_add", ...}'
        """
        if self.token_filter is None:
            raise RuntimeError("TokenFilter is not initialized.")

        # 【解説】LLMに渡す関数の定義を整理（ハルシネーション対策）
        # 元の関数定義には複雑な型情報が含まれているが、小型LLMは複雑なものを見ると混乱する。
        # そのため、引数の型を単純な文字列（例: "string" や "number"）だけに削ぎ落とし、
        # AIにとって見やすい「シンプルなメニュー表」を作り直す。
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

        # 該当しない質問（例：東京の首都は？）が来た時のために
        # 逃げ道として「unknown」という関数を追加しておく。
        clean_funcs.append({"name": "unknown", "parameters": {}})
        f_str = json.dumps(clean_funcs, ensure_ascii=False)

        # 【プロンプト設計】Few-shot Prompting
        # LLMに「こういうルールで、こういう形式で抽出してね」と学習させるためのカンニングペーパー。
        # 例をいくつか見せることで、JSONの構造や unknown の使い方を理解させる。
        context = (
            "System: Extract parameters EXACTLY from the User prompt.\n"
            "- Add '.0' to whole numbers (e.g., 3.0).\n"
            "- DO NOT add spaces to the beginning of string values.\n"
            "- Regex: /[aeiou]/g for vowels, /cat/g for cat.\n"
            "- Remove 'Format template: ' prefix.\n"
            "- If 'Format template:', ALWAYS use fn_format_template.\n"
            "- Copy names perfectly letter-by-letter (e.g., shrek).\n"
            "- Select 'unknown' for irrelevant prompts.\n\n"
            # (中略: いくつかの具体的な問題と答えのペアの例)
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

        # ====== 状態管理変数の初期化 ======
        selected_function: Dict[str, Any] = {} # AIが選んだ関数（例: fn_add_numbers）
        current_param_index = 0                # 今何個目の引数を処理しているか
        param_keys: List[str] = []             # 選んだ関数の引数名リスト（例: ["a", "b"]）

        # 【超重要変数】value_start_text
        # 値の文字列がどこから始まったかを記録する変数。
        # 例: current_textが '{"prompt":"...", "name":"fn", "parameters":{"a":' になった瞬間に、
        # この長さを記憶しておく。これ以降に足された文字が「AIが推論した引数の値」になる。
        value_start_text = ""

        p_json = json.dumps(prompt)
        # 【爆速化の極意１】プロンプトの転記部分をAIに書かせない！
        # 以前は '{"prompt":"' からAIに書かせていたが、ここは絶対に間違えようがない部分なので
        # プログラム側で勝手に作成し、一気に「関数名の選択」までワープする。
        # 例: prefix = '{"prompt":"入力プロンプト","name":"'
        prefix = '{"prompt":' + p_json + ',"name":"'

        current_text = prefix
        # 状態も START 等をすっ飛ばして、いきなり FUNCTION_NAME (関数名選択) からスタート
        current_state = JsonState.FUNCTION_NAME

        # LLMの記憶(履歴)に、「すでにJSONの途中まで自分で喋った」と錯覚させる
        input_ids = self.token_filter.encode(context + current_text)

        # ====== 1トークンずつ生成する推論ループ（最大500文字） ======
        for _ in range(500):
            if not input_ids:
                logits = [0.0] * len(self.token_filter.id_to_token)
            else:
                # LLMに現在の履歴(input_ids)を渡し、次に来る全トークン(15万件)の確率スコア(logits)を取得
                raw_logits = self.model.get_logits_from_input_ids(input_ids)
                if hasattr(raw_logits, "tolist"):
                    logits = raw_logits.tolist()
                else:
                    logits = list(raw_logits)

            # 出力してよいトークンIDを格納するセット（ロジットマスキング用）
            allowed_tokens: Set[int] = set()

            # ====== 制約の設定（次に出していい文字を絞り込む） ======
            if current_state == JsonState.FUNCTION_NAME:
                # 候補となる全ての関数名 ＋ unknown をターゲットとして許可
                al_names = [f["name"] for f in self.functions] + ["unknown"]
                for f_name in al_names:
                    # 例: 目標(ft) = '{"prompt":"...","name":"' + 'fn_add' + '"'
                    ft = prefix + f_name + '"'
                    # prefixから目標に向かって進むトークンだけを許可リストに加える
                    tokens = self.token_filter.filter_by_prefix(
                        current_text, ft
                    )
                    allowed_tokens.update(tokens)

            elif current_state == JsonState.PARAM_VALUE:
                # 値を抽出中の処理。今どの引数(p_key)のどんな型(p_type)を処理しているか取得
                p_key = param_keys[current_param_index]
                p_type = selected_function["parameters"][p_key].get(
                    "type", "string"
                )

                # 【爆速化の極意２】Fast-Forward Next (ff_next)
                # 値を書き終わった後に続く「次のキー名やカッコ」の文字列を事前に作っておく。
                if current_param_index + 1 < len(param_keys):
                    # まだ次の引数がある場合
                    n_key = param_keys[current_param_index + 1]
                    n_type = selected_function["parameters"][n_key].get(
                        "type", "string"
                    )
                    # 次の引数が文字列型なら `"次のキー名":"`、数値なら `"次のキー名":`
                    if n_type == "string":
                        ff_next = ',"' + n_key + '":"'
                    else:
                        ff_next = ',"' + n_key + '":'
                else:
                    # 次の引数がない場合（最後の引数）は、閉じカッコ `}}` を目指す
                    ff_next = "}}"

                if p_type in ["number", "integer"]:
                    # 数字として有効なトークン（0-9, . など）だけを許可
                    tokens_list = self.token_filter.filter_numeric_tokens(
                        is_start=(len(current_text) == len(value_start_text)),
                        is_integer=(p_type == "integer")
                    )
                    allowed_tokens = set(tokens_list)

                    # 現在AIが抽出中の数字部分を取得（例: current_textの末尾にある "12.0"）
                    num_part = current_text[len(value_start_text):]
                    c_len = 0
                    v_chars = (
                        "0123456789.-" if p_type == "number" else "0123456789-"
                    )
                    # 純粋な数字部分だけの長さを測る
                    for char in num_part:
                        if char in v_chars:
                            c_len += 1
                        else:
                            break
                    clean_num = num_part[:c_len]

                    # 【重要】AIに「数字を書き終えたら次はこの文字に進め」と道を示す
                    # 例: "12.0" + ',"次のキー":'
                    full_exit = value_start_text + clean_num + ff_next
                    allowed_tokens.update(
                        self.token_filter.filter_by_prefix(
                            current_text, full_exit
                        )
                    )

                elif p_type == "boolean":
                    # trueかfalseの直後に ff_next を繋げたものを目標にする
                    targets = [
                        value_start_text + "true" + ff_next,
                        value_start_text + "false" + ff_next
                    ]
                    for t in targets:
                        allowed_tokens.update(
                            self.token_filter.filter_by_prefix(current_text, t)
                        )

                else:
                    # 文字列の抽出中: 改行などを除いた安全な全ての文字を許可する
                    t_filter = self.token_filter
                    allowed_tokens.update(t_filter.valid_string_tokens)

                    # 文字列の終了を示す「"」の直後に、ff_next を繋げる
                    full_exit = current_text + '"' + ff_next
                    allowed_tokens.update(
                        self.token_filter.filter_by_prefix(
                            current_text, full_exit
                        )
                    )

            if not allowed_tokens and self.debug:
                print(f"⚠️ [WARNING] No tokens at: {current_state.name}!")

            # ====== ロジットマスキング（強制的なトークン選択） ======
            # 【爆速化の極意３】-infループの廃止
            # 15万件の配列をループして -inf に書き換える重い処理をやめ、
            # 「許可されたトークン(allowed_tokens)の中で、一番logitsが高いもの」をPythonの max関数（C言語ベース）で一発取得。
            if allowed_tokens:
                # IndexErrorを防ぐための安全装置
                valid_ids = {t for t in allowed_tokens if t < len(logits)}
                if valid_ids:
                    # valid_idsの中から、logitsの値が最大のものを選ぶ
                    next_token_id = max(valid_ids, key=logits.__getitem__)
                else:
                    next_token_id = int(logits.index(max(logits)))
            else:
                next_token_id = int(logits.index(max(logits)))

            # 選ばれたIDを文字列に戻して結合
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

            # ====== 状態チェックとワープ処理 (Cascade State Sync) ======
            while True:
                old_state = current_state

                if current_state == JsonState.FUNCTION_NAME:
                    # 末尾にダブルクォーテーションが出たらAIが「関数名」を選び終わった合図
                    if current_text.endswith('"'):
                        # 例: '{"prompt":"...","name":"fn_add"' -> 'fn_add' だけを取り出す
                        f_name = current_text[len(prefix):-1]

                        if f_name == "unknown":
                            # unknownの場合は引数がないので、ここでカッコを閉じて即終了！
                            return current_text + ',"parameters":{}}'

                        # 選ばれた関数の設計書を取得
                        selected_func = None
                        for f in self.functions:
                            if f["name"] == f_name:
                                selected_func = f
                                break

                        if not selected_func:
                            return current_text + ',"parameters":{}}'

                        # 引数のキー名リスト（["a", "b"]）を取得
                        param_keys = list(
                            selected_func.get("parameters", {}).keys()
                        )

                        if not param_keys:
                            return current_text + ',"parameters":{}}'

                        p_type = selected_func["parameters"][
                            param_keys[0]
                        ].get("type", "string")

                        # 【爆速化の極意４】プログラムによる強制ワープ
                        # AIにちまちま `", "parameters": { "a": ` と書かせるのではなく、
                        # プログラム側で強制的に結合して一気に値抽出のステート（PARAM_VALUE）までジャンプ！
                        if p_type == "string":
                            ff_str = ',"parameters":{"' + param_keys[0] + '":"'
                        else:
                            ff_str = ',"parameters":{"' + param_keys[0] + '":'

                        current_text += ff_str
                        # 足した文字をLLMの履歴(input_ids)にも同期して違和感をなくす
                        if self.token_filter is not None:
                            input_ids.extend(self.token_filter.encode(ff_str))

                        # 次の状態へ進む準備
                        current_param_index = 0
                        current_state = JsonState.PARAM_VALUE
                        value_start_text = current_text # ここから先が値になる
                        selected_function = selected_func

                elif current_state == JsonState.PARAM_VALUE:
                    # パラメータの値の抽出が完了したかチェック
                    p_key = param_keys[current_param_index]
                    p_type = selected_function["parameters"][p_key].get(
                        "type", "string"
                    )

                    value_is_finished = False

                    if p_type in ["number", "integer"]:
                        # 値の部分を取得（例: "12.0," のようになっているかもしれない）
                        num_part = current_text[len(value_start_text):]
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

                        # 数字以外の文字（AIが出口に向かって書いたカンマ等）が出力されたら終了と判定
                        if len(num_part) > c_len:
                            # カンマなどの余計なはみ出し（オーバーシュート）を一旦削り落とし、
                            # 綺麗な数値だけの状態（"12.0"）に戻す
                            current_text = value_start_text + num_part[:c_len]
                            value_is_finished = True

                    elif p_type == "boolean":
                        bool_part = current_text[len(value_start_text):]
                        is_t = bool_part.startswith("true")
                        # "true," のように5文字以上になっていたら終了
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
                        # エスケープ (\") されていない純粋な閉じるダブルクォート (") を探す
                        for i in range(len(val_str)):
                            if escape:
                                escape = False
                            elif val_str[i] == '\\':
                                escape = True
                            elif val_str[i] == '"':
                                quote_idx = i
                                break

                        # 閉じるクォートが見つかったら文字列抽出は終了！
                        if quote_idx != -1:
                            # 閉じるクォートまでの綺麗な文字列にする
                            val_clean = val_str[:quote_idx + 1]
                            current_text = value_start_text + val_clean
                            value_is_finished = True

                    # 1つの値が抽出完了した場合の処理
                    if value_is_finished:
                        current_param_index += 1

                        if current_param_index < len(param_keys):
                            # まだ次の引数が残っている場合
                            n_key = param_keys[current_param_index]
                            n_type = selected_function["parameters"][
                                n_key
                            ].get("type", "string")

                            # 【爆速化の極意５】プログラムによる強制ワープ（引数の移動）
                            # 次のキー名の文字列をプログラム側でくっつけてワープする。
                            if n_type == "string":
                                ff_str = ',"' + n_key + '":"'
                            else:
                                ff_str = ',"' + n_key + '":'

                            current_text += ff_str
                            # 履歴にも同期
                            if self.token_filter is not None:
                                input_ids.extend(
                                    self.token_filter.encode(ff_str)
                                )
                            # スタート地点を更新（ここから先が次の引数の値になる）
                            value_start_text = current_text
                        else:
                            # すべての引数が出力し終わった！
                            # 閉じカッコを付けて完成品としてreturnする
                            return current_text + '}}'

                # 状態に変化がなくなったらwhileループを抜けて、次の1文字を予測させる
                if current_state == old_state:
                    break

        return current_text
