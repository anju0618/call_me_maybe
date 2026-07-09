# src/json_generator.py

import json
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, ConfigDict, Field
from src.json_state import JsonState
from src.token_filter import TokenFilter
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class JsonGenerator(BaseModel):
    """
    ステートマシーンに基づき、正しいJSONのみをLLMに生成させるメインクラスです。
    """

    # Pydanticの設定：標準の型（intやstr）ではない自作クラス（Small_LLM_Model）を
    # クラスのプロパティとして持たせることを許可するための魔法のおまじないです。
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- 外部から受け取る必須データ ---
    # model: LLMの本体です。文章をIDに変換(encode)したり、確率(logits)を出したりします。
    model: Small_LLM_Model
    
    # vocab_path (str): ボキャブラリ（単語帳）JSONファイルのパスです。
    vocab_path: str
    
    # functions (List[Dict[str, Any]]): 関数定義書のリストです。
    # 例: [{"name": "fn_add", "parameters": {...}}, ...]
    functions: List[Dict[str, Any]]
    
    # debug (bool): 開発者用のデバッグモードON/OFFフラグです。デフォルトはFalse。
    debug: bool = Field(default=False)

    # --- 内部で自動生成するデータ ---
    # token_filter (Optional[TokenFilter]): 生成を制御する「絶対防壁フィルター」のインスタンス。
    # 初期化前はNoneですが、後で必ずセットされます。
    token_filter: Optional[TokenFilter] = Field(default=None)
    
    # all_ids (Set[int]): 単語帳に存在するすべてのトークンIDの集合（セット）です。
    # 例: {1, 2, 3, ..., 151643}
    all_ids: Set[int] = Field(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        """
        Pydanticによってクラスが初期化された直後に、自動で1回だけ呼ばれる関数です。
        ここで、単語帳のパスを使ってTokenFilterを組み立ててセットします。
        __context: Any はPydanticの仕様上必要な引数ですが、中身は使いません。
        """
        # TokenFilterクラスをインスタンス化
        tf = TokenFilter(vocab_path=self.vocab_path)
        # クラスの変数として保存
        self.token_filter = tf
        # フィルターから、全トークンIDの集合をもらって保存
        self.all_ids.update(tf.all_token_ids)

    def generate_function_call(self, prompt: str) -> str:
        """
        制約付きデコードを行って、完璧なJSON文字列を生成する推論のコアエンジンです。
        prompt (str): ユーザーからの入力文字列（例: "2と3を足して"）
        戻り値 (str): 完成したJSON文字列
        """
        # フィルターが準備できていない場合はエラーを出して止める（安全装置）
        if self.token_filter is None:
            raise RuntimeError("TokenFilter is not initialized.")

        # --- AIへの指示書（Context）の作成 ---
        # 使える関数リスト(List)を、そのままでは文字として繋げられないので、
        # json.dumpsを使ってJSON形式の「文字列(str)」に変換します。
        funcs_str = json.dumps(self.functions, ensure_ascii=False)
        
        # AIが「ただの穴埋め」ではなく「意味を理解して値を抽出」するように、
        # 事前にシステムプロンプトと関数リスト、ユーザーからの入力を合体させます。
        context = (
            "System: You are an expert JSON function calling assistant. "
            "You must strictly extract the exact parameter values from the "
            "User prompt. Do not use generic placeholders. "
            "For example, if the prompt is 'Greet shrek', the name is "
            "'shrek'. If the prompt is 'Reverse the string 'hello'', the "
            "string is 'hello'. Output ONLY valid JSON.\n"
            f"Functions: {funcs_str}\n"
            f"User: {prompt}\n"
            "JSON:"
        )

        # current_text (str): これからAIが1文字ずつ書き足していく、生成中の文字列（初期は空）
        current_text = ""
        # current_state (JsonState): 現在、JSONのどの部分を作っているかを表す状態（最初はSTART）
        current_state = JsonState.START

        # --- 状態管理用の変数たち ---
        # selected_function (Dict[str, Any]): AIが選んだ関数のルールを保存しておく辞書
        selected_function: Dict[str, Any] = {}
        # current_param_index (int): 今、何個目の引数(parameter)を処理しているかのカウンター
        current_param_index = 0
        # param_keys (List[str]): 処理すべき引数の名前のリスト（例: ["a", "b"]）
        param_keys: List[str] = []
        # is_numeric_start (bool): 数字の1文字目を処理しているかどうか
        is_numeric_start = True
        # param_base_text (str): 引数の処理を始めた時点の全体の文字列を記憶しておく変数
        param_base_text = ""
        # value_start_text (str): 引数の値(Value)を書き始めた時点の全体の文字列を記憶しておく変数
        value_start_text = ""

        # 【バグ対策】ユーザー入力の prompt に `"` などが含まれていてJSONが壊れるのを防ぐため、
        # json.dumps を使って安全な形（エスケープ済み）に変換しておきます。
        prompt_json = json.dumps(prompt)

        # 最大500回（500トークン分）のループを回します。
        for _ in range(500):
            # full_prompt (str): 背景知識(context)と、これまでに生成した文字(current_text)を合体
            full_prompt = context + current_text
            
            # AIモデルに文字列を渡し、AIが理解できる数字の配列(input_ids)に変換させます
            input_tensor = self.model.encode(full_prompt)
            
            # 帰ってきたデータがTensor(AI用配列)か、普通のPythonリストかを判定して取り出す安全装置
            if hasattr(input_tensor, "tolist"):
                input_ids = input_tensor[0].tolist()  # Tensorならリストに変換
            else:
                input_ids = input_tensor              # 既にリストならそのまま使う

            # ロジット(次に来る文字の確率スコア)を取得します
            if not input_ids:
                # 万が一入力が空っぽなら、全トークンのスコアを0.0にしておく
                logits = [0.0] * len(self.token_filter.id_to_token)
            else:
                raw_logits = self.model.get_logits_from_input_ids(input_ids)
                # これもTensorかリストかを判定して、普通のリスト(List[float])に変換します
                if hasattr(raw_logits, "tolist"):
                    logits = raw_logits.tolist()
                else:
                    logits = list(raw_logits)

            # allowed_tokens (Set[int]): このターンで「許可する」トークンIDを入れる空のセット
            allowed_tokens: Set[int] = set()

            # =====================================================================
            # フェーズ1: 現状態における絶対ターゲットの作成と、許可トークンの抽出
            # =====================================================================
            
            if current_state == JsonState.START:
                # まだ何も書いていない状態。まずは `{"prompt": "` を目指させる。
                full_target = '{"prompt": "'
                # filter_by_prefixに「今の文字」と「目標」を渡し、次に繋がるIDだけをもらう
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(current_text, full_target)
                )

            elif current_state == JsonState.PROMPT_VALUE:
                # プロンプトの文字列をそのまま出力させる状態。エスケープ済みの prompt_json を使う。
                full_target = '{"prompt": ' + prompt_json
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(current_text, full_target)
                )

            elif current_state == JsonState.NAME_KEY:
                # `", "name": "` の部分を出力させる状態。
                full_target = '{"prompt": ' + prompt_json + ', "name": "'
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(current_text, full_target)
                )

            elif current_state == JsonState.FUNCTION_NAME:
                # 利用可能な関数名（fn_add_numbers など）のどれかを出力させる状態。
                # 存在する関数の数だけループを回して、全パターンの目標ターゲットを作る。
                for func in self.functions:
                    full_target = (
                        '{"prompt": ' + prompt_json + ', "name": "'
                        + func["name"] + '"'
                    )
                    tokens = self.token_filter.filter_by_prefix(current_text, full_target)
                    allowed_tokens.update(tokens) # どの関数名に繋がるトークンも許可リストに入れる

            elif current_state == JsonState.PARAMS_START:
                # `", "parameters": {` の部分を出力させる状態。
                full_target = (
                    '{"prompt": ' + prompt_json + ', "name": "'
                    + selected_function["name"] + '", "parameters": {'
                )
                allowed_tokens = set(
                    self.token_filter.filter_by_prefix(current_text, full_target)
                )

            elif current_state == JsonState.PARAM_KEY:
                # 引数のキー（例: `"a": `）を出力させる状態。
                if current_param_index < len(param_keys):
                    # まだ処理すべき引数が残っている場合
                    p_key = param_keys[current_param_index]
                    full_target = param_base_text + f'"{p_key}": '
                    allowed_tokens = set(
                        self.token_filter.filter_by_prefix(current_text, full_target)
                    )
                else:
                    # すべての引数を処理し終わった場合、JSONを閉じる（} または }}）
                    if len(param_keys) == 0:
                        full_target = param_base_text + "}}" # 引数が元々無い関数用
                    else:
                        full_target = param_base_text + "}"  # 引数があった関数用

                    allowed_tokens = set(
                        self.token_filter.filter_by_prefix(current_text, full_target)
                    )

            elif current_state == JsonState.PARAM_VALUE:
                # 引数の「値（Value）」をAIに生成させる、一番複雑な状態。
                p_key = param_keys[current_param_index]
                # 選ばれた関数のルール（辞書）から、今の引数の型情報を取り出す
                p_info = selected_function["parameters"][p_key]
                p_type = p_info.get("type") # "number" か "string" かが入る

                if p_type == "number":
                    # 数字の場合、専用のフィルターを使って数字関連のトークンだけを許可する
                    tokens_list = self.token_filter.filter_numeric_tokens(is_start=is_numeric_start)
                    allowed_tokens = set(tokens_list)
                    is_numeric_start = False # 1文字目が終わったらフラグを折る

                    # AIがすでに出力した数字部分（num_part）を抽出する
                    num_part = current_text[len(value_start_text):]
                    c_len = 0
                    # 数字として正しい文字だけをカウントして切り出す
                    for char in num_part:
                        if char in "0123456789.-":
                            c_len += 1
                        else:
                            break
                    clean_num = num_part[:c_len]

                    # 「数字を書き終えたあとの出口（カンマかカッコ）」の目標ターゲットを作る
                    if current_param_index + 1 < len(param_keys):
                        # 次の引数があるなら、カンマで繋ぐ
                        n_key = param_keys[current_param_index + 1]
                        full_exit_target = value_start_text + clean_num + f', "{n_key}": '
                    else:
                        # これが最後の引数なら、カッコで閉じる
                        full_exit_target = value_start_text + clean_num + "}"

                    # 出口に向かうトークンも許可リストに追加する
                    allowed_tokens.update(
                        self.token_filter.filter_by_prefix(current_text, full_exit_target)
                    )

                elif p_type == "string":
                    # 文字列の場合の処理
                    s_part = current_text[len(value_start_text):]
                    if not s_part.startswith('"'):
                        # まだ最初のダブルクォート(")が出ていない場合
                        inv_chars_st = set("\n\rĊ{}") # 絶対に許さない記号リスト
                        for t_id, t_str in self.token_filter.id_to_token.items():
                            cl_str = t_str.replace("Ġ", " ").replace(" ", " ")
                            # " で始まるトークンを探す
                            if cl_str.startswith('"'):
                                # ただし、" の直後にまた " が来たり、改行が混ざる合体トークンは除外する
                                if '"' not in cl_str[1:] and not any(c in inv_chars_st for c in cl_str):
                                    allowed_tokens.add(t_id)

                        # " だけを出し終えたあとの出口ターゲットも用意する
                        if current_param_index + 1 < len(param_keys):
                            n_key = param_keys[current_param_index + 1]
                            full_exit = '"' + f', "{n_key}": '
                        else:
                            full_exit = '"}'

                        allowed_tokens.update(
                            self.token_filter.filter_by_prefix(current_text, current_text + full_exit)
                        )
                    else:
                        # すでに " が出ていて、中身の文字を書いている最中の場合
                        quote_idx = s_part.find('"', 1) # 2個目の " を探す
                        
                        if quote_idx == -1:
                            # まだ2個目の "（閉じるクォート）が出ていない場合
                            inv_chars = set("\"\n\rĊ{}") # "自体や改行を禁止
                            for t_id, t_str in self.token_filter.id_to_token.items():
                                cl_str = t_str.replace("Ġ", " ").replace(" ", " ")
                                # 危険な文字が含まれていないトークンなら許可する
                                if not any(c in inv_chars for c in cl_str):
                                    allowed_tokens.add(t_id)

                            # 「今書いた文字 + "」で閉じるための出口ターゲットを用意
                            if current_param_index + 1 < len(param_keys):
                                n_key = param_keys[current_param_index + 1]
                                full_exit = current_text + '"' + f', "{n_key}": '
                            else:
                                full_exit = current_text + '"}'

                            allowed_tokens.update(
                                self.token_filter.filter_by_prefix(current_text, full_exit)
                            )
                        else:
                            # 2個目の " が出たあとの場合（文字列の中身は完成している）
                            cl_str_val = s_part[1:quote_idx] # " と " の中身の文字
                            if current_param_index + 1 < len(param_keys):
                                n_key = param_keys[current_param_index + 1]
                                # 次の引数に繋げるターゲット
                                full_exit = (
                                    value_start_text + '"' + cl_str_val + '"'
                                    + f', "{n_key}": '
                                )
                            else:
                                # 最後の引数としてカッコを閉じるターゲット
                                full_exit = value_start_text + '"' + cl_str_val + '"}'

                            allowed_tokens.update(
                                self.token_filter.filter_by_prefix(current_text, full_exit)
                            )

            # =====================================================================
            # フェーズ2: ロジット・マスキングとトークン確定
            # =====================================================================
            
            # 許可トークンが0個（デッドロック）ならデバッグモードで警告を出す
            if not allowed_tokens and self.debug:
                print(f"⚠️ [WARNING] No tokens allowed at: {current_state.name}!")

            # logits（すべてのトークンの確率スコア配列）をループで回す
            for token_id in range(len(logits)):
                # もしこのIDが許可リスト(allowed_tokens)に入っていなければ...
                if token_id not in allowed_tokens:
                    # スコアを強制的に float("-inf") つまりマイナス無限大に書き換える（確率0%にする）
                    logits[token_id] = float("-inf")

            # スコアが一番高いトークンのID（インデックス）を取得する
            next_token_id = int(logits.index(max(logits)))
            
            # そのIDを文字（str）に変換する
            next_token_str = self.token_filter.id_to_token[next_token_id]
            # LLM特有の空白文字（Ġなど）を人間のスペースに置き換えて綺麗にする
            clean_next_str = next_token_str.replace("Ġ", " ").replace(" ", " ")
            
            # 綺麗にした文字を、これまでに生成した文字列の末尾にくっつける
            current_text += clean_next_str

            # デバッグモードONなら、どの状態(State)で何の文字(Token)が追加されたか表示する
            if self.debug:
                print(
                    f"  [State: {current_state.name:13}] "
                    f"Appended: {repr(clean_next_str):12} -> "
                    f"Current: {repr(current_text)}"
                )

            # =====================================================================
            # フェーズ3: 出力確定直後の完全ピッタリ同期処理 (Cascade State Sync)
            # =====================================================================
            
            # AIが巨大な合体トークン（", "name": "など）を出して一気に進んだ場合でも、
            # 現在地（State）がそれに追いつくまで、何度も状態を前進（Cascade）させます。
            while True:
                old_state = current_state

                if current_state == JsonState.START:
                    # {"prompt": " まで書き終わったら次の状態へ進む
                    if current_text.endswith('{"prompt": "'):
                        current_state = JsonState.PROMPT_VALUE

                elif current_state == JsonState.PROMPT_VALUE:
                    expected_pv = '{"prompt": ' + prompt_json
                    # プロンプトの中身を書き終わったら次の状態へ進む
                    if current_text == expected_pv:
                        current_state = JsonState.NAME_KEY

                elif current_state == JsonState.NAME_KEY:
                    expected_nk = '{"prompt": ' + prompt_json + ', "name": "'
                    # ", "name": " まで書き終わったら次の状態へ進む
                    if current_text == expected_nk:
                        current_state = JsonState.FUNCTION_NAME

                elif current_state == JsonState.FUNCTION_NAME:
                    # どの関数名を選んだのかを確認し、見つかったら設定を保存して次へ進む
                    for func in self.functions:
                        full_match = (
                            '{"prompt": ' + prompt_json + ', "name": "'
                            + func["name"] + '"'
                        )
                        if current_text == full_match:
                            selected_function = func
                            # 選ばれた関数の引数名リスト（param_keys）を取得しておく
                            param_keys = list(func.get("parameters", {}).keys())
                            current_param_index = 0
                            current_state = JsonState.PARAMS_START
                            break

                elif current_state == JsonState.PARAMS_START:
                    target_ps = (
                        '{"prompt": ' + prompt_json + ', "name": "'
                        + selected_function["name"]
                        + '", "parameters": {'
                    )
                    # parameters: { まで書き終わったら次の状態へ進む
                    if current_text == target_ps:
                        current_state = JsonState.PARAM_KEY
                        param_base_text = current_text # 現在地を記憶

                elif current_state == JsonState.PARAM_KEY:
                    if current_param_index < len(param_keys):
                        p_key = param_keys[current_param_index]
                        expected_pk = param_base_text + f'"{p_key}": '
                        # "引数名": まで書き終わったら、値を書かせる次の状態へ進む
                        if current_text == expected_pk:
                            current_state = JsonState.PARAM_VALUE
                            is_numeric_start = True
                            value_start_text = current_text # 値のスタート地点を記憶
                    else:
                        # 全ての引数が終わっている場合、JSONが綺麗に閉じていたら完成！（returnして終了）
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
                        suffix = num_part[c_len:] # 数字のあとに続く文字（カンマやカッコ）

                        if suffix:
                            if current_param_index + 1 < len(param_keys):
                                # カンマで終わっていたら、次の引数の処理（PARAM_KEY）に戻る
                                if suffix.startswith(", "):
                                    current_param_index += 1
                                    param_base_text = value_start_text + clean_num + ", "
                                    current_state = JsonState.PARAM_KEY
                            else:
                                # カッコで終わっていたら、最後の引数が終わった合図として PARAM_KEY に戻る（そして終了へ）
                                if suffix.startswith("}"):
                                    current_param_index += 1
                                    param_base_text = value_start_text + clean_num + "}"
                                    current_state = JsonState.PARAM_KEY

                    elif p_type == "string":
                        s_part = current_text[len(value_start_text):]
                        if s_part.startswith('"') and len(s_part) > 1:
                            quote_idx = s_part.find('"', 1)
                            if quote_idx != -1:
                                cl_str_val = s_part[1:quote_idx]
                                suffix = s_part[quote_idx + 1:] # " のあとに続く文字

                                if suffix:
                                    if current_param_index + 1 < len(param_keys):
                                        if suffix.startswith(", "):
                                            current_param_index += 1
                                            param_base_text = (
                                                value_start_text + '"'
                                                + cl_str_val + '", '
                                            )
                                            current_state = JsonState.PARAM_KEY
                                    else:
                                        if suffix.startswith("}"):
                                            current_param_index += 1
                                            param_base_text = (
                                                value_start_text + '"'
                                                + cl_str_val + '"}'
                                            )
                                            current_state = JsonState.PARAM_KEY

                # 状態(state)が変化しなくなったら（もう進めないなら）、whileループを抜けて次の文字予測(forループ)へ戻る
                if current_state == old_state:
                    break

        # もし500回のループを回し切っても return できなかったら、そこまでの結果を返す
        return current_text