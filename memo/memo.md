# 1. src/json_state.py (状態管理)
```python
# src/json_state.py
from enum import Enum, auto

class JsonState(Enum):
    START = auto()          # 1. 冒頭の `{"prompt": "` を出力する状態
    PROMPT_VALUE = auto()   # 2. プロンプト文字列をそのまま出力する状態
    NAME_KEY = auto()       # 3. `", "name": "` を出力する状態
    FUNCTION_NAME = auto()  # 4. 関数名（fn_add_numbers 等）を選ばせる状態
    PARAMS_START = auto()   # 5. `", "parameters": {` を出力する状態
    PARAM_KEY = auto()      # 6. 引数名（"a": や "name":）を出力する状態
    PARAM_VALUE = auto()    # 7. 引数の値をAIにパースさせる状態
```
* Enum (列挙型): これを使うことで、単なる数字（1, 2, 3...）ではなく、JsonState.START のような「意味のある名前」で状態を安全に管理できる

# 2. src/token_filter.py (フィルター)
```python
# src/token_filter.py
class TokenFilter(BaseModel):
    vocab_path: str

    # 辞書型 (Dict[int, str]): トークンIDをキー、文字列を値とする辞書
    # 例: { 1: '"', 2: 'a', 279: 'the', 11: 'Ċ' }
    id_to_token: Dict[int, str] = Field(default_factory=dict)

    # 集合型 (Set[int]): 全てのトークンIDのリスト（重複なし）
    # 例: { 1, 2, 3, ..., 151643 }
    all_token_ids: Set[int] = Field(default_factory=set)

    def filter_by_prefix(self, current_text: str, full_target: str) -> List[int]:
        """全体のターゲット文字列に対して、次に繋がり得るトークンのみ許可"""
        # 例: current_text = '{"prompt": '
        #     full_target  = '{"prompt": "'
        #     remainder (残り) = '"'
        remainder = full_target[len(current_text):]

        allowed_ids: List[int] = []
        for t_id, t_str in self.id_to_token.items():
            # LLM特有の空白文字(Ġ)を普通の空白に直す
            clean_str = t_str.replace("Ġ", " ").replace(" ", " ")

            # 残りの文字列が、このトークンから始まるなら許可！
            if remainder.startswith(clean_str):
                allowed_ids.append(t_id)

        # 許可されたIDのリスト（例: [1, 85, 203]）を返す
        return allowed_ids
```
* 重要なロジック (filter_by_prefix):
「すでに書いた文字」と「目標の文字」を引き算して、「次に書くべき文字（remainder）」を割り出します。そして、辞書の中からそれにピッタリ一致するトークンだけを許可（ホワイトリスト化）します。これにより、はみ出し（オーバーシュート）を物理的に防ぎます。

# 3. src/json_generator.py (推論エンジン)
```python
# src/json_generator.py (一部抜粋・解説付き)

    def generate_function_call(self, prompt: str) -> str:
        # 1. コンテキスト（前提条件）の作成
        # AIが「自分が何をしているか」を忘れないよう、毎回プロンプトの先頭に
        # System指示と、関数のルール（functions_def）をくっつけます。
        context = "System: You are an expert JSON..."

        # current_text (str): これからAIが書き足していく、生成中のJSON文字列
        current_text = ""
        current_state = JsonState.START # 初期状態

        for _ in range(500):
            # input_ids (List[int]): AIが読めるように文章を数字の配列に変換したもの
            # logits (List[float]): AIが「次はこのトークンが来る確率が高い」と予想したスコアの配列
            # 例: [ -10.5, 5.2, 12.8, -3.0, ... ] (辞書の単語数分ある)
            raw_logits = self.model.get_logits_from_input_ids(input_ids)
            logits = list(raw_logits)

            # allowed_tokens (Set[int]): 今回許可されたトークンIDの集合
            allowed_tokens: Set[int] = set()

            # --- フェーズ1: 絶対ターゲットの作成 ---
            # 今の状態に応じて、「次はここまで書け」という目標（full_target）を作る
            if current_state == JsonState.START:
                full_target = '{"prompt": "'
                allowed_tokens = set(self.token_filter.filter_by_prefix(current_text, full_target))

            # (中略: 他のステートでのターゲット生成処理...)

            # --- フェーズ2: ロジット・マスキング ---
            # 許可されていない(allowed_tokensに無い)トークンのスコアを -inf に塗りつぶす！
            # これにより、AIが文法違反の文字を選ぶ確率が 0% になる。
            for token_id in range(len(logits)):
                if token_id not in allowed_tokens:
                    logits[token_id] = float("-inf")

            # スコア(logits)が一番高いトークンを選び、文字に変換して current_text に足す
            next_token_id = int(logits.index(max(logits)))
            current_text += clean_next_str

            # --- フェーズ3: 連鎖同期処理 (Cascade State Sync) ---
            # AIが文字を足した直後、「チェックポイントを通過したか？」を確認する。
            # 合体トークンで一気に進んだ場合でも、状態(current_state)をAIに追いつかせる。
            while True:
                old_state = current_state
                if current_state == JsonState.START:
                    if current_text.endswith('{"prompt": "'):
                        current_state = JsonState.PROMPT_VALUE # 次の状態へ移行
                # (中略: 状態が変化しなくなるまで while ループで確認し続ける)
                if current_state == old_state:
                    break
```

# 4. src/__main__.py
```py
# src/__main__.py (一部抜粋)
def main() -> None:
    # args (argparse.Namespace): ターミナルから渡された引数（--debugなど）
    args = parse_arguments()

    # (JSONファイルの読み込みや、AIモデルの初期化処理...)
    generator = JsonGenerator(...)

    for i, item in enumerate(prompts):
        prompt_text = item.get("prompt", "")

        # max_retries (int): 失敗したときにやり直す最大回数。
        # AIがごく稀にループ上限(500回)に達してバグった時のための「自己修復機能」
        max_retries = 3
        for attempt in range(max_retries):
            # エンジンを回して文字列（json_str）を生成させる
            json_str = generator.generate_function_call(prompt_text)

            try:
                # json.loads: 出来上がった文字列が「本物のJSONか」をテストする
                parsed_result = json.loads(json_str)
                results.append(parsed_result)
                break # 成功したらリトライループを抜ける

            except json.JSONDecodeError as je:
                # パースエラーが起きたら、クラッシュさせずにやり直す
                if attempt < max_retries - 1:
                    print("Invalid JSON detected. Retrying...")
                else:
                    raise je # 3回失敗したら諦めてエラーを投げる
```

# 5. tests/test_token_filter.py

```py
# tests/test_token_filter.py (一部抜粋)

# @pytest.fixture: テストの前準備と後片付けを自動でやってくれる機能
@pytest.fixture
def dummy_vocab_path() -> Generator[str, None, None]:
    """テスト用に、必要最小限の「偽物の単語帳」を作る"""
    vocab_data = {
        "\"": 1,
        "a": 2,
        "}}Ċ": 13, # AIが出してくる「悪魔の合体トークン」
    }
    # (一時ファイルに保存してパスを貸し出し、終わったら消す処理...)
    yield path


def test_filter_by_prefix_overshoot_prevention(dummy_vocab_path: str) -> None:
    """目標を飛び越えるような合体トークンを正しくブロックするか"""
    tf = TokenFilter(vocab_path=dummy_vocab_path)

    # 現在地と目標が設定されたとき...
    current_text = '{"a": 2'
    target = '{"a": 2}'

    # allowed (List[int]): 許可されたトークンIDのリスト
    allowed = tf.filter_by_prefix(current_text, target)

    # id:12 の '}' は許可されるべきだから assert(断言) する！
    assert 12 in allowed

    # id:13 の '}}Ċ' は、目標の '}' を飛び越えてしまう（オーバーシュート）ため、
    # 許可リストには入っていない(not in) はずだと断言する！
    assert 13 not in allowed
```


