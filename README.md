*This project has been created as part of the 42 curriculum by amakino.*

# **Call_Me_Maybe**

## **Description**
This project implements a structured, constrained decoding and function calling pipeline that strictly regulates LLM outputs based on predefined schemas. Instead of allowing the LLM to generate arbitrary text, this custom engine force-guides token selection at every generation step, ensuring 100% compliance with a specified JSON format without breaking syntax.

---

## **Instructions**

### **Install**
To set up the local virtual environment and install all required dependencies, run:
```shell
make install
```

### Clean / Reset
To remove runtime caches and generated output files, run:
```shell
make clean
```
To completely delete the virtual environment (.venv) and perform a full factory reset, run:
```shell
make fclean
```
### Execution / Compilation
To execute the pipeline normally and save the JSON results to data/output/, run:
```shell
make run
```
ex output
```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2,
            "b": 3
        }
    }
]
```

【Developer Debug Mode】
If you want to visualize the token selection process and inspect the state machine transitions in real-time, run the module directly with the --debug flag:
```shell
.venv/bin/python3 -m src --debug
```
### **Command Line Arguments (Optional)**
The pipeline can be customized at runtime using the following command-line arguments. This allows you to process different test files or change output destinations without modifying the source code.

* `--input <path>`: Specify a custom path to the input JSON file containing user prompts. *(Default: `data/input/function_calling_tests.json`)*
* `--output <path>`: Specify a custom path where the generated JSON results will be saved. *(Default: `data/output/function_calling_results.json`)*
* `--functions_definition <path>`: Specify a custom path to the JSON file containing the available tool/function schemas. *(Default: `data/input/functions_definition.json`)*
* `--debug`: Enable real-time logging to visualize the token selection process and state machine transitions.

**Advanced Usage Example:**
To run the pipeline with a custom test file, save the output to a specific location, and enable debug logging simultaneously, run:
```shell
uv run python -m src --input custom_tests.json --output data/output/custom_results.json --debug
```

## Architecture & Processing Flow

To prevent the LLM from ever violating JSON syntax or schema rules, the pipeline executes the following 4-step loop for every single token generation:

Step 1: Dynamic Context Injection
Small-scale language models often suffer from "zombie generation" (e.g., repeating endless 0s or fabricating generic placeholders like "user_name") when trapped inside strict constraints without guidance. To restore the AI's contextual awareness, the System prompt and available tool schemas (Functions definitions) are dynamically pre-pended as a context prefix before every token inference step.

Step 2: Absolute Target Construction
Based on the current state of the machine (JsonState), the engine constructs the exact "absolute ideal JSON blueprint" expected up to that point. Rather than sequentially appending text recursively, the system recalculates the full structural road map dynamically, mathematically eliminating common syntax bugs like duplicate commas or broken brackets.

Step 3: Logit Masking
The engine inspects the raw vocabulary scores (Logits) predicted by the LLM. Any token ID that does not strictly match the next immediate characters of the absolute target prefix is masked by setting its score to -inf (minus infinity). This reduces the probability of invalid tokens to exactly 0%.

Step 4: Cascade State Sync
Immediately after a token is locked in, the system checks whether the text has crossed certain structural checkpoints (such as reaching ", "name": " or closing a parameter block). If a giant合体 (merged) token passes multiple checkpoints simultaneously, a while True synchronization loop triggers a Cascade, fast-forwarding the internal state until it perfectly synchronizes with the LLM's current generation coordinate.

## Design Decisions
Strict Prefix Matching: To ensure valid JSON generation, relying on absolute target recalculation and strict prefix matching proved to be the most reliable method. This physically eliminates common syntax errors such as duplicate commas that often occur with relative string appending.

Caching for Performance: Because searching tens of thousands of vocabularies at every token generation step is highly inefficient, the search logic was isolated as a pure function and optimized using functools.lru_cache.

## Performance Analysis
Reliability: By applying logit masking, the probability of generating invalid tokens is strictly forced to 0%, ensuring the pipeline outputs 100% valid, parseable JSON consistently.

Accuracy: Even while utilizing a lightweight model like Qwen3-0.6B, the system achieves exceptional accuracy (90%+) in selecting the correct function and extracting the exact parameters.

Speed: Thanks to lru_cache optimization, the vocabulary filtering overhead is drastically reduced, allowing the entire test suite to be processed rapidly within seconds to a few tens of seconds.

## Challenges Faced
Quote Collision Parsing Errors: We encountered issues where double quotes (") inside the user prompts broke the JSON structure. This was resolved by safely escaping the prompt strings using json.dumps() before integrating them into the absolute target.

BPE Token Overshoot: The AI occasionally output massive merged tokens (e.g., }}Ċ) containing multiple symbols and line breaks, jumping over the state machine's checkpoints. We addressed this by explicitly blocking merged tokens or newlines that do not strictly match the target prefix.

Sync Lag Deadlocks: Delays between the AI's output and the state machine's progression resulted in zero allowed tokens (deadlocks). We resolved this by implementing a "Cascade State Sync" (while True loop) that recursively fast-forwards the state to perfectly synchronize with the AI's current generation coordinate.

## Testing Strategy
To prove the robustness of the core logic (TokenFilter), we implemented a comprehensive unit testing suite using pytest.
Dynamically created a temporary dummy vocabulary (fixture).

Verified strict prefix matching against target strings.

Ensured malicious merged tokens (e.g., }}Ċ) are strictly blocked (overshoot prevention).

Validated the accurate extraction of numeric tokens, confirming the complete exclusion of alphabets and newlines.

## **Bonus Features Implemented**
This project successfully implements 7 of the optional bonus requirements:
1. **Recoding the tokenizer:** Completely avoided the direct use of the SDK's `encode` and `decode` in the main logic, replacing them with a custom Greedy Longest Prefix Match tokenizer using only the vocabulary file.
2. **Performance optimizations:** Implemented `@functools.lru_cache` and avoided redundant full-text re-encoding during the generation loop, significantly boosting execution speed.
3. **Public implementation of tokenizer:** Created a standalone tokenizer class integrating custom `encode` and `decode` methods.
4. **Comprehensive test suite:** Developed a robust testing suite using `pytest` to validate the generator, token filter, and custom tokenizer functionalities.
5. **Visualization of the generation process:** Expanded the `--debug` mode to provide real-time terminal visualization of the token generation and state transitions.
6. **Advanced error recovery mechanisms:** Implemented safe fallbacks, such as mapping irrelevant prompts to an `unknown` function and safely skipping file outputs to prevent crashes.
7. **Demonstration of integration:** Proved how the custom encoding/decoding pipeline smoothly integrates with the constrained decoding logic in the main loop.

## **Resorce and AI Usage**

### LLM
[【検証】ローカルLLMでコーディングはどこまでできる？（Qwen 3.5/Gemma 4）](https://note.com/iritec/n/n4f30c8373a77)

[[備忘録] Google Colabで30行！Qwen3-Embedding-0.6Bで日本語テキスト類似度計算](https://qiita.com/Tadataka_Takahashi/items/4ff6e114db134746c835)

[LogitsProcessorZoo で LLM の出力をコントロールする](https://zenn.dev/prgckwb/articles/logits-processor-zoo-explain)

[How Does an LLM Generate Text?](https://pub.towardsai.net/how-does-an-llm-generate-text-fd9c57781217)

### UV
[Python プロジェクト管理したくて uv に触れてみたメモ](https://qiita.com/0xmks/items/f5a4fcac81714ac2f803)

### Pydanatic
[pydanticによる型検証 [BaseModel]](https://qiita.com/uchksh/items/1cf6958dda52bb19c70b)

### JSON

[【Python】json.dumps() でJSON形式に変換する方法](https://qiita.com/enumura1/items/b50746357569a83db2c3)

## AI Usage
Algorithm Brainstorming & Debugging: Used AI to brainstorm solutions for deadlocks and BPE token overshoot issues (e.g., }}Ċ), and to analyze error logs.

Concept Comprehension: Utilized AI as a tutor to grasp the core concepts of constrained decoding and LogitsProcessors.

Code Generation: Used AI to bootstrap the pytest framework and assist in refactoring the codebase for lru_cache optimization (extracting logic into pure functions).

# **Call_Me_Maybe**

## **description**
この課題は、LLMからの出力を特定のルールの下で出力する（Constrained Decoding / Function Calling）ようなパイプラインを作るもの．
LLMが自由なテキストを生成するのではなく、事前に定義されたJSONスキーマに100%従った構造化データのみを出力するように、出力トークンをシステム側で強制的に制御・誘導するコードを実装している．

## **Instructions**

### **Instal**
仮想環境下で動かすため，
```shell
make install
```
で仮想環境をつくり，必要パッケージをインストールしてください．


### **初期化**
プログラムを動かす過程で生み出されるキャッシュや，outputをremoveするために
```shell
make clean
```
を使用してください．

**仮想環境.`venv`** も削除し，初期化したい場合は
```shell
make fclean
```
を使用してください．

### **Execution / Compilation**
通常の実行（出力を data/output/ に保存）を行う場合は以下のコマンドを実行してください．
```shell
make run
```
出力例
```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2,
            "b": 3
        }
    }
]
```

### 【デバッグモード】
LLMがどのトークンを選択し、システムがどのように状態（ステート）を遷移させているかをリアルタイムで可視化したい場合は、--debug フラグを付けて直接実行してください
```shell
.venv/bin/python3 -m src --debug
```
## Design Decisions（設計上の決定）
* **厳密前方一致 (Strict Prefix Matching):**
  LLMにJSONを生成させる際、最も確実な方法は「一から絶対座標のターゲットを再計算し、それに完全に一致するトークンのみを許可すること」だと判断しました。これにより、相対的な文字追加で発生しやすいカンマの重複などを設計レベルで排除しています。
* **パフォーマンスのためのキャッシュ化:**
  毎トークン生成時に数万件のボキャブラリをループ検索するのは非効率なため、検索ロジックを純粋関数として分離し、functools.lru_cache を導入して高速化を図りました。

## Performance Analysis（パフォーマンス分析）
* **Reliability**: ロジットマスキングにより、不正なトークンの確率を完全に0%にしているため、パース可能な有効なJSONを常に出力します。
* **精度 (Accuracy)**: Qwen3-0.6B という軽量モデルを使用しながらも、関数名の選択と引数の抽出において極めて高い精度を達成しています。
* **速度 (Speed)**: lru_cache による最適化のおかげで、ボキャブラリのフィルタリング負荷が劇的に下がり、全テストケースを数分で高速に処理可能です。

## Challenges Faced（直面した課題と解決策）
* **ダブルクォートの衝突によるパースエラー:**
  プロンプト内に含まれる " がJSONの構造を破壊する問題に直面しました。これは、プロンプト文字列を json.dumps() で安全にエスケープ処理してからターゲットに組み込むことで解決しました。
* **BPEトークンのオーバーシュート:**
  AIが }}Ċ などの複数の記号や改行が合体した巨大なトークンを出力し、ステートマシーンのチェックポイントを飛び越えてしまう問題が発生しました。ターゲットの接頭辞に厳密に一致しない合体トークンや改行を、フィルター側で明示的にブロック（除外）することで解決しました。
* **同期ラグによるデッドロック:**
  AIの出力に対してステートマシーンの進行が遅れ、許可トークンが0個になる問題は、while True ループによる連鎖的な状態同期（Cascade State Sync）を実装することでAIとプログラムの現在地をミ同期させて解決しました。

## Testing Strategy（テスト戦略）
最も複雑なコアロジックである TokenFilter の堅牢性を証明するため、pytest を用いた包括的なユニットテストを実装しました。

## **Bonus Features Implemented（実装済みのボーナス要件）**
本プロジェクトでは、評価シートに記載されているボーナス要件のうち、以下の7つを完全に実装しています。
1. **トークナイザーの再実装 (Recoding the tokenizer):** LLM SDKの `encode` / `decode` メソッドをメインコードから完全に排除し、`vocab.json` のみを用いた最長一致（Greedy Longest Prefix Match）による自作エンコーダに差し替えました。
2. **パフォーマンスの最適化 (Performance optimizations):** `@functools.lru_cache` を用いたキャッシュ処理と、推論ループ内での再エンコードを避ける配列操作（`.append`）の導入により、処理速度を劇的に向上（爆速化）させました。
3. **公開されたトークナイザーの実装 (Public implementation of tokenizer):** ボキャブラリファイルと完全に互換性のある `encode` および `decode` メソッドを自作し実装しました。
4. **包括的なテストスイート (Comprehensive test suite):** `pytest` を導入し、ダミーのLLMモデルやボキャブラリを用いた堅牢なユニットテスト（フィルター、ジェネレーター、トークナイザー）を構築しました。
5. **生成プロセスの可視化 (Visualization of the generation process):** `--debug` モードを実行することで、状態遷移とトークンの選択プロセスがターミナル上でリアルタイムに可視化されるようにしました。
6. **高度なエラー回復メカニズム (Advanced error recovery mechanisms):** 該当しないプロンプト（Tokyoの首都など）が入力された際にクラッシュや無限ループを起こさず、安全に `unknown` 関数としてスキップ処理を行う堅牢な例外処理を実装しました。
7. **エンコード/デコードと制約付きデコードの統合デモ (Demonstration of integration):** 自作したトークナイザーのエンコード結果が、メインループのロジットマスキングおよびデコード処理とシームレスに統合されていることをコード上で証明しました。

## **Architecture & Processing Flow**
本プロジェクトのパイプラインは、LLMがJSONの文法を絶対に壊さないよう、以下の4つのステップを1ループとして推論を行っています。

### Step 1: 動的コンテキストの注入 (Context Injection)
AIにただJSONの穴埋めをさせるだけでは、プロンプトの意図を無視したプレースホルダー（"user_name" など）や無限の 0 を出力する「ゾンビ化」を引き起こします。そのため、毎ターンの推論直前に System プロンプトや利用可能な Functions（関数定義書のJSON）をテキストの先頭にドッキングし、AIに「タスクの目的」を理解させた上でトークンを予測させます。

### Step 2: 絶対座標ターゲットの構築 (Absolute Target Construction)
現在の状態（ステート）に基づき、「次に目指すべきJSONの完成形（ターゲット文字列）」を一意に生成します。
（例：`{"prompt": "...", "name": "fn_add_numbers", "parameters": {"a": `）
相対的に文字を足していくのではなく、常に最初から最後までの「絶対座標」を再計算することで、カンマの重複などの構文エラーを物理的に防いでいます。

### Step 3: トークンのロジット・マスキング (Logit Masking)
LLMが予測した次トークンの候補（Logits）のうち、Step 2で作ったターゲット文字列の「純粋な続き（接頭辞）」に該当しないトークンのスコアを -inf（マイナス無限大）に書き換えます。これにより、JSONの文法から外れるトークンが選ばれる確率が完全に 0% になります。

### Step 4: 連鎖同期ステートマシーン (Cascade State Sync)
AIが新しいトークンを出力した直後、現在のテキストが「チェックポイント（例：", "name": "）」に到達したかを判定します。到達していれば、即座に次の状態へ前進させます。AIが巨大なトークンを出力して複数のチェックポイントを一度に通過した場合は、while True ループで状態が追いつくまで何段階でも連鎖的にワープ（Cascade）させ、AIとプログラムの現在地をミリ単位で同期します。


## **Features / Core Logic**
本プロジェクトのパイプラインは、主に以下の技術とロジックで構成されています．
1. **ステートマシーン (JsonState):**
  JSONの生成プロセスを START $\rightarrow$ PROMPT_VALUE $\rightarrow$ NAME_KEY $\rightarrow$ FUNCTION_NAME $\rightarrow$ PARAMS_START $\rightarrow$ PARAM_KEY $\rightarrow$ PARAM_VALUE という状態NI分割．AIの出力が確定した瞬間に、次の状態へと自動で前進させます．
2. **厳密前方一致によるトークンフィルタリング (TokenFilter):**
  AIが目標の文字列を飛び越えてしまう**オーバーシュートや**、他の関数の誤検知を防ぐため、常にその状態における**座標のターゲット文字列**を構築．現在地がその接頭辞（`prefix`）と完全に一致するルートのトークンのみを許可（ホワイトリスト化）します．
3. **動的コンテキスト注入:**
  LLMがただのJSONの穴埋めではなく「タスク」として文脈を理解できるよう、毎ターン System プロンプトや Functions（関数定義）のコンテキストをプレフィックスとして結合して推論させます．



## **Resorce and AI Usage**

### LLM関係
[【検証】ローカルLLMでコーディングはどこまでできる？（Qwen 3.5/Gemma 4）](https://note.com/iritec/n/n4f30c8373a77)

[[備忘録] Google Colabで30行！Qwen3-Embedding-0.6Bで日本語テキスト類似度計算](https://qiita.com/Tadataka_Takahashi/items/4ff6e114db134746c835)

[LogitsProcessorZoo で LLM の出力をコントロールする](https://zenn.dev/prgckwb/articles/logits-processor-zoo-explain)

[How Does an LLM Generate Text?](https://pub.towardsai.net/how-does-an-llm-generate-text-fd9c57781217)

### UV関係
[Python プロジェクト管理したくて uv に触れてみたメモ](https://qiita.com/0xmks/items/f5a4fcac81714ac2f803)

### Pydanatic関係
[pydanticによる型検証 [BaseModel]](https://qiita.com/uchksh/items/1cf6958dda52bb19c70b)

### JSON

[【Python】json.dumps() でJSON形式に変換する方法](https://qiita.com/enumura1/items/b50746357569a83db2c3)

### AI USAGE
本プロジェクトの開発において、AI（LLM）は主に以下の用途で活用しました。
* **アルゴリズムの壁打ちとデバッグ:** BPEトークナイザー特有の合体トークン（例: }}Ċ）によるオーバーシュート問題や、デッドロックの原因究明において、ログの解析と解決策のブレインストーミングにAIを使用しました。
* **概念の理解:** ロジットプロセッサ（LogitsProcessor）の仕組みや、制約付きデコードの基礎概念を学ぶための家庭教師として活用しました。
* **コード生成:** pytest を用いたテストフレームワークの雛形作成や、lru_cache をクラスメソッドに適用するためのリファクタリング（純粋関数への切り出し）の補助としてAIを使用しました。

---
## **用語 / メモ**
* **query**: LLMにおける「query」とは、ユーザーがAIに対して送信するプロンプト、または情報検索システム（RAGなど）が外部データベースや検索エンジンから情報を引っ張ってくるために生成する検索語句のこと。

* **outlines モジュール**: 大規模言語モデル（LLM）からJSON、正規表現、Pydanticモデルなどの構造化された出力を100%確実に生成するためのライブラリ(今回の課題では使用禁止)

* **出力ロジット(Logits)**: モデルが次のトークン（文字や単語の断片）を予測する際に生成する、正規化されていない生のスコア数値. このスコアが高いほど、モデルはそのトークンを出力する可能性が高いと判断する。
 普通のLLMは一番スコアが高いものをそのまま選びますが、それだと高確率でJSONの文法が壊れます。そこでこの課題でやるのは、「現在の文脈から考えて、JSONの文法やスキーマ的に絶対に次に来てはいけないトークン」のスコアを、強制的に -inf（マイナス無限大）に書き換えて抹殺するという処理をした

* **vocab.json**の中身: 以下のように「文字列」と「数字（ID）」がペアになった巨大な辞書（JSONオブジェクト）
```json
{
  "What": 3838,
  "is": 374,
  "the": 279,
  "sum": 2629,
  "1": 16,
  "2": 17,
  "3": 18
}
```

* **マスク処理**: AIにおける「マスク処理」とは、AIの視界に目隠し（マスク）をはめて、特定の言葉以外を物理的に見えなくする処理のこと.
```py
for token_id in range(len(logits)):
    if token_id not in allowed_tokens:
        logits[token_id] = float("-inf") # <-ここで、許可されていないトークンのロジットに-infを入れて、確率を０にしている
```
ボキャブラリデータそのものを消してるわけじゃないから、「マスク」

* **class ~(Basemodel)**: pydenticの検証機能を備えたクラスになる
```py
model_config = ConfigDict(arbitrary_types_allowed=True)
```
配布されたSmall_LLM_Model は標準的な型（int や str）ではないため、それをと許可する設定

* **オーバーシュート (Overshoot):** BPEトークナイザー特有の問題。AIが選んだ合体トークン（例：", "）が、システム側が想定しているターゲットの終端（例："）を飛び越えてしまう現象。これを許可してしまうと、システム側のステートマシーンが通過判定を逃して取り残されてしまう（同期ラグ）。本システムではこれを防ぐため、目標文字列からはみ出すトークンを許さない「厳密前方一致 (Strict Prefix Matching)」を採用している。

* **デッドロック:** フィルターのバグやステートの同期ラグにより、現在の状態で「許可されるトークン（allowed_tokens）」が0個になってしまう現象。すべてのロジットが -inf に塗りつぶされた結果、LLMは制御を失って「！」などを無限に出力する暴走状態に陥る。

* **Cascade State Sync（カスケード・ステート・シンク）**: ITシステムやネットワーク、またはマルチプレイゲームにおいて、「連鎖的な状態同期」を意味する技術用語

---

# 解説

## 入力データ (data/input/)
まずは、システムに読み込ませる2つのJSONファイル
* **functions_definition.json** (関数の仕様書)
LLMが利用できる「ツール」のリスト。fn_add_numbers（足し算）や fn_greet（挨拶）などの名前、説明、そして「どんな引数（parameters）が必要で、その型は何か（number か string か）」というルールが書かれています。LLMはこれを見て「どの関数を使い、何の値を抜き出すか」を考える。

* **function_calling_tests.json (テスト問題)**
ユーザーからのプロンプト（「2と3を足して」など）のリストです。これを1つずつLLMに入力し、正しい関数呼び出しJSONを作らせます。

## 2. json_state.py (状態)
LLMに一気にJSONを書かせるのではなく、「今、JSONのどの部分を書いているのか」をシステム側で管理するための定義

## 3. token_filter.py (フィルター)
LLMのボキャブラリから、「今この瞬間に、次に出力していい文字（トークン）だけを絞り込む」ためのフィルタークラス
* **filter_by_prefix メソッド (厳密前方一致 & オーバーシュート防止):**
「すでに書き終わった文字」と「次に目指すべきJSONの完成形」を比較し、ターゲットの残りの文字列にピッタリと繋がる（はみ出さない）トークンだけを許可（ホワイトリスト化）します。これにより、カンマの重複や、`}}Ċ `のようなトーークンによるJSON破壊をブロック

* **filter_by_prefix メソッド:**
「すでに書き終わった文字（current_text）」と、「次に目指すべき完成形（full_target）」を比較。そして、ターゲットの残りの文字列にピッタリと繋がる（はみ出さない）トークンだけを許可（ホワイトリスト化）します。これが、カンマの重複や謎の文字暴走を防ぐ「前方一致」

* **【Bonus】 functools.lru_cache によるパフォーマンス最適化:**
毎トークン生成時に発生する「数万件のボキャブラリ検索」と「文字列のクリーンアップ処理（replace）」の負荷を下げるため、検索ロジックを純粋関数としてクラス外に切り出し、lru_cache を適用しています。単語帳をTuple に事前変換してキャッシュに乗せることで、パフォーマンス最適化しています。

* **filter_numeric_tokens メソッド:**
引数が数字（number）だった場合に、数字や小数点（0-9, .）を含むトークンだけを許可する専用フィルターです。

## 4. json_generator.py
このプロジェクトのコアとなる、制約付きデコードの推論ループを回すエンジンです。最大500回のループを回し、1トークンずつ文字を紡ぎ出します。

**Context Injection:**
LLMを意図を理解するAIにするため、推論前に "System: You are an expert JSON..." という強プロンプトと関数定義書を結合します。

**絶対座標ターゲットの作成:**
現在の状態（`current_state`）に応じて、「次はここまで書かせたい」という絶対目標（`full_target`）を作ります。

**ロジット・マスキング (Logit Masking):**
token_filter が許可しなかったすべてのトークンのスコア（logits）を、強制的に -inf（マイナス無限大）に書き換えます。これにより、文法を壊すトークンが選ばれる確率を0% にします。

**Cascade State Sync:**
AIがトークンを出力した直後、「チェックポイント（目標文字列）を通過したか？」を確認します。BPEトークナイザーの都合でAIがデカい合体トークン（", "name": " など）を出して一気に進んだ場合でも、while True ループで状態をぐるぐる回し、AIの現在地にプログラムの状態を追いつかせます

## 5. __main__.py (実行スクリプト)

**コマンドライン引数:**
--input や --output でファイルの場所を指定できるほか、私たちがデバッグの終盤で実装した --debug フラグ を受け取る
**処理の進行:**
JSONファイルやLLMモデルを読み込み、JsonGenerator を起動します。1問ずつプロンプトを投げて処理を行い、パースエラーが起きないかを json.loads でテストしてから、最終的な結果を function_calling_results.json に書き出して保存します。

## 6. tests/test_token_filter.py(bonus: Comprehensive test suite)

ダミーのボキャブラリ生成し、システムが予期せぬ挙動を示さないかを検証
1. 初期化とボキャブラリのマッピングテスト
2. 前方一致によるトークンフィルタリングの精確性
3. 悪意ある合体トークン（例：}}Ċ）によるオーバーシュート（飛び越え）の完全ブロック
4. 数値トークンの正確な抽出

# test
```json
[
  {
    "prompt": "What is the sum of -12.5 and 1000000.001?"
  },
  {
    "prompt": "Is 42 an even number?"
  },
  {
    "prompt": "Turn on the living room light."
  },
  {
    "prompt": "Please turn off the bedroom light."
  },
  {
    "prompt": "Replace all vowels in 'Hello \"World\"' with '*'"
  },
  {
    "prompt": "Greet 42_Student!"
  },
  {
    "prompt": "What is the capital of Tokyo?"
  },
  {
    "prompt": "Who is the president of the United States?"
  }
]
```
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum. (Tests floats and negative numbers)",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  },
  {
    "name": "fn_is_even",
    "description": "Check if an integer is even. (Tests strict integer parsing without dots)",
    "parameters": {
      "n": { "type": "integer" }
    },
    "returns": { "type": "boolean" }
  },
  {
    "name": "fn_set_light_status",
    "description": "Turn a smart light on or off. (Tests boolean true/false parsing)",
    "parameters": {
      "is_on": { "type": "boolean" }
    },
    "returns": { "type": "string" }
  },
  {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace regex matches in a string. (Tests complex string escaping and multiple parameters)",
    "parameters": {
      "source_string": { "type": "string" },
      "regex": { "type": "string" },
      "replacement": { "type": "string" }
    },
    "returns": { "type": "string" }
  },
  {
    "name": "fn_greet",
    "description": "Greet a user. (Tests simple string parsing)",
    "parameters": {
      "name": { "type": "string" }
    },
    "returns": { "type": "string" }
  }
]
```


---

## **Review Guide (評価シート解説とテストケース)**

レビューア（評価者）の皆様へ。このセクションは、評価シート（Scale）に記載されている各項目の意味と、具体的な確認・テスト方法をまとめたものです。スムーズな評価にお役立てください。

### **1. Preliminaries (前提条件)**
*   **The project must be run using `uv run python -m src`**
    *   *(意味)* 指定されたコマンドで実行できること。
    *   *(テスト)* `make run` または `uv run python -m src` で正常に起動するか確認してください。
*   **All errors should be handled gracefully without crashing**
    *   *(意味)* クラッシュ（Tracebackを伴う異常終了）しないこと。
    *   *(テスト)* 存在しないファイルを指定するなどのエラーを起こしても、赤いエラーログを出さずに終了することを確認します。
*   **Check that constrained decoding is implemented (not just prompting)**
    *   *(意味)* ただLLMにプロンプトを投げるだけでなく、出力トークンをシステム側で制御（制約付きデコード）していること。
    *   *(テスト)* `src/json_generator.py` を確認し、`logits` を `-inf` でマスクする処理があることを確認してください。

### **2. Project Structure and Dependencies (プロジェクト構造と依存関係)**
*   **Run `uv sync` successfully**
    *   *(意味)* 依存関係が正常にインストールできること。
    *   *(テスト)* `make install` を実行し、エラーが出ないことを確認します。
*   **Check that all classes use pydantic for validation**
    *   *(意味)* すべてのクラスが Pydantic を使って定義されていること。
    *   *(テスト)* `src/token_filter.py` や `src/json_generator.py` が `BaseModel` を継承していることを確認します。

### **3. Input File Handling (入力ファイルの処理)**
*   **Test with invalid JSON in input files / missing input files**
    *   *(意味)* 不正なJSONや、ファイルが存在しない場合に適切にエラーを出すこと。
    *   *(テスト)* 以下のコマンドで、クラッシュせずにエラーメッセージが出力されるか確認します。
    ```shell
    # 存在しないファイルを指定する
    uv run python -m src --input nonexistent.json
    ```

### **4. Output File Format (出力フォーマット)**
*   **Verify the file contains 100% valid and retrievable JSON**
    *   *(意味)* 生成されたファイルが、構文エラーのない100%パース可能なJSONであること。
    *   *(テスト)* `data/output/function_calling_results.json` を開き、末尾のカンマや括弧の閉じ忘れがないことを確認します。

### **5. Function Calling Accuracy (関数呼び出しの精度)**
*   **Verify correct function selection / argument extraction accuracy (>90% expected)**
    *   *(意味)* 適切な関数を選び、引数を正確に抽出できていること。
    *   *(テスト)* `make run` 実行後の出力結果を見て、プロンプトの意図と関数のパラメータが正しく一致しているか確認します。

### **6. LLM SDK Usage (SDKの利用)**
*   **Ensure no private methods or attributes are accessed**
    *   *(意味)* 提供された `Small_LLM_Model` のプライベートメソッド（`_`で始まるものなど）を使用していないこと。
    *   *(テスト)* コード内で SDK に対して `encode()`, `get_logits_from_input_ids()`, `get_path_to_vocab_file()` のような公開インターフェース以外を使っていないか確認します（※本プロジェクトではボーナス要件のため `encode` すら使用していません）。

### **7. Error Handling and Robustness (エラー処理と堅牢性)**
*   **Test with prompts that don't match any function**
    *   *(意味)* 定義されたどの関数にも当てはまらないプロンプトの処理。
    *   *(テスト)* 「What is the capital of Tokyo?」などの無関係な質問が含まれたテストデータを実行した際、`{"name": "unknown"}` として処理し、出力への書き込みをスキップ（警告のみ出力）することを確認します。

### **8. Performance and Reliability (パフォーマンスと信頼性)**
*   **Check that all test prompts are processed in reasonable time (<5 minutes)**
    *   *(意味)* 全プロンプトの処理が5分以内に完了すること。
    *   *(テスト)* `make run` を実行し、完了までの時間が5分を余裕で下回っている（数秒〜数十秒程度である）ことを確認します。これは `lru_cache` などの最適化によるものです。

### **9. Code Quality and Documentation (コード品質とドキュメント)**
*   **Check that README.md explains the algorithm clearly**
    *   *(意味)* READMEにアルゴリズム、設計上の決定、直面した課題が明記されていること。
    *   *(テスト)* このREADMEの上部セクションを読んでいただき、内容が十分か判断してください。

### **10. Moulinette Evaluation (Moulinetteによる評価)**
*   **Run the moulinette evaluation**
    *   *(意味)* 配布されたMoulinette（評価用自動スクリプト）のプライベートテストを通過すること。
    *   *(テスト)* 評価者の環境にて、Moulinetteディレクトリ内で以下のコマンドを実行してください。
    ```shell
    uv sync
    uv run python -m moulinette prepare_exercises --set private
    uv run python -m moulinette grade_student_answers --set private --student_answer_path <学生の出力ファイルパス>
    ```

### **11. Bonus (ボーナス要件)**
本プロジェクトは以下のボーナス要件を満たしています。評価シートの Bonus セクションでチェックをお願いします。

*   **Recoding the tokenizer** / **Public implementation of tokenizer**
    *   *(テスト)* メインロジックがSDKの `encode` を使っていないことを確認してください。また、以下のコマンドで自作トークナイザーの実演が可能です。
    ```shell
    uv run python -m src.tokenizer --vocab llm_sdk/llm_sdk/vocab.json --text "Hello, World!"
    ```
*   **Performance optimizations (caching, batching)**
    *   *(テスト)* `src/token_filter.py` に `@functools.lru_cache` が実装されており、処理が高速化されていることを確認してください。
*   **Comprehensive test suite**
    *   *(テスト)* 以下のコマンドで、実装された包括的なユニットテストが実行され、すべてパスすることを確認してください。
    ```shell
    make test
    ```
*   **Visualization of the generation process**
    *   *(テスト)* 以下のコマンドでデバッグモードを起動し、トークン生成と状態遷移がリアルタイムで可視化されることを確認してください。
    ```shell
    uv run python -m src --debug
    ```
*   **Advanced error recovery mechanisms**
    *   *(テスト)* `try-except` による安全な例外処理や、プロンプトが関数に合致しない場合の `unknown` 回避機構（出力スキップ）が実装されていることを確認してください。