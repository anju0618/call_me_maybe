Markdown
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

【Developer Debug Mode】
If you want to visualize the token selection process and inspect the state machine transitions in real-time, run the module directly with the --debug flag:
```shell
.venv/bin/python3 -m src --debug
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
### 【デバッグモード】
LLMがどのトークンを選択し、システムがどのように状態（ステート）を遷移させているかをリアルタイムで可視化したい場合は、--debug フラグを付けて直接実行してください
```shell
.venv/bin/python3 -m src --debug
```

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

---
## **用語　/　メモ**
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

## 入力データ
まずは、システムに読み込ませる2つのJSONファイル
* **functions_definition.json** (関数の仕様書)
LLMが利用できる「ツール」のリスト。fn_add_numbers（足し算）や fn_greet（挨拶）などの名前、説明、そして「どんな引数（parameters）が必要で、その型は何か（number か string か）」というルールが書かれています。LLMはこれを見て「どの関数を使い、何の値を抜き出すか」を考える。

* **function_calling_tests.json (テスト問題)**
ユーザーからのプロンプト（「2と3を足して」など）のリストです。これを1つずつLLMに入力し、正しい関数呼び出しJSONを作らせます。

## 2. json_state.py (状態)
LLMに一気にJSONを書かせるのではなく、「今、JSONのどの部分を書いているのか」をシステム側で管理するための定義

## 3. token_filter.py (フィルター)
LLMのボキャブラリから、「今この瞬間に、次に出力していい文字（トークン）だけを絞り込む」ためのフィルタークラス

**filter_by_prefix メソッド:**
「すでに書き終わった文字（current_text）」と、「次に目指すべき完成形（full_target）」を比較。そして、ターゲットの残りの文字列にピッタリと繋がる（はみ出さない）トークンだけを許可（ホワイトリスト化）します。これが、カンマの重複や謎の文字暴走を防ぐ「前方一致」

**filter_numeric_tokens メソッド:**
引数が数字（number）だった場合に、数字や小数点（0-9, .）を含むトークンだけを許可する専用フィルターです。

## 4. json_generator.py
このプロジェクトのコアとなる、制約付きデコードの推論ループを回すエンジンです。最大500回のループを回し、1トークンずつ文字を紡ぎ出します。

**Context Injection:**
ループに入る前、"System: You are an expert JSON function calling assistant..." という指示と関数の仕様書をプロンプトの先頭に合体させます。これにより、LLMが「ただの穴埋め」ではなく「意味を理解して値を抽出する」ようになります。

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
