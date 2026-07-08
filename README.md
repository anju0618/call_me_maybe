# **Call_Me_Maybe**

## **description**
この課題は，LLMからの出力を特定のルールの下で出力するようなパイプラインを作るもの．

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
```shell
make run
```

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

---
