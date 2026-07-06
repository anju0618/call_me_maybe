## Resorce and AI Usage

[【検証】ローカルLLMでコーディングはどこまでできる？（Qwen 3.5/Gemma 4）](https://note.com/iritec/n/n4f30c8373a77)

[Python プロジェクト管理したくて uv に触れてみたメモ](https://qiita.com/0xmks/items/f5a4fcac81714ac2f803)

[[備忘録] Google Colabで30行！Qwen3-Embedding-0.6Bで日本語テキスト類似度計算](https://qiita.com/Tadataka_Takahashi/items/4ff6e114db134746c835)

[LogitsProcessorZoo で LLM の出力をコントロールする](https://zenn.dev/prgckwb/articles/logits-processor-zoo-explain)

[How Does an LLM Generate Text?](https://pub.towardsai.net/how-does-an-llm-generate-text-fd9c57781217)

## 用語　メモ
* **query**: LLMにおける「query」とは、ユーザーがAIに対して送信するプロンプト、または情報検索システム（RAGなど）が外部データベースや検索エンジンから情報を引っ張ってくるために生成する検索語句のこと。

* **outlines モジュール**: 大規模言語モデル（LLM）からJSON、正規表現、Pydanticモデルなどの構造化された出力を100%確実に生成するためのライブラリ(今回の課題では使用禁止)

* **出力ロジット(Logits)**: モデルが次のトークン（文字や単語の断片）を予測する際に生成する、正規化されていない生のスコア数値. このスコアが高いほど、モデルはそのトークンを出力する可能性が高いと判断する。
　普通のLLMは一番スコアが高いものをそのまま選びますが、それだと高確率でJSONの文法が壊れます。そこでこの課題でやるのは、「現在の文脈から考えて、JSONの文法やスキーマ的に絶対に次に来てはいけないトークン」のスコアを、強制的に -inf（マイナス無限大）に書き換えて抹殺するという処理をした
