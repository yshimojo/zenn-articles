---
title: "Vertex AI Vector Search のハイブリッド検索を日本語で試してみた"
emoji: "🔍"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["tech", "googlecloud", "vertexai", "vectorsearch", "embedding"]
published: true
---
[Google Cloud Japan Advent Calendar 2024](https://zenn.dev/google_cloud_jp/articles/7799cce9f23cf0) 6 日目です！

こんにちは、カスタマーエンジニアの下門 (しもじょう) です。

皆さんは [Vertex AI Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview) に [Hybrid Search](https://cloud.google.com/vertex-ai/docs/vector-search/about-hybrid-search) という機能があるのをご存知でしょうか？

簡単に言うと、セマンティック検索とキーワード検索を組み合わせることで、検索精度の向上を図る「ハイブリッド検索」を実現する機能です。

キーワード検索は、明確な関連度 (スコア) 計算により結果が解釈しやすく、チューニングも容易ですが、ユーザーの意図や文脈を捉えきれないため、関連性の低い結果が表示されたり、ゼロ件ヒットとなる可能性があります。
一方、セマンティック検索は、文脈を理解し意味的に近い結果を返すため、検索漏れを減らせますが、Embedding モデルの精度に依存し、説明可能性が低いという側面があります。
ハイブリッド検索ではこれらの長所を活かし、短所を補い合うことで、より良い検索体験の向上を図ります。

Vector Search では、従来からサポートしているセマンティック検索用の密ベクトル (Dense Vector) に加えて、キーワード検索用の疎ベクトル (Sparse Vector) を今年の 5 月からサポートするようになりました。
この 2 種類のベクトルを利用することでハイブリッド検索を実現します。

本記事では疎ベクトルについて解説した後に Vector Search を利用した日本語によるハイブリッド検索の実装方法を解説いたします。
(英語でも良いのでサクッと試したい方向けには[サンプルノートブック](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/embeddings/hybrid-search.ipynb)が公開されていますので、そちらをご参照ください)

:::message
本記事内のサンプルコードは全て [Colab Enterprise](https://cloud.google.com/colab/docs/introduction) 上での実行結果となります。ご利用の環境によっては追加でパッケージやモジュールのインポートが必要になる可能性があります。
:::

## 疎ベクトル (Sparse Vector) とは？

そもそも疎ベクトル (Sparse Vector) とは何でしょうか？
単的に言うと、少数の要素にのみ `0 以外の値`が入っており、その他ほとんどの要素には `0` が入っているベクトルのことです。

![token_search](https://cloud.google.com/static/vertex-ai/docs/vector-search/images/token_search.gif)

実例を見ていただいた方がイメージつきやすいと思いますので、日本語の疎ベクトルを実際に作成してみましょう。

例えば、以下のような 5 つの文書からなる日本語コーパスがあるとします。

```python
corpus_ja = [
    "東京は大阪の東にある",
    "大阪は東京の西にある",
    "京都は大阪の北にある",
    "札幌は東京の北にある",
    "那覇は大阪の南にある"
]
```

まずはじめに、日本語の形態素解析器などを利用して、全ての文書を意味のある最小単位の単語に分割します。(この処理をトークナイズと言います)

例えば、上記一行目の文書 (`doc0`) を日本語トークナイズすると `['東京', 'は', '大阪', 'の', '東', 'に', 'ある']` のように分割されます。

分割された各トークンが各ドキュメントに含まれるかどうかのマッピングを以下のようなデータとして保持します。

|      |   ある |   に |   の |   は |   京都 |   北 |   南 |   大阪 |   札幌 |   東 |   東京 |   西 |   那覇 |
|:-----|-------:|-----:|-----:|-----:|-------:|-----:|-----:|-------:|-------:|-----:|-------:|-----:|-------:|
| doc0 |      1 |    1 |    1 |    1 |      0 |    0 |    0 |      1 |      0 |    1 |      1 |    0 |      0 |
| doc1 |      1 |    1 |    1 |    1 |      0 |    0 |    0 |      1 |      0 |    0 |      1 |    1 |      0 |
| doc2 |      1 |    1 |    1 |    1 |      1 |    1 |    0 |      1 |      0 |    0 |      0 |    0 |      0 |
| doc3 |      1 |    1 |    1 |    1 |      0 |    1 |    0 |      0 |      1 |    0 |      1 |    0 |      0 |
| doc4 |      1 |    1 |    1 |    1 |      0 |    0 |    1 |      1 |      0 |    0 |      0 |    0 |      1 |

上記は単純化した例のため、単語の出現回数が多くなればなるほど値が大きくなってしまいますが、実際の全文検索エンジンなどでは単語の重要度に応じた重み付けが行われます。

本記事では最もベーシックなアルゴリズムである [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) (Term Frequency-Inverse Document Frequency) を利用して単語の重み付けを行った疎ベクトルを作成していきます。

TF-IDF のスコア計算は TF 値 (単語の出現頻度) と IDF 値 (逆文書頻度) という 2 つの指標に基づいて計算されます。
- TF 値 : ある文書の中である単語の出現回数が多ければスコアが増加する
- IDF 値 : 検索対象の全文書の中でその単語が出現する文書の数が少なければスコアが増加する

:::message
疎ベクトルの作成には、Elasticsearch や Solr のデフォルトアルゴリズムとして採用されている TF-IDF の進化系の [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) や BERT ベースのエンコーダモデルである [SPLADE](https://en.wikipedia.org/wiki/Learned_sparse_retrieval) なども利用可能です。
:::

### MeCab を利用した日本語トークナイズ

事前に日本語トークナイズをする必要がありますが、今回は OSS の形態素解析エンジンである [MeCab](https://ja.wikipedia.org/wiki/MeCab) を Python で利用します。

必要なパッケージをインストールして MeCab の Tagger オブジェクトを利用した日本語トークナイザ関数を定義します。

```python
! pip install mecab-python3 unidic-lite
```
```python
import MeCab

# MeCab の形態素解析器オブジェクトを作成
tagger = MeCab.Tagger()

# MeCab を用いた日本語トークナイザ関数
def mecab_tokenizer(text):
    """ 日本語テキストをトークン化 """
    node = tagger.parseToNode(text)
    tokens = []
    while node:
        if node.surface != "":  # 空白行を除外
            tokens.append(node.surface)
        node = node.next
    return tokens
```

### TF-IDF 疎ベクトル (Sparse Vector) を生成

TF-IDF 疎ベクトルの作成には [scikit-learn](https://scikit-learn.org/stable/) の [TfidfVectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) を利用します。
文書 (ドキュメント) は先ほどの `corpus_ja` を利用し、トーカナイザには先ほど定義した `mecab_tokenizer` 関数を指定しています。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF ベクトルを作成
# カスタムトークナイザを使用するため token_pattern=None を指定
tfidf_vectorizer = TfidfVectorizer(tokenizer=mecab_tokenizer, token_pattern=None)
tfidf_vectors = tfidf_vectorizer.fit_transform(corpus_ja)

import pandas as pd

# TF-IDF ベクトルを Pandas DataFrame に変換して表示
tfidf_vectors_df = pd.DataFrame(tfidf_vectors.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tfidf_vectors_df.head()
```

作成した疎ベクトルは以下のようになりました。 

|      |     ある |       に |       の |       は |     京都 |       北 |       南 |     大阪 |     札幌 |       東 |     東京 |       西 |     那覇 |
|:-----|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| doc0 | 0.291391 | 0.291391 | 0.291391 | 0.291391 | 0        | 0        | 0        | 0.344517 | 0        | 0.611516 | 0.409539 | 0        | 0        |
| doc1 | 0.291391 | 0.291391 | 0.291391 | 0.291391 | 0        | 0        | 0        | 0.344517 | 0        | 0        | 0.409539 | 0.611516 | 0        |
| doc2 | 0.280952 | 0.280952 | 0.280952 | 0.280952 | 0.589609 | 0.475693 | 0        | 0.332176 | 0        | 0        | 0        | 0        | 0        |
| doc3 | 0.27476  | 0.27476  | 0.27476  | 0.27476  | 0        | 0.465209 | 0        | 0        | 0.576615 | 0        | 0.386166 | 0        | 0        |
| doc4 | 0.265314 | 0.265314 | 0.265314 | 0.265314 | 0        | 0        | 0.556792 | 0.313687 | 0        | 0        | 0        | 0        | 0.556792 |

「ある」といった動詞や「に」「の」「は」といった助詞は全てのドキュメントにまんべんなく出現するため相対的に重みが小さく、一方で名詞は特定のドキュメントにのみ出現するため相対的に重みが大きくなっていることが分かります。

### クエリとのスコアを計算

実際にクエリを作成してドキュメントとの間のスコアを計算してみましょう。
今回クエリには「`大阪は京都の南にある`」というテキストを入力してみます。

```python
query = "大阪は京都の南にある"

# TF-IDF クエリベクトルを作成
query_vector = tfidf_vectorizer.transform([query])

# 各文書ベクトルとクエリベクトル間のスコア (ドット積) を計算
scores = (tfidf_vectors * query_vector.T).toarray()

# 各文書とスコアを DataFrame に変換して降順で表示
scores_df = pd.DataFrame({'docs': corpus_ja, 'scores': scores.flatten()})
scores_df.sort_values('scores', ascending=False).head()
```

スコアの結果です。

|      | docs                 |   scores |
|:-----|:---------------------|---------:|
| doc2 | 京都は大阪の北にある | 0.730651 |
| doc4 | 那覇は大阪の南にある | 0.689983 |
| doc0 | 東京は大阪の東にある | 0.417311 |
| doc1 | 大阪は東京の西にある | 0.417311 |
| doc3 | 札幌は東京の北にある | 0.291591 |

「`京都`」や「`大阪`」といった固有名詞がマッチした `doc2` が最も高いスコアとなっており、次点で「`南`」という一般名詞がマッチした `doc4` の順になっています。

ちなみに `tfidf_vectors` の中身を見ると実際には以下のデータ形式となっています。

```python
print(tfidf_vectors)
```
```
  (0, 10)	0.4095392593497724
  (0, 3)	0.29139055604235914
  (0, 7)	0.34451733585505817
  (0, 2)	0.29139055604235914
  ...
  (4, 1)	0.26531423910850577
  (4, 0)	0.26531423910850577
  (4, 12)	0.5567917225517395
  (4, 6)	0.5567917225517395
```

上記は、例えば `0` 行 `10` 列目 の値が `0.4095392593497724`であることを意味しています。

というのも、疎ベクトルは密ベクトルとは異なり、**可変長のベクトル** となります。
トークン数 = ベクトルの次元数となるため、大量に `0` の値が入ったベクトルをそのまま扱うと非常に冗長となります。したがって上記のように値が入っている次元のみインデックスと値がペアになった状態で格納されます。

後ほど解説しますが、実際に Vector Search に疎ベクトルを登録する際にも同様に `0` 以外の値が入っている次元のインデックスと値のみをインプットします。

## Vector Search でハイブリッド検索を実装

本題である Vector Search におけるハイブリッド検索を実装していきます。

### 日本語データセットの準備

はじめに日本語のデータセットを用意します。

今回は Python の Wikipedia モジュールを利用して、各都道府県ごとに存在する「○○県の観光地」というタイトルの Wikipedia ページをインポートして利用します。

```python
! pip install wikipedia
```
```python
import wikipedia
import pandas as pd

# Wikipedia の言語を日本語に設定
wikipedia.set_lang("ja")

# 都道府県名のリスト
prefectures = ["北海道","青森県","岩手県","秋田県","宮城県","山形県","福島県","茨城県","栃木県","群馬県","埼玉県","千葉県","東京都","神奈川県","新潟県","富山県","石川県","福井県","山梨県","長野県","岐阜県","静岡県","愛知県","三重県","滋賀県","京都府","大阪府","兵庫県","奈良県","和歌山県","鳥取県","島根県","岡山県","広島県","山口県","徳島県","香川県","愛媛県","高知県","福岡県","佐賀県","長崎県","熊本県","大分県","宮崎県","鹿児島県","沖縄県"]

# 各都道府県ごとに「〇〇の観光地」というタイトルの Wikipedia ページを取得
pages = [wikipedia.page(prefecture + "の観光地", auto_suggest=False) for prefecture in prefectures]

# 抽出したデータを Pandas DataFrame に格納
df = pd.DataFrame({
    'title': [page.title for page in pages],  # 各 Wikipedia ページのタイトル
    'url': [page.url for page in pages],  # 各 Wikipedia ページの URL
    'content': [page.content for page in pages]  # 各 Wikipedia ページの内容
})

# 各 Wikipedia ページの内容を corpus_ja に格納
corpus_ja = df.content.tolist()
```

各ページごとの **title**, **url**, **content** を Pandas DataFrame に変換して `df.head()` で先頭行を表示した結果は以下の通りです。

|    | title           | url                                                        | content                                                                     |
|----|--------------------|-------------------------------------------------------------|-----------------------------------------------------------------------------|
| 0  | 北海道の観光地     | [https://ja.wikipedia.org/wiki/%E5%8C%97%E6%B5%...](https://ja.wikipedia.org/wiki/%E5%8C%97%E6%B5%...) | 北海道の観光地（ほっかいどうのかんこうち）は、北海道内の主要な観光地に関する項目である。「北... |
| 1  | 青森県の観光地     | [https://ja.wikipedia.org/wiki/%E9%9D%92%E6%A3%...](https://ja.wikipedia.org/wiki/%E9%9D%92%E6%A3%...) | 青森県の観光地（あおもりけんのかんこうち）は、青森県内の主要な観光地等に関する項目である。\... |
| 2  | 岩手県の観光地     | [https://ja.wikipedia.org/wiki/%E5%B2%A9%E6%89%...](https://ja.wikipedia.org/wiki/%E5%B2%A9%E6%89%...) | 岩手県の観光地（いわてけんのかんこうち）は、岩手県内の主要な観光地等に関する項目である。\n... |
| 3  | 秋田県の観光地     | [https://ja.wikipedia.org/wiki/%E7%A7%8B%E7%94%...](https://ja.wikipedia.org/wiki/%E7%A7%8B%E7%94%...) | 秋田県の観光地（あきたけんのかんこうち）は、秋田県内の主要な観光地等に関する項目である。\n... |
| 4  | 宮城県の観光地     | [https://ja.wikipedia.org/wiki/%E5%AE%AE%E5%9F%...](https://ja.wikipedia.org/wiki/%E5%AE%AE%E5%9F%...) | 宮城県の観光地（みやぎけんのかんこうち）は、宮城県内の主要な観光地等に関する項目である。\n... |

### 疎ベクトル (Sparse Vector) 取得関数を定義

疎ベクトルを取得する関数を定義します。
先ほど定義した `mecab_tokenizer` 関数をトーカナイザに指定して `TfidfVectorizer` で TF-IDF ベクトルを作成します。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# 日本語トークナイザを指定して TF-IDF ベクトルを学習
vectorizer = TfidfVectorizer(tokenizer=mecab_tokenizer, token_pattern=None)
vectorizer.fit(corpus_ja)

# Sparse Vector (疎ベクトル) 取得関数
def get_sparse_embedding(text):
    """ 入力テキストを TF-IDF 疎ベクトルに変換 """
    tfidf_vector = vectorizer.transform([text])
    values = []
    dims = []
    for i, tfidf_value in enumerate(tfidf_vector.data):
        values.append(float(tfidf_value))
        dims.append(int(tfidf_vector.indices[i]))
    return {"values": values, "dimensions": dims}
```

Vector Search にインプットするフォーマットに合わせて `{"values": [0.1, 0.2], "dimensions": [1, 4]}` といった形式で出力するようにしています。

### 密ベクトル (Dense Vector) 取得関数を定義

密ベクトルの取得には [Vertex AI Text Embeddings API](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings) の [Multilingual モデル](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#language_coverage_for_textembedding-gecko-multilingual_models)を利用します。

```python
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

model = TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual")

# ドキュメント用 Dense Vector (密ベクトル) 取得関数
def get_document_dense_embedding(text):
    """ 入力ドキュメントを密ベクトルに変換 """
    input = TextEmbeddingInput(text=text, task_type="RETRIEVAL_DOCUMENT")
    return model.get_embeddings([input])[0].values

# クエリ用 Dense Vector (密ベクトル) 取得関数
def get_query_dense_embedding(text):
    """ 入力クエリを密ベクトルに変換 """
    input = TextEmbeddingInput(text=text, task_type="RETRIEVAL_QUERY")
    return model.get_embeddings([input])[0].values
```

今回ドキュメント用とクエリ用に関数を分けていますが、その理由は次の通りです。

以下の図で説明している通り、単純にドキュメントから類似のドキュメントを検索する、例えばレコメンデーションのようなタスクにおいては単一の Encoder を利用した Embedding モデルでも有効かもしれませんが、一方で、今回のようにクエリからドキュメントを検索する、いわゆる情報検索のタスクにおいては単純な類似検索では欲しい情報との**関連度**を測ることが難しいため、Text Embeddings API の[タスクタイプ](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types)を利用して、ドキュメント用には `RETRIEVAL_DOCUMENT` を、クエリ用には `RETRIEVAL_QUERY` を指定しています。

![](https://storage.googleapis.com/zenn-user-upload/bebed8fb310a-20241204.png)

タスクタイプについては[こちらのノートブック](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/embeddings/task-type-embedding.ipynb)や[こちらの動画](https://t.co/xioh0kku69)で詳細に解説されていますので、詳しく知りたい方はぜひご参照ください。

### Vector Search 用のインデックスを作成

Vector Search にインプットするインデックスファイルを作成します。

```python
# Vector Search 用のインデックスファイルを作成
items = []
for i in range(len(df)):
    id = i
    title = df.title[i]
    url = df.url[i]
    content = df.content[i]
    dense_embedding = get_document_dense_embedding(content)
    sparse_embedding = get_sparse_embedding(content)
    items.append({"id": id, "title": title, "url": url, "embedding": dense_embedding, "sparse_embedding": sparse_embedding})
items[0]
```

`item[0]` の中身は以下の形式となっています。

```json
{
   "id":0,
   "title":"北海道の観光地",
   "url":"https://ja.wikipedia.org/wiki/%E5%8C%97%E6%B5%B7%E9%81%93%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0",
   "embedding":[
      0.0598582848906517,
      -0.03103564865887165,
      ...
      0.031308043748140335,
      -0.02314840629696846
   ],
   "sparse_embedding":{
      "values":[
         0.0049297180245891905,
         0.0015858856872395704,
         ...
         0.03098300411558551,
         0.0015858856872395704
      ],
      "dimensions":[
         0,
         3,
         ...
         13257,
         13259
      ]
   }
}
```

従来からある密ベクトル用の `embedding` フィールドに加えて、疎ベクトル用に追加された `sparse_embedding` オブジェクトの中に `values` と `dimensions` という 2 つのフィールドが追加されています。

次に items の内容を出力したインデックスファイルを GCS に格納します。

```python
# Project ID & リージョンを設定
PROJECT_ID = ! gcloud config get project
PROJECT_ID = PROJECT_ID[0]
LOCATION = "us-central1"
```
```python
# インデックスファイル格納用の GCS バケットを作成
BUCKET_URI = f"gs://{PROJECT_ID}-vs-hybridsearch-ja"
! gsutil mb -l $LOCATION -p $PROJECT_ID $BUCKET_URI
```
```python
# インデックスファイルを GCS バケットに格納
with open("items.json", "w") as f:
    for item in items:
        f.write(f"{item}\n")
! gsutil cp items.json $BUCKET_URI
```

### Vector Search にインデックスをデプロイ

Vector Search にインデックスをデプロイしてクエリを実行する手順は以下の通りです。

1. Index の作成
2. Index Endpoint の作成
3. Index Endpoint に対して Index をデプロイ (Deployed Index が作成)
4. Deployed Index に対してクエリを実行

![](https://storage.googleapis.com/zenn-user-upload/cc98f4493db7-20241204.png)

まずは Vertex AI の Python SDK をインポートして初期化します。

```python
# Vertex AI を初期化
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=LOCATION)
```

先ほど作成して GCS に保管したインデックスファイルを指定して Vector Search の Index を作成します。

```python
# Index を作成
my_hybrid_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="vs-hybridsearch-ja-index",
    contents_delta_uri=BUCKET_URI,
    dimensions=768,
    approximate_neighbors_count=20,
    shard_size="SHARD_SIZE_SMALL"
)
```

Index Endpoint を作成します。

```python
# Index Endppoint を作成
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name=f"vs-hybridsearch-ja-index-endpoint",
    public_endpoint_enabled=True
)
```

Index Endpoint に Index をデプロイします。
(初回のデプロイが完了するまでには 20 ~ 30 分かかります)

```python
# Index を Index Endpoint にデプロイ
DEPLOYED_HYBRID_INDEX_ID = f"vs_hybridsearch_ja_deployed"
my_index_endpoint.deploy_index(
    index=my_hybrid_index,
    deployed_index_id=DEPLOYED_HYBRID_INDEX_ID,
    min_replica_count=1
)
```

### ハイブリッドクエリを実行

HybridQuery クラスをインポートしてクエリを作成します。
今回クエリには「`文化遺産`」というテキストを入力してみます。

```python
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    HybridQuery,
)

# ハイブリッドクエリを作成
query_text = "文化遺産"
query_dense_emb = get_query_dense_embedding(query_text)
query_sparse_emb = get_sparse_embedding(query_text)
query = HybridQuery(
    dense_embedding=query_dense_emb,
    sparse_embedding_dimensions=query_sparse_emb["dimensions"],
    sparse_embedding_values=query_sparse_emb["values"],
    rrf_ranking_alpha=0.5
)
```

クエリを Deployed Index に対して送信します。

```python
# ハイブリッドクエリを送信
response = my_index_endpoint.find_neighbors(
    deployed_index_id=DEPLOYED_HYBRID_INDEX_ID,
    queries=[query],
    num_neighbors=10,
)

# 結果を表示
for idx, neighbor in enumerate(response[0]):
    title = df.title[int(neighbor.id)]
    url = df.url[int(neighbor.id)]
    dense_dist = neighbor.distance if neighbor.distance else 0.0
    sparse_dist = neighbor.sparse_distance if neighbor.sparse_distance else 0.0
    print(f"{title:<9}: dense_dist: {dense_dist:.3f}, sparse_dist: {sparse_dist:.3f}, url: {url}")
```

検索結果は以下の通りです。

```
奈良県の観光地  : dense_dist: 0.722, sparse_dist: 0.049, url: https://ja.wikipedia.org/wiki/%E5%A5%88%E8%89%AF%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
宮崎県の観光地  : dense_dist: 0.710, sparse_dist: 0.034, url: https://ja.wikipedia.org/wiki/%E5%AE%AE%E5%B4%8E%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
島根県の観光地  : dense_dist: 0.702, sparse_dist: 0.031, url: https://ja.wikipedia.org/wiki/%E5%B3%B6%E6%A0%B9%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
佐賀県の観光地  : dense_dist: 0.698, sparse_dist: 0.031, url: https://ja.wikipedia.org/wiki/%E4%BD%90%E8%B3%80%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
大分県の観光地  : dense_dist: 0.699, sparse_dist: 0.026, url: https://ja.wikipedia.org/wiki/%E5%A4%A7%E5%88%86%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
滋賀県の観光地  : dense_dist: 0.725, sparse_dist: 0.000, url: https://ja.wikipedia.org/wiki/%E6%BB%8B%E8%B3%80%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
岩手県の観光地  : dense_dist: 0.000, sparse_dist: 0.044, url: https://ja.wikipedia.org/wiki/%E5%B2%A9%E6%89%8B%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
秋田県の観光地  : dense_dist: 0.000, sparse_dist: 0.035, url: https://ja.wikipedia.org/wiki/%E7%A7%8B%E7%94%B0%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
熊本県の観光地  : dense_dist: 0.000, sparse_dist: 0.035, url: https://ja.wikipedia.org/wiki/%E7%86%8A%E6%9C%AC%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
京都府の観光地  : dense_dist: 0.703, sparse_dist: 0.000, url: https://ja.wikipedia.org/wiki/%E4%BA%AC%E9%83%BD%E5%BA%9C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
```

密ベクトルの検索順位と疎ベクトルの検索順位がマージされていることが分かります。
こちらは [Reciprocal Rank Fusion (RRF)](https://cloud.google.com/vertex-ai/docs/vector-search/about-hybrid-search#rrf) というアルゴリズムが利用されています。

RRF は複数のランキングリストがある場合に、それぞれの逆順位 (順位を逆数にした値) を合計した値でリランキングする手法です。

以下は簡単な RRF の例です。

| ドキュメント | 密ベクトルの順位 | 密ベクトルの逆順位 | 疎ベクトルの順位 | 疎ベクトルの逆順位 |
|---|---|---|---|---|
| Doc A | 1 | 1.0 | 2 | 0.5 |
| Doc B | 2 | 0.5 | 4 | 0.25 |
| Doc C | 3 | 0.333 | 1 | 1.0 |
| Doc D | 4 | 0.25 | 3 | 0.333 |

| ドキュメント | 密ベクトルの逆順位 | 疎ベクトルの逆順位 | RRF スコア |
|---|---|---|---|
| Doc A | 1.0 | 0.5 | 1.5 |
| Doc C | 0.333 | 1.0 | 1.333 |
| Doc B | 0.5 | 0.25 | 0.75 |
| Doc D | 0.25 | 0.333 | 0.583 |

Vector Search の HybridQuery では以下の通り `rrf_ranking_alpha` というパラメータでマージされる際の重みを調整できます。

- `1` または`指定なし`: 疎ベクトルは無視され密ベクトルのみを使用
- `0`: 密ベクトルは無視され疎ベクトルのみを使用
- `0` ~ `1`: 密ベクトルと疎ベクトルの両方を使用。`0.5` は同じ重みで検索結果をマージ

試しに `rrf_ranking_alpha=1.0` に変更して再度クエリを送信してみました。

```
滋賀県の観光地  : dense_dist: 0.725, sparse_dist: 0.000, url: https://ja.wikipedia.org/wiki/%E6%BB%8B%E8%B3%80%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
奈良県の観光地  : dense_dist: 0.722, sparse_dist: 0.000, url: https://ja.wikipedia.org/wiki/%E5%A5%88%E8%89%AF%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
宮崎県の観光地  : dense_dist: 0.710, sparse_dist: 0.000, url: https://ja.wikipedia.org/wiki/%E5%AE%AE%E5%B4%8E%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
京都府の観光地  : dense_dist: 0.703, sparse_dist: 0.000, url: https://ja.wikipedia.org/wiki/%E4%BA%AC%E9%83%BD%E5%BA%9C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
島根県の観光地  : dense_dist: 0.702, sparse_dist: 0.000, url: https://ja.wikipedia.org/wiki/%E5%B3%B6%E6%A0%B9%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
東京都の観光地  : dense_dist: 0.700, sparse_dist: 0.000, url: https://ja.wikipedia.org/wiki/%E6%9D%B1%E4%BA%AC%E9%83%BD%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
大分県の観光地  : dense_dist: 0.699, sparse_dist: 0.000, url: https://ja.wikipedia.org/wiki/%E5%A4%A7%E5%88%86%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
佐賀県の観光地  : dense_dist: 0.698, sparse_dist: 0.000, url: https://ja.wikipedia.org/wiki/%E4%BD%90%E8%B3%80%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
愛媛県の観光地  : dense_dist: 0.697, sparse_dist: 0.000, url: https://ja.wikipedia.org/wiki/%E6%84%9B%E5%AA%9B%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
高知県の観光地  : dense_dist: 0.697, sparse_dist: 0.000, url: https://ja.wikipedia.org/wiki/%E9%AB%98%E7%9F%A5%E7%9C%8C%E3%81%AE%E8%A6%B3%E5%85%89%E5%9C%B0
```

意図した通りに、疎ベクトルの検索順位は無視されて密ベクトルのスコアのみでランキングされていました。

## まとめ

- [Vertex AI Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview) ではセマンティック検索用の密ベクトル (Dense Vector) に加えてキーワード検索用の疎ベクトル (Sparse Vector) をサポートしており、密ベクトルと疎ベクトルの検索結果を [Reciprocal Rank Fusion (RRF)](https://cloud.google.com/vertex-ai/docs/vector-search/about-hybrid-search#rrf) でマージしたハイブリッド検索が実現できます
- 疎ベクトルとは、ほとんどの要素に `0` が入っているベクトルのことで、本記事では MeCab による日本語トーカナイズ + TF-IDF による疎ベクトル作成の流れを解説しました
- ハイブリッド検索に加えて、[Vertex AI Text Embeddings API](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings) の[タスクタイプ](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types)を併用することで、従来のセマンティック検索では置き換えが難しかった情報検索タスクにも適用しやすくなりました