---
title: "Vertex AI Vector Search 2.0 登場 ―― ANN 特化から「データストア統合型」の包括的な検索基盤へ"
emoji: "📁"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["googlecloud", "vertexai", "vectorsearch", "情報検索", "RAG"]
published: false
---

[Google Cloud Japan Advent Calendar 2025](https://zenn.dev/google_cloud_jp/articles/ba1f810503bfd2) AI/ML 特集版 23 日目の記事です。

## はじめに

Google Cloud のマネージドなベクトル検索サービスである **Vertex AI Vector Search** が、メジャーバージョンアップとなる **2.0** へと進化しました。

今回のアップデートでは、従来の「**パワフルな近似近傍探索 (ANN) エンジン**」としての強みは維持しつつ、より「**汎用的な検索データストア**」としても利用できるよう進化しています。

本記事では、[Vertex AI Vector Search 2.0](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/overview) の主要な変更点と、それによって開発者体験がどう変わるのかを解説します。

## 特徴 (1): ANN 特化から「データストア統合型」の包括的な検索基盤へ

Vector Search 2.0 最大の進化点は、その立ち位置が「**ベクトル検索特化の ANN インデックス**」から、「**ベクトルとドキュメントを統合管理する包括的な検索データストア**」へとシフトした点にあります。

### ANN 検索基盤としての強みはそのままにアーキテクチャを刷新

従来からの特徴である Google 検索や YouTube を支える [ScaNN](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/) (Scalable Nearest Neighbors) アルゴリズムによる高いパフォーマンスと大規模なスケーラビリティはそのままに、アーキテクチャが大幅に刷新されました。

従来の Vector Search 1.0 でも 2025 年 8 月のアップデートにて [embedding_metadata](https://docs.cloud.google.com/vertex-ai/docs/vector-search/using-metadata) フィールドを利用してベクトル以外の属性情報 (メタデータ) を格納できる機能がパブリックプレビューリリースされましたが、これはあくまでもベクトル検索のコンテキストを補完する付加的な情報という位置付けでした。

これに対して 2.0 では、**ドキュメントデータ (実データ) とベクトルデータの両方を格納すること**を前提に設計されています。これにより、ベクトル検索とデータ取得が一元化され、Vector Search 単体で完結するユースケースが大幅に拡大しました。

この設計思想の変化により、2.0 では「ANN インデックスの作成」が必須ではなくなりました。ANN を利用しない小規模なセマンティック検索やテキスト検索などのユースケースも想定されており、より汎用的なデータストアとしての側面が強化されています。

### 「Query API」と「Search API」：2 つのアプローチ

データの操作には、目的別に明確に分離された 2 種類の API が提供されています。

#### 1. [Query API](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/query-search/query) (データ取得・フィルタリング)

リレーショナルデータベースの `WHERE` 句のようなデータ操作を提供します。ID 指定による取得や条件フィルタリングにより、ピンポイントでデータを操作・取得する際に使用します。

#### 2. [Search API](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/query-search/search) (検索・ランキング)

単一のエンドポイントで以下の多様な検索手法を提供しています。

- **セマンティック検索**: 密ベクトル (Dense Vector) を利用した意味検索。
- **キーワード検索**: 疎ベクトル (Sparse Vector) を利用したトークンベース検索。
- **[NEW] フルテキスト検索**: 疎ベクトルを生成せずに利用可能な全文検索の機能が新たに追加されました。
- **ハイブリッド検索**: これらを組み合わせて、組み込みの [RRF (Reciprocal Rank Fusion)](https://cloud.google.com/vertex-ai/docs/vector-search/about-hybrid-search#rrf) アルゴリズム、または [Vertex AI Ranking API](https://docs.cloud.google.com/generative-ai-app-builder/docs/ranking) を用いて高精度なリランキングを行います。

## 特徴 (2): 開発者ファーストなフルマネージドサービス

2.0 では「フルマネージド」の意味合いが一段階深まりました。インフラストラクチャの複雑さが徹底的に抽象化され、開発者がアプリケーションロジックの実装により集中できる環境が整えられています。

### 専用 SDK の提供

クライアントライブラリの面でも開発者体験が向上しています。従来の 1.0 では汎用的な [Vertex AI SDK](https://docs.cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-sdk) を利用していましたが、2.0 からは直感的に利用できる専用の Vector Search SDK が提供されています。

- [Vector Search Python SDK](https://docs.cloud.google.com/python/docs/reference/google-cloud-vectorsearch)
- [Vector Search Node.js SDK](https://github.com/googleapis/google-cloud-node/tree/main/packages/google-cloud-vectorsearch)
- [Vector Search Java SDK](https://github.com/googleapis/google-cloud-java/tree/main/java-vectorsearch)
- [Vector Search Go SDK](https://github.com/googleapis/google-cloud-go/tree/main/vectorsearch)

### インフラ管理からの解放と自動最適化

2.0 では、インフラ構成やチューニングがサービス側にオフロードされ、より手軽に高性能な検索基盤を利用できるようになりました。

- **サイジングと構成**: 1.0 では事前に[インデックスサイズ](https://docs.cloud.google.com/vertex-ai/docs/vector-search/create-manage-index#index_size)を見積もった上で、適切なマシンタイプやシャードサイズ、可用性を考慮したレプリカ数を決定する必要がありました。2.0 ではこれらのインフラ構成を意識する必要がなくなり、システムがワークロードに応じてリソースを管理します。
- **パフォーマンスチューニング**: 1.0 では ANN の性能をチューニングするために、[TreeAhConfig](https://docs.cloud.google.com/vertex-ai/docs/vector-search/configuring-indexes#tree-ah-config) (ScaNN パラメータ) をデフォルト値以外に調整することができましたが、この方法はアルゴリズムへのある程度の理解が必要でした。2.0 では Automatic Performance Tuning により、これらの設定が自動で最適化されます。

### Auto-Embeddings によるベクトル化の統合

インフラ管理に加え、ベクトル化 (Embedding 生成) もフルマネージドになりました。

- **Auto-Embeddings (自動エンベディング)**: Gemini Embeddings などの[組み込みモデル](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings#google-models)を指定することで、テキストデータ等から自動的にエンベディングフィールドを生成・入力できるようになりました。
- **BYOE (Bring Your Own Embeddings)**: 従来どおり、独自のモデルで生成したベクトルを持ち込むことも引き続きサポートされています。

## 特徴 (3): シンプルで柔軟な料金体系

記事執筆時点では、Vector Search 2.0 はパブリックプレビュー中のため**無料**でご利用いただけます。

詳細な料金体系はまだ公開されていませんが、[公式ドキュメント](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/overview)によると、小規模ワークロード向けの「**従量課金 (Usage-based)**」と、パフォーマンスの調整が可能な「**リソースベース課金 (Resource-based)**」の 2 つの利用形態・課金体系が提供されるようです。

:::details (参考) Vector Search 1.0 の料金体系
現行の 1.0 では、基本的に「インデックスをホストする VM (ノード) のマシンタイプ × レプリカ数」の時間単価と、インデックス作成・更新費用で構成されています。

デフォルトではインデックスを RAM に載せて低レイテンシを実現しますが、2025 年 10 月に GA となった **[Storage-optimized Tier](https://docs.cloud.google.com/vertex-ai/docs/vector-search/storage-optimized-vector-search)** を選択することも可能です。こちらは RAM の代わりに SSD を活用することで、QPS やレイテンシとのトレードオフを許容しつつ、よりコスト効率よく運用できるオプションです。
:::

## データ構造 (Data Structure)

Vector Search 2.0 では、[Collection](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/collections/collections)、[Schema](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/collections/collections#collection_schema)、[Data Object](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/data-objects/data-objects)、[Index](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/indexes/indexes) といった新たな概念が導入されましたので、まずはこれらの主要コンポーネントを理解する必要があります。

1.0 とのデータ構造の違いについては、公式ドキュメント「[Migrate from Vector Search 1.0](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/migration-from-vs-1_0)」もご参照ください。

![](https://storage.googleapis.com/zenn-user-upload/4364bbc45d9c-20251221.png)

### Collection (コレクション)

関連する JSON データ オブジェクトを格納する最上位のコンテナ。コレクション作成時にスキーマを指定する必要があります。Elasticsearch におけるインデックス (Index)、Solr におけるコレクション (Collection) に相当します。

### Collection Schema (コレクション スキーマ)

コレクション内のデータ オブジェクトの構造や制約を定義するもの。ドキュメントデータ (実データ) に対するユーザー定義の **Data Schema** とコレクション内のベクトル フィールドを定義する **Vector Schema** から構成されます。Elasticsearch におけるマッピング (Mappings)、Solr におけるスキーマ (Schema) に相当します。

### Data Object (データ オブジェクト)

コレクション内に保存される個々の JSON オブジェクト。Elasticsearch および Solr におけるドキュメント (Document) に相当します。

### Collection Index (コレクション インデックス)

データ オブジェクト内の各ベクトル フィールドに対して ANN インデックスを作成することが可能です。インデックスがない場合は kNN を利用したブルートフォース (総当たり) 検索となります。

:::message
Elasticsearch や Solr との対比は、あくまでも直感的な理解を助けるための概念的な対応関係を示したものです。厳密な機能の等価性や、アーキテクチャの完全な一致を意味するものではない点、あらかじめご理解ください。
:::

## 実装の流れ

ここからは、実際のコードベースで実装の流れを追ってみましょう。 公式チュートリアルとして、架空の E コマースデータセット [theLook eCommerce](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce) を利用した[サンプルノートブック](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/embeddings/vector-search-2-intro.ipynb)が公開されています。

以下では、このノートブックからコード (Python SDK) の一部を抜粋しながら実装手順を解説します。

### 1. SDK クライアントの定義

まず、必要なクライアントを初期化します。2.0 からは目的別にクライアントが分かれています。

```python
from google.cloud import vectorsearch_v1beta

vector_search_service_client = vectorsearch_v1beta.VectorSearchServiceClient()
data_object_service_client = vectorsearch_v1beta.DataObjectServiceClient()
data_object_search_service_client = vectorsearch_v1beta.DataObjectSearchServiceClient()
```

各クライアントの役割は以下の通りです。

1. [`VectorSearchServiceClient`](https://docs.cloud.google.com/python/docs/reference/google-cloud-vectorsearch/latest/google.cloud.vectorsearch_v1beta.services.vector_search_service.VectorSearchServiceClient): コレクションやインデックスの管理 (CRUD 操作)
2. [`DataObjectServiceClient`](https://docs.cloud.google.com/python/docs/reference/google-cloud-vectorsearch/latest/google.cloud.vectorsearch_v1beta.services.data_object_service.DataObjectServiceClient): データ オブジェクトの管理 (作成、更新、削除)
3. [`DataObjectSearchServiceClient`](https://docs.cloud.google.com/python/docs/reference/google-cloud-vectorsearch/latest/google.cloud.vectorsearch_v1beta.services.data_object_search_service.DataObjectSearchServiceClient): 検索およびクエリ操作の実行

### 2. コレクションの作成

スキーマ (`data_schema` / `vector_schema`) を定義して、データの器となるコレクションを作成します。

```python
# Create the product Collection with schemas that match our dataset

request = vectorsearch_v1beta.CreateCollectionRequest(
    parent=f"projects/{PROJECT_ID}/locations/{LOCATION}",
    collection_id=collection_id,
    collection={
        # Data Schema: Product data (id, name, category, retail_price)
        "data_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},           # Product ID
                "name": {"type": "string"},         # Product name
                "category": {"type": "string"},     # Product category (Dresses, Jeans, etc.)
                "retail_price": {"type": "number"}, # Product price in USD
            },
        },
        # Vector Schema: Product name-based embeddings for semantic and keyword search
        "vector_schema": {
            # Dense embedding: Captures semantic meaning of product names
            # Auto-generated by Vertex AI using gemini-embedding-001 model
            "name_dense_embedding": {
                "dense_vector": {
                    "dimensions": 768,  # Using 768 dimensions for gemini-embedding-001
                    "vertex_embedding_config": {
                        # Auto-generate dense embeddings from product name
                        "model_id": "gemini-embedding-001",
                        "text_template": "{name}",
                        "task_type": "RETRIEVAL_DOCUMENT",
                    },
                },
            },
        },
    }
)

operation = vector_search_service_client.create_collection(request=request)
```

ここで重要なのが [`vertex_embedding_config`](https://docs.cloud.google.com/python/docs/reference/google-cloud-vectorsearch/latest/google.cloud.vectorsearch_v1beta.types.VertexEmbeddingConfig) です。こちらを指定することで Auto-Embeddings が有効になります。
- `text_template`: エンベディングの元となるテキストを指定します。ここではシンプルに `name` フィールドのみを指定していますが、例えば、`Movie Title: {title} ---- Movie Plot: {plot}` のように複数フィールドをテンプレートとして結合し、追加のコンテキストを付与することも可能です。
- [`task_type`](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types): モデルがどのようなタスク (検索ドキュメント、質問応答など) に使われるかを指定します。

### 3. データ オブジェクトの作成

コレクションに対し、データを投入します。

```python
# Add the first product as a demonstration

request = vectorsearch_v1beta.CreateDataObjectRequest(
    parent=f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{collection_id}",
    data_object_id=products[0]["id"],
    data_object={
        "data": products[0]["data"],  # Data: id, name, category, retail_price
        "vectors": {},  # Empty vectors - dense embedding will be auto-generated!
    },
)
result = data_object_service_client.create_data_object(request=request)
```

上記は単一オブジェクトの作成例です。Auto-Embeddings が有効なため、`vectors` フィールドは空のままで構いません (自動生成されます)。

なお、サンプルコードでは [`BatchCreateDataObjectsRequest`](https://docs.cloud.google.com/python/docs/reference/google-cloud-vectorsearch/latest/google.cloud.vectorsearch_v1beta.types.BatchCreateDataObjectsRequest) (最大 1,000 件/リクエスト) を利用して一括登録を行っていますが、背後で呼び出される `gemini-embedding-001` モデルのレート制限等を考慮し、バッチサイズを `250` に設定して処理しています。

:::message alert
**大量データの場合の推奨事項**: 数十万件以上の非常に大規模なデータセットを扱う場合は、API によるバッチリクエスト (`BatchCreateDataObjectsRequest`) を繰り返す方法ではなく、Cloud Storage からの一括インポート ([`ImportDataObjectsRequest`](https://docs.cloud.google.com/python/docs/reference/google-cloud-vectorsearch/latest/google.cloud.vectorsearch_v1beta.types.ImportDataObjectsRequest)) を利用することが推奨されています。
:::

### 4. クエリ操作 (Query API)

[`QueryDataObjectsRequest`](https://docs.cloud.google.com/python/docs/reference/google-cloud-vectorsearch/latest/google.cloud.vectorsearch_v1beta.types.QueryDataObjectsRequest) を使用すると、SQL の `WHERE` 句のようなフィルタリングが可能です。

```python
# Example 3: Category browsing with price exclusion
# Useful for: "Show me Dresses or premium Clothing Sets (over $150)"
nested_conditionals_request = vectorsearch_v1beta.QueryDataObjectsRequest(
    parent=f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{collection_id}",
    filter={
        "$or": [
            {"category": {"$eq": "Dresses"}},
            {
                "$and": [
                    {"category": {"$eq": "Clothing Sets"}},
                    {"retail_price": {"$gte": 150}},
                ]
            },
        ]
    },
    output_fields=vectorsearch_v1beta.OutputFields(data_fields=["*"]),
)
nested_conditionals = data_object_search_service_client.query_data_objects(
    nested_conditionals_request
```

この例では、「カテゴリが `Dresses`」または「`150` ドル以上かつカテゴリが `Clothing Sets`」という条件でデータを抽出しています。

### 5. セマンティック検索 (Search API)

[`SearchDataObjectsRequest`](https://docs.cloud.google.com/python/docs/reference/google-cloud-vectorsearch/latest/google.cloud.vectorsearch_v1beta.types.SearchDataObjectsRequest) を利用して意味検索を行います。

```python
query_text = "Men's outfit for beach"

# Semantic search automatically generates embeddings from the query text
semantic_search_request = vectorsearch_v1beta.SearchDataObjectsRequest(
    parent=f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{collection_id}",
    semantic_search=vectorsearch_v1beta.SemanticSearch(
        search_text=query_text,
        search_field="name_dense_embedding",  # The vector field to search
        task_type="QUESTION_ANSWERING",
        top_k=10,
        output_fields=vectorsearch_v1beta.OutputFields(data_fields=["name", "category", "retail_price"]),
    ),
)

results = data_object_search_service_client.search_data_objects(semantic_search_request)
```

ユーザーは自然言語 (`search_text`) を渡すだけです。クエリテキストについても、ここで指定した [`task_type`](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types) に基づいて自動的に適切なベクトル化が行われます。

### 6. テキスト検索 (Search API)

テキスト検索も、同じ [`SearchDataObjectsRequest`](https://docs.cloud.google.com/python/docs/reference/google-cloud-vectorsearch/latest/google.cloud.vectorsearch_v1beta.types.SearchDataObjectsRequest) で実行できます。

```python
query_text = "Short"

text_search_request = vectorsearch_v1beta.SearchDataObjectsRequest(
    parent=f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{collection_id}",
    text_search=vectorsearch_v1beta.TextSearch(
        search_text=query_text,
        data_field_names=["name"],  # Search in product name field
        top_k=10,
        output_fields=vectorsearch_v1beta.OutputFields(data_fields=["name", "category", "retail_price"]),
    ),
)
results = data_object_search_service_client.search_data_objects(text_search_request)
```

この機能は、疎ベクトルを生成せずに実行されるフルテキスト検索です。

デフォルトの挙動では、検索クエリに複数の単語を入力した場合 (例: "Blue Jeans")、それらすべてを含むドキュメントを探す**暗黙的な AND 検索**が行われます。

なお、[公式ドキュメント](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/query-search/search#text_search)によると、より高度な検索を行いたい場合は `enhanced_query` オプションを `true` に設定します。これにより、ステミング (語形変化の統一) やストップワードの削除、および以下の検索演算子が利用できるようになります。

:::message
記事執筆時点では `enhanced_query` オプションはまだ有効化できないようでした。今後のアップデートに期待しましょう。
:::

### 7. ハイブリッド検索 (RRF)

セマンティック検索とテキスト検索を組み合わせて、[RRF (Reciprocal Rank Fusion)](https://cloud.google.com/vertex-ai/docs/vector-search/about-hybrid-search#rrf) でリランキングする例です。

```python
# Hybrid search: combine semantic and text searches with built-in RRF
query_text = "Men's short for beach"

batch_search_request = vectorsearch_v1beta.BatchSearchDataObjectsRequest(
    parent=f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{collection_id}",
    searches=[
        vectorsearch_v1beta.Search(
            semantic_search=vectorsearch_v1beta.SemanticSearch(
                search_text=query_text,
                search_field="name_dense_embedding",
                task_type="QUESTION_ANSWERING",
                top_k=20,
                output_fields=vectorsearch_v1beta.OutputFields(data_fields=["id", "name", "category", "retail_price"]),
            )
        ),
        vectorsearch_v1beta.Search(
            text_search=vectorsearch_v1beta.TextSearch(
                search_text=query_text,
                data_field_names=["name"],
                top_k=20,
                output_fields=vectorsearch_v1beta.OutputFields(data_fields=["id", "name", "category", "retail_price"]),
            )
        ),
    ],
    combine=vectorsearch_v1beta.BatchSearchDataObjectsRequest.CombineResultsOptions(
        ranker=vectorsearch_v1beta.Ranker(
            rrf=vectorsearch_v1beta.ReciprocalRankFusion(weights=[1.0, 1.0])
        )
    ),
)

batch_results = data_object_search_service_client.batch_search_data_objects(batch_search_request)
```

[`BatchSearchDataObjectsRequest`](https://docs.cloud.google.com/python/docs/reference/google-cloud-vectorsearch/latest/google.cloud.vectorsearch_v1beta.types.BatchSearchDataObjectsRequest) 内で複数の検索定義 (`searches`) を渡し、[`CombineResultsOptions`](https://docs.cloud.google.com/python/docs/reference/google-cloud-vectorsearch/latest/google.cloud.vectorsearch_v1beta.types.BatchSearchDataObjectsRequest.CombineResultsOptions) でそれらを統合しています。

また、RRF によるリランキングでは、セマンティック検索とテキスト検索の結果を `1:1` の重み付けでマージしています。

:::details (参考) Vector Search 1.0 のハイブリッド検索
現行の 1.0 で密ベクトル (Dense Vector) と疎ベクトル (Sparse Vector) を利用したハイブリッド検索を実現する方法については、以前の記事「[Vertex AI Vector Search のハイブリッド検索を日本語で試してみた](https://zenn.dev/google_cloud_jp/articles/vs-hybridsearch-japanese)」もあわせてご参考ください。
:::

### 8. ANN インデックス作成

ここまでのセマンティック検索は、裏側では kNN によるブルートフォース検索で実行されていました。これは正確ですが、データ量やベクトルの次元数が増えると現実的な遅延の範囲内でレスポンスを返すのが難しくなってしまいます。そこで、大規模データに対して高速に検索を行うため、ANN インデックスを作成します。

このインデックスは、データオブジェクト全体に対してではなく、特定のベクトルフィールドごとに作成します。

```python
## Creating an ANN Index for Dense Embeddings
request = vectorsearch_v1beta.CreateIndexRequest(
    parent=f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/{collection_id}",
    index_id="name-dense-index",  # Use hyphens instead of underscores
    index={
        "index_field": "name_dense_embedding",  # Index the product name dense embeddings
        "filter_fields": ["category", "retail_price"],  # Enable filtering by category and price
        "store_fields": ["name"],  # Store product name for quick retrieval
    },
)
dense_index_lro = vector_search_service_client.create_index(request)
```

この操作により、高速な ANN 検索が可能になります。ここで指定している `filter_fields` と `store_fields` の使い分けは、パフォーマンスとコストの観点で重要です。

- `index_field`: インデックスを作成する対象のベクトルフィールドを指定します。
- `filter_fields`: 検索時のフィルタリング (絞り込み) 条件として利用したいフィールドを指定します。
- `store_fields`: フィルタリングには使用しませんが、検索結果 (ペイロード) として取得したいフィールドを指定します。

なお、サンプルでは 10,000 件程度のデータセットに対し、完了まで 30 分ほど要すると記載されていました。そのためインデックス作成ジョブは非同期 (LRO = Long Running Operation) で行われ、より大規模なデータセットをインデックスする場合は、これよりも長い待ち時間が発生することが予想されます。

## まとめ

[Vertex AI Vector Search 2.0](https://docs.cloud.google.com/vertex-ai/docs/vector-search-2/overview) は、従来の「高速な ANN エンジン」という枠を超え、実データとベクトルを統合管理する「包括的な検索プラットフォーム」へと進化しました。

これにより、従来から得意としてきた大規模な情報検索やレコメンデーションシステムに加え、RAG や多様な検索アプリケーションの基盤としても、より手軽かつ強力に活用できるようになりました。

また、インフラ管理やチューニングが自動化されたことで、開発者は「インフラの調整」ではなく「ユーザー体験の向上」により集中できます。

現在はパブリックプレビュー期間中で、無料 (記事執筆時点) でお試しいただけますので、ぜひこの機会に、Google の検索技術が詰まった新しい Vector Search に触れてみてください！