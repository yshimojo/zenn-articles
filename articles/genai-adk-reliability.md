---
title: "Gen AI SDK & ADK で実装する 429 エラーのリトライ＆フォールバック戦略"
emoji: "🔁"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["gemini", "claude", "vertexai", "adk", "python"]
published: false
---

## はじめに

Gemini API をはじめとする LLM API を本番環境のサービスやシステムに組み込む際に、多くの開発者が直面する問題として、API サービスのレート制限やバックエンドのリソース不足による **429 エラー**があります。この 429 エラーに適切に対処することで、ユーザー体験を損なうことなく、サービスの安定的な稼働と信頼性を維持することができます。

Vertex AI の Gemini API では、各ユーザーが利用可能なキャパシティを動的にコントロールする [Dynamic Shared Quota (DSQ)](https://cloud.google.com/vertex-ai/generative-ai/docs/dynamic-shared-quota) という仕組みを導入しており、現在は従来のような固定の Quota (上限値) は撤廃されました。一方で共有キャパシティ全体で一時的に高い需要が発生すると、一時的なリソース競合状態となり、429 "Resource Exceeded" エラーが発生することがあります。

本記事では、この 429 エラーに対処するため、Gen AI SDK や ADK (Agent Development Kit) を利用した**リトライ**および**フォールバック**の実装戦略をコード実例とともに解説します。

:::message
本記事では Python 版の Gen AI SDK および ADK を使用した実装例を解説します。他の言語の SDK をご利用の方は、エラー対処の考え方としてご参考ください。
:::

## 429 エラーの回避方法

Vertex AI の Gemini API を利用する際に 429 エラーを解決する方法として、[公式ドキュメント](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput/error-code-429)では次の方法が推奨されております。
- [Provisioned Throughput](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput) (PT) の利用
- [Global Endpoint](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#global-endpoint) の利用
- [切り捨て型指数バックオフ](https://cloud.google.com/storage/docs/retry-strategy#exponential-backoff)によるリトライの実装

上記に加えて、大量のデータ (最大 20 万件) に対して非同期で Gemini による推論を行いたいユースケースにおいては、[バッチ予測 API](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini) の活用も有効な選択肢の一つです。

本記事では、これらのオプションの中から、指数バックオフによるリトライに焦点を当てて具体的な実装方法を解説していきます。その後、PT と従量課金制 (オンデマンド) をハイブリッドに併用することでコストと安定性を両立させるフォールバック戦略とその実装方法についても解説していきます。

## リトライ戦略と実装方法

Google Cloud では、API へのリクエストをリトライする際のベストプラクティスとして、ジッターを伴う[切り捨て型指数バックオフ](https://cloud.google.com/iam/docs/retry-strategy#overview)を使用することを推奨しております。これは、以下の 3 つの要素を組み合わせた堅牢なリトライ戦略です。

- **指数バックオフ(Exponential Backoff)**: リトライを繰り返すたびに、待機時間を指数関数的に長くしていく方式です。429 エラーのように一時的な負荷が原因の場合、リトライの集中を避けて API バックエンドの負荷を軽減する効果があります。
- **切り捨て型 (Truncated)**: 待機時間が増え続けないように、上限値 (最大バックオフ時間) を設ける仕組みです。これにより、リトライ間隔が非現実的な長さになることを防ぎます。
- **ジッター (Jitter)**: 各待機時間に短いランダムな遅延を追加する手法です。エラーを検知した複数のクライアントが全く同じタイミングでリトライを再開すると、再び負荷が集中する「[Thundering Herd 問題](https://en.wikipedia.org/wiki/Thundering_herd_problem)」を引き起こす可能性があります。ジッターは、このリトライのタイミングを意図的にずらすことで問題を回避します。

### Gen AI SDK のリトライ

[v1.21.0](https://github.com/googleapis/python-genai/releases/tag/v1.21.0) から導入された `HttpRetryOptions` を利用することで切り捨て型指数バックオフによるリトライ処理を簡単に実装することが可能です。

具体的な実装方法は、Client オブジェクトを初期化する際に、次の通り `retry_options` に [`HttpRetryOptions`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptions) を指定するだけです。

```python
from google import genai
from google.genai import types

client = genai.Client(
   http_options=types.HttpOptions(
       api_version='v1',
       retry_options=types.HttpRetryOptions(
           attempts=10,  # Default: 5
           initial_delay=10,  # Default: 1.0
           max_delay=100,  # Default: 60.0
           exp_base=1.5,  # Default: 2
           jitter=0.5,  # Default: 1
           http_status_codes=[429],  # Default: [408, 429, 500, 502, 503, 504]
       )
   )
)
```

`HttpRetryOptions` を引数なしで指定するとデフォルト値が適用され、引数として各パラメータを明示的に指定すると設定値をオーバーライドすることが可能です。

ちなみに、[v1.28.0](https://github.com/googleapis/python-genai/releases/tag/v1.28.0) からは `generate_content` メソッド実行時に `GenerateContentConfig` の [`http_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.GenerateContentConfigDict.http_options) 内で `HttpRetryOptions` を指定することで、リクエスト送信時にリトライの設定をオーバーライドすることも可能となりました。

```python
response = client.models.generate_content(
    model='gemini-2.5-flash',
    config=types.GenerateContentConfig(
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions()
        ),
    ),
    contents='Why is the sky blue?',
)
```

それでは `HttpRetryOptions` の各パラメータの意味を見ていきましょう。

内部的には Tenacity というライブラリの [`wait_exponential_jitter`](https://tenacity.readthedocs.io/en/latest/api.html#tenacity.wait.wait_exponential_jitter) 関数を利用して指数バックオフによるリトライを実現しています。

$n$ 回目のリトライまでの待機時間 $Delay_n$ を表す計算式と各パラメータの意味は次の通りとなります。

$$ \small Delay_n = min(initial\_delay \times exp\_base^n + random.uniform(0, jitter), max\_delay)) $$

| パラメータ | 説明 |
| :--- | :--- |
| `attempts` | 初回リクエストを含む総試行回数 |
| `n` | リトライ回数 ($0 \le n \le \text{attempts} - 2$) |
| `initial_delay` | 初回リトライまでの待機時間 (秒) |
| `exp_base` | 指数関数の底 |
| `jitter` | ランダムな遅延 (秒)<br>`0` から `jitter` までの範囲で乱数を加算 |
| `max_delay` | 最大待機時間 (秒) |

:::message
`HttpRetryOptions` の引数 `attempts` には**初回リクエストを含む**総試行回数を指定しますので、リトライを行いたい場合は `2` 以上を設定する必要があります。
:::

実際に `HttpRetryOptions` のデフォルト値を適用した $Delay_n$ は次の通りとなります。

$$ Delay_n = min(1.0 \times 2^n + random.uniform(0, 1), 60.0)$$

デフォルト値では `attempts=5` が設定されておりますが、前述の通り、こちらには初回リクエストが含まれるため、実際のリトライ回数は `4 回`となります。

まとめると、デフォルト値における各リトライごとのバックオフ時間とジッターによるランダム値を含めた待機時間は次の通りとなります。($n$ の初期値 = `0`)

| リトライ回数 (n) | バックオフ時間 | ジッター (乱数) | 待機時間 |
| ---- | ---- | ---- | ---- |
| 0 | 1 | 0.0 ~ 1.0 秒 | 1.0 ~ 2.0 秒 |
| 1 | 2 | 0.0 ~ 1.0 秒 | 2.0 ~ 3.0 秒 |
| 2 | 4 | 0.0 ~ 1.0 秒 | 4.0 ~ 5.0 秒 |
| 3 | 8 | 0.0 ~ 1.0 秒 | 8.0 ~ 9.0 秒 |

### ADK (Gemini) のリトライ

ADK においても [v1.9.0](https://github.com/google/adk-python/releases/tag/v1.9.0) のアップデートにて、モデルとして Gemini を利用する場合には Gen AI SDK の `HttpRetryOptions` が指定できるようになりました。こちらをご活用いただくとシンプルに実装が可能です。

具体的には、次の通り `LlmAgent` オブジェクト作成時に [`Gemini`](https://google.github.io/adk-docs/api-reference/python/google-adk.html#google.adk.models.Gemini) を直接指定することで設定が可能です。

```python
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai.types import HttpRetryOptions

root_agent = LlmAgent(
    name="gemini_agent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=HttpRetryOptions(
            attempts=10,  # Default: 5
            initial_delay=10,  # Default: 1.0
            max_delay=100,  # Default: 60.0
            exp_base=1.5,  # Default: 2
            jitter=0.5,  # Default: 1
            http_status_codes=[429],  # Default: [408, 429, 500, 502, 503, 504]
        )
    ),
    instruction="You are an assistant powered by Gemini 2.5 Flash",
)
```

### ADK (LiteLLM) のリトライ

ADK ではLiteLLM という様々な LLM API をラップした OSS のライブラリを指定することができ、こちらを活用すると、Gemini / Vetex AI 以外の LLM API を利用することも可能となります。具体的には、[こちらのドキュメント](https://google.github.io/adk-docs/agents/models/#using-cloud-proprietary-models-via-litellm)に記載の通り `LlmAgent` のモデルとして `LiteLlm` を指定します。

LiteLLM では、様々な LLM API を統一されたインターフェース経由で利用することが可能なことに加えて、ビルトインのリトライやフォールバックの機能も提供しており、具体的には、[`Router`](https://docs.litellm.ai/docs/routing) (`litellm.router`) の機能を利用して、次のようにリトライを設定することができます。

```python
import os

os.environ["VERTEXAI_PROJECT"] = "your-gcp-project-id"
os.environ["VERTEXAI_LOCATION"] = "your-gcp-location"

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from litellm.router import RetryPolicy

retry_policy = RetryPolicy(
    RateLimitErrorRetries=5,
)

root_agent = LlmAgent(
    model=LiteLlm(
        model="vertex_ai/claude-sonnet-4-5",
        retry_policy=retry_policy,
        retry_strategy='exponential_backoff_retry',
    ),
    name="litellm_caude_agent",
    instruction="You are an assistant powered by Claude Sonnet 4.5.",
)
```

:::message
今回モデルには Vertex AI の Claude Sonnet 4.5 API を指定しているため、事前に[こちらの手順](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude#before_you_begin)に沿って API の有効化が必要になります。
:::

上記のコード例では明示的に `retry_strategy` に `exponential_backoff_retry` を指定することにより指数バックオフによるリトライを実現していますが、こちらを指定しない場合、デフォルトでは `constant_retry` が指定されるため注意が必要です。

こちらも Gemini の `HttpRetryOptions` と同様に、内部的には Tenacity ライブラリが利用されていますが、`HttpRetryOptions` との違いとして、ジッターなしの [`wait_exponential`](https://tenacity.readthedocs.io/en/latest/api.html#tenacity.wait.wait_exponential) 関数が利用されております。また、同関数のパラメータには `multiplier=1` (バックオフ時間に乗算する係数), `max=10` (最大待機時間の秒数) があらかじめ設定されており、この値をオーバーライドすることはできないようです。

:::message
Gen AI SDK の `HttpRetryOptions(attempts=5)` が初回リクエスト 1 回 + リトライ 4 回 を意味するのに対し、LiteLLM の `RateLimitErrorRetries=5` はリトライを 5 回行うことを意味します。ライブラリによってパラメータの定義が異なる点にご注意ください。
:::

LiteLLM には、より高度なルーティング処理を実装可能な LiteLLM Proxy という機能が提供されております。こちらは [ADK 経由でも利用可能](https://docs.litellm.ai/docs/tutorials/google_adk#5-using-litellm-proxy-with-adk)ではあるのですが、この方式だと別途 Proxy を実行する必要があり、今回は可搬性を考えて、`LlmAgent` オブジェクトにロジックを集約できる `litellm.router.RetryPolicy` を利用した方式をご紹介いたしました。

## フォールバック戦略と実装方法

フォールバック戦略を考える前に、Provisioned Throughput (PT) の[デフォルト](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput/use-provisioned-throughput#default)の動作について解説します。
- デフォルトでは、購入したスループット量を超えると、超過分は自動的に従量課金制 (オンデマンド) のリクエストとして処理されます。(= **spillover**)
- 明示的に PT のみを使用する場合は、リクエスト送信時に `X-Vertex-AI-LLM-Request-Type` HTTP ヘッダーを `dedicated` に、従量課金制のみを使用する場合は、同 HTTP ヘッダーを `shared` に設定します。

![](https://storage.googleapis.com/zenn-user-upload/10552d67f386-20251014.png)

PT は必要なスループットを事前に予約できるため、高い信頼性と安定稼働が必要な本番環境のアプリケーションに Gemini API を組み込む際に最適な購入オプションです。

一方でサービスの立ち上げフェーズなど、PT の購入量をできるだけ抑えつつ、コストと安定性のバランスを取りたいという場面も多いかと思います。

そのようなニーズに対して考え得るアプローチとして、通常時は従量課金制でリクエストを行いつつ、万が一 429 エラーが返ってきた場合にのみ、PT のリクエストとしてフォールバックするという戦略が考えられます。

![](https://storage.googleapis.com/zenn-user-upload/9733b90ca371-20251014.png)

### Gen AI SDK のフォールバック

Gen AI SDK を利用して前述のフォールバック戦略をシンプルに実装する方法としては、Python の `try...except` により 429 エラーを捕捉して例外処理の中でフォールバック処理を実装するアプローチが考えられます。

例えば、次のコードでは `generate_content` メソッドをラップした内部関数である `_generate_content` を定義して、その中で `"X-Vertex-AI-LLM-Request-Type"` ヘッダーの中身を動的に変更できるようにしています。

```python
from google import genai
from google.genai import types
from google.genai import errors

client = genai.Client(
    http_options=types.HttpOptions(
        api_version='v1',
    )
)

def _generate_content(request_type: str):
    """Generates content with a specific request type."""
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        config=types.GenerateContentConfig(
            system_instruction='You are an assistant powered by Gemini 2.5 Flash.',
            http_options=types.HttpOptions(
                headers={
                    "X-Vertex-AI-LLM-Request-Type": request_type,
                },
            )
        ),
        contents='Why is the sky blue?',
    )
    print(response.text)

try:
    _generate_content(request_type='shared')
except errors.ClientError as e:
    if e.code == 429:
        _generate_content(request_type='dedicated')
    else:
        raise
```

### ADK (LiteLLM) のフォールバック

ADK においては、現在フォールバックを実現する直接的な機能は提供されておりませんが、こちらも LiteLLM を利用することで、次のようにフォールバックを簡単に実装することが可能です。

```python
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

root_agent = LlmAgent(
    model=LiteLlm(
        model="vertex_ai/gemini-2.5-flash",
        extra_headers={
            "X-Vertex-AI-LLM-Request-Type": "shared",
        },
        fallbacks=[
            {
                "model": "vertex_ai/gemini-2.5-flash",
                "extra_headers": {"X-Vertex-AI-LLM-Request-Type": "dedicated"},
            }
        ],
    ),
    name="throttling_fallback_agent",
    instruction="You are an assistant powered by Gemini 2.5 Flash.",
)
```

LiteLLM を利用するその他のメリットとして、フォールバック先のモデル自体を切り替えることも容易に実装できます。

例えば、次のコードはプライマリのモデルに `vertex_ai/claude-sonnet-4-5` を指定して、フォールバック先のモデルに `vertex_ai/gemini-2.5-pro` を指定する実装例となります。

```python
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

root_agent = LlmAgent(
    model=LiteLlm(
        model="vertex_ai/claude-sonnet-4-5",
        fallbacks=["vertex_ai/gemini-2.5-pro"],
        # context_window_fallback_dict={"vertex_ai/claude-sonnet-4-5": "vertex_ai/gemini-2.5-pro"},
    ),
    name="model_fallback_agent",
    instruction="You are an assistant powered by Claude Sonnet 4.5 with Gemini 2.5 Pro as a fallback model.",
)
```

尚、汎用的なエラーに対するフォールバック先を意味する `fallbacks` の代わりに上記コード内でコメントアウトしている `context_window_fallback_dict` オプションを利用することで、入力トークンを超過した場合にのみ、よりコンテキストウィンドウの大きい Gemini 2.5 Pro にフォールバックするような処理を実装することも可能です。こちらのアプローチは現実のユースケースにおいても使い所が多そうです。

## 実際に 429 エラーを発生させてみよう

これまで解説してきたリトライやフォールバックの実装方法が、想定した通りに機能するのか、意図的に 429 エラーを発生させて実際に確認してみましょう。

とは言え、従量課金制で意図的に 429 エラーを発生させるには、同時に相当な量のトークンを処理させる必要がありますので、手元で簡単にテストすることは現実的ではありません。

そこで、あくまで意図的に 429 エラーを発生させる目的の構成として、PT の割り当てがないプロジェクト内で `X-Vertex-AI-LLM-Request-Type` HTTP ヘッダーを `dedicated` に設定してリクエストを送信することで、擬似的なリソース不足の状態を作り出し、429 エラーを発生させます。

![](https://storage.googleapis.com/zenn-user-upload/39839fb661d4-20251014.png)

:::message alert
この構成はあくまでも意図的に 429 エラーを発生させるためのものです。実際の本番環境には適用しないでください。
:::

:::message
各 SDK は記事執筆時点の最新版である次のバージョンを利用してテストしています。
- google-genai 1.44.0
- google-adk 1.16.0
:::

### Gen AI SDK のテストコード

これまで解説してきた `HttpRetryOptions` によるリトライと `try...except` によるフォールバックを実装していきます。

次のテストコードでは、リトライやフォールバックの挙動を実行結果に出力するために `logging.basicConfig()` を利用してログレベルとログフォーマットを設定しています。

```python
import os

os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'
os.environ['GOOGLE_CLOUD_PROJECT'] = 'your-project-id'
os.environ['GOOGLE_CLOUD_LOCATION'] = 'global'

import logging

logging.basicConfig(
    level=logging.INFO,
    # Log format: timestamp - log message
    format='%(asctime)s.%(msecs)03d - %(message)s',
    # Date format: YYYY-MM-DD
    datefmt='%Y-%m-%d %H:%M:%S'
)

from google import genai
from google.genai import types
from google.genai import errors

client = genai.Client(
    http_options=types.HttpOptions(
        api_version='v1',
        retry_options=types.HttpRetryOptions()
    )
)

def _generate_content(request_type: str):
    """Generates content with a specific request type."""
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        config=types.GenerateContentConfig(
            system_instruction='You are an assistant powered by Gemini 2.5 Flash.',
            http_options=types.HttpOptions(
                headers={
                    "X-Vertex-AI-LLM-Request-Type": request_type,
                },
            )
        ),
        contents='Tell me your role briefly.',
    )
    print(response.text)

try:
    _generate_content(request_type='dedicated')
except errors.ClientError as e:
    if e.code == 429:
        _generate_content(request_type='shared')
    else:
        raise
```

実行結果

```
2025-10-15 06:29:51.183 - HTTP Request: POST https://aiplatform.googleapis.com/v1/projects/your-project-id/locations/global/publishers/google/models/gemini-2.5-flash:generateContent "HTTP/1.1 429 Too Many Requests"
2025-10-15 06:29:51.185 - Retrying google.genai._api_client.BaseApiClient._request_once in 1.3357772089194895 seconds as it raised ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Too many requests. Exceeded the provisioned throughput. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.', 'status': 'RESOURCE_EXHAUSTED'}}.
2025-10-15 06:29:52.641 - HTTP Request: POST https://aiplatform.googleapis.com/v1/projects/your-project-id/locations/global/publishers/google/models/gemini-2.5-flash:generateContent "HTTP/1.1 429 Too Many Requests"
2025-10-15 06:29:52.642 - Retrying google.genai._api_client.BaseApiClient._request_once in 2.07533216206925 seconds as it raised ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Too many requests. Exceeded the provisioned throughput. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.', 'status': 'RESOURCE_EXHAUSTED'}}.
2025-10-15 06:29:54.967 - HTTP Request: POST https://aiplatform.googleapis.com/v1/projects/your-project-id/locations/global/publishers/google/models/gemini-2.5-flash:generateContent "HTTP/1.1 429 Too Many Requests"
2025-10-15 06:29:54.968 - Retrying google.genai._api_client.BaseApiClient._request_once in 4.53247254651655 seconds as it raised ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Too many requests. Exceeded the provisioned throughput. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.', 'status': 'RESOURCE_EXHAUSTED'}}.
2025-10-15 06:29:59.579 - HTTP Request: POST https://aiplatform.googleapis.com/v1/projects/your-project-id/locations/global/publishers/google/models/gemini-2.5-flash:generateContent "HTTP/1.1 429 Too Many Requests"
2025-10-15 06:29:59.580 - Retrying google.genai._api_client.BaseApiClient._request_once in 8.42558444295076 seconds as it raised ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Too many requests. Exceeded the provisioned throughput. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.', 'status': 'RESOURCE_EXHAUSTED'}}.
2025-10-15 06:30:08.347 - HTTP Request: POST https://aiplatform.googleapis.com/v1/projects/your-project-id/locations/global/publishers/google/models/gemini-2.5-flash:generateContent "HTTP/1.1 429 Too Many Requests"
2025-10-15 06:30:08.349 - AFC is enabled with max remote calls: 10.
2025-10-15 06:30:09.072 - HTTP Request: POST https://aiplatform.googleapis.com/v1/projects/your-project-id/locations/global/publishers/google/models/gemini-2.5-flash:generateContent "HTTP/1.1 200 OK"
I am a large language model, trained by Google. My purpose is to assist users with information, generate text, and engage in helpful conversations.
```

期待した通り、429 エラーの後、4 回のリトライを試みていることが分かります。5 回目のエラーの後にフォールバックが実行され、最終的には正常に回答が生成されていることが確認できました。

また、ログのタイムスタンプを見ると、指数バックオフによりリトライまでの待機時間が指数関数的に長くなっていることが分かります。

### ADK (LiteLLM) のテストコード

続いて、ADK + LiteLLM を利用したリトライおよびフォールバックの挙動を確認していきます。

今回はテスト用に 2 つのエージェントを作成し、`adk run` コマンドを利用して非対話モードで実行していくため、次のプロジェクト (ディレクトリ) 構成としています。

```
project_folder
├── replay.json
├── .env
└── throttling-fallback-test
    ├── __init__.py
    ├── agent.py
└── model-fallback-test
    ├── __init__.py
    ├── agent.py
```

各 `agent.py` 内でエージェントの定義をしており、今回は非対話モードで実行しますので `replay.json` というファイル内に次の通りクエリを記述しています。

```json:replay.json
{"state": {}, "queries": ["Tell me your role briefly."]}
```

`.env` の中身は次の通りです。

```
# For Gemini
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=global

# For Claude
VERTEXAI_PROJECT=your-project-id
VERTEXAI_LOCATION=global
```

以上で事前の準備が整いましたので、まずはじめに `throttling-fallback-test` エージェントを実行していきます。

こちらは先ほどの Gen AI SDK を利用したテストコードと同様に、意図的に 429 エラーを発生させた上で、リトライおよびフォールバックの挙動を確認するテストコードです。

```python:agent.py
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from litellm.router import RetryPolicy

retry_policy = RetryPolicy(
    RateLimitErrorRetries=5,
)

root_agent = LlmAgent(
    model=LiteLlm(
        model="vertex_ai/gemini-2.5-flash",
        extra_headers={
            "X-Vertex-AI-LLM-Request-Type": "dedicated",
        },
        retry_policy=retry_policy,
        retry_strategy='exponential_backoff_retry',
        fallbacks=[
            {
                "model": "vertex_ai/gemini-2.5-flash",
                "extra_headers": {"X-Vertex-AI-LLM-Request-Type": "shared"},
            }
        ],
    ),
    name="throttling_fallback_agent",
    instruction="You are a resilient assistant powered by Gemini 2.5 Flash.",
)
```

次のコマンドで実行します。

```shell-session
$ adk run throttling-fallback-test --replay replay.json
```

実行結果 (一部抜粋)

```
[user]: Tell me your role briefly.
06:16:37 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= gemini-2.5-flash; provider = vertex_ai
...
06:16:37 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= gemini-2.5-flash; provider = vertex_ai
...
06:16:39 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= gemini-2.5-flash; provider = vertex_ai
...
06:16:41 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= gemini-2.5-flash; provider = vertex_ai
...
06:16:45 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= gemini-2.5-flash; provider = vertex_ai
...
06:16:53 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= gemini-2.5-flash; provider = vertex_ai
...
06:16:53 - LiteLLM:ERROR: fallback_utils.py:62 - Fallback attempt failed for model vertex_ai/gemini-2.5-flash: litellm.RateLimitError: litellm.RateLimitError: Vertex_aiException - {
  "error": {
    "code": 429,
    "message": "Too many requests. Exceeded the provisioned throughput. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.",
    "status": "RESOURCE_EXHAUSTED"
  }
}
 LiteLLM Retried: 5 times
...
httpx.HTTPStatusError: Client error '429 Too Many Requests' for url 'https://aiplatform.googleapis.com/v1/projects/your-project-id/locations/global/publishers/google/models/gemini-2.5-flash:generateContent'
...
litellm.llms.vertex_ai.common_utils.VertexAIError: {
  "error": {
    "code": 429,
    "message": "Too many requests. Exceeded the provisioned throughput. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.",
    "status": "RESOURCE_EXHAUSTED"
  }
}
...
litellm.exceptions.RateLimitError: litellm.RateLimitError: litellm.RateLimitError: Vertex_aiException - {
  "error": {
    "code": 429,
    "message": "Too many requests. Exceeded the provisioned throughput. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.",
    "status": "RESOURCE_EXHAUSTED"
  }
}
 LiteLLM Retried: 5 times
06:16:53 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= gemini-2.5-flash; provider = vertex_ai
[throttling_fallback_agent]: I am an agent named "throttling_fallback_agent," a resilient assistant powered by Gemini 2.5 Flash.
```

期待した通り、初回リクエスト後、5 回のリトライを試みた後に、フォールバックが実行され、最終的には正常に回答が生成されていることが確認できました。

続いて、モデルのフォールバック (Claude Sonnet 4.5 -> Gemini 2.5 Pro) が想定通りに動作するかどうかも確認してみましょう。

```python:agent.py
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from litellm.router import RetryPolicy

retry_policy = RetryPolicy(
    RateLimitErrorRetries=5,
)

root_agent = LlmAgent(
    model=LiteLlm(
        model="vertex_ai/claude-sonnet-4-5",
        headers={
            "X-Vertex-AI-LLM-Request-Type": "dedicated",
        },
        fallbacks=[
            {
                "model": "vertex_ai/gemini-2.5-pro",
                "extra_headers": {"X-Vertex-AI-LLM-Request-Type": "shared"},
            }
        ],
        retry_policy=retry_policy,
        retry_strategy='exponential_backoff_retry',
    ),
    name="model_fallback_agent",
    instruction="You are an assistant powered by Claude Sonnet 4.5 with Gemini 2.5 Pro as a fallback model.",
)
```

次のコマンドで実行します。

```shell-session
$ adk run model-fallback-test --replay replay.json
```

実行結果 (一部抜粋)

```
[user]: Tell me your role briefly.
06:22:21 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= claude-sonnet-4-5; provider = vertex_ai
...
06:22:23 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= claude-sonnet-4-5; provider = vertex_ai
...
06:22:26 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= claude-sonnet-4-5; provider = vertex_ai
...
06:22:30 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= claude-sonnet-4-5; provider = vertex_ai
...
06:22:35 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= claude-sonnet-4-5; provider = vertex_ai
...
06:22:45 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= claude-sonnet-4-5; provider = vertex_ai
...
06:22:47 - LiteLLM:ERROR: fallback_utils.py:62 - Fallback attempt failed for model vertex_ai/claude-sonnet-4-5: litellm.RateLimitError: litellm.RateLimitError: Vertex_aiException - {
  "error": {
    "code": 429,
    "message": "Too many requests. Exceeded the provisioned throughput. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.",
    "status": "RESOURCE_EXHAUSTED"
  }
}
 LiteLLM Retried: 5 times
...
httpx.HTTPStatusError: Client error '429 Too Many Requests' for url 'https://aiplatform.googleapis.com/v1/projects/your-project-id/locations/global/publishers/anthropic/models/claude-sonnet-4-5:rawPredict'
...
litellm.llms.anthropic.common_utils.AnthropicError: {
  "error": {
    "code": 429,
    "message": "Too many requests. Exceeded the provisioned throughput. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.",
    "status": "RESOURCE_EXHAUSTED"
  }
}
...
litellm.exceptions.RateLimitError: litellm.RateLimitError: litellm.RateLimitError: Vertex_aiException - {
  "error": {
    "code": 429,
    "message": "Too many requests. Exceeded the provisioned throughput. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.",
    "status": "RESOURCE_EXHAUSTED"
  }
}
 LiteLLM Retried: 5 times
06:22:47 - LiteLLM:INFO: utils.py:3373 - 
LiteLLM completion() model= gemini-2.5-pro; provider = vertex_ai
[model_fallback_agent]: I am an AI assistant powered by Claude Sonnet 4.5, with Gemini 2.5 Pro as a fallback model. My role is to help you with your requests.
```

こちらも期待した通りにリトライとフォールバックが動作していることを確認できました。

## まとめ

- 本記事では、LLM API の安定稼働のため、リトライとフォールバックの 2 つの戦略と、Gen AI SDK ならびに ADK を利用したそれぞれの実装方法を紹介しました。
- Gemini を利用したシンプルなリトライなら Gen AI SDK ならびに ADK の両方で利用できる `HttpRetryOptions` を指定するだけで、Google Cloud 推奨のジッターを伴う切り捨て型指数バックオフによるリトライ処理を簡単に実装できます。
- ADK を利用する場合、LiteLLM を活用することで複数モデルの利用や高度なフォールバックを簡単に実装できます。
- フォールバック先として Provisioned Throughput (PT) を用意しておくことで、コスト効率と安定性を両立させることが可能です。