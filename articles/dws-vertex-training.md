---
title: "Dynamic Workload Scheduler on Vertex AI Training で NVIDIA H100 を確保する"
emoji: "🕐"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["googlecloud", "vertexai", "機械学習", "gpu", "llm"]
published: true
---

こんにちは、カスタマーエンジニアの下門 (しもじょう) です。

[2024 年 9 月のリリース](https://cloud.google.com/vertex-ai/docs/release-notes#b430cad1)で、従来 GCE や GKE 経由でのみ利用可能であった **Dynamic Workload Scheduler** (以下 DWS) という仕組みが Vertex AI Training でも利用できるようになりました。

そこで本記事では DWS on Vertex AI Training の概要をご紹介したあとに、実際に DWS を活用して NVIDIA H100 GPU を利用した Gemma 2 (27B) のファインチューニングジョブを実行する流れを解説いたします。

:::message
本記事では DWS に焦点を当てますので、学習コードやデータに関する解説、チューニングの方法などについては触れません。
:::

## Dynamic Workload Scheduler (DWS) とは？

DWS は GPU や TPU などの AI アクセラレータを含む HPC 向けに設計されたリソース管理とジョブスケジューリングのための仕組みです。

GCE 経由では [ResizeRequest](https://cloud.google.com/compute/docs/instance-groups/about-resize-requests-mig) として、GKE 経由では [ProvisioningRequest](https://cloud.google.com/kubernetes-engine/docs/how-to/provisioningrequest) としてリクエストを行います。そしてこの度 Vertex AI Custom Training においても [Scheduling](https://cloud.google.com/vertex-ai/docs/training/schedule-jobs-dws) として利用できるようになりました。

DWS にはリソースが確保でき次第、リソースをプロビジョニングする **Flex Start モード**と、指定した開始日時にリソースをプロビジョニングする **Calendar モード**が存在し、現在 Vertex AI Training では Flex Start モードのみがサポートされていますので、以降、本記事では Flex Start モードの前提で解説いたします。

![](https://storage.googleapis.com/zenn-user-upload/deb01f19590a-20250120.png)

*ご参考: [Dynamic Workload Scheduler: リソースへのアクセスと AI / ML ワークロード経済性を最適化](https://cloud.google.com/blog/ja/products/compute/introducing-dynamic-workload-scheduler)*

DWS が向いているユースケースとしては以下などが挙げられます。
- A100 GPU や H100 GPU など需要の大きいハイエンドな GPU リソースが必要なジョブ
- 複数のワーカーノードが同時にプロビジョニングされる必要のある分散トレーニングジョブ
- ファインチューニングジョブ (7 日以内に完了するもの)
- リアルタイム性の必要のないジョブ (オフラインバッチ推論など)

逆に向いていないユースケースとしては以下などが挙げられます。
- 大規模な基盤モデルの事前学習 (Fex Start モードの最大実行時間が [7 日まで](https://cloud.google.com/vertex-ai/docs/training/schedule-jobs-dws#requirements)のため)
- リアルタイム推論 (リソースがプロビジョニングされるまでに待機時間が発生するため)

## DWS on Vertex AI を活用する利点

昨今の世界的な GPU 需要の増加に伴い「自社の機械学習プロジェクトのために GPU を必要な時に即座に活用できるようリソースを確保しておきたい！」といったニーズは依然として非常に高いのではないでしょうか。

Google Cloud 上で GPU リソースのキャパシティを確実に確保するためのソリューションとしては、[Compute Engine の予約](https://cloud.google.com/compute/docs/instances/reservations-overview)が提供されており、[2024 年 12 月のリリース](https://cloud.google.com/vertex-ai/docs/release-notes#af2308f3)で Vertex AI でも利用できるようになりました。しかし、予約は常にコストが発生するため、GPU を確保していても、GPU を使用していないアイドル時間が発生すると、無駄なコストが発生してしまう可能性があります。

一方で Vertex AI Training は本来トレーニングジョブが開始すると必要なワーカーを起動し、終了すると自動的にワーカーを終了しますので、GPU インスタンスの使用状況に合わせて課金され、アイドル時間のコストを削減できるというメリットがあります。しかし、予約をしているわけではないため、必要な時に GPU が取得できない可能性があります。

そこで Vertex AI のコスト効率の高さの恩恵を受けつつ GPU リソースの取得可能性を向上させる仕組みが DWS となります。DWS は予約とオンデマンドの両方の利点を兼ね備えており、この点で Vertex AI と非常に相性が良い機能だと言えます。

### 通常の Vertex AI Training との違い

通常の Vertex AI Custom Training では、[こちらのドキュメント](https://cloud.google.com/vertex-ai/docs/training/understanding-training-service#common-errors)に記載の通り、ジョブ実行時に GPU が使用できない場合、ジョブがエラーとなってしまい、3 回まで再試行されますが、その後はユーザー側でリソースの空き状況を確認しながらジョブを再実行する必要があります。
DWS を利用することでリソースが使用可能になるまで自動的に待機してくれるため、ユーザーの負担を軽減できます。

## DWS を使ってみよう

DWS を利用した Gemma 2 ファインチューニングジョブ用のサンプルノートブックが公開されていますので、今回はそちらを利用していきます。

:::message
今回 H100 GPU を 8 個搭載したアクセラレータ最適化 VM である `a3-highgpu-8g` インスタンス 1 台を起動します。短時間の実行のため高額な金額にはなりませんが、意図しない料金が発生しないように事前に[料金表](https://cloud.google.com/compute/all-pricing#a3-machine-types)をもとにおおよその費用を試算した上で実行ください。
:::

### Quota の引き上げ申請

GPU リソースをリクエストする際には、十分な Quota が事前に割り当てられている必要があり、十分な Quota がない場合は[引き上げ申請](https://cloud.google.com/docs/quotas/view-manage#requesting_higher_quota)が必要となります。

[こちらのドキュメント](https://cloud.google.com/vertex-ai/docs/training/schedule-jobs-dws#quota)に記載の通り、DWS ではオンデマンド Quota ではなく、プリエンプティブル Quota を使用します。例えば、H100 GPU を利用したい場合には、`aiplatform.googleapis.com/custom_model_training_nvidia_h100_gpus` ではなく、`aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h100_gpus` を使用します。

Google Cloud に精通されている方ほど、「プリエンプティブル」と聞くと、通常料金より安価に利用できるプリエンプティブル VM (現 Spot VM) を思い浮かべるかもしれません。しかし、**DWS では、Quota 上はプリエンプティブル Quota を使用しますが、料金は通常のオンデマンド料金が適用されます。** この点は少し分かりづらいので十分にご注意ください。

今回は H100 GPU を 8 個搭載した `a3-highgpu-8g` インスタンスを利用しますので、リージョンは `us-central1` を指定して、割り当て量を `8` に変更して引き上げ申請を行います。

![](https://storage.googleapis.com/zenn-user-upload/7ff4b8bb72b8-20250120.png)

:::message
Quota の引き上げ申請が承認されるまでに数営業日かかることがあります。
:::

### Gemma 2 ファインチューニング用のサンプルノートブックを実行

同僚の Ax さんの[記事](https://zenn.dev/google_cloud_jp/articles/03786c734091b4#vertex-ai-training)でも Cloud Console (GUI) 上で Gemma 2 のファインチューニングが簡単に実行できることを解説されていますが、今回はノートブックを使って実行していきます。

まず Model Garden 上の [Gemma 2 モデルカード](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemma2)に遷移して`ノートブックを開く`を選択します。
どのサンプルノートブックを利用するか選択画面が表示されますので、今回はファインチューニング用の `model_garden_gemma2_finetuning_on_vertex.ipynb` を選択すると、Colab Enterprise 上にノートブックが読み込まれます。

![](https://storage.googleapis.com/zenn-user-upload/685737e0c7c0-20250120.png  =650x)

今回利用するサンプルノートブックは [GitHub 上に公開](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_gemma2_finetuning_on_vertex.ipynb)されていますので、詳細を確認されたい方はそちらもご参照ください。

こちらのノートブックは Colab の [Forms](https://colab.research.google.com/notebooks/forms.ipynb) 機能により、ノートブック内にテキストボックスやドロップダウンなどの GUI が表示され、簡単にパラメータを設定できるようになっています。

今回は直接コードの変更はせずに、Forms の以下項目を設定していきます。

- GCS バケットの指定
- Hugging Face トークンの入力
- ベースモデルの選択
- アクセラレータの選択

はじめにご自身の `BUCKET_URI` と `HF_TOKEN` (Hugging Face トークン) を入力します。

![](https://storage.googleapis.com/zenn-user-upload/deb1314ad8d8-20250120.png =550x)

ベースモデルとして今回は `gemma-2-27b` を選択します。

![](https://storage.googleapis.com/zenn-user-upload/ff6b6943abc4-20250120.png =550x)

アクセラレータとして今回は `NVIDIA_H100_80GB` を選択します。

![](https://storage.googleapis.com/zenn-user-upload/a8729aad2bd9-20250120.png =550x)

設定箇所は以上ですが、DWS に関連する下記のコードについてもざっと見ていきましょう。

![](https://storage.googleapis.com/zenn-user-upload/42608f55956c-20250120.png =650x)
![](https://storage.googleapis.com/zenn-user-upload/bf0d1b1dfe21-20250121.png)

上記のコードスニペットより、`accelerator_type` が `NVIDIA_H100_80GB` の場合は DWS を利用し、また、[CustomContainerTrainingJob.run](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomContainerTrainingJob#google_cloud_aiplatform_CustomContainerTrainingJob_run) 実行時に DWS に関する次のパラメータを指定していることが分かります。
- **max_wait_duration**: 要求したリソースがプロビジョニングされるまでの最大待機時間 (秒単位)。デフォルトは 30 分
- **scheduling_strategy**: DWS (Flex Start モード) を利用する場合には `gca_custom_job_compat.Scheduling.Strategy.FLEX_START` を指定

ここまででジョブを実行する準備が整いましたのでノートブック内の **Finetune** のセルまで順番に実行していきます。(今回は DWS の動作を確認するのが目的のため、後半の **Deploy** および **Predict** のセル実行は割愛します)

### Vertex AI Custom Training jobs を確認

ノートブック上で Finetune ジョブを実行したら、Cloud Console の `Vertex AI` > `トレーニング` > `カスタムジョブ`の画面に遷移してジョブが作成されていることを確認します。
この時点ではステータスが `保留` / `Pending` となっております。

![](https://storage.googleapis.com/zenn-user-upload/74c9cf01f5ff-20250120.png)

しばらく待つと無事にリソースがプロビジョニングされトレーニングが開始されました。

![](https://storage.googleapis.com/zenn-user-upload/1a526c0f171d-20250120.png)

ジョブの詳細画面も見てみましょう。
今回、作成日が `2025/01/17 12:55:34` で開始時間が `2025/01/17 13:11:59` となっているため、ジョブ作成から 16 分 25 秒後に a3-highgpu-8g VM (H100 GPU x8) 1 ノードがプロビジョニングされたことが分かります。

![](https://storage.googleapis.com/zenn-user-upload/f593e6bc7bc1-20250120.png)

:::message
DWS 経由であっても GPU リソースの確保やリソースがプロビジョニングされるまでの待機時間の保証はしてはおりませんので、あくまで目安としてご参考ください。
:::

トレーニングが終了しました。

![](https://storage.googleapis.com/zenn-user-upload/3aa5f38dc804-20250120.png)

### 学習状況をモニタリング

TensorBoard 用のログが GCS に出力されていますので、Cloud Shell 上で以下のようなコマンドを実行して TensorBoard インスタンスを起動し、学習状況を可視化してみます。

```shell
tensorboard --logdir gs://BUCKET_NAME/temporal/gemma2-lora-train-20250117-XXXXXX/logs
```

学習も問題なく収束していました。

![](https://storage.googleapis.com/zenn-user-upload/8579f730ef9b-20250120.png)

最後に GPU 使用率 / GPU メモリ使用率も確認します。

![](https://storage.googleapis.com/zenn-user-upload/7a2b5ba60d95-20250120.png)

今回トレーニング開始後すぐには GPU の使用率が上がらず、実際に GPU リソースをフルに使い出したのは後半の 30 分弱ぐらいでした。前半はデータの送受信や前処理など GPU を利用した演算処理以外で時間を要していた可能性もあり、GPU リソースの有効活用の観点では改善の余地があるかもしれません。

## まとめ

DWS on Vertex AI は Vertex AI の従量課金の恩恵を受けながら需要の高い GPU リソースの取得可能性を向上させるリソース管理の仕組みで、特に次のような方にはおすすめです！
- 自社のビジネス/ユースケースに最適化する目的で OSS の基盤モデルのファインチューニング (LoRA など) を気軽に実行したい方
- これまでリソース確保容易性やコスト効率の問題で旧世代の GPU (T4, V100, etc.) を中心に利用していたが、より Perf/$ の高い新しい世代の GPU (A100, H100, etc.) を活用されたい方

Vertex AI Custom Training をすでにご活用いただいている方はジョブ実行時にパラメータを追加するだけで簡単に利用できますので、ぜひ試してみてください！