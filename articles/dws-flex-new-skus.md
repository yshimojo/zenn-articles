---
title: "ハイエンド GPU が最大約半額に！DWS の新料金プランで GPU コストを大幅削減"
emoji: "💰"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["googlecloud", "gpu", "vertexai", "gce", "機械学習"]
published: false
---

## TL;DR

DWS Flex-Start の新料金プラン (新 SKUs) により、Google Cloud のハイエンド GPU を搭載したアクセラレータ最適化 VM が、オンデマンドの**最大約半額**で利用可能になりました。
AI 開発者にとっては、これまで高価で手が出しにくかった、パワフルなハイエンド GPU を利用した学習や推論をより気軽に試せるようになりました。

本記事では、[Vertex AI Model Garden 経由で利用可能な Gemma 3](https://cloud.google.com/vertex-ai/generative-ai/docs/open-models/use-gemma#use-in-vertex-ai) を利用して、実際に Vertex AI の Custom Training と Online Prediction をそれぞれ実行し、オンデマンドと DWS Flex-Start の料金を比較します。

## Dynamic Workload Scheduler (DWS) とは？

Dynamic Workload Scheduler (a.k.a. DWS) は、Google Cloud 上で需要の高い GPU や TPU を確保しやすくするためのリソース管理・ジョブスケジューリングの仕組みです。
これに加えて、この度リリースされた新料金体系により、ハイエンドな GPU リソース (A100, H100, H200 など) をコストを抑えながら利用できるようになりました。

DWS の詳細については、以前の記事「[Dynamic Workload Scheduler on Vertex AI Training で NVIDIA H100 を確保する](https://zenn.dev/google_cloud_jp/articles/dws-vertex-training)」もあわせてご参照ください。

## Vertex AI Custom Training 料金比較

まずはじめに Vertex AI Training を動かして料金を見ていきましょう。

今回は `us-central1` リージョンで、NVIDIA H100 GPU を 8 個搭載したマシンタイプ `a3-highgpu-8g` を利用して Custom Training ジョブを実行します。

**[a3-highgpu-8g スペック](https://cloud.google.com/compute/docs/gpus#h100-gpus)**

| マシンタイプ | GPU 数 | GPU メモリ (GB) | vCPU 数 | VM メモリ (GB) | ローカル SSD (GiB) |
|---|---|---|---|---|---|
| a3-highgpu-8g| 8 | 640 | 208 | 1,872 | 6,000 |

事前に十分な Quota が割り当てられているか確認し、必要であれば Quota の[引き上げ申請](https://cloud.google.com/docs/quotas/view-manage#requesting_higher_quota)を行います。[こちらのドキュメント](https://cloud.google.com/vertex-ai/docs/training/schedule-jobs-dws#quota)に記載の通り、DWS ではオンデマンド Quota ではなく、プリエンプティブル Quota を使用します。

**対象の割り当て (Quota)**

- オンデマンド:
  `aiplatform.googleapis.com/custom_model_training_nvidia_h100_gpus`
- DWS Flex-Start:
  `aiplatform.googleapis.com/custom_model_training_preemptible_nvidia_h100_gpus`

なお、[こちらのブログ](https://cloud.google.com/blog/ja/products/compute/announcing-smaller-machine-types-for-a3-high-vms)に記載の通り、`a3-highgpu-8g` より小さなマシンタイプはオンデマンドではサポートされていないため、今回は上記 Quota の割り当て量がそれぞれ `8` 必要となります。

### Gemma 3 (1B) のファインチューニングジョブを実行

Model Garden 上の [Gemma 3 モデルカード](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemma3)から`ノートブックを開く`を選択し、`model_garden_axolotl_gemma3_finetuning.ipynb` という名前のサンプルノートブックを Colab Enterprise 上で開きます。

![model_garden_gemma3_notebook](https://storage.googleapis.com/zenn-user-upload/5ac9b22d7a13-20250815.png)

ノートブックの詳細は [GitHub](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/model_garden/model_garden_axolotl_gemma3_finetuning.ipynb) 上でもご確認いただけます。

今回は GPU を利用してジョブを実行すること自体が目的のため、チューニング内容の解説は割愛させていただき、以下のセルのみを順次実行していきます。

1. **Import utility packages for fine-tuning**
    - デフォルト値のままで実行
    ※ エラー発生するが続行して問題なし
2. **Setup Google Cloud project**
    - 自身の GCS バケットを設定
    - リージョンは `us-central1` を指定
3. **Set model to fine-tune**
    - `google/gemma-3-1bit-it `を指定
4. **Set Axolotl config**
    - デフォルト値のままで実行
5. **Setup HF token**
    - 自身の Hugging Face Token を入力
6. **Setup Axolotl Flags**
    - デフォルト値のままで実行
7. **Vertex AI fine-tuning job**
    - `NVIDIA_H100_80GB` を選択して実行

今回のノートブックは、デフォルトで DWS Flex-Start を利用する設定になっていますので、DWS を利用して実行する場合はデフォルトのままで特に変更は必要なく、オンデマンドで実行する際には、上記 7 のセル内のコードを `is_dynamic_workload_scheduler = False` に変更し、`train_job.run` 実行時に指定している DWS に関するパラメータ `dws_kwargs` をコメントアウトします。

```python
is_dynamic_workload_scheduler = False
```

```python
train_job.run(
    args=train_job_args,
    replica_count=replica_count,
    machine_type=training_machine_type,
    accelerator_type=training_accelerator_type,
    accelerator_count=per_node_accelerator_count,
    boot_disk_size_gb=boot_disk_size_gb,
    service_account=SERVICE_ACCOUNT,
    base_output_dir=TRAINING_JOB_OUTPUT_DIR,
    sync=False,  # Non-blocking call to run.
    # **dws_kwargs,
)
```

### オンデマンド

ノートブック上のセルを実行したら、Cloud Console の `Vertex AI` > `トレーニング` > `カスタムジョブ`の画面に遷移してジョブが作成されていることを確認します。

![vai_training_od_customjob_training](https://storage.googleapis.com/zenn-user-upload/07140a68e8e3-20250815.png)

しばらく待つと無事にジョブが完了していました。

![vai_training_od_customjob_finished](https://storage.googleapis.com/zenn-user-upload/d2f6a45217bb-20250815.png)

ジョブの詳細は次のようになっており、GPU もきちんと使えていそうでした。

![vai_training_od_customjob_details](https://storage.googleapis.com/zenn-user-upload/d70609a52d86-20250816.png)

続いて、Cloud Billing の[レポート](https://cloud.google.com/billing/docs/how-to/reports)上で料金を確認してみましょう。
※ 最新の請求情報が反映されるまでに通常 1 日程度かかります。[^1]

![vai_training_od_billing_report](https://storage.googleapis.com/zenn-user-upload/a1e7124e9277-20250816.png)

上記のレポートには今回実際に掛かった費用が含まれていますが、費用はジョブの実行時間によって変動するため、今回は実際に発生した費用を直接見比べるのではなく、対象となる課金項目 (SKU) を特定し、それらを元に 1 時間あたりの費用を再計算したもので料金比較をしていきたいと思います。

具体的には、オンデマンドの料金を SKU 単位で 1 時間あたりに換算して再計算します。[^2]

| SKU | サービス | SKU ID | 単価 | 単位 | 数量 | 小計 |
|---|---|---|---|---|---|---|
| Vertex AI: Training/Pipelines on NVIDIA H100 80GB in Iowa | Vertex AI | [3F95-001E-B747](https://cloud.google.com/skus?currency=USD&filter=3F95-001E-B747) | $11.2660332 | per 1 hour | 8 | $90.1282656 |
| Vertex AI: Training/Pipelines on A3 Instance Core in Iowa | Vertex AI | [2E83-99F1-391A](https://cloud.google.com/skus?currency=USD&filter=2E83-99F1-391A) | $0.0293227 | per 1 hour | 208 | $6.0991216 |
| Vertex AI: Training/Pipelines on A3 Instance RAM in Iowa | Vertex AI | [B045-9255-8314](https://cloud.google.com/skus?currency=USD&filter=B045-9255-8314) | $0.0025534 | per 1 gigabyte hour | 1,872 | $4.7799648 |
| Vertex AI: Training/Pipelines on SSD backed PD Capacity | Vertex AI | [A005-98FE-36CC](https://cloud.google.com/skus?currency=USD&filter=A005-98FE-36CC) | $0.0002678 [^3] | per 1 gigabyte hour | 2,000 [^4] | $0.53561644 |

合計: **$101.5429684** per hour

### DWS Flex-Start

DWS Flex-Start についても、ノートブックを実行し、`カスタムジョブ`の画面からジョブが作成されていることを確認します。

![vai_training_dws_customjob_training](https://storage.googleapis.com/zenn-user-upload/0d5d9de2a519-20250815.png)

こちらもしばらく待つとジョブが完了していました。

![vai_training_dws_customjob_finished](https://storage.googleapis.com/zenn-user-upload/dffc0ba7c66e-20250816.png)

ジョブの詳細を確認すると、今回は `VM プロビジョニング モデル`が `Flex Start` になっておりました。

![vai_training_dws_customjob_details](https://storage.googleapis.com/zenn-user-upload/b916d7f5b08f-20250816.png)

続いて請求レポートを見ていきましょう。

![vai_training_dws_billing_report](https://storage.googleapis.com/zenn-user-upload/84bf510dbb22-20250816.png)

オンデマンドと異なり、DWS 利用時には Compute Engine と Vertex AI に分かれて費用が計上されていました。

少々複雑で分かりづらいのですが、DWS 利用時には GPU を含む VM に係る費用は Compute Engine 側の SKU [^5] が適用され、Vertex AI 側は Management Fee (管理手数料) の SKU が適用される、ハイブリッドな課金体系となっております。[^6]

DWS Flex-Start の料金も SKU 単位で 1 時間あたりに換算して再計算します。

| SKU | サービス | SKU ID | 単価 | 単位 | 数量 | 小計 |
|---|---|---|---|---|---|---|
| Nvidia H100 80GB GPU attached to DWS Defined Duration VMs running in Americas | Compute Engine | [341A-49A5-0C07](https://cloud.google.com/skus?currency=USD&filter=341A-49A5-0C07) | $4.200761 | per 1 hour | 8 | $33.606088 |
| DWS Defined Duration A3 Core running in Americas | Compute Engine | [9A32-36B2-7FBC](https://cloud.google.com/skus?currency=USD&filter=9A32-36B2-7FBC) | $0.010934 | per 1 hour | 208 | $2.274272 |
| DWS Defined Duration A3 Ram running in Americas | Compute Engine | [ED19-A584-4A84](https://cloud.google.com/skus?currency=USD&filter=ED19-A584-4A84) | $0.000952 | per 1 gigabyte hour | 1,872 | $1.782144 |
| DWS Defined Duration SSD backed Local Storage running in Americas | Compute Engine | [2303-1D6A-9C08](https://cloud.google.com/skus?currency=USD&filter=2303-1D6A-9C08) | $0.0001096 [^3] | per 1 gibibyte hour | 6,000 | $0.65753425 |
| Vertex AI: Training/Pipelines management fee on NVIDIA H100 80GB in Iowa | Vertex AI | [5853-505F-A5C4](https://cloud.google.com/skus?currency=USD&filter=5853-505F-A5C4) | $1.4694826 | per 1 hour | 8 | $11.7558608|
| Vertex AI: Training/Pipelines management fee on A3 Instance Core in Iowa | Vertex AI | [9AC1-7D9B-C47E](https://cloud.google.com/skus?currency=USD&filter=9AC1-7D9B-C47E) | $0.0038247 | per 1 hour | 208 | $0.7955376 |
| Vertex AI: Training/Pipelines management fee on A3 Instance RAM in Iowa | Vertex AI | [D30D-69C9-F057](https://cloud.google.com/skus?currency=USD&filter=D30D-69C9-F057) | $0.0003331 | per 1 gigabyte hour | 1,872 | $0.6235632 |
| Vertex AI: Training/Pipelines on SSD backed PD Capacity | Vertex AI | [A005-98FE-36CC](https://cloud.google.com/skus?currency=USD&filter=A005-98FE-36CC) | $0.0002678 [^3] | per 1 gigabyte hour | 2,000 [^4] | $0.53561644 |

合計: **$52.0306163** per hour

オンデマンドと比べると、DWS Flex-Start 利用時には **48.76%** も安く利用できていました。

## Vertex AI Online Prediction 料金比較

続いて Vertex AI Online Prediction を動かして料金を比較していきます。

[2025 年 7 月のリリース](https://cloud.google.com/vertex-ai/docs/release-notes#July_11_2025)にて Online Prediction でも DWS Flex-Start が利用できるようになりましたが、こちらは A3 High VM のより小さなマシンタイプもサポートされていますので、今回は `us-central1` リージョンで、NVIDIA H100 GPU を 1 個搭載したマシンタイプ `a3-highgpu-1g` を利用してエンドポイントにモデルをデプロイします。

**[a3-highgpu-1g スペック](https://cloud.google.com/compute/docs/gpus#h100-gpus)**

| マシンタイプ | GPU 数 | GPU メモリ (GB) | vCPU 数 | VM メモリ (GB) | ローカル SSD (GiB) |
|---|---|---|---|---|---|
| a3-highgpu-1g| 1 | 80 | 26 | 234 | 750 |

こちらも十分な Quota が割り当てられているか事前に確認します。

**対象の割り当て (Quota)**

- オンデマンド:
  `aiplatform.googleapis.com/custom_model_serving_nvidia_h100_gpus`
- DWS Flex-Start:
  `aiplatform.googleapis.com/custom_model_serving_preemptible_nvidia_h100_gpus`

今回は上記 Quota の割り当て量がそれぞれ `1` 必要となります。

### Gemma 3 (1B) モデルをエンドポイントにデプロイ

Online Precition の方はシンプルな GUI 操作だけでデプロイが可能で、今回は [Gemma 3 モデルカード](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemma3)の `Deploy options` から `Vertex AI` を選択します。

![model_garden_gemma3_deploy](https://storage.googleapis.com/zenn-user-upload/5db6d0de336c-20250815.png)

画面右側にデプロイの設定を入力する枠が表示されますので、リソース ID とリージョンを以下の通り設定します。

- リソース ID: `gemma-3-1b-it`
- リージョン: `us-central1`

### オンデマンド

オンデマンドの場合、リソース ID とリージョンを指定したら、あとはデフォルト値のままでデプロイを実行します。

![vai_prediction_od_deploy](https://storage.googleapis.com/zenn-user-upload/f7f1e56f3faa-20250815.png =400x)

無事にデプロイが完了しました。

![vai_prediction_od_deployed_model](https://storage.googleapis.com/zenn-user-upload/b254c80a90de-20250815.png)

Training と同様に Cloud Billing の[レポート](https://cloud.google.com/billing/docs/how-to/reports)上で料金を確認します。

![vai_prediction_od_billing_report](https://storage.googleapis.com/zenn-user-upload/01155f6bb906-20250815.png)

上記の SKU を元に 1 時間あたりの料金に換算して再計算します。

| SKU | サービス | SKU ID | 単価 | 単位 | 数量 | 小計 |
|---|---|---|---|---|---|---|
| Vertex AI: Online/Batch Prediction Nvidia H100 80gb GPU running in Iowa | Vertex AI | [AE9C-DB60-DF46](https://cloud.google.com/skus?currency=USD&filter=AE9C-DB60-DF46) | $11.2660332 | per 1 hour | 1 | $11.2660332 |
| Vertex AI: Online/Batch Prediction A3 Predefined Instance Core running in Iowa | Vertex AI | [A002-4323-D900](https://cloud.google.com/skus?currency=USD&filter=A002-4323-D900) | $0.0293227 | per 1 hour | 26 | $0.7623902 |
| Vertex AI: Online/Batch Prediction A3 Predefined Instance Ram running in Iowa | Vertex AI | [2424-9F04-82A8](https://cloud.google.com/skus?currency=USD&filter=2424-9F04-82A8) | $0.0025534 | per 1 gigabyte hour | 234 | $0.5974956 |

合計: **$12.625919** per hour

### DWS Flex-Start

DWS Flex-Start の場合は、デプロイ設定画面の`デプロイの設定`で`詳細`を選択します。次に`可用性ポリシー`にある `VM プロビジョニング モデル`を `Flex Start` に変更してデプロイを実行します。

![vai_prediction_dws_deploy](https://storage.googleapis.com/zenn-user-upload/e939b8e59f1b-20250815.png =400x)

こちらも無事にデプロイが完了しました。

![vai_prediction_dws_deployed_model](https://storage.googleapis.com/zenn-user-upload/bc37d38596da-20250815.png)

Cloud Billing の請求レポートを見ていきましょう。

![vai_prediction_dws_billing_report](https://storage.googleapis.com/zenn-user-upload/e72775d8df90-20250815.png)

Training 同様に、Prediction でも DWS Flex-Start 利用時には Compute Engine と Vertex AI に分かれて費用が計上されていました。

こちらも 1 時間あたりの料金に換算して再計算します。

| SKU | サービス | SKU ID | 単価 | 単位 | 数量 | 小計 |
|---|---|---|---|---|---|---|
| Nvidia H100 80GB GPU attached to DWS Defined Duration VMs running in Americas | Compute Engine | [341A-49A5-0C07](https://cloud.google.com/skus?currency=USD&filter=341A-49A5-0C07) | $4.200761 | per 1 hour | 1 | $4.200761 |
| DWS Defined Duration A3 Core running in Americas | Compute Engine | [9A32-36B2-7FBC](https://cloud.google.com/skus?currency=USD&filter=9A32-36B2-7FBC) | $0.010934 | per 1 hour | 26 | $0.284284 |
| DWS Defined Duration A3 Ram running in Americas | Compute Engine | [ED19-A584-4A84](https://cloud.google.com/skus?currency=USD&filter=ED19-A584-4A84) | $0.000952 | per 1 gigabyte hour | 234 | $0.222768 |
| DWS Defined Duration SSD backed Local Storage running in Americas | Compute Engine | [2303-1D6A-9C08](https://cloud.google.com/skus?currency=USD&filter=2303-1D6A-9C08) | $0.0001096 [^3] | per 1 gibibyte hour | 750 | $0.0821918 |
| Vertex AI: Online/Batch Prediction management fee on NVIDIA H100 80GB in Iowa | Vertex AI | [1797-967A-274D](https://cloud.google.com/skus?currency=USD&filter=1797-967A-274D) | $1.4694826 | per 1 hour | 1 | $1.4694826|
| Vertex AI: Online/Batch Prediction management fee on A3 Instance Core in Iowa | Vertex AI | [9AC1-7D9B-C47E](https://cloud.google.com/skus?currency=USD&filter=9AC1-7D9B-C47E) | $0.0038247 | per 1 hour | 26 | $0.0994422 |
| Vertex AI: Online/Batch Prediction management fee on A3 Instance RAM in Iowa | Vertex AI | [5692-009D-2057](https://cloud.google.com/skus?currency=USD&filter=5692-009D-2057) | $0.0003331 | per 1 gigabyte hour | 234 | $0.0779454 |

合計: **$6.436875** per hour

Prediction においても DWS Flex-Start 利用時には、オンデマンド比で **49.02%** の割引率で利用できていました。

## まとめ

- DWS Flex-Start の新しい料金プランにより、NVIDIA H100 GPU を搭載した A3 High VM を利用した場合、Vertex AI Training / Online Prediction いずれの場合も、オンデマンド比で**最大約半額**の割引価格で利用できることが確認できました。
- Vertex AI では、オンデマンド利用時と DWS 利用時で使用される SKU 体系が異なり、DWS では Compute Engine の VM 利用料と Vertex AI の管理手数料を組み合わせたハイブリッドな課金体系となっておりました。

## 最後に

- DWS Flex-Start を実際に試される際には、制約や要件 ([Training](https://cloud.google.com/vertex-ai/docs/training/schedule-jobs-dws#requirements) / [Prediction](https://cloud.google.com/vertex-ai/docs/predictions/use-flex-start-vms#limitations)) についても事前にご確認いただいた上でご利用ください。
- 今回は Vertex AI を利用した解説をしましたが、[GCE](https://cloud.google.com/compute/docs/instance-groups/about-resize-requests-mig) や [GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/dws) Standard を活用して Vertex AI 以外から DWS Flex-Start を利用する際も DWS の新料金の恩恵を受けることができます。その場合 Vertex AI における管理手数料が発生しないため、より深い割引率で利用することが可能です。
- マシンタイプやリージョンによっては割引率が異なることもありますので、実際に利用する際には、以下の SKU グループ等を参照して事前に費用を試算いただくことをお勧めします。

**Compute Engine SKU Groups**
- [Flex-start Mode A2 VMs](https://cloud.google.com/skus/sku-groups/flex-start-mode-a2-vms)
- [Flex-start Mode A3 VMs](https://cloud.google.com/skus/sku-groups/flex-start-mode-a3-vms)
- [Flex-start Mode A3 Mega VMs](https://cloud.google.com/skus/sku-groups/flex-start-mode-a3-mega-vms)
- [Flex-start mode A3 Ultra VMs](https://cloud.google.com/skus/sku-groups/flex-start-mode-a3-ultra-vms)
- [Flex-start Mode A4 VMs](https://cloud.google.com/skus/sku-groups/flex-start-mode-a4-vms)
- [Flex-start Mode Local SSD](https://cloud.google.com/skus/sku-groups/flex-start-mode-local-ssd)

**Vertex AI SKU Groups**
- [Vertex Training](https://cloud.google.com/skus/sku-groups/vertex-training)
- [Vertex Prediction](https://cloud.google.com/skus/sku-groups/vertex-prediction)
- [Vertex Management Fee](https://cloud.google.com/skus/sku-groups/vertex-management-fee)

[^1]: Cloud Billing レポート [よくある質問](
https://cloud.google.com/billing/docs/how-to/reports#faqs)
[^2]: SKU 上の単価はあくまでも Unit Price となりますので、費用合計を算出するためには、単価に対して GPU 数、vCPU 数、RAM 容量などの数量を掛けて算出する必要があります。
[^3]: SKU 上の単位は `per 1 gibibyte month` ですが、単位を `per 1 gibibyte hour` にあわせるために、1 ヶ月 ≒ 730 時間で割った金額を単価としています。
[^4]: Training ジョブ実行時のブートディスクのサイズによります。今回はノートブック内で `2,000 GiB` と指定しています。
[^5]: Compute Engine 側で DWS Flex-Start を直接利用する場合も同じ SKU が適用されます。
[^6]: [Vertex AI の料金](https://cloud.google.com/vertex-ai/pricing#custom-trained_models)によると、Vertex AI で Spot VM や Reservations (予約) を利用する場合も、DWS 利用時と同様に Compute Engine の SKU と Vertex AI の Management Fee (管理手数料) SKU の組み合わせの課金体系となることが明記されています。


