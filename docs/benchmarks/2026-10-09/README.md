# Gen5 / 262K検証データ

[結論・採否・制約](../../optimization-gen5-262k.md)。値は2026-10-09の保存結果であり、再現コマンドを後から実行した結果ではありません。

| ファイル | 内容 |
|---|---|
| [measurements.json](measurements.json) | 成功・失敗を含む15構成の実測、コマンド、token数、出力SHA、メモリ、swap |
| [broker-smoke.json](broker-smoke.json) | 通常設定・共通APIを通した日本語 / clamp / tool JSON |
| [requests.json](requests.json) | 比較に使った6つの人工的なコーディング要求。T0・seed42・512 tokens、コードは未実行 |
| [frozen-code-seed.txt](frozen-code-seed.txt) | 比較時のexpert使用頻度。文章・モデル重みは含まない |
| [mapped-transfer.json](mapped-transfer.json) | mapped read / DMAの独立転送試験の集計 |
| [latency-slot-candidate.json](latency-slot-candidate.json) | オフライン配置候補。実装済みの速度改善を意味しない |

モデルrevisionは`efa699effe7e3114ac89a87bab2ee9f56ccddba3`、Q4_K_Mの全5シャードとSHA256は[固定manifest](../../../configs/orcarouter-q4-model.json)。測定バイナリSHA256は`13a97bae0d030dd02e79877b05c44750e3635dc426c7d56e5600ebdb02b56373`。そのままのバイナリは公開せず、ソースからビルドします。同じソースでもtoolchainによりバイナリSHAは変わります。

## 長文試験を再実行する

モデル・pack・ビルド済みエンジン・A1を準備し、他のGPU工程がidleであることを確認します。モデルとベンチマークは別端末で起動してください。以下は測定時の固定予算です。通常の可変RAM設定とは分けます。RAM headroom 12 GiBで測った際はzramへのswap-outが増えたため、通常運用では16 GiBにしています。

```bash
# Strata-GLMのルートで実行。各自の絶対パスへ置換する
export MANGAI_ROOT=/absolute/path/to/Mang-AI
export GLM_PACK="$MANGAI_ROOT/models/llm/glm-5.3-flash-orcarouter/Q4_K_M/pack-maya-q4"
.venv/bin/python tools/run_glm.py --pack "$GLM_PACK" --context 262144 --port 1244 --model-name glm-orca-q4-profile-test --mangai-root "$MANGAI_ROOT" --cpu-threads 8 --cpu-spin-us 1000 --cpu-affinity 0-15 --expert-pool-gib 66.7 --ram-tier-gib 95 --ram-headroom-gib 12 --usage-profile docs/benchmarks/2026-10-09/frozen-code-seed.txt --benchmark --uniform-expert-slots --run
```

GPU pool上限66.7 GiBから空き容量・contextで制限され、実際は63.13 GiBでした。CPU affinityはこのi9のPコアlogical CPU配置です。他のCPUへ流用しないでください。RAM不足でguardが拒否する場合は固定予算を減らし、条件が変わったことを記録します。モデルを停止した後もメモリ回収を待ちます。

```bash
# 別端末。同じGLM_PACKを設定し、モデルready後に実行する
.venv/bin/python tools/bench_glm_long_context.py --pack "$GLM_PACK" --url http://127.0.0.1:1244 --model glm-orca-q4-profile-test --tokens 260100 --output-tokens 512 --output build/validation/long-262k
```

スクリプトは同じ記録形式から実token数を数えて上限まで入力を作り、10%・50%・90%の検索結果、API usageとの一致、prefix再利用0を検査します。260100という上限に対し今回の実入力は260090でした。出力プログラムを実行しません。`--output`は存在しない保存先を指定します。長文だけを測る場合も新規起動などでprefix再利用を避けてください。

## 短文比較・通常設定の確認

`measurements.json`のコマンドと`requests.json`を対応させます。`${MANGAI_ROOT}`はclone先、`${ENGINE_ROOT}`はこのリポジトリ、`${LOCAL_EVIDENCE}`は未公開のローカル保存先を表す記録上の表記です。各課題で同じHTTP JSONを送信し、返却された`usage`、`timings`、本文SHAを保存します。CUDA profileありの探索群と、profileなしの未使用3課題群を混ぜないでください。

```bash
# 共通brokerに通常のGLMが登録済みの場合
.venv/bin/python tools/bench_glm.py --url http://127.0.0.1:1234 --model glm-5.3-flash-orcarouter-q4-mangai --pack "$GLM_PACK" --case smoke
```

smokeのclampは限定ASTで8入力を検査します。tool JSONは構造だけを検査し実行しません。短いsmoke出力のtokens/sを持続生成の性能としません。

次の評価では反復と課題数を増やし、初期ロード・prefill・decodeを分け、zram、空きRAM/VRAM、温度、A1の未使用タスクのp95を記録します。品質は校正に使わないコード実行・tool・日本語・長文課題をFP8参照と比較します。それまでは品質同等や50 tokens/s達成を主張しません。
