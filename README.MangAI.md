# Mang-AI用 GLM Strata

GLM-5.3-Flashを単一RTX PRO 6000 Blackwell 96GB・RAM 128GBで動かす公開派生。
公開元は [Project Maya](https://github.com/mw00/project-maya)
`5932f601373f53fc021f75dc55159a722c772571`。その元になった
[Strata](https://github.com/Niko1221/Strata)とともにMITの著作権表示を維持する。
現行の実測は[Gen5・262K検証](docs/optimization-gen5-262k.md)、再現用データは[benchmark資料](docs/benchmarks/2026-10-09/README.md)を参照。

## 変更

- Unsloth UD-IQ4_XSに含まれるQ6_K expertをGPU decodeで処理。Q4_K/Q5_Kも追加。
- 未対応expert形式は起動時に拒否。万一dispatchに到達してもsticky errorで推論を失敗扱いにする。
- CUDA 13で未定義templateが実体化される分岐を修正。CUDA profiler/NVTXは任意のビルド機能へ分離。
- APIでpackの`pre=glm4`、`special_ids`を復元。GLMをQwen方式でtokenizeしない。
- GLMの`<arg_key>`/`<arg_value>`ツール構文をAPIのJSON tool_callsへ変換。
  GLMのツール引数は閉じタグ到着時に一括送信。通常テキストはstreaming。
- A1のhealth、GPU/RAM下限、所有プロセスの終了処理、任意のMang-AI共通GPUキュー。
- Nixで有効にならなかったCPU命令セットを明示し、RAM側expertをAVX2/AVX-VNNIで計算。
  prefillは8 GiBの作業予算、最大16,384 tokens/chunk。256 tokens未満は短文用経路を使う。
- OrcaRouter Q4_K_Mの取得・SHA256検証と、Q4_K gate/up → Q6_K downの数値検査を追加。
- CPUワーカーの同期負荷を削減。CPU配置・使用頻度に応じたVRAM配分を比較できる。
  [追加最適化と実測](docs/optimization-round2.md)、
  [Devin CLI / Opus 5.5の提案](docs/opus55-optimization-consultation.md)を参照。

## ビルド

llama.cppの固定版は`3cf03257f219afbe7334045ff7c6a06ac68c627d`。
`setup.get_llama_cpp()`で取得できる。モデル取得を伴う`maya.py`の自動セットアップは今回使わない。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-mangai.txt
.venv/bin/python -c 'import setup; setup.get_llama_cpp()'
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX_VNNI=ON
cmake --build build --target strata glm_moe_quant_parity glm_workers_test glm_profile_test --parallel 2
```

`/etc/nixos#cuda`は検証PC固有のflakeであり本リポジトリに含まれない。一般環境ではCUDA 13・CMake・Ninja・対応C++コンパイラを準備する。
このPCでは各コマンドを`nix develop /etc/nixos#cuda -c`経由で実行する。
Nixの`NIX_ENFORCE_NO_NATIVE`は`-march=native`を除去するため、上記では確認済みのi9命令セットを明示する。
他のCPUでは対応ISAを確認し、AVX-VNNI非対応なら同オプションをOFFにする。
OS設定、A1用エンジン、既存Qwen用Strataは変更しない。
プロファイラーヘッダーを導入済みの環境では`-DSTRATA_GLM_PROFILING=ON`が利用可能。

## モデルとpack

現在の対象は[OrcaRouter GLM-5.3-Flash-Uncensored-GGUF](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-GGUF)の
`Q4_K_M`、revision `efa699effe7e3114ac89a87bab2ee9f56ccddba3`。
5ファイル合計192,974,979,872 bytes。本体のrouted expertsは約169.91 GiBで、
Q4_KとQ6_Kの混合。32-bitやBF16へ全展開しない。
SHA256付きの固定manifestは`configs/orcarouter-q4-model.json`。
既存のHFアクセス権を使い、取得ツールは利用条件への同意操作を行わない。

```bash
.venv/bin/python tools/fetch_glm.py --manifest configs/orcarouter-q4-model.json --out /absolute/model --env-file /absolute/secrets/models.env
STRATA_GGUF_PY="$PWD/third_party/llama.cpp/gguf-py" .venv/bin/python tools/iq_pack.py --gguf /absolute/model/Q4_K_M/GLM-5.3-Flash-Uncensored-Q4_K_M-00001-of-00005.gguf --out /absolute/model/Q4_K_M/pack-maya-q4 --compat-bf16
```

packはGGUFと同じディレクトリ直下に置く。`--experts-bin`は指定しない。
モデル、pack、実行ログ、会話本文、秘密情報をGitへ入れない。

## 起動

2026-10-09以降、`--thinking-budget 0`（既定）で人工的な思考打切りを行わない。
旧smokeの256-token上限を再現する場合は`--thinking-budget 256`を明示する。
Mang-AI broker配下は継承したGPU利用権を`--external-gpu-lease`で検証し、二重取得を避ける。
このオプションを単独で使ってGPUキューを省略することはできない。
`--trace-experts`でtier選択を記録し、`tools/analyze_glm_tiers.py`でVRAM hit・RAM CPU・RAM転送・SSD missを分けて集計する。
`--ram-policy`と`--ram-protect-tokens`は既存のRAM入替方針の比較用。既定はprotected LFU・64 tokens。
`--prefetch-experts 1`は次層のexpertを予測してRAMからVRAMへ先にコピーする既存機能。0〜2を指定できる。
`--lookahead-layers 2`は最大4層先のSSD読み取りを予測する既存機能。両方とも既定0で無効。
予測が外れると余分な転送・退避が増えるので、別々に比較してから採用する。expertの省略やrouter変更は行わない。

```bash
.venv/bin/python tools/run_glm.py --pack /absolute/model/Q4_K_M/pack-maya-q4
.venv/bin/python tools/run_glm.py --pack /absolute/model/Q4_K_M/pack-maya-q4 --mangai-root /mnt/solidigm-b/Mang-AI --cpu-threads 8 --cpu-spin-us 1000 --cpu-affinity 0-15 --uniform-expert-slots --run
```

`--run`なしは検査だけ。APIは`http://127.0.0.1:1243/v1`。
起動には空きRAM94 GiB・VRAM80 GiBとA1の正常応答が必要。VMは利用者自身が停止する。
launcherの既定contextは131,072。通常Mang-AIでは`--context 262144 --ram-headroom-gib 16`を明示する。262144は入出力の合計枠。最新の260090実入力検証は上のリンクを参照し、下記の旧checkpoint測定と区別する。
GPU全体の使用上限目標90 GiBからさらに2 GiBを予約、RAMは空き容量から12 GiBを残してexpertへ使用。
MTPは単一GPUの実行経路がないため無効。CPU laneは起動時のCPU/PCIe校正に任せる。
`--cpu-lane off`で比較可能。`--cpu-plan 001122334`では、その層でRAMから必要となるexpert数0〜8に対するCPU担当数を指定する。
9桁のASCII数字で各桁は対象数以下とし、残りはGPUへ転送する。省略時は従来の起動時校正を使う。
明示した配分は起動記録へ保存する。`--cpu-lane off`との併用は拒否する。性能を確認してから通常設定へ採用する。CPU 8 P-core/8 E-coreに固定の万能スレッド数は仮定しない。
APIのmodel IDは`glm-5.3-flash-orcarouter-q4-mangai`。
別のcheckpointを比較する場合は`--model-name`で識別名も明示する。
上のCPU指定はこのPCのi9-12900KS向けで、`0-15`がPコアのlogical CPU。
別のCPUではaffinityを確認するか省略する。省略時は物理コア数に応じたauto設定。
packの`expert_usage.txt`でGPUへ置くexpertを優先する。このPCにはコード課題由来のseedを配置済み。
通常運用では使用履歴を更新し、次回の起動へ反映する。新規packでは初回は履歴なしで起動する。
`--benchmark`は履歴を実行ごとのコピーへ書き、元を保持する。
比較試験では`--expert-pool-gib 67 --ram-tier-gib 92`等でexpert領域の予算を固定できる。
前者はGPU poolの上限で、VRAM全体のcap・空き容量からさらに制限される。後者はpinned RAMの予算で、
設定したheadroomに加えて起動処理用4 GiBを残せなければ拒否する。実際の割当はslot単位に丸められる。
通常運用では省略し、従来通り空き容量に合わせる。固定予算の試験と従来の自動予算を混ぜて速度比較しない。

Ctrl+Cでこの起動が所有するAPIとengineを終了し、GPUキューを解放する。
空きRAM8 GiB未満、VRAM1 GiB未満、A1 healthの連続失敗でも同じ終了処理をする。
メモリ使用は`build/runtime/*-resources.jsonl`、engineログも同じディレクトリに保存。

層別の待ち時間を調べる診断起動には`--profile-gpu-after 0`を追加する。既存のCUDA event計測を用い、
`build/runtime/*-gpu-profile.0.csv`へfast pass・消費token位置・層・処理区間を記録する。
`0`は全fast passを記録し、正の値は先頭のpassをその数だけ集計対象から外す。prefillとdecodeの
自動分離ではない。単一要求ならAPIのprompt token数を下の`--min-position`へ指定すると、
生成tokenを入力にしたpassだけを集計できる（最初の出力は最終prompt tokenから生成されるため除外）。

```bash
.venv/bin/python tools/analyze_glm_gpu_profile.py build/runtime/RUN-gpu-profile.0.csv --min-position 142
.venv/bin/python -m unittest discover -s tests -p test_glm_gpu_profile.py -v
```

これは通常無効の診断機能。event挿入・読み取り・CSV保存の負荷があるので、この速度を通常性能と混ぜない。
`moe_cpu_wait`は主GPU stream上でCPU結果を待つ区間、`moe_wait`はhost応答待ち、`moe_fetch`は
取得処理の区間であり、単独CPU計算時間・物理SSD読取時間・純粋PCIeコピー時間とはそれぞれ異なる。
記録範囲外のembedding・出力head・境界処理・HTTP時間を含まず、区間合計をtoken全体の時間と呼ばない。
MTPや複数要求の解析では位置が重複し得るため、まずMTP無効・単一要求で診断する。

## 検証

```bash
.venv/bin/python -m unittest discover -s tests -p test_glm_frontend.py -v
build/glm_moe_quant_parity
build/glm_workers_test
build/glm_profile_test
# 上のrun_glm.pyを --benchmark --run 付きで起動してから実行する
.venv/bin/python tools/bench_glm.py --pack /absolute/model/Q4_K_M/pack-maya-q4 --case smoke
.venv/bin/python tools/bench_glm.py --pack /absolute/model/Q4_K_M/pack-maya-q4 --case decode --repeat 3 --output-tokens 768
.venv/bin/python tools/bench_glm.py --pack /absolute/model/Q4_K_M/pack-maya-q4 --case needle --tokens 128000
```

OrcaRouter版の検査・実機測定は[追加最適化記録](docs/optimization-round2.md)を正本とする。

## 旧Unsloth UD-IQ4_XSでの初回検証

以下は旧checkpointの履歴であり、OrcaRouter Q4_K_Mの性能値ではない。
旧モデルは[Unsloth GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)の
`UD-IQ4_XS`、revision `a38483c8cd5df544f53d70fb281afe97369d5ab6`。
expertはIQ3_S/IQ4_XS/Q6_K混合、約133.822 GiB。

2026-10-08: CUDA13/GCC15/sm120でビルド成功。APIの4回帰テスト合格。
GPU数値比較8ケース（Q4_K/Q5_K/Q6_K各1・8expert、IQ3_S→Q6_K、IQ3_S→IQ4_XS）合格。
down計算のCPU double参照に対する相対L2誤差は1.4e-7以下。
gate/up後のQ8活性化誤差を含む比較は0.00559～0.00624。
未知形式の拒否・sticky errorも確認済み。

最終設定（UD-IQ4_XS、context 131,072、A1常駐）の短文試験はすべて合格。
以下の生成速度はdecode区間であり、入力処理を含む所要時間とは異なる。

| 試験 | 入力 / 出力tokens | 生成速度 | API全体の所要時間 | 判定 |
| --- | ---: | ---: | ---: | --- |
| 日本語の指定文 | 37 / 10 | 18.7 tokens/s | 3.22秒 | 一致 |
| clamp関数 | 73 / 37 | 16.7 tokens/s | 6.65秒 | 8入力で一致 |
| ツールJSON | 197 / 21 | 10.6 tokens/s | 17.71秒 | 名前・引数一致 |
| 128K長文検索 | 127,990 / 42 | 17.5 tokens/s | 612.38秒 | 3箇所すべて一致 |

短文の生ログは`build/validation/20261008T073400-smoke/`。
CPU最適化前の生成速度5.3～6.7 tokens/sに比べ改善したが、短い試験だけの測定値である。
長文検索は最適化前に31,989 / 63,999 tokensで3箇所とも成功。
最終設定でも127,990 tokensの実入力で成功した。入力処理は609.63秒（209.9 tokens/s）、
生成は2.40秒。生ログは`build/validation/20261008T073454-needle/`。
すべてキャッシュ再利用0。短い回答の計測であり、長時間の生成速度を保証しない。
最終起動の78回の監視でA1 health失敗0、空きRAMの最小31.97 GiB、空きVRAMの最小6.41 GiB。
GLM終了後もA1のPIDは変わらず、GPUキューを解放した。
22.2 tokens/sという投稿値を、このQ4構成の実測値として扱わない。
`bench_glm.py`はモデルと同じchat template/tokenizerで実入力を数え、API usageとの一致を検査する。
長文は10%・50%・90%位置の3つの合言葉を検索する限定テストであり、長文コーディング全般の保証ではない。
APIの`tool_choice=required`など強制ツール選択は元実装の未対応項目。現在は通常のauto選択を使う。

## PCIeの状態

2026-10-09追記：UEFIをAuto／Gen5へ変更して再起動した後、行列計算・転送・GLM生成中のGen5 x16を確認。
pinned H2Dは3回の中央値36.89 GB/s、D2Hは36.99 GB/s。同じ短文課題の計測付きdecodeは9.2→14.5 tokens/s。
出力本文や起動時の自動配分も異なるため、全ての差をPCIeだけに帰属させない。50 tokens/sは未達。

以下はGen1時の記録。

推論中もGPUとCPU root portのPCIeリンクが2.5 GT/s x16（Gen1）だった。
両端の最大対応はGen5。GPU使用率100%でもGen1で、入力処理の転送を制限している。
CPU側expertの高速化でdecode中のRAM→GPU転送を抑えたが、リンク自体は改善していない。
`--prefer-max-performance`は同一NVML接続でhintを設定・検証し、終了時に元へ戻す任意機能。
今回の短い比較ではGen1のままだったため既定では無効。BIOS・ドライバー設定は変更していない。
