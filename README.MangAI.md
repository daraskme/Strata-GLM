# Mang-AI用 GLM Strata

GLM-5.3-Flashを単一RTX PRO 6000 Blackwell 96GB・RAM 128GBで動かすprivate派生。
公開元は [Project Maya](https://github.com/mw00/project-maya)
`5932f601373f53fc021f75dc55159a722c772571`。その元になった
[Strata](https://github.com/Niko1221/Strata)とともにMITの著作権表示を維持する。
このリポジトリを公開する操作はしていない。

## 変更

- Unsloth UD-IQ4_XSに含まれるQ6_K expertをGPU decodeで処理。Q4_K/Q5_Kも追加。
- 未対応expert形式は起動時に拒否。万一dispatchに到達してもsticky errorで推論を失敗扱いにする。
- CUDA 13で未定義templateが実体化される分岐を修正。CUDA profiler/NVTXは任意のビルド機能へ分離。
- APIでpackの`pre=glm4`、`special_ids`を復元。GLMをQwen方式でtokenizeしない。
- GLMの`<arg_key>`/`<arg_value>`ツール構文をAPIのJSON tool_callsへ変換。
  GLMのツール引数は閉じタグ到着時に一括送信。通常テキストはstreaming。
- A1のhealth、GPU/RAM下限、所有プロセスの終了処理、任意のMang-AI共通GPUキュー。
- Nixで有効にならなかったCPU命令セットを明示し、RAM側expertをAVX2/AVX-VNNIで計算。
  prefillは4 GiBの作業領域、最大8,192 tokens/chunk。256 tokens未満は短文用経路を使う。

## ビルド

llama.cppの固定版は`3cf03257f219afbe7334045ff7c6a06ac68c627d`。
`setup.get_llama_cpp()`で取得できる。モデル取得を伴う`maya.py`の自動セットアップは今回使わない。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-mangai.txt
.venv/bin/python -c 'import setup; setup.get_llama_cpp()'
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp" -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX_VNNI=ON
cmake --build build --target strata glm_moe_quant_parity --parallel 2
```

このPCでは各コマンドを`nix develop /etc/nixos#cuda -c`経由で実行する。
Nixの`NIX_ENFORCE_NO_NATIVE`は`-march=native`を除去するため、上記では確認済みのi9命令セットを明示する。
他のCPUでは対応ISAを確認し、AVX-VNNI非対応なら同オプションをOFFにする。
OS設定、A1用エンジン、既存Qwen用Strataは変更しない。
プロファイラーヘッダーを導入済みの環境では`-DSTRATA_GLM_PROFILING=ON`が利用可能。

## モデルとpack

対象は[Unsloth GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)の
`UD-IQ4_XS`、revision `a38483c8cd5df544f53d70fb281afe97369d5ab6`。
5ファイル合計156,822,111,200 bytes。IQ4_XSという名前でもexpertはIQ3_S/IQ4_XS/Q6_Kの混合。
本体のrouted expertsは約133.822 GiB。32-bitやBF16へ全展開しない。

```bash
STRATA_GGUF_PY="$PWD/third_party/llama.cpp/gguf-py" .venv/bin/python tools/iq_pack.py --gguf /absolute/model/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf --out /absolute/model/pack-maya-iq4 --compat-bf16
```

packはGGUFと同じディレクトリ直下に置く。`--experts-bin`は指定しない。
モデル、pack、実行ログ、会話本文、秘密情報をGitへ入れない。

## 起動

```bash
.venv/bin/python tools/run_glm.py --pack /absolute/model/pack-maya-iq4
.venv/bin/python tools/run_glm.py --pack /absolute/model/pack-maya-iq4 --mangai-root /mnt/solidigm-b/Mang-AI --run
```

`--run`なしは検査だけ。APIは`http://127.0.0.1:1243/v1`。
起動には空きRAM94 GiB・VRAM80 GiBとA1の正常応答が必要。VMは利用者自身が停止する。
既定contextは131,072で、実入力の品質検証結果は下記と区別する。
GPU全体の使用上限目標90 GiBからさらに2 GiBを予約、RAMは空き容量から20 GiBを残してexpertへ使用。
MTPは単一GPUの実行経路がないため無効。CPU laneは起動時のCPU/PCIe校正に任せる。
`--cpu-lane off`で比較可能。CPU 8 P-core/8 E-coreに固定の万能スレッド数は仮定しない。

Ctrl+Cでこの起動が所有するAPIとengineを終了し、GPUキューを解放する。
空きRAM8 GiB未満、VRAM1 GiB未満、A1 healthの連続失敗でも同じ終了処理をする。
メモリ使用は`build/runtime/*-resources.jsonl`、engineログも同じディレクトリに保存。

## 検証

```bash
.venv/bin/python -m unittest discover -s tests -p test_glm_frontend.py -v
build/glm_moe_quant_parity
# 上のrun_glm.pyを --benchmark --run 付きで起動してから実行する
.venv/bin/python tools/bench_glm.py --pack /absolute/model/pack-maya-iq4 --case smoke
.venv/bin/python tools/bench_glm.py --pack /absolute/model/pack-maya-iq4 --case needle --tokens 32000
.venv/bin/python tools/bench_glm.py --pack /absolute/model/pack-maya-iq4 --case needle --tokens 128000
```

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

## このPCに残る転送制約

推論中もGPUとCPU root portのPCIeリンクが2.5 GT/s x16（Gen1）だった。
両端の最大対応はGen5。GPU使用率100%でもGen1で、入力処理の転送を制限している。
CPU側expertの高速化でdecode中のRAM→GPU転送を抑えたが、リンク自体は改善していない。
`--prefer-max-performance`は同一NVML接続でhintを設定・検証し、終了時に元へ戻す任意機能。
今回の短い比較ではGen1のままだったため既定では無効。BIOS・ドライバー設定は変更していない。
