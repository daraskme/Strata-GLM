# Strata-GLM for Mang-AI

GLM-5.3-FlashをVRAM・RAM・NVMe SSDへ分け、単一GPUで実行するカスタム推論エンジンです。[Mang-AI](https://github.com/daraskme/Mang-AI)のA1常駐環境と組み合わせ、GLMをコーディング担当として使います。

ベースは[Project Maya](https://github.com/mw00/project-maya)と[Strata](https://github.com/Niko1221/Strata)。元のMITライセンスと履歴を維持しています。[上流README](README.upstream.md)は参考資料で、Maya-Sや複数GPUの上流測定を本派生のOrcaRouter Q4性能とは扱いません。

## 確認した結果

**2026-10-09、RTX PRO 6000 Blackwell 96GB + i9-12900KS + DDR4-3200 128GiB + NixOS。** OrcaRouter Uncensored Q4_K_M、A1 4B Q8常駐、context 262144。

| 実入力 / 出力 | prefill | decode | 3位置検索 |
|---|---:|---:|---|
| 8180 / 512 tokens | 9.77秒 | 15.9 tokens/s | 成功 |
| 260090 / 512 tokens | 426.49秒 | 17.1 tokens/s | 成功 |

prefix再利用0。長文HTTP全体457秒、初期ロード約70秒は別。固定RAM tier 94.98GiB / GPU expert pool 63.13GiBでの測定です。通常の可変RAM設定の再測定値ではありません。

**50 tokens/sは未達。FP8に近い品質も未検証です。** 512-tokenのコード出力は実行せず、検索の正答と生成速度だけを確認しました。VRAM/RAMの余裕、zram、PCIe、温度、出力枠の制約も[詳細な検証記録](docs/optimization-gen5-262k.md)を参照してください。

## 導入・使い方

- [日本語のビルド・モデル取得・pack・起動手順](README.MangAI.md)
- [測定JSON・再現用入力・固定seed](docs/benchmarks/2026-10-09/README.md)
- [Mang-AIへ登録する手順](https://github.com/daraskme/Mang-AI/blob/main/manga-studio/docs/glm-coding-session.md)
- [以前のQ4 / 128K検証](docs/optimization-round2.md)

実機で確認したビルドはCUDA 13・sm_120。モデルのGGUF約179.72GiBとpack、ビルド領域を別途準備してください。モデル重み、認証情報、会話、ビルド済み実機バイナリは配布しません。`data/glm5-synth.gguf`は上流の小さな人工的試験データです。

`README.MangAI.md`の`/etc/nixos#cuda`は検証PCのローカル開発環境で、同flakeは配布していません。他の環境ではCUDA、CMake、Ninja、C++コンパイラを用意してください。AVX-VNNIとCPU affinityは自分のCPUに合わせます。

## 追加機能

- Q4_K / Q5_K / Q6_K expert decode、混合gate/up/down形式の検査と未対応形式の停止。
- GLMのtokenizer・chat template、GLM tool構文からOpenAI JSONへの変換。
- A1のhealth、空きメモリ、GPU leaseの検査と所有プロセスの終了処理。
- CPU/GPU分担、RAM/GPU予算、先読み、思考budgetの設定。tier別・層別の診断ツール。
- 同じ履歴seedを使う比較試験と、入力・生成・prefix reuseを分けた測定。

通常設定はCPU自動分担・8 workers、均等slots、先読み0。独自再量子化、DMAへの変更、単一GPUのMTPは未実装です。`--thinking-budget 0`は思考無効ではなく人工的な打切りなしを意味します。APIはlocalhost用で、強制`tool_choice`は未対応です。

## 検査

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_glm*.py'
```

公開前に15件成功。GPU数値検査・実推論は[導入手順](README.MangAI.md)に分けて記載しています。詳細計測を有効にした速度を通常性能と混ぜず、同じ比較条件で測定してください。

## ライセンス

[MIT](LICENSE)。第三者コードの表示を維持します。モデルには配布元の利用条件が適用されます。
