# OrcaRouter Q4への切替と追加最適化

2026-10-08。利用者の追加指定により、対象を
[orcarouter/GLM-5.3-Flash-Uncensored-GGUF](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-GGUF)
の`Q4_K_M`へ切り替える。旧Unsloth `UD-IQ4_XS`の速度を新モデルの測定値として扱わない。

## 取得と再現

固定revisionと全5ファイルのSHA256は`configs/orcarouter-q4-model.json`。
既存のHugging Faceアクセス権が必要。取得ツールは利用条件への同意操作をしない。

```bash
.venv/bin/python tools/fetch_glm.py --manifest configs/orcarouter-q4-model.json --out /mnt/solidigm-b/Mang-AI/models/llm/glm-5.3-flash-orcarouter --env-file /mnt/solidigm-b/Mang-AI/secrets/models.env
STRATA_GGUF_PY="$PWD/third_party/llama.cpp/gguf-py" .venv/bin/python tools/iq_pack.py --gguf /mnt/solidigm-b/Mang-AI/models/llm/glm-5.3-flash-orcarouter/Q4_K_M/GLM-5.3-Flash-Uncensored-Q4_K_M-00001-of-00005.gguf --out /mnt/solidigm-b/Mang-AI/models/llm/glm-5.3-flash-orcarouter/Q4_K_M/pack-maya-q4 --compat-bf16
```

このPCではコマンドの前に`nix develop /etc/nixos#cuda -c`を付ける。
環境変数の指定を含むpackコマンドは`nix develop /etc/nixos#cuda -c env STRATA_GGUF_PY=...`とする。
モデル本体、認証情報、実行ログはGitに入れない。

全5シャードのヘッダーでは、本体expertは169.91015625 GiB。
gate/upはQ4_K、downは24層がQ4_K・18層がQ6_K。
同じ128K構成のdense VRAM見積りは5.34 GiB、state/KVは4.44 GiB。
配布カードの「MTPなし」と異なり、実ファイルには`blk.45`のMTP tensorが29個ある。
metadataの46 blocks−nextn1を使い、本体45層を`NO_MTP=1`で実行する。
新モデルでも旧モデルと同じthinking-off変換を適用できることをtemplateから確認した。

## 実装

- CPUワーカーの完了通知を仕事ごとからワーカーごとへまとめ、共有atomicを別cache lineへ配置。
  世代ticketを先に無効化し、次の仕事の境界とcallbackを公開する。計算の行内加算順序は変えない。
- 本番と同じWorkersを単独ヘッダーへ移し、大小の仕事、待機・復帰、取りこぼし・二重実行を検査。
  20回のプロセス起動で12万バッチ合格。ThreadSanitizerでも合格。
- `--cpu-threads`・`--cpu-affinity`・`--cpu-spin-us`を追加。spinは0～20,000 µs。
  affinityはこのGLMプロセスと子だけへ適用し、A1や他アプリを変更しない。
- `--usage-expert-slots`で保存済み使用頻度から層ごとのVRAM配分を決められる。
  元の`expert_counts.txt`しか参照しない処理に`expert_usage.txt`対応を追加。混合量子化のbyteコストも考慮。
  性能は入力に依存するので任意指定。`--uniform-expert-slots`で対照比較できる。
- RAM tierの計算元となる空き容量と予約量、pin失敗による縮小を記録。
- `--benchmark`時は使用頻度を実行ごとのコピーへ保存。比較が元のprofileを上書きしない。
- 768 tokens ×3回の持続生成試験を追加。これは速度試験であり、生成コードは実行しない。
  別のsmoke試験で日本語指定文、clampの8入力、tool JSONを確認する。
- GPU数値検査に新モデルのQ4_K gate/up → Q6_K downを追加し、9ケース合格。
  追加ケースのgate Q8相対L2誤差は0.00599822、downは0.00000008。

```bash
cmake --build build --target strata glm_workers_test glm_profile_test glm_moe_quant_parity --parallel 2
build/glm_workers_test
build/glm_profile_test
build/glm_moe_quant_parity
.venv/bin/python -m unittest discover -s tests -p test_glm_frontend.py -v
```

## Opus 5.5への相談

[提案全文](opus55-optimization-consultation.md)を保存。
Devin CLIのexportで`Claude Opus 5.5 High`を確認し、追加ツール呼び出し0。
利用者の明示承認に基づき、限定したコード抜粋と構成・測定値を送信した。

今回の検証対象はCPU同期・配置、長文chunkの拡大、頻度に応じたVRAM配分。
super-chunkの導入、CPU/GPU混合のbatched prefill、decode中の動的昇格、speculative decodingは
状態管理や数値検証の追加が必要なため、提案として保存し、今回の既定動作には入れない。

## 測定

旧モデルで追加した基準測定（変更前、Unsloth UD-IQ4_XS）は
`build/validation/20261008T132153-decode/`。同じ117-token入力から各768 tokens生成し、
12.0 / 12.4 / 12.0 tokens/s、中央値12.0。キャッシュ再利用0、3回の本文は同一。
短い試験の最大18.7 tokens/sを持続速度として扱わない。

新OrcaRouter Q4_K_Mは全5ファイルのSHA256照合、pack化、実推論に成功。
初期構成はcontext 131,072、GPU cap90 GiB、RAM予約12 GiB、CPU auto16 threads、
spin20 ms、prefill8,192 tokens / 4 GiB、各層114 slots、使用頻度seedなし。
runtimeは`20261008T135603-3320323`。GPU expert pool67.26 GiB、RAM tier94.91 GiBを確保。

| 初期構成の試験 | 入力 / 出力tokens | decode tokens/s | API全体の秒 | 結果 |
| --- | ---: | ---: | ---: | --- |
| 日本語指定文 | 37 / 10 | 7.2 | 9.09 | 一致 |
| clamp | 73 / 29 | 6.1 | 15.71 | 8入力一致 |
| tool JSON | 197 / 21 | 8.5 | 32.19 | 名前・引数一致 |
| 持続生成1 | 117 / 768 | 8.7 | 101.83 | 速度測定 |
| 持続生成2 | 117 / 768 | 9.7 | 91.70 | 速度測定 |
| 持続生成3 | 117 / 768 | 9.9 | 89.34 | 速度測定 |
| 32K検索 | 31,989 / 37 | 8.0 | 222.17 | 3箇所一致 |

smokeは`20261008T135803-smoke`、持続生成は`20261008T135912-decode`、
32K検索は`20261008T140605-needle`。いずれもprefix cache再利用0。
32K入力処理は217.344秒、147.2 tokens/s。持続生成の中央値は9.7 tokens/s。
新モデルの3回の生成本文は一致しておらず、同じ入力・token数の速度比較として扱う。
CPU/GPUへのexpert配置に伴う数値差もあるため、逐語一致や品質の同一性をこの試験から主張しない。

初期GPU配置はID順で、持続生成のVRAM hitは約44%。CPU laneはRAM expertをGPUへ昇格しないため、
過去の使用頻度で起動時にhot expertを配置する既存機能を次の比較で使う。
固定seedは短文試験と3回のLRU生成の履歴から作り、32K検索の結果を含めない。
`build/runtime/orca-code-seed.txt`のSHA256は
`4b61919f4a58cf412e8df1fb7b1ffbae0dfb923f6cc12853049c524e8c96d8e7`。
同じ題材のusageを利用するため、未知の入力への改善率は別途確認が必要。

使用頻度配置＋16K prefill、CPU auto16 threads / spin20 msの比較は
runtime `20261008T141137-3328085`。GPU expert pool67.85 GiB / 115 slots、RAM95.49 GiB。
固定seedを読み込んだことを42層すべてのログで確認。

| 使用頻度配置＋16K prefill | 入力 / 出力tokens | decode tokens/s | API全体の秒 | 結果 |
| --- | ---: | ---: | ---: | --- |
| 持続生成1 | 117 / 768 | 18.7 | 47.40 | 速度測定 |
| 持続生成2 | 117 / 768 | 24.6 | 36.38 | 速度測定 |
| 32K検索 | 31,989 / 42 | 8.6 | 135.30 | 3箇所一致 |

生成のVRAM hitは78.15% / 76.78%、CPU expertは約73～78個/tokenまで減った。
32K入力は130.257秒、245.6 tokens/s。初期構成の217.344秒から約40.1%短縮。
使用頻度とchunkの両方を変えた構成比較であり、個々の変更の効果を分離した測定ではない。
長文検索ではdecodeのVRAM hit48.1%、8.6 tokens/sだった。LRU課題の24.6を全入力へ一般化しない。
記録は`20261008T141334-decode`、`20261008T141515-needle`。

CPU12 threads / spin1 msでは、同じseedと16K prefillで
19.9 / 21.8 / 24.5 tokens/s（各768 tokens、中央値21.8）だった。
runtime `20261008T141806-3331057`、記録`20261008T142006-decode`。
CPU auto16の2回と比べ明確な優劣はなく、thread数やspin単独の改善を主張しない。
RAM tierはその時の空き容量によって95.90 GiBになった。

Pコアのlogical CPU `0-15`へGLMだけを配置し、CPU8 workers / spin1 msで
19.9 / 22.8 / 24.3 tokens/s（各768 tokens、中央値22.8）だった。
runtime `20261008T142303-3333319`、記録`20261008T142509-decode`。
GPU expert pool67.26 GiB / 114 slots、RAM96.04 GiB。A1のCPU配置は変更していない。
小規模比較のためCPU12との1 tokens/s差を有意な改善とは扱わず、
Eコアをほかの用途へ残すこの配置を、このPC向けの起動例に採用する。

履歴の作成に使っていないJSONL集計コード課題（157 tokens入力、768 tokens生成）では
18.0 / 18.6 tokens/s、API全体52.46 / 49.41秒だった。
記録は`20261008T142801-decode-heldout`。LRU試験の後に同じサーバーで実行し、prefix再利用は0。
この2回もコードの実行・正しさを評価する試験ではない。

このPCのpackには上記のコード由来seedだけを`expert_usage.txt`として配置した。
長文検索の大量のrouting回数で初期seedを上書きしていない。
通常運用では利用者の使用履歴が同ファイルへ蓄積され、次回起動へ反映する。
モデル重み、routingのtop-k、量子化形式は変更していない。

最終構成（Pコア8 workers、spin1 ms、16K prefill）の128K検索は
**127,990 tokensの実入力から10%・50%・90%位置の3つの合言葉をすべて正答**。
入力処理553.375秒（231.3 tokens/s）、42 tokensの回答生成4.930秒（8.5 tokens/s）、
API全体558.65秒。prefix再利用0。記録は`20261008T143039-needle`。
この検索試験は長文コーディング全般の品質保証ではなく、262K以上は実入力未検証。
PCIeは高負荷中もGen1 x16のままで、長い入力の初回処理が残る制約である。

128K処理後のsmokeも3項目すべて合格（`20261008T144045-smoke`）。
日本語37/10 tokensで14.9 tokens/s、clamp73/29で15.4、tool JSON197/21で13.2。
API全体はそれぞれ4.04 / 6.46 / 19.17秒。

4回の起動で合計268回の監視を行い、A1 health失敗0。
最終起動112回の監視では空きRAMの最小8.98 GiB、空きVRAMの最小6.65 GiB。
全実測の後にGLMだけを終了し、空きRAM110.79 GiB・VRAM85.17 GiBへの回復と、
A1が同じPID2406155でactiveのまま正常応答することを確認した。
次回の起動計画も確認済み。モデルの自動起動は追加していない。

[全試行の測定JSON](benchmarks-orcarouter-q4.json)には設定・token数・timing・判定・監視集計を保存。
プロンプトや生成本文を含む生ログはGit対象外の`build/validation/`に残す。
親Mang-AIの監査はNix開発環境で14項目合格。最初の素のシェルでは子プロセスの
`python3`がPATHに無く1項目を実行できなかったが、同じスクリプトを開発環境で再実行して解消した。
