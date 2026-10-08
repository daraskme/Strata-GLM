# Devin CLI / Opus 5.5による最適化案

2026-10-08。利用者の指示と、非公開コード抜粋・構成・測定値の送信への明示承認に基づく相談。
Devin CLI 3000.10.48、`claude-opus-5-5-high`。exportの実モデル名は`Claude Opus 5.5 High`。
追加ツール呼び出しは0。APIキー・モデル本体・漫画・会話データは送っていない。

相談時の測定モデルはUnsloth UD-IQ4_XS。その後、利用者の指定でOrcaRouter Q4_K_Mへ変更した。
以下は外部モデルの提案であり、実装済み・実測済みという意味ではない。採否と実機結果はREADME.MangAI.mdに記録する。

---

# GLM-5.3-Flash on Maya 4623f19: 最適化コンサルテーション(分析のみ)

ツールは使っていません。測定もしていません。以下の数値は、いただいた値とコード片から算術で出した推定です。それぞれ「推定」「仮定」と明記しています。

---

## 0. 提示データからの推定

**デコード**
- 18.7 / 16.7 / 10.6 t/s は、1トークンあたり約 53 / 60 / 94 ms です。
- CPU lane は 44–82 ms/token なので、範囲がほぼ重なります。デコード時間の大半は CPU lane がクリティカルパスになっている可能性が高いです。
- PCIe は 5.8 ms/expert、CPU は 0.37 ms/expert です。`fast_cpu_lane_setup` の split 式では f 個全部を CPU が取り、GPU は待つだけになります。
- 効く手段は「CPU に回る expert の数を減らす」か「1 expert あたりの CPU 時間を減らす」の二つだけです。
- 8→12→16 threads が 0.566→0.410→0.341 ms です。完全な帯域律速ではありません。ただし 12→16 の伸びは 1.2倍に留まります。これが E-core の遅さによるのか帯域飽和によるのかは、まだ区別できません(P2 で判定します)。

**Prefill**
- `prefill()` の式から、127,990 tokens は 16 チャンク、Trun = 8000 になります。
- 609.63 s ÷ 16 ≈ 38 s/チャンクです。
- 以前の 64K / 4672 chunk 構成は、64K = 65,536 と仮定すると 15 チャンク、485 s、約 32 s/チャンクです。
- チャンクあたりのトークン数が約 1.7 倍、文脈も長いのに、チャンクあたりの時間はほとんど変わっていません。チャンクごとに RAM-tier expert を PCIe Gen1 で再ステージングする固定費が支配的、という仮説と整合します。
  - 72 GiB ÷ 30〜38 s ≈ 2〜2.6 GB/s 程度です。Gen1 x16 の実効値として不自然ではありません。
  - これは仮説です。P1 で確認が必要です。

---

## 1. ランキング

| # | 施策 | 期待効果 | 確度 | 工数 | 厳密性リスク | 種別 |
|---|---|---|---|---|---|---|
| 1 | 長文 prefill: residual を保持する super-chunk と、計算単位の sub-chunk を分離し、PCIe 再ステージング回数を減らす | 大(長文 prefill) | 中(P1 次第) | 高 | 低(順序を保てば bitwise 一致を狙える) | コード |
| 2 | 短〜中プロンプトの hybrid batched prefill(RAM-tier expert を CPU で batched 計算) | 大(TTFT: tools / chat) | 中 | 中〜高 | 丸め誤差レベル(現行 batched と同程度) | コード |
| 3 | CPU lane のトポロジ最適化(pin、粒度、hugepage、形式別計測) | 小〜中(デコード) | 高(計測可能性) | 低〜中 | なし(bitwise 一致) | まずプロファイル、次に小コード |
| 4 | デコード中の決定的な適応 residency(アイドルの PCIe で昇格) | 中(ルーティング局所性次第) | 低〜中 | 中〜高 | 丸め誤差レベル(lane が変わるため) | まずオフラインシミュレーション、次にコード |
| 5 | (保留)prompt-lookup による複数トークン検証 | 不明(code / tool 向き) | 低 | 高 | greedy なら lossless(batched 経路基準) | コード |

デコード t/s だけを最優先するなら、3 → 4 の順で着手してください。P2 で CPU lane がほぼ帯域律速だと判定されたら、3 の上限は低くなり、4 が本命になります。

---

## 2. 各施策

### #1 長文 prefill: super-chunk の導入

**根拠**
- 現行はチャンク単位で処理し、各チャンクで全層を回します。そのため非常駐 expert をチャンク数ぶん再転送しています。
- チャンクを大きくすると、`carve_moe(m, T, g)` が T に比例して VRAM を消費します。実効値は 4096 MiB / 8192 ≈ 0.5 MiB/token です。
- residual(`R`)は hop サイズのコードから `T*4*E*sizeof(float)`、つまり 16E bytes/token です。E = 4096 と仮定すると 64 KiB/token です。

**変更案**
- residual と MoE 入力だけを S tokens(例: 32K)保持します。概算 20E bytes/token、E = 4096 なら約 2.5 GiB です。
- 層 l ごとに次の順で処理します。
  1. sub-chunk(現行の T = 8000 のまま)を昇順に、attention、gating、KDA / DSA の状態更新まで実行する。
  2. 層 l の非常駐 expert を landing ring で 1 回だけ流す。
  3. 流した各 group について、全 sub-chunk の該当トークンを計算する。
- 層 l における sub-chunk c+1 の attention や KDA 状態は、層 l の入力にしか依存しません。c の MoE 出力には依存しないので、MoE を後回しにしても数学的に同一です。
- PCIe の通過回数は N/T から N/S に減ります(128K、S = 32K なら 16 → 4)。代わりに借用 VRAM が増え、pool からの evict が増えるので、1 回あたりの転送量はやや増えます。

**厳密性**
- MoE カーネルを sub-chunk ごとに現行と同じ shape で起動します。トークンごとの expert 出力の加算順を現行と一致させれば、bitwise 一致を狙えます。
- 現行が atomicAdd で加算しているなら、もともと非決定的です。先に確認してください。

**事前確認(プロファイルのみ)**
- P1: チャンクごとの所要時間と H2D bytes を記録します。チャンク時間が位置にほぼ依存せず一定なら PCIe 律速です。後半ほど増えるなら attention 律速で、この施策の価値は下がります。

**暫定策(設定のみで比較)**
- `STRATA_GLM_PREFILL_CHUNK=16384` に大きい budget を組み合わせます。ただし実行前に以下を確認してください。
  - `h_bounds`(8192 ints)、`h_counts` / `h_base`(4096)がトークン数で索引されていないこと。T > 8192 で溢れる可能性があります。
  - attention 側は `ts = min(T, kSub)` で有界ですが、MoE 側は T 全体で確保されます。
- トレードオフ: 借用分は pool から evict されます。prefill 中は 1 回あたりの転送が増え、prefill 後は pool の再充填コストか、デコード初期の hit 率低下として返ってきます。後者は §3 の「post-prefill window」で必ず測ってください。

### #2 Hybrid batched prefill(32〜数千 tokens)

**根拠**
- `STRATA_GLM_PREFILL_MIN=256` 未満は token path を通ります。tools 197 tokens / 17.71 s の大半は、ほぼデコード速度での入力処理です。
- batched path は、触れた expert をすべて PCIe で流すので、小さい入力では不利になります。

**変更案**
- batched path で、常駐 expert は従来どおり GPU の MMQ で計算します。
- RAM-tier expert は CPU pool で「重みを 1 回読み、m_e 個のトークンに適用」します。
- `native_gu_rows_split` / `native_down_rows_split` は活性ベクトルの配列と個数を受け取るシグネチャです(現状は `a, 1, o`)。複数ベクトル化の素地がある可能性があるので、実装を確認してください。
- 効果の上限は、層ごとの「token-expert ペア数 ÷ 異なり expert 数」(再利用率)で決まります。

**拡張: 長文 prefill への適用**
- expert ごとに、m_e が小さいものを CPU、大きいものを PCIe に振り分け、並行に実行します。
- 振り分けは、既存の split 式 `max(PCIe(f-k)p, over + kc)` を一般化した形にします。

  min over S: max( Σ_{e∉S} bytes_e / BW_pcie , Σ_{e∈S} (bytes_e / BW_mem + m_e · c_tok) )

  を m_e の昇順 greedy で解きます。

**事前確認(プロファイルのみ)**
- CPU batched microbench: m = 1, 2, 4, …, 64 で固定費と c_tok を測ります。
- `PREFILL_MIN` の crossover を 64 / 128 / 256 / 512 / 1024 / 2048 で測ります。

**厳密性**
- CPU 部分は、各 (row, vector) の dot が単一ベクトル呼び出しと同一なら、token path の CPU 部分と bitwise 一致します。
- GPU 部分は MMQ で、現行 batched と同じ丸めです。
- `STRATA_GLM_CPU_LANE_VERIFY` 相当の相対 L2 と、end-to-end の指標で検証します。

### #3 CPU lane のトポロジ最適化(bitwise 一致)

**コード上の事実**
- `physical_cores()` は distinct な core_id を数えるので、この CPU では 16(P 8 + E 8)を返します。結果は Workers(15) + caller です。
- pin していないため、同じ P-core の SMT 兄弟に 2 worker が乗る一方で E-core が遊ぶ、という配置が起き得ます。
- caller は service thread で、`native_quant_act` と ne 個の `native_quant_h` を逐次で担当します。service thread が E-core に乗ると、逐次部分と最遅ワーカーの両方を E-core が担うことになります。
- down 相のジョブ数は `n_embd / kDnRows`(E = 4096 なら 32)です。16 スレッドで約 2 ジョブずつになり、E-core が取った最後のジョブが tail になりやすいです。

**プロファイルのみ**
- P2(最重要の判定): 実効帯域 = blob bytes / c_ms を出し、同じスレッド構成で測った読み出し帯域(STREAM 類似)と比べます。
  - 実効帯域がそれに近ければ、スレッド調整の伸びしろは小さいです。#4 か、転送量を減らす手段が本命になります。
  - 遠ければ、計算律速です。カーネル改善と E-core の活用が効きます。
- 形式別の c_ms: 既存の `STRATA_GLM_CPU_LANE_CHECK=<layer>` は `il_cal` を差し替えるので、IQ3_S / IQ4_XS / Q6_K の各層で校正値を出せます。
  - 現状の校正は 1 層の形式しか見ていません。IQ3_S(grid lookup)だけ遅い、といった偏りが見つかればカーネル改善の対象になります。
  - VNNI 命令が各形式で実際に使われているか、逆アセンブルで確認してください。
- taskset による粗い比較。注意点として、mask を絞ると `physical_cores()` の値も変わるので、`STRATA_GLM_CPU_LANE` で明示的に指定してください。
  - a) `0-15`, LANE=16
  - b) `0,2,…,14`, LANE=8
  - c) `0,2,…,14,16-23`, LANE=16
  - d) 現行の `0-23`, LANE=16

**小さなコード変更**
- worker ごとの affinity リスト(env)を追加し、service thread は P-core に固定します。兄弟スレッドを空けるかどうかも比較します。
- `kDnRows` を 32 / 64 / 128 で可変にします(n_embd を割り切る値に限る)。
- `native_quant_h` は「各 expert の gu チャンクを最後に完了したスレッドがその expert を量子化する」形にします。per-expert の atomic counter で実現でき、バリア追加も逐次部分もなくなります。同じ関数を同じデータに適用するので bitwise 一致です。
- spin_us を env 化し、20 ms / 2 ms / 0.3 ms を比較します。turbostat でパッケージ電力とクロックも併記してください。15 スレッドが常時 spin すると、電力枠経由で P-core のクロックに影響する可能性があります。
- RAM tier の hugepage 化: mmap に MADV_HUGEPAGE(または hugetlbfs)を付け、`cudaHostRegister` で登録します。
  - 4K ページでは HW prefetcher がページ境界を越えないので、大きな連続 blob のストリーミングで不利になり得ます。
  - まず `/proc/<pid>/smaps` の AnonHugePages で現状を確認し、64-expert のスタンドアロンベンチで比較してください。
  - メモリ量は変わりません。

**検証**
- 変更前後でロジットの hash がトークンごとに完全一致すること。行分割が行内の加算順に影響しないことの確認も兼ねます。

### #4 デコード中の適応 residency

**根拠**
- デコード中の PCIe はほぼアイドルです(全ミスが CPU に回るため)。
- device 側には、すでに route count(`md.dcnt`、LFU で seed 済み)があります。

**変更案**
- 別ストリームで、K トークンごとに最大 1〜2 expert を昇格させます。
- 有界な staging slot を 2 個(最大 blob サイズ)用意し、コピーが完了してからトークン境界で slot table を切り替えます。
- victim は減衰付き LFU で選び、ヒステリシスで thrash を防ぎます。
- 決定性のため、昇格の判断と切り替えのタイミングはトークン番号だけで決めます。時間には依存させず、未完了なら同期して待ちます。
- DMA が奪うホストメモリ帯域は、CPU lane が使う帯域と比べて小さい見込みです。ただし計測してください。

**プロファイル先行**
- P4: 3 種のワークロードと長出力で (layer, expert) のルーティングトレースを記録します(挙動は変えないロギングのみ)。
- オフラインで次の方針を比較し、hit 率を予測します。
  - 静的 LFU
  - 窓付き LFU + 昇格予算 1/token
  - Belady(上限)
- 予測 hit 率の伸びが小さければ、この施策は中止します。

**厳密性**
- 同じ expert でも GPU 経路と CPU lane 経路(`native_quant_act`)では丸めが異なります。residency の状態に依存した差が出ます。これは現行の LFU 状態依存と同種のリスクです。

### #5(保留)prompt-lookup による検証

- CPU lane は重みを 1 回読めば複数トークンに適用できるので、k トークンの検証は k 倍より安くなり得ます。
- ただし、トークンごとに expert の和集合が増えるので、削減幅はその分小さくなります。
- 複数トークンのデコードカーネルが必要で、工数は大きいです。#2 の CPU batched カーネルができてから再評価してください。

---

## 3. ベンチマーク方法(短い出力の cherry-pick を避ける)

1. **主指標は teacher-forced デコード**
   - baseline の greedy 出力(≥ 1024 tokens)を固定トークン列として保存します。
   - 各構成で、そのトークン列をデコード経路に 1 トークンずつ強制入力します。
   - ルーティングが同一になるので、エンジン速度だけを比べられます。MTP は無効なので、各ステップの仕事量は同じです。
2. **副指標は自然な長出力**
   - 本質的に長い出力が必要なプロンプト(長文の日本語、長いコード生成、複数ツール呼び出し)を使い、max 1024 / greedy で走らせます。
   - EOS 禁止は推奨しません。反復ループでルーティングが偏り、hit 率が過大に出ます。
3. **窓別の集計**
   - tokens 1–64 / 65–256 / 257–1024 の窓ごとに、次を集計します: t/s、トークン遅延の p50 / p90 / p99、GPU hit 率、CPU lane ms/token。
   - prefill 直後の窓は、#1 の借用影響を見るため別に報告します。
4. **状態管理**
   - 「新規プロセス」と「固定スクリプトで warm-up 後」の両方を報告します。
   - prefix reuse は無効のままにします。
   - prefix reuse の正しさは別テストで確認します: 同一会話の 2 ターン目で、reuse ありとなしのロジットを比べます。同じ経路・同じチャンク境界なら bitwise 一致、経路が異なれば許容差で比較します。
5. **統計**
   - ABBA の交互実行で 1 構成あたり 5 回以上走らせ、中央値と bootstrap 信頼区間を出します。
   - CI が 0 を跨ぐ改善は採用しません。
   - turbostat でクロック、電力、温度を併記します。
6. **厳密性ゲート**
   - bitwise を想定する変更(#1、#3): ステップごとのロジット hash の一致。
   - 丸め誤差を伴う変更(#2、#4): teacher-forced 列での top-1 一致率、max |Δlogit|、KL、needle テスト 3 種、固定テキストの perplexity。
7. **Prefill**
   - 同一テキストを 16K / 32K / 64K / 128K に切って、budget と chunk を格子状に振ります。
   - 報告項目: チャンクごとの時間、H2D bytes、ピーク VRAM、prefill 後の再充填時間または初期窓の hit 率。
   - 今回の 64K と 128K の比較は長さと構成が同時に変わっていて、対照比較になっていません。

---

## 4. 推奨順序

1. まずプロファイルのみで P1、P2、形式別 c_ms、taskset 比較、P4 のトレース記録を行います。
2. #3 の小さなコード変更(pin、粒度、量子化の融合)を、bitwise 確認付きで入れます。
3. P1 で PCIe 律速が確認できたら #1、P4 で局所性が確認できたら #4 に進みます。#2 は TTFT が重要なら並行して進めます。

ここに挙げた期待効果はすべて、上記の推定と仮定に基づく見込みです。速度の約束ではありません。
