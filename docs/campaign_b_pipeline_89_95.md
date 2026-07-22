# Campaign B パイプライン処理仕様（Notebook 89 / 95）

> **推奨運用（現行）:** **89∥97**（旧 89∥95 と同じ producer/consumer）。  
> 待ち行列が増えてもよい。Notebook **96** は任意の単一ノート統合モード。  
> 設計: [campaign_b_parallel_split_design.md](./campaign_b_parallel_split_design.md)、  
> 96 詳細: [campaign_b_end_to_end_design.md](./campaign_b_end_to_end_design.md)。  
> 監視: **98** status、**99** CERTIFIED カタログ。  
> GPU: `{PERSIST}/campaign_b/_locks/gpu_lane.json`（96∥97 同時フル起動は fail closed）。

Paperspace 上で Notebook **89**（mass explore）と **95**（pipeline_to_m6；現行推奨 consumer は **97**）を運用するときの、**各ステージが実際に何を計算・読み書きするか**の技術仕様。要約ではなく、コード上の処理ステップに沿って記述する。

| ノート | エントリ | 実装 |
|---|---|---|
| 89 | `notebooks/89_campaign_b_mass_explore.ipynb` | `src/campaign_b/mass_explore.py` → `driver.py` |
| 95 | `notebooks/95_campaign_b_pipeline_to_m6.ipynb` | `src/campaign_b/pipeline_to_m6.py`（90→94 を直列呼出） |
| 97 | `notebooks/97_campaign_b_post_m2_pipeline.ipynb` | `post_m2_pipeline.py` → 既定で `pipeline_to_m6`（95 相当） |

関連モジュール（95 が内部で呼ぶ段）:

| 段 | Notebook | モジュール |
|---|---|---|
| advance | 90 | `src/campaign_b/advance_selected.py` |
| M3 | 91 | `src/campaign_b/gpu_m3_batch.py` + `src/m3_orchestrator.py` |
| pre_m6 (M4→M5) | 92 | `src/campaign_b/pre_m6_batch.py` |
| obligations | 93 | `src/campaign_b/close_obligations.py` |
| m6 | 94 | `src/campaign_b/m6_batch.py` |

**不変条件（全段共通）:** バッチ成果物の `claim_scope` は常に `SCREENING_ONLY`、集約 JSON の `certification_status` は常に `NOT_CERTIFIED`。production paperspace M6 gate（notebook 81）は起動しない。

---

## 1. Notebook 89 — Mass Explore

### 1.1 全体フロー

`run_mass_explore(config_path)`（`mass_explore.py`）が:

1. ベース YAML（通常 `configs/campaign_b_mass_explore.yaml`）を読む。
2. `mass_explore.space_paths` の各 space YAML について wave を起動。
3. 各 wave で **別 `campaign_run_id`** の Campaign B を `run_campaign_b`（`driver.py`）として実行。
4. wave 終了後、その campaign の `queue.json` から `normalized_scheme_key` を収穫し、永続 `seen_normalized_schemes.json` にマージ。
5. `max_waves` まで、または expanded space の追加 wave で novelty が尽きたら終了。

Wave 設定は runtime に書き直される（`never_stop: true`、`stop_after_first_verified_q_lt_1: false`、`inherit_deadline: false` を強制）。

```
{PERSIST}/campaign_b/_mass_explore/
  seen_normalized_schemes.json
  LATEST_MASS_SESSION.json
  MASS-..._summary.json
  runtime/
    wave_00_space.yaml
    wave_00_config.yaml
    exclude_normalized_keys.json
    ...
```

### 1.2 Space の笛卡尔積（候補列挙）

`candidate_generator.generate_campaign_b_queue_candidates` が search-space YAML の軸を **ネストループ** で列挙する。軸（expanded v1 例）:

| 軸 | YAML パス | expanded v1 例 |
|---|---|---|
| `j2` | `staging.j2_values` | `[2, 3, 4]`（`j2>=2` 必須、`forbid_j2_1: true`） |
| `target_rank` | `rank.values` | `[16,25,...,100]`（**降順**で優先） |
| `oversampling` | `rsvd.oversampling` | `[8..32]` |
| `power_iterations` | `rsvd.power_iterations` | `[1..4]` |
| `seed` | `rsvd.seeds` | 複数 |
| `perron_weight_strategy` | `layers.perron_weight_strategy` | `all_ones` 等 |
| `coupling_policy` | `layers.coupling_policy` | `uniform_full` |
| `residual_tolerance` | `residual.tolerances` | `[0.0, 0.01, 0.05]` |
| `residual_norm_model` | `residual.norm_models` | `frobenius`, `spectral` |

各組合せで scheme を組み立て、`effective_projected_rank` を完全平方リフト（M4 互換）で付与。`num_steps=3`、`execution_mode=staged`、`change_class=S2` 固定。

- **exact 重複**・**normalized_scheme_key 重複**・`exclude_normalized_keys`（seen）はスキップ。
- `normalized_scheme_key` は数値に効くフィールドだけをハッシュ（cosmetic 除外）。定義は `candidate_generator.normalized_scheme_key`。

`campaign_b_s2_space_v1.yaml` は狭い軸、`campaign_b_s2_space_expanded_v1.yaml` は広い軸。mass explore はまず v1→expanded の順、その後 novelty があれば expanded を再 wave。

### 1.3 Wave 内ドライバ（`driver.run_campaign_b`）

#### Preflight / manifest

- `campaign_b/{campaign_run_id}/campaign_manifest.json` を作成または resume。
- Shared M2 の `structural_key` / `proof_key` を解決（設定明示、または `m2_compatibility.keys_from_project`）。
- `allow_production_m6` / `allow_campaign_c` / `CERTIFIED` は hard fail → `FAIL_CLOSED`。
- source hash drift は soft（rebind して継続）。

#### 候補キュー

1. 笛卡尔積で候補生成 → `assign_priorities`。
2. `queue.json` に永続化。lease 切れは `PENDING` に戻す。
3. M2 キーを全候補へ rebind；詰まった状態の候補は `PENDING` に戻す。

#### 1 候補の処理パイプライン

状態遷移の実処理:

| 状態 | 処理 | 成果物 |
|---|---|---|
| `SCREENING` | `screening.run_primary_screening` → `m7_lineage.screen_s2_candidate`（FP enclosure、**非証明**） | `candidates/{id}/screening.json` |
| 分類 | `state_machine.classify_q`：`q < 1-margin` → `SCREENED_Q_LT_1`；`|q-1|≤margin` → `BORDERLINE_Q`；それ以外 `Q_GE_1` | — |
| archive | `BORDERLINE_Q` / `Q_GE_1` | `archive/` + 理由コード |
| `M2_RESOLVE` | `lineage.resolve_shared_m2`（registry lookup、なければ `canonical_m2.ensure_canonical_shared_m2`） | binding dict |
| `S0` | screening 結果を M2 binding に結び付けた記録のみ | `candidates/{id}/s0.json` |
| `INDEPENDENT_VERIFY` | **別 Python プロセス**で `run_primary_screening` を再実行；双方 `q<1` かつ atol/rtol で一致 | verify 結果 |
| `PACKAGE_AUDIT` | `build_lineage_package` → `audit_lineage_package` → `hashes.sha256` | `selected/{id}/` |
| `SELECTED` | ledger に記録。`never_stop` なら次候補へ継続 | — |

`never_stop: true`（mass explore 強制）時:

- 最初の SELECTED で止めない。
- wall-clock finalize は `enforce_wall_clock: false` なら無視。
- audit 失敗・例外は原則 archive（`NUMERICAL_INSTABILITY` 等）して継続。
- **例外:** `CERTIFIED` / production M6 / staged-only 不変条件違反は fail-closed。

Shared M2: `shared_m2.j2_max`（mass YAML では 2）が canonical generate の上限。候補の screening `j2`（2/3/4）とは別物——後段 M3 は **親 M2 の `j2_max` に束縛**する（§3）。

### 1.4 `seen_normalized_schemes` / 収穫

- パス: `{PERSIST}/campaign_b/_mass_explore/seen_normalized_schemes.json`
- mass explore は `generate_campaign_b_queue_candidates` をパッチし、`exclude_normalized_keys=seen` を注入。
- wave 後 `harvest_seen_from_campaign` が `queue.json` の全候補 key を union。
- これにより wave 横断で同じ正規化スキームを再スクリーニングしない。

### 1.5 `selected/` 成果物レイアウト（正確）

`lineage.build_lineage_package` + driver 後処理が書くファイル:

```
campaign_b/{campaign_run_id}/selected/{candidate_id}/
  candidate_manifest.json      # 候補全体 + SCREENING_ONLY
  campaign_manifest.json
  scheme.json
  structural_key.json
  proof_key.json
  m2_binding.json              # READY_SHARED 等
  shared_m2_audit.json
  s0_result.json
  independent_verification.json
  package_audit.json
  source_tree_manifest.json
  environment_manifest.json
  README.md                    # NOT_CERTIFIED / SCREENING-ONLY 明示
  hashes.sha256
  COMPLETED.json
```

後段 advance / M3 が追記するファイル（SELECTED 時点では未存在可）:

```
  ADVANCE.json / advance_result.json
  lineage_plan.json
  fixture_residual_result.json   # parent M6 がある場合
  child_run_ids.json
  m3_config_overrides.json
  audits/m2_shared_parent.json   # package-local M2 audit
  GPU_M3.json
  PRE_M6.json
  m4_config_overrides.json
  M6_GATE.json
  M6_STATUS.json
```

---

## 2. Advance — `advance_selected.py`（Notebook 90 / pipeline 段）

### 2.1 発見・ソート

`discover_selected_packages`: `{PERSIST}/campaign_b/*/selected/*` で `candidate_manifest.json` があるディレクトリ（`_*` campaign は除外）。

ソート: discovery / パス順。未 ADVANCE のみ拾い、`max_candidates` で打ち切る（**`q_upper` ソートなし**）。`max_advance<=0` ならスキップ。

### 2.2 `advance_one_selected` の処理

1. 既に `ADVANCE.json` が `LINEAGE_PLANNED` / `FIXTURE_RESIDUAL_DONE` / `READY_FOR_M3` ならスキップ（`force` で上書き可）。
2. `build_s2_lineage_plan` → `lineage_plan.json`（子 run id: M3/M4/M5/M6 等）。
3. Parent M6 package（`final_influence_matrix.json` + `final_bound.json`）があれば `evaluate_s2_fixture_residual` → `fixture_residual_result.json`。失敗は `fixture_error` に記録して継続。
4. `m2_binding.json` が `READY_SHARED` / `READY` なら status = `READY_FOR_M3`；fixture のみなら `FIXTURE_RESIDUAL_DONE`；それ以外 `LINEAGE_PLANNED`。
5. 書くファイル: `ADVANCE.json`、`advance_result.json`、上記 plan/fixture。

セッション台帳: `{PERSIST}/campaign_b/_advance/LATEST_ADVANCE_SESSION.json`。

**この段は CPU のみ。GPU M3 は起動しない。**

---

## 3. M3 — GPU Triad-ATRG（深掘り）

実装の層:

| 層 | モジュール | 役割 |
|---|---|---|
| バッチ / キュー | `gpu_m3_batch.py` | SELECTED パッケージ発見・準備・1 GPU 逐次実行 |
| オーケストレータ | `m3_orchestrator.py` | フェーズキュー・チェックポイント・受入 |
| 作用素 | `linear_operator.py` | ブロック対角 armillary 作用素・matvec |
| RSVD | `rsvd.py` | 乱択 SVD |
| Triad | `triad_atrg.py` | RSVD 因子 → triad 残差 |
| 次元 | `cutoff_dims.py` | `sector_count=(j2+1)^6`、`op_dim=[Σ(j2+1)]^6` |
| 設定 | `m3_config.py` | FP64・CUDA・6h セッション閾値 |

### 3.1 キュー発見とソート（`list_gpu_m3_queue`）

対象条件:

- `ADVANCE.json.status == READY_FOR_M3`、または `m2_binding` が `READY_SHARED` / `READY`。
- `GPU_M3.json.status` が `M3_COMPLETE` / `M3_BLOCKED_BAD_M2` のものは通常除外（`include_complete` で含める可）。

ソートキー（昇順）:

1. `queue_tier` 0: 健全な `M3_RUNNING` / `M3_CHECKPOINT`（**resume のみ優先**）。
2. `queue_tier` 1: 未着手 READY（`GPU_M3` なし）など。
3. `queue_tier` 2: 繰り返し失敗 / エラー（通常はデフォルトキューから除外）。
4. 同一 tier 内は **パス順**（discovery）。**`q_upper` ソートはしない**（drain 運用向け）。

デフォルト除外（`include_errors=False`）:

- `M3_COMPLETE` / `M3_BLOCKED_BAD_M2` / `M3_BLOCKED_NONFINITE`
- `M3_ERROR`
- `consecutive_failures >= 2` の再開候補

NaN/Inf で JSON シリアライズに失敗したセッションは fail-closed で
`M3_BLOCKED_NONFINITE`（`nonfinite_values_present: true`、**CERTIFIED にしない**）。
`--include-errors` / `include_errors=True` で再試行可能。

バッチは `max_sessions` 件だけ **逐次** `run_one_gpu_m3`（単一 GPU 前提）。

**キュー索引（v2）:** `{PERSIST}/campaign_b/_indexes/` に
`gpu_m3` / `pre_m6` / `obligation` / `m6` の JSON（resume / stage tier のみ、通常 **<1 MiB**）。
`max_*=1` では `fetch_limit = min(max_*×8, max_queue)` だけ検証（`MAX_QUEUE=2000` でも 8 件）。
旧 v1 索引は schema mismatch で自動再構築。`VALIDATED_RG_DISABLE_QUEUE_INDEX=1` で全走査に戻す。
ラウンド末 reclaim は preferred が空なら full scan しない（セッション開始の `force_full_scan` のみ）。
pre_m6 は `NEED_M5` を `NEED_M4` より先、あとはパス順。

### 3.2 `prepare_package_for_m3`

パッケージを M3 実行可能な状態にする（orchestrator 起動前）:

1. `m2_binding.json` 必須・`READY_SHARED`。`canonical_run_id` → `{PERSIST}/runs/{m2_id}/`。
2. `reports/M2_acceptance.json` の存在確認。
3. **`j2_max` は候補の screening j2 ではなく、親 M2 run の `run_config.json` / `M2_report.json` から読む**（`_parent_m2_j2_max`）。  
   候補 j2 で上書きすると equivalence gate 不一致で落ちる（コードコメント通りの既知失敗モード）。
4. `_preflight_m2_equivalence`: `M2_EQUIVALENCE.result` の `exact_match_count` が `expected_m2_gate_counts(j2_max)` と一致し、`mismatches==[]`、comparison が許可された exact 証明書文字列であること。不一致 → `M3_BLOCKED_BAD_M2`。
5. `cutoff_dimension_payload(j2)` で `sector_count` / `operator_dimension` を確定。
6. scheme から `target_rank` / `oversampling` / `power_iterations` / `seed`。`1 <= target_rank < op_dim`。
7. `lineage_plan.json` が無ければ再生成。`child_run_ids.json` に M2=親 id、M3=計画 id を書く。
8. `m3_config_overrides.json` を書く。
9. package-local shared M2 audit（`write_package_m2_shared_audit`）が無ければ作成。

`build_m3_config` は overrides + audit から `M3Config` を組み立て、`certification_status='NOT_CERTIFIED'`、`require_cuda=True`。

### 3.3 セッションポリシー（`M3Config` / `SessionGuard`）

| 閾値 | 値 | 意味 |
|---|---|---|
| `checkpoint_interval_s` | 15 min | 周期チェックポイント |
| `no_long_task_after_s` | 5 h | 以降は長い work item 開始抑制 |
| `drain_after_s` | 5 h 15 min | drain |
| `final_save_after_s` | 5 h 20 min | 最終保存開始 |
| `hard_return_s` | 5 h 30 min | 強制 return |
| memory headroom | normal ≥25%、checkpoint ≥35% | 不足なら `BlockedResourceError` |
| OOM | shard を半減、最大 3 retry | RSVD 時 |

数値: FP64 必須、TF32 無効、deterministic algorithms。  
バッチは `VALIDATED_RG_M3_ALLOW_CODE_DRIFT=1` を setdefault（source/notebook hash drift でも resume 可；config_hash と M2 parent ピンは維持）。
あわせて `VALIDATED_RG_M3_CHECKPOINT_KEEP=1` と
`VALIDATED_RG_CHECKPOINT_KEEP=2` を setdefault（検証済み ckpt 直後に
同一 run の古い COMMITTED を prune；CheckpointManager の一時 keep は 2。
Triad/RSVD 同内容 shard は hardlink。詳細は
[campaign_b_m3_storage_reclaim.md](./campaign_b_m3_storage_reclaim.md) の
minimal-storage contract）。

`run_until_checkpoint` は「次の item を安全に始められない」「drain/final/hard return」「全フェーズ完了」のいずれかで戻る。未完了なら `GPU_M3.json` は `M3_CHECKPOINT` → 再実行で resume。

### 3.4 フェーズ一覧（`M3_PHASES`）

キューは content-addressed work item として以下を順に実行。各 item 前後でチェックポイント。成果は:

```
runs/{M3-...}/
  artifacts/{item_id}/attempt_NNN/result.json
  work_items/{item_id}.done          # result_relpath + sha256
  checkpoints/ckpt_XXXXXX/
  cache/contraction_paths.json
  reports/                           # 完了時
  run_config.json
  run_manifest.json
  logs/events.jsonl
```

#### Phase A — `M3_BACKEND_DIAGNOSTIC`

- **入力:** CUDA backend（`require_cuda`）。
- **操作:** FP64 単位行列 `I₄` の matmul プローブ；TF32 が無効であることを記録。
- **出力:** `status=PASS`、memory before/after、backend selection。
- **失敗:** プローブ不一致 → `ArithmeticError`。

#### Phase B — `M3_OPERATOR_BUILD`

- **入力:** 親 M2 checkpoint から復元した projector tensors（`verify_accepted_m2_parent`）。
- **操作:** `build_armillary_operator`（下記 §3.5）。Frobenius ノルムと metadata（sector 数、graph_hash、shard plan）。
- **出力:** dimension、sector_count、operator_frobenius_norm、metadata。
- **失敗:** 欠落/形状不一致/非有限 projector。

#### Phase C — `M3_MATRIX_FREE_VALIDATE`

- **入力:** 同じ作用素。
- **操作:**
  1. 乱数ベクトル `x,y`。
  2. matrix-free `matvec(x)` と **block-explicit CPU reference**（大域 dense 行列を
     立てない）の max abs 誤差 ≤ 1e-12。`j2_max=2` では約 17 GiB の
     `46656²` FP64 割り当てを回避。
  3. 随伴整合: `<Ax,y>` vs `<x,A*y>` 相対誤差 ≤ 1e-12。
  4. path cache が 2 回目 matvec で hit すること。
- **出力メモ:** `reference_mode=block_explicit_no_global_dense`、
  `avoided_dense_matrix_bytes`、`explicit_matrix_bytes=0`。
- **失敗:** いずれか非有限または閾値超過 → fail-closed。

#### Phase D — `M3_RSVD`

- **入力:** 作用素、`target_rank` / `oversampling` / `power_iterations` / `seed`。
- **操作（`randomized_svd`）:**
  1. GPU 上で Gaussian `Ω`（列数 = target_rank + oversampling）。
  2. `Y = A Ω`、power iteration（QR 直交化を挟んだ `(A A*)^q` 風サンプル）。
  3. 左基底から縮小系の SVD → 上位 `target_rank` の `U, Σ, Vᵀ`。
  4. OOM 時は `sectors_per_shard` を下げて再試行（失敗/OOM した operator は
     shard キャッシュから外し、再構築する）。
  5. **検証:** 安全な直交 projector ブロックは rank スペクトル
    （`|weight|` を rank 回）を使い、診断に落ちたブロックだけ明示 SVD に
     fallback。上位 rank と max abs 誤差 ≤ 1e-5；残差 Frobenius が
     「参照最適残差」比 ≤ 1.00001。
- **出力:** singular values、残差、`influence_proxy`（ヒューリスティック）、
  `reference_spectrum_mode`、テンソル `rsvd_*` を checkpoint に保存。
- **厳密性:** `rigor: EXPLORATORY_FIXED_SEED_NOT_A_CERTIFICATE`。証明書に使わない。
- **高速化メモ:** 作用素インスタンスは `sectors_per_shard` ごとにキャッシュし、
  build / validate / RSVD / Triad で再利用。

#### Phase E — `M3_TRIAD`

- **入力:** checkpoint 済み RSVD テンソル + 作用素。
- **操作（`triad_from_rsvd`）:**
  - `left = U`、`core = diag(Σ)`、`right = Vᵀ`。
  - ブロックごとに `‖ wP − U Σ Vᵀ ‖_F` を積み上げて残差。
  - 相対残差 = 残差 / `‖A‖_F`。
- **出力:** `triad_left/core/right` テンソルと summary（`EXPLORATORY_TRIAD_FACTORIZATION_NOT_A_CERTIFICATE`）。
- **役割:** 後段 M4 が固定基底として使う低ランク因子を確定する。ATRG 名だが実装は「RSVD 因子の triad 配置 + 作用素残差」であり、フル ATRG 粗視化ループではない。

#### Phase F — `M3_REPORT` → 受入

- 先行フェーズ成果をロードし、`validate_m3_acceptance`。
- `state.bounds` に明示:
  - matrix-free / adjoint: `FP64_CORE_REPRODUCED`
  - rsvd / triad / influence_proxy: **exploratory / heuristic、境界ではない**
- `phase = M3_COMPLETE`、`write_m3_report_package` → `M3_report.json` / `M3_acceptance.json`。
- 常に `certification_status: NOT_CERTIFIED`。

### 3.5 ブロック対角 / sector-local matvec の意味

`ArmillaryLinearOperator`（`linear_operator.py`）:

- 親 M2 の各 link-star sector ラベル（`all_link_star_keys(j2_max)`、辞書順）ごとに projector 行列 `P_s` がある。
- 大域作用素は **sector ブロック対角**:
  - オフセット `offset_s` にサイズ `∏(j₂,ℓ+1)` のブロック。
  - ブロック値は `w_s · P_s`、`w_s = (1/2)^{Σ j₂}`（`weight_base=0.5`）。
- `matvec` / `matmat` は **ブロック外成分を混ぜない**。各 shard 内で sector ごとに:
  - `result[start:stop] = w * contract(P, x[start:stop])`（随伴なら `Pᵀ`）。
- よって「4D armillary を密行列化せず、不変部分空間ごとに行列フリー適用する」実装。dense `explicit_matrix()` は検証用にブロックを対角配置するだけ。

次元対応（`cutoff_dims.py`）:

| `j2_max` | `sector_count=(j2+1)^6` | `operator_dimension=[Σ(j2+1)]^6` |
|---:|---:|---:|
| 1 | 64 | 729 |
| 2 | 729 | 46656 |
| 3 | 4096 | …（巨大；dense 禁止、matrix-free 必須） |

Campaign B の shared M2 は通常 `j2_max=2` なので M3 も 2 にピンされる。

### 3.6 `GPU_M3.json` と `runs/M3-*`

パッケージ側 `GPU_M3.json`（および `ADVANCE.json` の `gpu_m3_status`）:

| `status` | 意味 |
|---|---|
| `M3_RUNNING` | セッション開始直後 |
| `M3_CHECKPOINT` | 時間切れ/OOM/未完了で return；resume 可 |
| `M3_COMPLETE` | 受入ゲート通過 |
| `M3_BLOCKED_BAD_M2` | equivalence / j2 不整合 |
| `M3_ERROR` | その他例外 |

フィールド例: `m2_run_id`, `m3_run_id`, `phase`, `run_root`, `result`, `error`。

ラン側主要パス:

```
{PERSIST}/runs/M3-.../
  run_config.json              # immutable canonical M3Config
  run_manifest.json            # parent hashes, source/notebook hash, NOT_CERTIFIED
  test_report.json
  checkpoints/ckpt_*/
  artifacts/                   # フェーズ結果
  work_items/*.done
  reports/M3_report.json
  reports/M3_acceptance.json
  reports/code_drift.json      # allow_code_drift resume 時のみ
  cache/contraction_paths.json
  logs/events.jsonl
```

バッチ台帳: `{PERSIST}/campaign_b/_gpu_m3/LATEST_GPU_M3_SESSION.json`。

`create_or_resume_m3` の不変条件: 既存 run の `run_config.json` が変わったら拒否。parent M2 report/acceptance/checkpoint ピン不一致は `M3CompatibilityError`（code drift 緩和フィールドを除く）。

---

## 4. pre_m6 — M4 次いで M5（`pre_m6_batch.py`）

前提: `child_run_ids.json` あり、かつ `GPU_M3.json == M3_COMPLETE` または disk 上で M3 report/acceptance 完了。

キュー優先: `NEED_M5`（M4 完了済）を `NEED_M4` より先。同一 stage 内はパス順（**`q_upper` ソートなし**）。

デフォルト除外（`include_errors=False`、M3 の `M3_BLOCKED_NONFINITE` と同型）:

- `M4_BLOCKED` / `M5_BLOCKED` / `M5_BLOCKED_M4_REGRESSION`
- 例: `M5ParentError: M4 centered finite difference lacks second-order convergence.`
  （`MIN_CENTERED_FD_ACCEPTANCE_ORDER=1.8` の fail-closed；**緩和しない**・**CERTIFIED にしない**）
- ブロック済み head はキューから外れ、次候補が `MAX_PRE_M6=1` でも進む。
- 再試行: `--include-errors` / `include_errors=True`。解除は M4 側を直したうえで
  `PRE_M6.json` の blocked status を消すか上書き（下記 §10）。

### 4.1 M4（`run_m4_session`）

処理概要（`m4_orchestrator` / `M4_PHASES`）:

| フェーズ | 内容 |
|---|---|
| `M4_SOURCE_CHANNELS` | 親 M3 triad から投影 dual（primal + source-class tangents）を構築 |
| `M4_DUAL_PIPELINE` | forward-AD: `dual_matmul` → regroup；CPU/GPU パリティ ≤ 1e-12 |
| `M4_NORMALIZATION` | dual 正規化；非有限なら fail；`normalization_lower_bound_rigorous=False` |
| `M4_FINITE_DIFFERENCE` | AD 接線を有限差分回帰で検証（証明境界ではない） |
| `M4_ERROR_LEDGER` | 基底固定ポリシー下の誤差台帳 |
| `M4_REPORT` | 受入 |

準備:

- `projected_rank = effective_projected_rank(m3_target_rank)`（完全平方）。
- `write_child_m3_acceptance_audit` で **グローバル** `audit/m3_accepted_parent.json` を候補ごとに書換（同時 1 パッケージ制限の理由）。
- `m4_config_overrides.json` をパッケージに保存。

状態は `PRE_M6.json`（`M4_RUNNING` / `M4_COMPLETE` / `M4_CHECKPOINT`）。未完了なら `max_stage_sessions` 回 resume。

### 4.2 M5（`run_m5_session`）

- `write_child_m4_acceptance_audit` → グローバル M4 audit 書換。
- `default_m5_config(..., mode='staged_child')`；`cutoff=j2_max`（親 M3 run_config 優先）、`bond_dimension=projected_rank`。
- `create_or_resume_m5` → `run_until_checkpoint`。
- a posteriori 側の要点（`m5_orchestrator`）:
  - M4 親を検証し、one-step certificate パッケージ組立を試みる。
  - `evaluate_all_obligations` で証明義務を評価 → `M5_obligation_report.json`。
  - 義務が全て閉じかつ組立成功なら `M5_COMPLETE` / 次マイルストーン M6 受入可能；さもなくば義務 open のまま。
- **バッチは M6 を起動しない。** `PRE_M6.json` = `PRE_M6_READY`、`M6_GATE.json` = `BLOCKED_PRE_M6`（後で obligations が開ける）。
- 出力の `certification_status` は強制的に `NOT_CERTIFIED`（M5 が偶発的に別語彙でも）。

台帳: `{PERSIST}/campaign_b/_pre_m6/LATEST_PRE_M6_SESSION.json`。

---

## 5. obligations — `close_obligations.py`

### 5.1 キュー

M4 完了済みで、`M5_obligation_report.json` が無い、または `all_closed` でない / `open_obligations` が残るパッケージ。パス順（**`q_upper` ソートなし**）。索引: `campaign_b/_indexes/obligation_queue.json`。

デフォルト除外（`include_errors=False`、pre_m6 と同型）:

- `M4_BLOCKED`
- `M5_BLOCKED`
- `M5_BLOCKED_M4_REGRESSION`

義務ステージで `M5ParentError` / FD 回帰などが起きたときは `PRE_M6.json` に durable block を書き、索引から外す。`--include-errors` / `include_errors=True` で再試行可能。

### 5.2 再評価（`reevaluate_one`）

1. 再度 `run_m5_session`（staged M5 を回し直し、live 組立と義務評価を更新）。
2. `M5_obligation_report.json` を読む。
3. j2-aware: `evaluate_all_obligations` は親 M3/M4 近傍から `j2_max` を読み、projector の **網羅カバー**（`exhaustive_projector_cover_at_frozen_j2_max`）や表現テール義務などを評価。偽の `CERTIFIED` は作らない；`RIGOROUS` な閉鎖のみカウント。
4. 状態:
   - `OBLIGATIONS_CLOSED_M5_COMPLETE` — `all_closed` かつ `M5_acceptance` PASS/COMPLETE
   - `OBLIGATIONS_CLOSED_AWAITING_ASSEMBLY` — 義務は閉じたが acceptance 未完
   - `OBLIGATIONS_STILL_OPEN`
5. `all_closed && m5_complete` なら `M6_GATE.json` → `READY_FOR_STAGED_M6`。  
   そうでなければ `BLOCKED_PRE_M6`。

台帳: `{PERSIST}/campaign_b/_obligations/LATEST_OBLIGATION_SESSION.json`。

---

## 6. m6 — `m6_batch.py`（`live_parent`）

### 6.1 起動条件（`_m5_ready_for_m6`）

全て満たすこと:

- `M5_acceptance.json`: `phase=M5_COMPLETE`, `status=PASS`, `accepted_for_next_milestone=M6`
- `certification_status` ∈ {`NOT_CERTIFIED`, `ONE_STEP_CERTIFIED`}
- `M5_obligation_report.json`: `all_closed`
- `artifacts/one_step_certificate/` 存在、`verdict.json.independent_verifier == PASS`

### 6.2 実行

- `default_m6_config(..., mode='live_parent')`（**paperspace production gate 81 ではない**）。
- パラメータ: `j2_max` / `bond_dimension` / `num_steps`（scheme または M5/M3 overrides から）。
- `create_or_resume_m6` → parent verify → multi-step certificate パッケージ組立。
- 主に計算する majorant 系（orchestrator / package 経路）:
  - M5 one-step influence majorant を継承した coarse 合成（`composition_policy` に記録）。
  - Perron / outside-tail / residual budget / `z_min` 等を束ねた **有限カットオフ・有限ステップ** の `q_cert_upper` / `q_cert_lower`。
  - 独立検証レポート。

### 6.3 `CERTIFIED` vs `NOT_CERTIFIED`（バッチの扱い）

| orchestrator 結果 | バッチの解釈 |
|---|---|
| `CERTIFIED` | 宣言 majorant が `q_cert < 1` を示した有限証明。`m6_certified_count` に計上。ただし Campaign B の `claim_scope` は **`SCREENING_ONLY` のまま**；continuum / mass-gap 禁止。`campaign_b_note` を付記。 |
| `NOT_CERTIFIED` | majorant が `q_cert<1` を示せなかった **検証済み証明書失敗**。真の RG が拡大的であることの証明ではない。`m6_not_certified_count`。 |
| 例外 / parent 失敗 | `M6_FAILED` / `M6_ERROR`；cert は `NOT_CERTIFIED` |

パッケージ: `M6_STATUS.json`、`M6_GATE.json`（`M6_DONE` / `M6_FAILED`）。  
台帳: `{PERSIST}/campaign_b/_m6/LATEST_M6_SESSION.json`。

---

## 7. pipeline_to_m6 — Notebook 95

`run_pipeline_to_m6` が 1 ラウンドで:

```
advance → gpu_m3 → pre_m6 → obligations → m6
```

をこの順で呼ぶ。`max_rounds` まで繰り返し、**ラウンドの progress 合計が 0 なら早期終了**（スタックしたキューで無限スピンしない）。

### 7.1 progress の定義

| 段 | 加算 |
|---|---|
| advance | `advanced` |
| m3 | `m3_complete` + `m3_checkpoint`（`sessions_ok` は二重計上しない） |
| pre_m6 | `pre_m6_ready` + `m4_checkpoint` |
| obligations | `all_closed_count` |
| m6 | `m6_complete` |

### 7.2 スキップ knobs

`skip_advance` / `skip_m3` / `skip_pre_m6` / `skip_obligations` / `skip_m6`（API；ノート 95 のセルは通常すべて実行）。

### 7.3 Notebook 95 の典型 knobs

| 変数 | デフォルト（ノート） | 意味 |
|---|---|---|
| `MAX_ROUNDS` | 20 | 89 が SELECTED を足し続ける間の吸い上げ回数 |
| `MAX_ADVANCE` | `None` | 全 SELECTED |
| `MAX_M3_SESSIONS` | 16 | ラウンドあたり M3 セッション数 |
| `MAX_PRE_M6_PACKAGES` | 16 | M4/M5 対象パッケージ数 |
| `MAX_STAGE_SESSIONS` | 6 | 1 パッケージあたり M4 resume 回数 |
| `MAX_OBLIGATION_PACKAGES` | 16 | |
| `MAX_M6_PACKAGES` | 16 | |
| `MAX_QUEUE` | 2000 | 各段のキュー上限 |
| `ONLY_CAMPAIGN` | `None` | 特定 `campaign_run_id` に限定可 |

台帳: `{PERSIST}/campaign_b/_pipeline_to_m6/LATEST_PIPELINE_SESSION.json`。

---

## 8. Persist ディレクトリマップ

```
{PERSIST}/                          # 例: /storage/validated_4d_su2_rg
  campaign_b/
    {campaign_run_id}/              # 89 の各 wave
      campaign_manifest.json
      preflight.json
      queue.json
      ledger / events               # QueueStore
      candidates/{id}/screening.json, s0.json
      selected/{id}/                # §1.5
      archive/
    _mass_explore/                  # 89 セッション
    _advance/                       # 90
    _gpu_m3/                        # 91
    _pre_m6/                        # 92
    _obligations/                   # 93
    _m6/                            # 94
    _pipeline_to_m6/                # 95
    RESUME 系ポインタ               # resume_pointer
  runs/
    M2-.../                         # shared / canonical M2
    M3-.../
    M4-.../
    M5-.../
    M6-.../
  LATEST_M3_RUN.json                # 汎用 M3 ポインタ（バッチは child id を優先）
```

プロジェクト側（リポジトリ）のグローバル audit は pre_m6 が候補ごとに上書きする:

```
{PROJECT}/audit/m3_accepted_parent.json
{PROJECT}/audit/m4_accepted_parent.json
```

---

## 9. 認定ルール（fail-closed）

コード上の強制ポイント（抜粋）:

1. **浮動小数の探索結果だけでは `CERTIFIED` を出さない。** M3 RSVD/Triad、M4 AD、screening `estimated_q` はすべて exploratory / screening。
2. **欠測境界・NaN/Inf・非正の正規化境界・未検証残差** → そのマイルストーンは受入失敗または義務 open。
3. Campaign B バッチ JSON は `screening_only_payload()` で常に:
   - `certification_status: NOT_CERTIFIED`
   - `claim_scope: SCREENING_ONLY`
4. M3 フェーズ成果に `certification_status != NOT_CERTIFIED` があれば load 時に reject。
5. M6 `CERTIFIED` が出ても **有限カットオフ・有限ステップ majorant のみ**；バッチは continuum 主張に昇格させない。production gate 81 は別経路。
6. M2 equivalence / parent pin 不一致は M3 を始めない（`M3_BLOCKED_BAD_M2`）。
7. `allow_production_m6` / `allow_campaign_c` / 予期せぬ `CERTIFIED` 語彙 → Campaign B preflight fail-closed。

ヒューリスティックのまま残る境界（証明書に使ってはいけない）:

- screening `q_upper` / fixture residual（親 M6 がある場合の探索）
- M3 `influence_proxy`、RSVD/Triad 残差
- M4 有限差分回帰誤差、`normalization_lower_bound_rigorous=False`
- M5 で閉じなかった義務；閉じても production 81 前は human review

---

## 10. 運用ノート（89 ∥ 95 / 推奨 89 ∥ 97）

- **推奨:** 89∥**97**（95 と同じ思想の改良 consumer）。待ち行列が増えてもよい。
- **並行可:** 89 は SELECTED を生産、95/97 は消費。同一 GPU を 89 が使わない前提（89 は主に CPU screening；canonical M2 生成時は GPU を奪い得る点に注意）。
- **96:** 任意の単一ノート。throttle は必須ではない。**96∥97 同時フル GPU は不可**（GPU lane lease）。
- **バックログ動態:**
  - 89 の wave が SELECTED を増やす → 95/97 の次ラウンド `advance` が拾う。
  - M3 が `M3_CHECKPOINT` で戻ると progress > 0 のためラウンド継続；時間予算で何度も resume。
  - M4 も同様に `m4_checkpoint` が progress に入る。
  - progress=0 で停止したら、89 側の新規 SELECTED 待ちか、全キューが blocked（bad M2 / 義務 open / M6 未準備）か。
- **単一 GPU:** 95/97 内の M3→M4 は逐次。lease は `campaign_b/_locks/gpu_lane.json`。
- **j2 の二層:** screening 候補の `j2∈{2,3,4}` はスキーム探索軸。実行系 M2/M3 のカットオフは shared M2 の `j2_max`（通常 2）に固定。高 j2 候補でも M3 次元は親 M2 に従う。
- **タイミング:** 1 M3 セッションが 6h 枠に収まる設計。j2=2・rank 大の RSVD 実時間はマシン依存 → `[要確認]` Paperspace GPU 実測で更新すること。
- **再実行:** 各段はパッケージ内 status JSON と `runs/` チェックポイントで resume。`force` は advance のみ明示フラグ。
- **pre_m6 poison package:** `PRE_M6.json` が `M4_BLOCKED` / `M5_BLOCKED` /
  `M5_BLOCKED_M4_REGRESSION` の候補はデフォルトキューから除外される（例:
  `B-0b31d2ec0d8be5ce` の FD 2次収束失敗）。放置してよい。再挑戦する場合のみ
  (1) M4 FD / parent を修正し (2) `PRE_M6.json` の blocked status を削除または
  `M4_CHECKPOINT` 等に戻し (3) 必要なら `--include-errors`。FD しきい値は緩めない。

---

## 11. ステータストークン早見

```
# Campaign B candidate
PENDING → SCREENING → SCREENED_Q_LT_1 → M2_RESOLVE → READY_SHARED
  → S0 → INDEPENDENT_VERIFY → PACKAGE_AUDIT → SELECTED
# または ARCHIVED (Q_GE_1 | BORDERLINE_Q | M2_NOT_AVAILABLE | ...)

# ADVANCE.json
LINEAGE_PLANNED | FIXTURE_RESIDUAL_DONE | READY_FOR_M3

# GPU_M3.json
M3_RUNNING | M3_CHECKPOINT | M3_COMPLETE | M3_BLOCKED_BAD_M2 | M3_ERROR

# PRE_M6.json
M4_RUNNING | M4_COMPLETE | M4_CHECKPOINT | M5_RUNNING | PRE_M6_READY
M4_BLOCKED | M5_BLOCKED | M5_BLOCKED_M4_REGRESSION
  # ↑ 3 blocked: default list_pre_m6_queue 除外（include_errors で再試行）

# M6_GATE.json
BLOCKED_PRE_M6 | READY_FOR_STAGED_M6 | M6_DONE | M6_FAILED

# M6_STATUS.json
M6_RUNNING | M6_COMPLETE | M6_FAILED | M6_ERROR

# M6 orchestrator certification_status
NOT_CERTIFIED | CERTIFIED   # 後者でも Campaign B claim_scope は SCREENING_ONLY
```

---

## 参照モジュール一覧

- `src/campaign_b/mass_explore.py`, `driver.py`, `screening.py`, `candidate_generator.py`, `lineage.py`, `independent_verifier.py`, `schemas.py`
- `src/campaign_b/advance_selected.py`, `gpu_m3_batch.py`, `pre_m6_batch.py`, `close_obligations.py`, `m6_batch.py`, `pipeline_to_m6.py`
- `src/m3_orchestrator.py`, `m3_config.py`, `linear_operator.py`, `rsvd.py`, `triad_atrg.py`, `cutoff_dims.py`, `m3_reporting.py`
- `src/m4_orchestrator.py`, `m5_orchestrator.py`, `m5_obligations.py`, `m6_orchestrator.py`
- `configs/campaign_b_mass_explore.yaml`, `configs/campaign_b_s2_space_v1.yaml`, `configs/campaign_b_s2_space_expanded_v1.yaml`
- `notebooks/89_campaign_b_mass_explore.ipynb`, `notebooks/95_campaign_b_pipeline_to_m6.ipynb`
