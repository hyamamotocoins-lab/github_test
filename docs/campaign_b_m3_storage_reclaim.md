# Campaign B — M3 ストレージ最小契約（minimal-storage contract）

Paperspace 永続ディスク上の screening M3 は、パッケージから参照されたまま
`checkpoints/*/tensors/` が ~640 MiB/run 残ることが多い。本ドキュメントは
**何を残し・何を消すか**の契約と、安全な reclaim 実装をまとめる。

実装の単一ソース: `src/campaign_b/m3_reclaim.py`  
CLI（dry-run 既定）: `scripts/persist_reclaim_m3.py`  
自動 strip（notebook 96 / 97）: `run_end_to_end` / `run_pipeline_to_m6` /
`run_post_m2_pipeline`  
即時 strip: `pre_m6_batch.run_m4_session`（`M4_COMPLETE` 直後）

---

## Minimal-storage contract（既定パス）

### 残す（必須・小さい）

| 対象 | 理由 |
|------|------|
| `reports/M3_report.json` / `M3_acceptance.json` / `M3_RECIPE.json` / `M3_RECEIPT.json` / `M3_storage_cleanup.json` | resume 判定・監査・下流 status |
| `run_config.json` / `run_manifest.json` / `work_items/*.done` | 再起動・成果物ピン |
| `artifacts/*/attempt_NNN/result.json`（参照中の最新のみ） | phase 成果（JSON、小さい） |
| `logs/events.jsonl` | 運用証跡 |
| **最新 1 個**の検証済み `checkpoints/ckpt_*`（`COMMITTED` + SHA-256） | M3 resume と **M4 親テンソル**源 |
| `campaign_b/*/selected/*` パッケージ JSON | 候補メタ・ゲート（触らない） |

### 消す / 書かない（既定で積極的）

| 対象 | いつ |
|------|------|
| 古い / 未 COMMITTED `ckpt_*` | 検証済み保存の直後（keep=1）＋ session keep-latest |
| 同一バイトの RSVD/Triad shard 二重書き込み | 保存時に **hardlink 共有**（名前は M4 契約どおり 6 本） |
| obsolete / `.tmp-attempt-*` | 各 checkpoint 後 + M3_COMPLETE |
| `cache/contraction_paths.json`（再生成可） | M3_COMPLETE |
| `checkpoints/` 全体 | **M4_COMPLETE 直後**（即時）および 96/97 session strip |
| 未参照 / archived の run 全体 | CLI `--allow-delete-run` のみ |

### 明示的にやらないこと

- **incomplete** M3 の sole checkpoint は消さない
- M3_COMPLETE 時点では final ckpt を消さない（M4 が RSVD/Triad を読む）
- CERTIFIED / ONE_STEP_CERTIFIED 系統は既定 skip
- 数値規約・認証 fail-closed は変更しない
- parent M2 テンソルを M3 ckpt に複製しない（都度 M2 から検証ロード）

完了後・下流後の理想ディスク像（1 screening パッケージあたり）:

```
runs/M3-.../          # checkpoints/ は M4_COMPLETE 後に STRIPPED マーカーのみ
  reports/*.json      # 小さい receipt / recipe / report
  work_items/ artifacts/ logs/ run_*.json
runs/M4-.../          # M4 自身の ckpt（親テンソルはここへコピー済み）
```

---

## 近期限定 strip（実装済み）

### 1. 現行基準（CLI + 97 共通）

**条件（すべて必須・fail-closed）:**

- `M3_COMPLETE`（`reports/M3_report.json` phase + `M3_acceptance.json` PASS）
- 下流が既に消費済み: パッケージの `child_run_ids` 経由で  
  `M4_COMPLETE` **または** M5 進捗 **または** M6 進捗
- CERTIFIED / ONE_STEP_CERTIFIED の M6 系統ではない（既定で skip）
- `checkpoints/` が存在し、まだ `STRIPPED_FOR_RECLAIM.json` でない

**動作:** `runs/M3-*/checkpoints/` を削除し、マーカー JSON を残す。  
reports / acceptance / config / artifacts / work_items は残す。  
selected パッケージ・レポートは **絶対に削除しない**。

Strip 後、その M3 は **M4 parent resume 源としては使えない**（意図的）。

### 1b. M4_COMPLETE 即時 strip（実装済み・既定 ON）

`run_m4_session` が `M4_COMPLETE` になった瞬間に
`strip_m3_after_m4_complete(m3_run_id)` を呼ぶ。  
M4 は親検証を完了し自前 ckpt にテンソルを持つため、M3 ckpt を待たせず解放する。  
結果は `PRE_M6.json` の `m3_reclaim_after_m4` に記録。

### 2. 96 / 97 内自動 strip（実装済み・既定 ON）

`auto_strip_m3_checkpoints=True`（notebook 96 `run_end_to_end` /
notebook 97 / `run_pipeline_to_m6` 既定）のとき:

1. **セッション開始時にフル安全スキャンを 1 回**（`force_full_scan=True`）  
   → 既存の COMPLETE+downstream バックログを strip
2. 各ラウンドの **pre_m6 / m6 結果**から増分 strip  
3. `persist_m3_cap_gib`（既定 **32.0**）が設定されていれば、strip 後に
   `runs/M3-*` 合計がキャップを超える限り最古 eligible を追加 strip

手動一括回収:

```bash
export VALIDATED_RG_PERSIST_ROOT=/storage/validated_4d_su2_rg
python scripts/persist_reclaim_m3.py --mode strip-checkpoints          # dry-run
python scripts/persist_reclaim_m3.py --mode strip-checkpoints --execute
python scripts/persist_reclaim_m3.py --mode strip-tensors --execute
python scripts/persist_reclaim_m3.py --mode keep-latest-checkpoint --execute
```

### 3. keep-latest（進行中 M3・**96 / 97 ホットパス既定 ON**）

- 最新しい `COMMITTED` `ckpt_*` だけ残し、古い / 未 COMMITTED を削る
- **quota 逼迫のため notebook 96 / 97 / `gpu_m3_batch` 既定 ON**
  （`auto_keep_latest_m3_checkpoint=True`）
- **セッション開始時に全 `runs/M3-*` へフル keep-latest**
- 各 M3 セッションの resume 前・セッション後に、その run だけ trim
- CLI: `--mode keep-latest-checkpoint`

#### 3b. M3 orchestrator 内 prune + mid-run cleanup（実装済み）

| env | 既定 | 意味 |
|-----|------|------|
| `VALIDATED_RG_M3_CHECKPOINT_KEEP` | `1`（範囲 1–8） | 最新 N 個だけ残す |
| `VALIDATED_RG_CHECKPOINT_KEEP` | `2`（gpu_m3_batch setdefault） | CheckpointManager の一時 keep（下限 2） |

検証済み ckpt 保存の直後に:

1. keep=1 prune（symlink は追わない）
2. obsolete / tmp attempt 削除
3. （完了時）`cache/` 削除 + `reports/M3_RECEIPT.json` 書き込み

同一バイトの Triad/RSVD shard は `tensors/` 内で hardlink 共有
（`storage_dedup.json` に会計。symlink は使わない）。

**最新 final ckpt は M4 親として残す**。下流 `M4_COMPLETE` で即 strip（§1b）。

### 4. tensors のみ削除（実装済み）

- 同一 fail-closed 基準で `checkpoints/*/tensors/` のみ削除
- マーカー: `STRIPPED_TENSORS_FOR_RECLAIM.json`
- CLI: `--mode strip-tensors`

### 5. persist size-cap（実装済み）

- `persist_m3_cap_gib`（既定 **32.0**; `None` で無効）
- `enforce_persist_m3_cap`: `runs/M3-*` 合計がキャップ超過のあいだ、
  最古の COMPLETE+downstream を strip-checkpoints

### 6. 絶対に strip / 削除しないもの

| 対象 | 理由 |
|------|------|
| CERTIFIED / ONE_STEP_CERTIFIED 系統の M3 | 監査・再現のため保持 |
| incomplete M3（keep-latest 以外） | まだ resume が必要 |
| 下流未完了（no M4_COMPLETE/M5/M6） | まだ live M4 parent |
| `campaign_b/*/selected/*` パッケージ | 候補メタ・ゲートの durable state |
| reports / acceptance / receipt / recipe | 監査証跡 |

---

## 中期: recipe / receipt（stub + receipt 実装済み）

| フィールド | 意味 |
|------------|------|
| `m3_execution_key` | 既存 execution key |
| `m2_hash` / M2 run id | 親 M2 |
| `target_rank` / `seed` / `weight_strategy` / `sector_ordering` | 再生成ヒント |
| `M3_RECEIPT.json` | final ckpt 名・サイズ・tensor 名・dedup 会計 |

**tensors は M3_COMPLETE では削除しない**（M4 が必要）。  
`M4_COMPLETE` 直後の即時 strip と session reclaim で解放。

---

## Paperspace: 既存 fat tree の一括回収 + 96 / 97

1. `git pull` 最新 `main`
2. **緊急一括（推奨・dry-run 後に execute）:**

   ```bash
   export VALIDATED_RG_PERSIST_ROOT=/storage/validated_4d_su2_rg
   python scripts/persist_reclaim_m3.py --mode keep-latest-checkpoint --execute
   python scripts/persist_reclaim_m3.py --mode strip-checkpoints --execute
   ```

3. notebook **97** またはスタンドアロン **96**（同時起動しない）を再実行。  
   セッション開始で strip フルスキャン + keep-latest フルスキャン。
4. セル knobs:

   ```python
   AUTO_STRIP_M3_CHECKPOINTS = True
   AUTO_KEEP_LATEST_M3_CHECKPOINT = True
   PERSIST_M3_CAP_GIB = 32.0              # None で無効
   ```

5. 要約フィールド: `m3_reclaim.stripped` / `bytes_freed_human` /
   `session_start_keep_latest` / `keep_latest_bytes_freed_human`  
   M4 完了セッション: `m3_reclaim_after_m4`

### 安全メモ

- 96 / 97 / 即時 strip は **既存 reclaim 基準を満たすものだけ**削除（fail-closed）
- CLI の既定は dry-run；`--execute` が必要
- keep-latest は incomplete M3 にも効く（最新 COMMITTED のみ残す）

---

## Dry-run 実績メモ（参考）

ある時点の Paperspace スキャン例:

- strip 可能: ~78 runs ≈ 48.8 GiB（M3_COMPLETE + downstream）
- skip: ~160（`no_downstream` など）

97 を回しながら下流が進むたびに消化済み分が空く。  
`M4_COMPLETE` 即時 strip で、ラウンド末を待たず解放される。
