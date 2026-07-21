# Campaign B — M3 ストレージ回収戦略

Paperspace 永続ディスク上の screening M3 は、パッケージから参照されたまま
`checkpoints/*/tensors/` が ~640 MiB/run 残ることが多い。
未参照 run 削除だけでは ~1 GiB しか空かない。本ドキュメントは
**安全な strip** と、中期の **recipe 化** の方針をまとめる。

実装の単一ソース: `src/campaign_b/m3_reclaim.py`  
CLI（dry-run 既定）: `scripts/persist_reclaim_m3.py`  
自動 strip（notebook 96 / 97）: `run_end_to_end` / `run_pipeline_to_m6` /
`run_post_m2_pipeline`

---

## 近期限定 strip（いま実装してよいもの）

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

### 2. 96 / 97 内自動 strip（実装済み・既定 ON）

`auto_strip_m3_checkpoints=True`（notebook 96 `run_end_to_end` /
notebook 97 / `run_pipeline_to_m6` 既定）のとき:

1. **セッション開始時にフル安全スキャンを 1 回**（`force_full_scan=True`）  
   → 既存の COMPLETE+downstream バックログ（例: ~48 GiB）を、このラウンドで
   PRE_M6 が進まなくても strip する
2. 各ラウンドの **pre_m6 / m6 結果**から増分 strip  
   （候補 ID が無いときだけフルスキャンにフォールバック）
3. `persist_m3_cap_gib`（既定 **80.0**）が設定されていれば、strip 後に
   `runs/M3-*` 合計がキャップを超える限り最古 eligible を追加 strip

**なぜ以前 `stripped=0` になり得たか:** 増分 strip は
`PRE_M6_READY` / `M4_COMPLETE` / `M6` のパッケージ ID に依存する。
セッションが `m3_complete:0, pre_m6_ready:0` のまま中途 M3 ばかり進むと、
増分候補が空（または未 eligible）で、セッション開始フルスキャンが無い版では
既存の ~78 eligible が触れられない。セッション開始フルスキャンで修正済み。

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
- **セッション開始時に全 `runs/M3-*` へフル keep-latest**（このバッチで触れない
  古い M3 の ckpt 積み上げも回収。`m3_reclaim.session_start_keep_latest`）
- 各 M3 セッションの resume 前・セッション後に、その run だけ trim（再蓄積防止）
- CLI: `--mode keep-latest-checkpoint`（一括・緊急用。96/97 再実行後は通常不要）

### 4. tensors のみ削除（実装済み）

- 同一 fail-closed 基準で `checkpoints/*/tensors/`（および類似 bulky 名）のみ削除
- チェックポイントメタ / LATEST は残す
- マーカー: `STRIPPED_TENSORS_FOR_RECLAIM.json`
- CLI: `--mode strip-tensors`

### 5. persist size-cap（実装済み）

- `persist_m3_cap_gib`（既定 80.0; `None` で無効）
- `enforce_persist_m3_cap`: `runs/M3-*` 合計がキャップ超過のあいだ、
  最古の COMPLETE+downstream を strip-checkpoints
- 97 / pipeline / 96 (`run_end_to_end`) から strip 後に自動呼び出し

### 6. 絶対に strip / 削除しないもの

| 対象 | 理由 |
|------|------|
| CERTIFIED / ONE_STEP_CERTIFIED 系統の M3 | 監査・再現のため保持 |
| incomplete M3（keep-latest 以外） | まだ resume が必要（ただし ckpt 積み上げは keep-latest で抑制） |
| 下流未完了（no M4_COMPLETE/M5/M6） | まだ live M4 parent |
| `campaign_b/*/selected/*` パッケージ | 候補メタ・ゲートの durablestate |
| reports / acceptance | 監査証跡 |

---

## 中期: recipe 化（stub 実装済み）

巨大テンソルを永続に置き続ける代わりに、**再生成可能なレシピ**を残す。

### 保存すべき recipe フィールド

| フィールド | 意味 |
|------------|------|
| `m3_execution_key` | 既存 execution key（内容アドレス / stub） |
| `m2_hash` / M2 run id | 親 M2（shared / canonical） |
| `target_rank` | Triad 目標ランク |
| `weight_strategy` | 重み戦略識別子 |
| `backend` | `legacy_rsvd`（現状） |
| `sector_ordering` | セクター順（規約固定） |
| `seed` | 乱数シード（あれば） |

**実装:** `M3_COMPLETE` 時に `runs/M3-.../reports/M3_RECIPE.json` を書き、
パッケージへ `m3_recipe.json` をコピー。**tensors は M3_COMPLETE では削除しない**
（M4 がまだ必要）。下流消費後の strip と併用。

### `projector_exact` / shared M3

- 因子が安い再計算で足りる経路では **642 MiB tensors を保持しない**
- recipe + M2 があれば factors を再生成 → 必要時だけ materialize
- **実装ステータス:** 設計のみ。現状は下流消費後に strip する運用 + recipe stub

### 移行パス

1. いま: strip after downstream + session-start full strip + session-start full keep-latest + per-session keep-latest（本ドキュメント近期）
2. 次: M3 完了時に recipe JSON を必須出力（tensors と併記）← **stub 実装済み**
3. その後: shared / projector_exact 系は tensors を短寿命化し、recipe のみ永続

---

## 追加 strip アイデア（文書化）

| アイデア | 状態 | 備考 |
|----------|------|------|
| tensors のみ削除（`checkpoints/*/tensors`）、LATEST メタ残す | **実装済み** | `--mode strip-tensors` |
| PRE_M6_READY / M6_COMPLETE での strip | **現行基準に含まれる** | M4_COMPLETE 以降なら strip 可。97 は PRE_M6_READY で増分候補化 |
| persist 木 > N GiB で最古 COMPLETE+downstream を自動 strip | **実装済み** | `persist_m3_cap_gib`（既定 80） |
| delete-run（未参照 / archived のみ） | CLI のみ | `--allow-delete-run` 必須。パッケージは消さない |
| keep-latest on hot path | **実装済み（96 / 97 既定 ON）** | session-start 全 M3 + 進行中 M3 の ckpt 積み上げ抑制 |

---

## Paperspace: notebook 96 / 97 の自動掃除

1. `git pull` 最新 `main`（本機能含む）
2. **推奨:** notebook **97**（89 と並走）を再実行。または **スタンドアロンで 96 だけ**
   （**97 と同時起動しない**）。セッション開始時に **strip フルスキャン + keep-latest
   フルスキャン**が走り、dry-run で見えていた ~GiB 級の古い ckpt も回収される
   （別途 CLI は不要。緊急で今すぐ空けたいときだけ下の CLI）。
3. 緊急回収（再実行前）:

   ```bash
   export VALIDATED_RG_PERSIST_ROOT=/storage/validated_4d_su2_rg
   python scripts/persist_reclaim_m3.py --mode strip-checkpoints --execute
   python scripts/persist_reclaim_m3.py --mode keep-latest-checkpoint --execute
   ```

4. notebook 97 を開く。セル 2 で（96 も同名 knobs）:

   ```python
   AUTO_STRIP_M3_CHECKPOINTS = True       # 既定 ON（session-start フルスキャン含む）
   AUTO_KEEP_LATEST_M3_CHECKPOINT = True  # 既定 ON（session-start 全 M3 + per-session）
   PERSIST_M3_CAP_GIB = 80.0              # None で無効
   ```

5. セル 3:
   - 97: `run_post_m2_pipeline(..., auto_strip_m3_checkpoints=..., auto_keep_latest_m3_checkpoint=..., persist_m3_cap_gib=...)`
   - 96: `CFG.auto_strip_...` をセットして `run_end_to_end(CFG)`
6. セッション要約 / ledger:
   - 97: `campaign_b/_post_m2/LATEST_POST_M2_SESSION.json`
   - 96: `campaign_b/_end_to_end/LATEST_END_TO_END_SESSION.json`

   共通フィールド:

   - `auto_strip_m3_checkpoints` / `auto_keep_latest_m3_checkpoint` / `persist_m3_cap_gib`
   - `m3_reclaim.stripped` / `m3_reclaim.bytes_freed_human`
   - `m3_reclaim.session_start_full_scan`
   - `m3_reclaim.session_start_keep_latest`（trimmed / bytes_freed）
   - `m3_reclaim.keep_latest_bytes_freed_human`

7. オフにしたいときだけ各 flag を False / None  
   （ディスク逼迫時は ON 推奨。dry-run 一括確認は CLI）

### 安全メモ

- 96 / 97 は **既存 reclaim 基準を満たすものだけ**削除する（fail-closed）
- CLI の既定は引き続き dry-run；`--execute` が必要
- CERTIFIED 系統・未完了 M3（strip 対象）・下流未到達は strip しない
- keep-latest は incomplete M3 にも効く（最新 COMMITTED のみ残す）

---

## Dry-run 実績メモ（参考）

ある時点の Paperspace スキャン例:

- strip 可能: ~78 runs ≈ 48.8 GiB（M3_COMPLETE + downstream）
- skip: ~160（`no_downstream` など）

97 を回しながら下流が進むたびに、上記のうち消化済み分が自動で空いていく。
セッション開始フルスキャンで、PRE_M6 が進まないセッションでもバックログは回収される。
