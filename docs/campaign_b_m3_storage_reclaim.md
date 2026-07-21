# Campaign B — M3 ストレージ回収戦略

Paperspace 永続ディスク上の screening M3 は、パッケージから参照されたまま
`checkpoints/*/tensors/` が ~640 MiB/run 残ることが多い。
未参照 run 削除だけでは ~1 GiB しか空かない。本ドキュメントは
**安全な strip** と、中期の **recipe 化** の方針をまとめる。

実装の単一ソース: `src/campaign_b/m3_reclaim.py`  
CLI（dry-run 既定）: `scripts/persist_reclaim_m3.py`  
自動 strip（notebook 97）: `run_pipeline_to_m6` / `run_post_m2_pipeline`

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

### 2. 97 内インクリメンタル（実装済み・既定 ON）

`auto_strip_m3_checkpoints=True`（notebook 97 / `run_pipeline_to_m6` 既定）のとき:

- 各ラウンドの **pre_m6 / m6 結果**から、`PRE_M6_READY` / `M4_COMPLETE` /
  `M6_COMPLETE` になったパッケージの M3 run_id を収集
- それらに対して上記基準で即 strip（成功した M4 パッケージ完了直後）
- 候補 ID が取れないラウンドは **1 回だけフル安全スキャン**にフォールバック

手動一括回収:

```bash
export VALIDATED_RG_PERSIST_ROOT=/storage/validated_4d_su2_rg
python scripts/persist_reclaim_m3.py --mode strip-checkpoints          # dry-run
python scripts/persist_reclaim_m3.py --mode strip-checkpoints --execute
```

### 3. keep-latest（長期 M3 中・任意）

- 最新しい `COMMITTED` `ckpt_*` だけ残し、古い / 未 COMMITTED を削る
- **97 ホットパスの既定には入れない**（resume 中の M3 を誤って薄くしない）
- CLI: `--mode keep-latest-checkpoint`

### 4. 絶対に strip / 削除しないもの

| 対象 | 理由 |
|------|------|
| CERTIFIED / ONE_STEP_CERTIFIED 系統の M3 | 監査・再現のため保持 |
| incomplete M3 | まだ resume が必要 |
| 下流未完了（no M4_COMPLETE/M5/M6） | まだ live M4 parent |
| `campaign_b/*/selected/*` パッケージ | 候補メタ・ゲートの durablestate |
| reports / acceptance | 監査証跡 |

---

## 中期: recipe 化（設計・stub）

巨大テンソルを永続に置き続ける代わりに、**再生成可能なレシピ**を残す。

### 保存すべき recipe フィールド（案）

| フィールド | 意味 |
|------------|------|
| `m3_execution_key` | 既存 execution key（内容アドレス） |
| `m2_hash` / M2 run id | 親 M2（shared / canonical） |
| `target_rank` | Triad 目標ランク |
| `weight_strategy` | 重み戦略識別子 |
| `backend` | CPU/GPU・dtype・コード版ハッシュ |
| `sector_ordering` | セクター順（規約固定） |
| `seed` | 乱数シード（あれば） |

### `projector_exact` / shared M3

- 因子が安い再計算で足りる経路では **642 MiB tensors を保持しない**
- recipe + M2 があれば factors を再生成 → 必要時だけ materialize
- **実装ステータス:** 設計のみ。現状は下流消費後に strip する運用で代用

### 移行パス

1. いま: strip after downstream（本ドキュメント近期）
2. 次: M3 完了時に recipe JSON を必須出力（tensors と併記）
3. その後: shared / projector_exact 系は tensors を短寿命化し、recipe のみ永続

---

## 追加 strip アイデア（文書化 / 容易なら実装）

| アイデア | 状態 | 備考 |
|----------|------|------|
| tensors のみ削除（`checkpoints/*/tensors`）、LATEST メタ残す | **未実装** | strip より弱い。resume 不能は同じ。メタ監査用に検討可 |
| PRE_M6_READY / M6_COMPLETE での strip | **現行基準に含まれる** | M4_COMPLETE 以降なら strip 可。97 は PRE_M6_READY で増分候補化 |
| persist 木 > N GiB で最古 COMPLETE+downstream を自動 strip | **未実装** | キャップ運用。N は Paperspace 枠に合わせて設定。実装時は fail-closed 同一基準 |
| delete-run（未参照 / archived のみ） | CLI のみ | `--allow-delete-run` 必須。パッケージは消さない |

---

## Paperspace: notebook 97 の自動掃除

1. `git pull` 最新 `main`（本機能含む）
2. notebook 97 を開く。セル 2 で:

   ```python
   AUTO_STRIP_M3_CHECKPOINTS = True  # 既定 ON
   ```

3. セル 3 の `run_post_m2_pipeline(..., auto_strip_m3_checkpoints=AUTO_STRIP_M3_CHECKPOINTS)`
4. セッション要約 / ledger `campaign_b/_post_m2/LATEST_POST_M2_SESSION.json` に:

   - `auto_strip_m3_checkpoints`
   - `m3_reclaim.stripped` / `m3_reclaim.bytes_freed_human`

5. オフにしたいときだけ `AUTO_STRIP_M3_CHECKPOINTS = False`  
   （ディスク逼迫時は ON 推奨。dry-run 一括確認は CLI）

### 安全メモ

- 97 は **既存 reclaim 基準を満たすものだけ**削除する（fail-closed）
- CLI の既定は引き続き dry-run；`--execute` が必要
- CERTIFIED 系統・未完了 M3・下流未到達は触らない

---

## Dry-run 実績メモ（参考）

ある時点の Paperspace スキャン例:

- strip 可能: ~78 runs ≈ 48.8 GiB（M3_COMPLETE + downstream）
- skip: ~160（`no_downstream` など）

97 を回しながら下流が進むたびに、上記のうち消化済み分が自動で空いていく。
