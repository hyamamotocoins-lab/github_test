# Campaign B 統合 end-to-end 設計（Notebook 96）

## 1. 結論

Paperspace 上の **89 / 95 二重ループ**を、単一ノート **96** の
**backlog-aware scheduler** に置き換える。

| ノート | 役割 |
|---|---|
| **96** | 統合スケジューラ（本設計・Phase 1） |
| 97 | 分割レーンの post-M2 pipeline（補完オプション） |
| 98 | 読み取り専用ダッシュボード（補完オプション） |
| 99 | M6 `CERTIFIED` 永続カタログ |

関連ドキュメント:

- [campaign_b_parallel_split_design.md](./campaign_b_parallel_split_design.md)
  — 96/97/98 分割レーン案（Downloads 由来）。**Phase 1 は本設計の単一ノート 96 を採用。**
  分割レーンは将来オプションとして残す。
- [campaign_b_pipeline_89_95.md](./campaign_b_pipeline_89_95.md)
  — 現行 89/95 の処理仕様（参照用）。

## 2. Phase 1 スコープ

- **M3 数学・backend は変更しない**（既存 `run_gpu_m3_batch` 等を再利用）。
- `m3_backend` フィールドは YAML に `legacy_rsvd` デフォルトで置けるが **Phase 2 まで無視**。
- セッション壁時計: `VALIDATED_RG_DISABLE_SESSION_WALLCLOCK=1`
  （`src/session_guard.py`）。アイテム級 checkpoint / fail-closed は維持。
- 永続スナップショット: `{PERSIST}/campaign_b/_end_to_end/`。

## 3. ループ順序

各ラウンド:

1. **M3 / downstream を先に実行**  
   `run_gpu_m3_batch` → `run_pre_m6_batch` → `run_close_obligations_batch` → `run_m6_batch`
2. **バックログゲート**  
   `len(list_gpu_m3_queue(...)) < selected_backlog_target`（既定 8）のときだけ  
   screening chunk + `run_advance_selected`
3. progress == 0 なら停止（idle）

### progress の定義（完了のみ）

次の合計。**`m3_checkpoint` 単独では progress に含めない。**

- `m3_complete`
- `pre_m6_ready`
- obligations closed（`all_closed_count`）
- `m6_complete`
- `advanced`
- screening で得た `selected`（wave の selected_count）

（95 の pipeline は resume 用に checkpoint も progress 扱いだが、96 Phase 1 は完了ベース。）

## 4. Screening chunk

`configs/campaign_b_mass_explore.yaml` を基に runtime YAML を書き:

- `mass_explore.candidates_per_wave` / `candidate_limit` ≈ 32
- `mass_explore.max_waves: 1`
- `persistent_root` を実行時パスに固定

その後 `run_mass_explore` を呼ぶ（1 wave）。

## 5. 設定キー（YAML）

```yaml
selected_backlog_target: 8
screening_chunk_size: 32
max_rounds: 100
max_m3_sessions: 8
max_pre_m6_packages: 8
max_stage_sessions: 6
max_obligation_packages: 8
max_m6_packages: 8
max_queue: 500
m3_backend: legacy_rsvd   # Phase 2 まで無視
mass_explore_config: campaign_b_mass_explore.yaml
disable_session_wallclock: true
```

## 6. 回復・実行キー（最小）

- `pipeline_recovery.recover_interrupted_work` — `*.tmp` 掃除、stale lease ディレクトリ stub
- `execution_keys` — Phase 2+ 用 stub（GPU lock / shared M3 key）

## 7. 不変条件

- Campaign B 集約の `claim_scope` は `SCREENING_ONLY`
- 集約 JSON の `certification_status` は `NOT_CERTIFIED`（オーケストレータが
  M6 で `CERTIFIED` を出しても continuum 主張禁止）
- production paperspace gate 81 は起動しない
- CERTIFIED を捏造しない（検出は notebook 99）

## 8. Paperspace 運用（Phase 1）

1. main を pull
2. Notebook **96** を実行（CUDA 必須）
3. 必要なら **98** で状態確認、**99** で CERTIFIED カタログ更新
4. 分割レーン（97）は M2 並行構築が必要なときのみ

## 9. 後続フェーズ（非 Phase 1）

- Phase 2: `m3_backend` 切替、execution_keys 実体化
- 分割レーン全面採用時は parallel_split 設計の Lane A–D へ移行
