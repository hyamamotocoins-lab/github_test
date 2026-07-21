# Campaign B 統合 end-to-end 設計（Notebook 96）

## 1. 結論

Notebook **96** は **任意の単一ノート統合スケジューラ**（backlog-aware）。
**推奨の運用は 89∥97**（旧 89∥95 と同じ producer/consumer）。待ち行列が増えてもよい。
96 の backlog throttle を主運用にしない。

| ノート | 役割 |
|---|---|
| **89** | screening / mass-explore **producer**（推奨） |
| **97** | post-M2 **consumer**（95 相当；推奨、89 と並走可） |
| **96** | 統合スケジューラ（**任意**；単独利用 OK） |
| **98** | 読み取り専用ダッシュボード |
| **99** | M6 `CERTIFIED` 永続カタログ |

関連ドキュメント:

- [campaign_b_parallel_split_design.md](./campaign_b_parallel_split_design.md)
  — 89∥97 推奨と GPU lane lease。
- [campaign_b_pipeline_89_95.md](./campaign_b_pipeline_89_95.md)
  — 89/95 処理仕様（97 が 95 後継 consumer）。

## 2. Phase 1 スコープ（96 実装）

- **M3 数学・backend は変更しない**（既存 `run_gpu_m3_batch` 等を再利用）。
- `m3_backend` フィールドは YAML に `legacy_rsvd` デフォルトで置けるが **Phase 2 まで無視**。
- セッション壁時計: `VALIDATED_RG_DISABLE_SESSION_WALLCLOCK=1`
  （`src/session_guard.py`）。アイテム級 checkpoint / fail-closed は維持。
- 永続スナップショット: `{PERSIST}/campaign_b/_end_to_end/`。
- GPU lane lease: `{PERSIST}/campaign_b/_locks/gpu_lane.json`
  （96 / 97 / `pipeline_to_m6` が取得。別プロセスが保持中なら fail closed）。

## 3. ループ順序

各ラウンド:

1. **M3 / downstream を先に実行**  
   `run_gpu_m3_batch` → `run_pre_m6_batch` → `run_close_obligations_batch` → `run_m6_batch`
2. **バックログゲート（96 専用オプション）**  
   `len(list_gpu_m3_queue(...)) < selected_backlog_target`（既定 8）のときだけ  
   screening chunk + `run_advance_selected`  
   ※ 89∥97 運用ではこの throttle は使わない（backlog 増は許容）。
3. progress == 0 なら停止（idle）

### progress の定義（完了のみ）

次の合計。**`m3_checkpoint` 単独では progress に含めない。**

- `m3_complete`
- `pre_m6_ready`
- obligations closed（`all_closed_count`）
- `m6_complete`
- `advanced`
- screening で得た `selected`（wave の selected_count）

（95/97 の pipeline は resume 用に checkpoint も progress 扱いだが、96 は完了ベース。）

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

## 6. 回復・GPU lane lease

- `pipeline_recovery.recover_interrupted_work` — `*.tmp` 掃除、stale lease ディレクトリ stub
- `execution_keys` — **排他 GPU lease**（heartbeat + pid/hostname）。
  死んだ PID または古い heartbeat は acquire 時に reclaim。
  生存プロセスの lease は奪わない（`GpuLaneHeldError`）。

## 7. 不変条件

- Campaign B 集約の `claim_scope` は `SCREENING_ONLY`
- 集約 JSON の `certification_status` は `NOT_CERTIFIED`（オーケストレータが
  M6 で `CERTIFIED` を出しても continuum 主張禁止）
- production paperspace gate 81 は起動しない
- CERTIFIED を捏造しない（検出は notebook 99）
- **96∥97 をフル GPU consumer として同時起動しない**（lease が第二側を fail closed）

## 8. Paperspace 運用（推奨）

1. main を pull
2. **推奨:** Notebook **89**（CPU/screening）∥ **97**（CUDA consumer）
3. 待ち行列（SELECTED / READY_FOR_M3）が増えてもよい。M2 が先に終わってもよい。
4. **任意:** 単独で **96** を使う（統合スケジューラ）。throttle は必須ではない。
5. **98** で状態確認、**99** で CERTIFIED カタログ更新
6. 誤って 96 と 97 を両方フル起動した場合、後から lease を取れない側が明確にエラーする

## 9. 後続フェーズ

- Phase 2: `m3_backend` 切替、execution key の更なる細分化
- 分割レーン全面採用時は parallel_split 設計の Lane A–D へ
