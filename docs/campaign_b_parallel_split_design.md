# Campaign B 分割・並行実行設計

> **番号の対応（2026-07 更新）**  
> Downloads 原案では `96=M2 builder / 97=post-M2 / 98=dashboard` だった。  
> 現行リポジトリの番号付けは次のとおり:
>
> | ノート | 役割 |
> |---|---|
> | **89** | mass-explore **producer**（推奨・並走可） |
> | **97** | post-M2 **consumer**（95 相当；**推奨**、89∥97） |
> | **96** | 統合 backlog-aware end-to-end（**任意**の単一ノート） |
> | **98** | 読み取り専用 status dashboard |
> | **99** | M6 CERTIFIED 永続カタログ |
>
> 原案の「96 = M2 shared build」は **未採番の将来レーン**（`m2_shared_batch`）として残す。  
> **推奨 ops = 89∥97**（旧 89∥95）。待ち行列が増えてもよい。96 の throttle を主運用にしない。

## 1. 結論

**推奨:** Notebook **89**（producer）∥ **97**（consumer）。  
backlog（SELECTED / READY_FOR_M3）の成長は許容する。M2 が先に終わってもよい。

安全なレーン構成:

1. （将来）canonical shared M2 builder
   - sector 単位で checkpoint する。
   - CPU process pool を使用する。
2. `97_campaign_b_post_m2_pipeline.ipynb`
   - **デフォルトは 95 相当の消化レーン**: 既に溜まっている
     SELECTED / `READY_FOR_M3` / `m2_binding` READY を
     `advance → M3 → M6`（`run_pipeline_to_m6`）で消費する。
     `drain_existing_backlog=True` / `skip_screening=True`。
   - `M2_READY.json` は検出のみ（待機ゲートではない）。
   - opt-in: `drain_existing_backlog=False` で Phase-1 end_to_end
     （screening 可）。
   - **排他 GPU lane lease**（`campaign_b/_locks/gpu_lane.json`）。
3. `96_campaign_b_end_to_end.ipynb`（**任意**）
   - 単独利用 OK。backlog throttle は効率化のオプションであり必須ではない。
   - **96∥97 をフル GPU consumer として同時起動しない**（lease が第二側を fail closed）。
4. `98_campaign_b_status_dashboard.ipynb`
   - 読み取り専用。lock / status を変更しない。

## 2. 並行実行可能性

### 2.1 並行可能

- M2 sector 計算と Campaign B screening
- screening と M3
- screening と M5 / obligations / M6
- M2 の異なる sector
- M3 完了候補に対する CPU 後処理と、別候補の GPU M3

### 2.2 直列にすべき箇所

- 同一 candidate の M2 → M3
- 単一 GPU 上の M3 と GPU 使用 M4
- 同一 run の同一 work item
- 現行の global audit ファイルを書き換える M4/M5

現行仕様では `audit/m3_accepted_parent.json` と
`audit/m4_accepted_parent.json` が候補ごとに上書きされるため、
M4/M5 は候補間で安全に並行化できない。並行化するには、
config に package-local audit path を渡し、global mutable audit を廃止する。

## 3. 実行レーン

### Lane A: M2 builder（将来）

- CPU process pool
- sector または symmetry orbit representative 単位
- 最大 worker 数は物理 CPU 数ではなく、メモリに応じて制限
- GPU は原則使用しない
- 完了時に `M2_READY.json` を atomic commit

### Lane B: producer

- search-space 列挙
- screening
- independent verification
- selected package 作成
- M2 未完了なら `WAITING_FOR_M2`

### Lane C: GPU consumer

- READY_FOR_M3 を q_upper 順で処理
- GPU worker は 1
- M3 checkpoint を最優先で resume
- M4 が GPU を使う場合は同じ GPU lock を取得

### Lane D: downstream CPU

- M5
- obligations
- M6
- package-local audit 化後は複数 worker 可
- それまでは worker=1

## 4. M2 と後段の接続

M2 notebook は次を作る。

```text
{PERSIST}/runs/{M2_RUN_ID}/
  work_items/
  sectors/
  reports/M2_report.json
  reports/M2_acceptance.json
  M2_READY.json
```

`M2_READY.json` の条件:

- expected sector 数と completed sector 数が一致
- exact equivalence gate PASS
- acceptance PASS
- source/config/parent hash が固定
- active lease がゼロ

後段 notebook は `M2_READY.json` の存在だけでなく、
その内部ハッシュと acceptance を再検証して binding する。

## 5. 後段 notebook の待機方式

M2 が未完了の場合でも notebook 97 は停止しない。

```text
SCREENING
  ↓
SCREENED_Q_LT_1
  ↓
WAITING_FOR_M2
```

M2 が READY になった後、reconciler が

```text
WAITING_FOR_M2 → READY_SHARED → S0 → VERIFY → SELECTED
```

へ移す。

これにより M2 と screening を並行実行できる。

ただし M3 は親 M2 が READY になるまで開始しない。

## 6. バックログとバックプレッシャー

**89∥97（推奨）:** backlog 成長は許容する。unified throttle を主運用にしない。  
disk 逼迫時のみ screening を止めるなど、資源ガードは別途。

**96（任意・単独）:** backlog-aware gate（`selected_backlog_target`）は
効率化のオプション。必須ではない。

将来の分割レーン向け参考値（96 や opt-in end_to_end 用）:

```yaml
selected_backlog_target: 8
waiting_for_m3_limit: 16
screening_chunk_size: 32
gpu_workers: 1
downstream_cpu_workers: 1
```

- disk free ratio が閾値未満なら新規 candidate を作らない
- stale GPU lane lease（死 PID / 古い heartbeat）は consumer 起動時に reclaim
- 生存プロセスの GPU lease は奪わない（fail closed）

## 7. 停止判定

停止判定は一種類にまとめず、以下を区別する。

### RUNNING

- active lease がある
- runnable work item がある
- producer に未列挙 scheme がある

### WAITING_FOR_M2

- M2 未完了
- WAITING_FOR_M2 candidate がある
- M2 builder の active lease または未完了 sector がある

### IDLE_COMPLETE

- search space exhausted
- active lease がゼロ
- runnable queue がゼロ
- 全 candidate が terminal
- M2 が READY
- 直近二回の scan で新規 `.done` commit がゼロ

### IDLE_BLOCKED

- active lease がゼロ
- runnable queue がゼロ
- 非 terminal candidate が存在
- 全てが bad M2 / open obligation / missing parent / resource block

### ERROR_STALE

- stale RUNNING lease がある
- heartbeat が期限超過
- recovery が未実施

### PAUSED_RESOURCE

- disk / memory / CUDA 条件で新規 work を開始できない

## 8. progress の定義

progress は session の開始回数や checkpoint 回数ではなく、
新しく atomic commit された work item 数で測る。

```text
progress_delta =
  current_done_marker_count - previous_done_marker_count
```

表示する主要指標:

### M2

- expected sectors
- completed sectors
- running sectors
- failed sectors
- completion ratio
- last committed sector
- last heartbeat

### Campaign B

- candidate status 別件数
- selected 数
- archived 数
- WAITING_FOR_M2 数
- READY_FOR_M3 数

### M3–M6

- M3_RUNNING / CHECKPOINT / COMPLETE
- M4_RUNNING / CHECKPOINT / COMPLETE
- PRE_M6_READY
- obligations open / closed
- M6_RUNNING / COMPLETE / FAILED
- active package と current phase

### 停止診断

- stop verdict
- runnable item 数
- active lease 数
- stale lease 数
- blocked reason 上位
- search space exhausted 여부
- disk free ratio

## 9. atomicity と resume

全 work item は次の順で commit する。

1. attempt directory に一時出力
2. fsync
3. sha256
4. final path へ atomic rename
5. `.done` marker を atomic write
6. heartbeat / aggregate を更新

Notebook 再実行時:

- `.done` がある item は再実行しない
- result があって `.done` がなければ未完了
- stale RUNNING を CHECKPOINT に戻す
- M3 checkpoint を fresh work より先に処理
- notebook memory の変数は resume 判定に使わない

## 10. 必要な実装修正

### 新設

```text
src/campaign_b/m2_shared_batch.py      # 将来（Lane A）
src/campaign_b/post_m2_pipeline.py     # notebook 97
src/campaign_b/pipeline_recovery.py
src/campaign_b/pipeline_status.py      # notebook 98
src/campaign_b/execution_keys.py
src/campaign_b/end_to_end.py           # notebook 96
src/campaign_b/m6_certified_catalog.py # notebook 99
```

### 修正（後続）

```text
src/campaign_b/driver.py
  WAITING_FOR_M2 を追加

src/campaign_b/gpu_m3_batch.py
  GPU lock と shared M3 execution key

src/campaign_b/pre_m6_batch.py
  package-local audit path 対応

src/m4_orchestrator.py
src/m5_orchestrator.py
  global audit 依存を config path に置換
```

## 11. 推奨運用

1. **推奨:** Notebook **89**（producer）∥ **97**（GPU consumer）。
   待ち行列が増えてもよい。95 と同じ思想。
2. 97 のデフォルトは既存 M2-ready / READY_FOR_M3 backlog の drain
   （`drain_existing_backlog=True` / `skip_screening=True`）。
3. **任意:** 単独で Notebook **96**（統合スケジューラ）。throttle は必須ではない。
4. **禁止:** 96∥97 を両方フル GPU consumer として起動しない。
   `campaign_b/_locks/gpu_lane.json` が第二側を `GpuLaneHeldError` で止める。
   97 drain は `owner=pipeline_to_m6` のみ（outer `notebook_97_post_m2` なし）。
   死んだ PID の lease 回収:
   `python scripts/release_gpu_lane_lease.py --force-if-dead`。
5. 既存 `READY_FOR_M3` / binding READY は `M2_READY` 待ちなしで M3→M6 へ進む。
6. Notebook 98 は別 kernel で必要時に（読み取り専用）。
7. Notebook 99 で CERTIFIED を永続カタログへマージする。
8. M2 worker 数を制限し、GPU pipeline 用メモリを残す。

この構成なら、89 の screening と 97 の GPU 消化を並走でき、
単一 GPU と global audit の競合は lease で直列化する。
