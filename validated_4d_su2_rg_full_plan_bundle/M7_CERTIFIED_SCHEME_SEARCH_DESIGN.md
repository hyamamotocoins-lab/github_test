# M7 — Certified Scheme Search 設計書

## 1. 目的

M0–M6は、固定scheme \(\theta\) に対して

\[
q_{\mathrm{cert}}^{\mathrm{upper}}(\theta)<1
\]

が成立するかをfail-closedで判定する基盤である。M7は、許可されたscheme空間 \(\Theta_{\mathrm{allowed}}\) から候補を生成し、依存関係に従って必要なmilestoneだけを再実行し、最終的に

\[
\exists\theta\in\Theta_{\mathrm{allowed}}:\quad q_{\mathrm{cert}}^{\mathrm{upper}}(\theta)<1
\]

を満たす独立検証済みschemeを見つける。

成功条件は次である。

```text
M7_COMPLETE
CERTIFIED_SCHEME_FOUND
independent_verifier = PASS
q_cert_upper < 1
```

## 2. 基本原則

1. 完了済みM0–M6 runは不変とする。
2. scheme変更ごとに新しいscheme IDとrun lineageを作る。
3. exploratory screeningとrigorous certificationを分離する。
4. 未評価誤差を0として扱わない。
5. certificate条件、誤差ledger、数学的scopeを探索中に変更しない。
6. 失敗候補も再現可能なartifactとして保存する。
7. 独立verifierが `q_cert_upper < 1` を再計算するまで成功扱いにしない。

## 3. M7のsubphase

| Subphase | 名称 | 役割 |
|---|---|---|
| M7.0 | Search initialization | 親M6、探索空間、予算、LOCKを固定 |
| M7.1 | Failure diagnosis | `q_cert`の支配項を分類 |
| M7.2 | Candidate proposal | 許可parameterから候補生成 |
| M7.3 | Impact analysis | dependency DAGで再実行範囲を決定 |
| M7.4 | Exploratory screening | 非rigorous高速評価 |
| M7.5 | Rigorous replay | 新しいimmutable lineageで再計算 |
| M7.6 | Independent acceptance | final packageを独立再計算 |
| M7.7 | Search update | 履歴・除外規則・次候補を更新 |
| M7.8 | Finalization | 成功または停止理由をpackage化 |

## 4. scheme parameterの分類

schemeを

\[
\theta=(\theta_{\mathrm{alg}},\theta_{\mathrm{tensor}},\theta_{\mathrm{rank}},\theta_{\mathrm{norm}},\theta_{\mathrm{family}},\theta_{\mathrm{majorant}})
\]

と分ける。

### 4.1 Algebraic scheme

- block geometry
- coarse-link definition
- orientation convention
- fusion tree
- representation cutoff \(j_{\max}\)
- channel cutoff
- tensor normalization convention

### 4.2 Tensor / low-rank scheme

- target rank
- oversampling
- power iteration count
- deterministic basis construction
- residual estimator

### 4.3 Norm / influence scheme

- `weight_m`
- source component weights
- Perron / Collatz vector
- stage-dependent weighted norm
- influence block partition
- diagonal scaling

### 4.4 Input-family scheme

- interval subdivision
- family center / radius
- singleton / cell / polytope representation
- source-family partition
- tail-budget allocation

### 4.5 Multi-step policy

- RG step数 \(K\)
- direct product majorant
- stage-dependent weighting
- stepwise recentering
- parent-majorant inheritance policy

最初の重要な変更候補は、現行の親M5 majorant継承を次のpolicyへ置き換えることである。

```text
PARENT_ONE_STEP_INHERITANCE
DIRECT_MULTI_STEP_PRODUCT
STAGE_DEPENDENT_WEIGHTED_PRODUCT
RECENTERED_CELLWISE_PRODUCT
```

## 5. 変更クラスと再実行範囲

### S0 — final-majorant-only

例:

- Perron weight
- Collatz vector
- diagonal scaling

再実行:

```text
M6.final_collatz
M6.final_certificate
M6.independent_verifier
```

### S1 — M5/M6 influence scheme

例:

- `weight_m`
- influence partition
- source grouping
- interval subdivision
- direct multi-step influence policy

再実行:

```text
M5 influence leaves
M5 acceptance
M6 composition
M6 final verifier
```

### S2 — M3–M6 numerical representation

例:

- target rank
- oversampling
- power iteration
- deterministic low-rank basis

再実行:

```text
M3 -> M4 -> M5 -> M6
```

M4 tangent basisがM3 basisに依存するため、M4も必ず再実行する。

### S3 — M2–M6 algebraic scheme

例:

- representation cutoff
- channel cutoff
- fusion tree
- block geometry
- tensor normalization

再実行:

```text
M2 -> M3 -> M4 -> M5 -> M6
```

M1 tail theoremのparameterizationを変える場合はM1も再実行する。

### S4 — mathematical scheme change

例:

- RG mapの定義
- normの数学的意味
- gauge-fixing convention
- positivity cone
- source class
- certificate判定不等式

自動探索対象外とし、人間承認、新しいgoverning document、schema version更新を必須とする。

## 6. LOCKとlineage

### 6.1 不変条件

1. 完了run directoryはread-only。
2. candidateごとに新しい `scheme_id` と `candidate_id` を発行。
3. rigorous replayごとに新しいM2–M6 run IDを発行。
4. parent artifact hashを全manifestに記録。
5. invalidated artifactを再利用しない。
6. reusable artifactはcontent hashとconvention hashの一致を要求。
7. exploratory artifactをrigorous packageへ混入させない。

### 6.2 Candidate LOCK例

```json
{
  "schema_version": 1,
  "search_run_id": "M7-...",
  "candidate_id": "CAND-000014-...",
  "scheme_hash": "sha256:...",
  "parent_scheme_hash": "sha256:...",
  "parent_m6_run_id": "M6-20260720T061700Z-7c4e91a2b850",
  "change_class": "S1",
  "changed_parameters": ["majorant_policy"],
  "invalidated_nodes": [],
  "reused_artifacts": [],
  "status": "LOCKED_FOR_SCREENING"
}
```

既存LOCKを書き換えず、state transitionはappend-only event logへ追記する。

## 7. dependency DAG

nodeはparameter、config、artifact、ledger leaf、acceptance、final certificateから構成する。

必須edge:

```text
parameter -> generated config
config -> task artifact
artifact -> ledger leaf
ledger leaf -> M5 acceptance
M5 acceptance -> M6 composition
M6 composition -> final_collatz
final_collatz -> final_certificate
final_certificate -> independent_verifier
```

例:

```text
target_rank
  -> M3 low-rank basis
  -> M3 deterministic residual
  -> M4 tangent basis
  -> M5 rank error
  -> M5 influence
  -> M6 final majorant
```

```text
perron_weight
  -> M6 final_collatz
  -> M6 final_certificate
```

candidate diffに含まれるparameter nodeの全descendantをinvalidatedとする。

## 8. failure diagnosis

\[
q_{\mathrm{cert}}
=
q_{\mathrm{core}}
+\varepsilon_{\mathrm{round}}
+\varepsilon_{\mathrm{rank}}
+\varepsilon_{\mathrm{rep}}
+\varepsilon_{\mathrm{channel}}
+\varepsilon_{\mathrm{input}}
+\varepsilon_{\mathrm{norm}}
+\varepsilon_{\mathrm{source}}
+\varepsilon_{\mathrm{composition}}.
\]

### D0 — inherited-majorant failure

条件:

```text
q_final == q_parent
majorant_policy == PARENT_ONE_STEP_INHERITANCE
```

候補:

- direct multi-step product
- stage-dependent weights
- recentered cellwise product

### D1 — core expansion

`q_core` が支配。候補はblock geometry、source block、追加RG step、recentring。

### D2 — truncation dominated

rank、representation、channel tailが支配。候補はrank/cutoff増加、basis改善。

### D3 — interval dependency dominated

input radiusやinterval幅が支配。候補はsubdivision、affine arithmetic、cellwise recentering。

### D4 — normalization dominated

分母下界が支配。候補はnormalization enclosure改善、cellwise denominator。

### D5 — arithmetic dominated

roundingが支配。候補はcontraction order、scaling、compensated summation。

### D6 — unresolved analytic leaf

ledger leafがOPENならrigorous replayへ進めず `M7_BLOCKED_MATH` とする。

## 9. 初期探索空間

### 第一層: 低コスト

```yaml
majorant_policy:
  - DIRECT_MULTI_STEP_PRODUCT
  - STAGE_DEPENDENT_WEIGHTED_PRODUCT
perron_weight_strategy:
  - interval_power
  - collatz_lp
  - block_diagonal
source_partition:
  - current
  - symmetry_blocks
input_subdivision: [1, 2, 4, 8]
```

### 第二層: numerical representation

```yaml
target_rank: [16, 24, 32, 48, 64]
oversampling: [8, 16, 24]
power_iterations: [1, 2, 3]
```

### 第三層: algebraic/high-cost

```yaml
j2_max: [1, 2, 3, 4]
channel_policy:
  - complete_at_cutoff
  - certified_pruned
block_geometry:
  - current
  - approved_geometry_B
```

第三層は第一・第二層で改善傾向が確認できた場合だけ解放する。

## 10. exploratory screening

目的はrigorous replay候補を減らすこと。許可されるもの:

- floating point
- reduced cutoff/rank
- sampled cells
- approximate spectral radius
- approximate resource prediction

screening status:

```text
SCREEN_PROMISING
SCREEN_REJECTED
SCREEN_INCONCLUSIVE
SCREEN_RESOURCE_FAILURE
```

screeningから `CERTIFIED` を出してはならない。

promotion条件例:

```text
estimated_q < 0.90
estimated_error_margin < 0.08
no_open_ledger_leaf
resource_prediction within budget
```

## 11. rigorous replay

promoted candidateごとに、最初のinvalidated milestoneから新lineageを作る。

```text
S0: new M6 child
S1: new M5 -> M6
S2: new M3 -> M4 -> M5 -> M6
S3: new M2 -> M3 -> M4 -> M5 -> M6
```

要求:

- outward-rounded interval arithmetic
- deterministic residual
- complete error ledger
- checkpoint hash chain
- immutable final package
- independent verifier

## 12. 真のmulti-step influence

各stepについて

\[
D\mathcal R_r(\mathcal K_r)\subseteq\mathcal B_r
\]

を囲い、

\[
D\mathcal R^K(\mathcal K_0)
\subseteq
\mathcal B_{K-1}\cdots\mathcal B_0
\]

を直接計算する。

interval dependency blow-upを抑えるため、以下を許可parameterとする。

1. stage-dependent diagonal scaling
2. block cone majorant
3. affine center-radius product
4. cellwise recentering
5. symmetry block decomposition

最終的に正ベクトル \(w>0\) と \(q\) について

\[
\overline B^{(K)}w\le q w
\]

を成分ごとに検証する。\(q<1\) なら収縮certificate成立。

## 13. candidate state machine

```text
PROPOSED
 -> LOCKED_FOR_SCREENING
 -> SCREEN_RUNNING
 -> SCREEN_REJECTED | SCREEN_PROMISING
 -> LOCKED_FOR_RIGOROUS_REPLAY
 -> RIGOROUS_RUNNING
 -> RIGOROUS_FAILED | M6_NOT_CERTIFIED | M6_CERTIFIED
 -> INDEPENDENT_VERIFIER_PASS
 -> ACCEPTED_CERTIFIED_SCHEME
```

異常系:

```text
INVALID_SCHEME
DEPENDENCY_ERROR
OPEN_LEDGER_LEAF
RESOURCE_LIMIT
CHECKPOINT_CORRUPT
VERIFIER_MISMATCH
```

## 14. search state machine

```text
M7_INITIALIZED
M7_DIAGNOSIS_COMPLETE
M7_SEARCHING
M7_CANDIDATE_RUNNING
M7_CERTIFIED_SCHEME_FOUND
M7_SEARCH_SPACE_EXHAUSTED
M7_RESOURCE_LIMIT_REACHED
M7_BLOCKED_MATH
M7_BLOCKED_POLICY
M7_COMPLETE
```

`SEARCH_SPACE_EXHAUSTED` は、固定した探索空間内でcertificateが見つからなかったことだけを意味し、理論的不存在は主張しない。

## 15. 予算と停止条件

例:

```yaml
max_candidates_total: 200
max_rigorous_replays: 20
max_gpu_hours: 1500
max_wall_days: 90
max_storage_gb: 2000
stop_on_first_certified: true
required_q_cert_upper: 0.99
```

数学的certificate条件は \(q<1\) だが、運用上は再現性marginを設定してよい。

## 16. notebook / module構成

```text
72_m7_initialize_search.ipynb
73_m7_failure_diagnosis.ipynb
74_m7_generate_candidates.ipynb
75_m7_screen_candidates.ipynb
76_m7_promote_candidate.ipynb
77_m7_resume_search.ipynb
78_m7_final_review.ipynb
```

```text
src/m7_config.py
src/m7_models.py
src/m7_lock.py
src/m7_dependency.py
src/m7_diagnosis.py
src/m7_generator.py
src/m7_screening.py
src/m7_replay.py
src/m7_acceptance.py
src/m7_orchestrator.py
src/m7_independent_verifier.py
```

## 17. directory構成

```text
/storage/validated_4d_su2_rg/searches/M7-<run-id>/
  LOCK.json
  search_space.lock.json
  dependency_graph.json
  state/
    search_state.json
    append_only_events.jsonl
  candidates/
    CAND-000001-.../
      candidate.lock.json
      scheme.json
      impact_report.json
      screening/
      rigorous_lineage.json
      acceptance.json
  reports/
    failure_diagnosis.json
    candidate_ranking.csv
    best_so_far.json
    search_summary.md
  final_package/
    accepted_scheme.json
    accepted_lineage.json
    M6_final_certificate/
    M7_acceptance.json
    independent_verifier_report.json
```

## 18. independent verifier

M7 verifierはGPUを再実行せず、次を独立再構成する。

1. canonical scheme hash
2. parent/child lineage
3. invalidationの正当性
4. reused artifactの依存非汚染
5. ledger合計
6. multi-step majorant
7. Collatz inequality
8. `q_cert_upper < 1`
9. package hash chain

producer側helperをimportしない独立実装とする。

## 19. checkpoint / restart

checkpoint対象:

- candidate queue
- generator RNG state
- completed screening
- rigorous lineage状態
- budget counters
- best-so-far
- append-only event offset

resume時は既存search LOCK一致を必須とする。accepted candidateが存在する場合は新規探索を開始しない。

## 20. test計画

### Unit

- canonical scheme hashing
- dependency descendants
- invalidation class
- duplicate exclusion
- budget accounting
- state transition validation
- manifest schema

### Integration

1. S0でM6 final leavesだけ開く
2. rank変更でM3–M6が開く
3. geometry変更でM2–M6が開く
4. invalid artifact reuseを拒否
5. screeningからCERTIFIEDを出せない
6. verifier mismatchでacceptance拒否
7. resume後にcandidate順序が再現
8. search exhaustionを正しく出力

### Adversarial

- forged parent hash
- modified checkpoint
- missing ledger leaf
- NaN/Inf score
- interval endpoint reversal
- reordered JSON duplicate
- old schema artifact reuse
- S4自動promotion拒否

## 21. 初回campaign

### Campaign A — majorant改善

1. `DIRECT_MULTI_STEP_PRODUCT`
2. stage-dependent Perron weight
3. symmetry source block
4. cellwise recentering
5. input subdivision 2, 4, 8

目的は親M5 majorant継承による構造的停滞を除去すること。

### Campaign B — finite approximation改善

Campaign Aでcoreが1未満に近づいた場合のみ:

- rank 24, 32, 48
- deterministic residual tightening
- cutoff/channel拡張

### Campaign C — geometry変更

Campaign A/Bで改善傾向がなくcoreが支配する場合に、人間レビュー後に実行。

## 22. 完了条件

個別M6 run:

```text
M6_COMPLETE
SCHEME_REJECTED | CERTIFIED
```

M全体:

```text
M7_COMPLETE
CERTIFIED_SCHEME_FOUND
independent_verifier = PASS
q_cert_upper < 1
```

M6は固定schemeの厳密oracle、M7はそのoracleを用いてcertifiable schemeを探索するcontrollerである。
