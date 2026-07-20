# 4次元 SU(2) validated RG enclosure
## Codex実装用・詳細設計書

**設計版:** 0.2.0  
**主成果物:** `validated_4d_su2_rg_gpu_driver.ipynb`  
**対象環境:** NVIDIA CUDA GPUを持つJupyter環境  
**セッション制約:** 連続利用6時間。5時間30分以内に永続checkpointを完了して正常終了する。  
**基本原則:** GPUは有限coreと候補低rank空間の高速計算に使い、証明判定は解析tailとa posteriori残差評価で行う。

---

# 1. 数学的な目的

有限回RGを

\[
T_{r+1}=\mathcal R(T_r)
\]

とし、各段階を

\[
T_r=\widetilde T_r+E_r,
\qquad \|E_r\|\le \varepsilon_r
\]

という有限coreと誤差上界の組で保持する。最終的に時間方向weighted influence matrixのentrywise上界 \(\overline B_m\) と解析tailを構成し、正ベクトル \(w>0\) に対して

\[
q_{\mathrm{cert}}
=
\max_i \frac{(\overline B_m w)_i}{w_i}
+\varepsilon_{\mathrm{tail}}<1
\]

を検証する。

この不等式が、すべてのvalidation flagが真である状態で成立した場合だけ、run statusを `CERTIFIED` にする。近似計算だけでは絶対に `CERTIFIED` にしない。

---

# 2. 非目標

初版では次を行わない。

1. 任意のcompact simple groupへの一般化。
2. continuum limit全体の証明。
3. 巨大tensor全成分の区間演算。
4. dense rank-8 tensorの明示的生成。
5. GPU結果だけに依存した厳密証明。
6. notebookのRAM状態に依存した再開。
7. 6時間ぎりぎりまで新規長時間kernelを開始すること。

---

# 3. 成果物とproject構造

Notebookを主実行画面とする。Notebook内の `%%writefile` cellでmoduleを永続ストレージへ生成する。

```text
project_root/
├── validated_4d_su2_rg_gpu_driver.ipynb
├── src/
│   ├── config.py
│   ├── session_guard.py
│   ├── checkpoint.py
│   ├── work_queue.py
│   ├── representations.py
│   ├── fusion.py
│   ├── armillary.py
│   ├── sparse_tensor.py
│   ├── contraction_backend.py
│   ├── triad_atrg.py
│   ├── forward_ad.py
│   ├── tail_bounds.py
│   ├── residual_validation.py
│   ├── influence.py
│   ├── certificate.py
│   ├── orchestrator.py
│   └── reporting.py
├── tests/
├── cache/
└── runs/
```

Notebook内のmodule生成cellを正本とする。外部moduleを手で変更した場合は、次回notebook実行で上書きされる可能性を明記する。

---

# 4. 永続ストレージ

## 4.1 必須条件

checkpoint rootはruntime shutdown後も残る場所でなければならない。

- Google Colab: Google Drive内の専用directory。
- クラウドVM: 永続volume。
- JupyterHub: home directoryまたは永続project volume。
- その他: shutdown後も残ることを利用者が確認した外部保存先。

`/tmp`、`/content`直下、ephemeral local SSDだけをcheckpoint rootにしてはならない。

## 4.2 設定

環境変数を最優先する。

```python
PERSIST_ROOT = os.environ["VALIDATED_RG_PERSIST_ROOT"]
```

未設定、書込不能、または明らかにephemeralなpathの場合は計算を開始しない。

## 4.3 run directory

```text
PERSIST_ROOT/
├── project/
├── cache/
│   ├── wigner/
│   ├── fusion/
│   ├── armillary/
│   └── contraction_paths/
└── runs/RUN_ID/
    ├── run_config.json
    ├── run_manifest.json
    ├── logs/
    ├── reports/
    ├── artifacts/
    ├── work_items/
    └── checkpoints/
        ├── ckpt_000001/
        │   ├── meta.json
        │   ├── state.json
        │   ├── bounds.json
        │   ├── work_queue.json
        │   ├── rng_state.pt
        │   ├── tensors/
        │   ├── hashes.json
        │   └── COMMITTED
        └── LATEST.json
```

---

# 5. 5時間30分session protocol

monotonic clockを使う。

```text
0:00  bootstrap / resume
5:00  新しい長時間taskを開始しない
5:15  drain mode
5:20  final checkpoint開始
5:25  checkpoint再読込・hash検証
5:30  runnerを正常return
```

既定値:

```python
SESSION_BUDGET_S = 5.5 * 3600
NO_LONG_TASK_AFTER_S = 5.0 * 3600
DRAIN_AFTER_S = 5.25 * 3600
FINAL_SAVE_AFTER_S = (5 + 20/60) * 3600
HARD_RETURN_S = 5.5 * 3600
CHECKPOINT_INTERVAL_S = 15 * 60
MAX_WORK_ITEM_S = 20 * 60
```

## 5.1 task粒度

単一work itemは20分以内を目標とする。

- fusion sector列挙: key batch。
- armillary build: sector shard。
- contraction: output sector batch。
- RSVD: sketch column batch。
- derivative: symmetry class。
- interval validation: matrix row block。

過去のtimingのp95を保存し、残り時間が

\[
t_{\mathrm{left}}
<1.3t_{\mathrm{p95}}+t_{\mathrm{reserve}}
\]

なら開始しない。

## 5.2 save trigger

- 15分ごと。
- phase boundary。
- RG substep完了時。
- sector shard完了時。
- RSVD basis確定時。
- validation開始前。
- `KeyboardInterrupt` 捕捉時。
- drain mode移行時。
- final return前。

`atexit`は補助であり、主checkpoint手段にはしない。

---

# 6. atomic checkpoint

1. `checkpoints/.tmp-UUID/` へ書く。
2. GPU tensorをCPUへ移す。
3. tensorをshard単位で保存する。
4. JSON metadataを書く。
5. 全fileのSHA-256を `hashes.json` に記録する。
6. flushし、可能なら`fsync`する。
7. final checkpoint名へatomic renameする。
8. `COMMITTED` markerを作る。
9. `LATEST.json`を一時file経由でatomic replaceする。
10. save直後に再読込してhash検証する。

復元時は新しいcheckpointから順に検証し、破損していれば一つ前へfallbackする。

## 6.1 保存対象

- phase / subphase / RG step / direction。
- work queueのpending/running/done。
- core tensorsとderivative tensors。
- analytic tail、RSVD residual、rounding bound。
- normalization logs。
- candidate Perron vector。
- Python / NumPy / torch CPU / CUDA RNG state。
- config hash、source hash、notebook hash。
- CUDA、driver、GPU、PyTorch、cuQuantumのversion。
- contraction pathとRSVD seed。
- task timing history。
- certification status。

## 6.2 work item recovery

work item IDはcontent-addressedにする。

```text
item_id = sha256(phase + input_hash + parameters)
```

処理規約:

1. `pending -> running` を保存。
2. resultをtemp pathへ書く。
3. result hashを検証。
4. `.done` markerをatomic作成。
5. `running -> done` を保存。

再起動時、`.done`のない`running` itemは`pending`へ戻す。`.done`がありhashが正しければ`done`へ修復する。

---

# 7. 数学データ構造

## 7.1 irrep

半整数をfloatで保存しない。

```python
@dataclass(frozen=True, order=True)
class Irrep:
    j2: int  # j2 = 2j
```

\[
d_j=j2+1,
\qquad
C_2(j)=\frac{j2(j2+2)}4.
\]

## 7.2 fusion

\(SU(2)\)ではmultiplicityは0または1。triangle conditionとparityを整数`j2`で処理する。

## 7.3 sector key

```python
@dataclass(frozen=True)
class SectorKey:
    external_j2: tuple[int, ...]
    fusion_j2: tuple[int, ...]
    orientation: tuple[int, ...]
    parity: int
```

格子対称性で同値なsectorをcanonicalizeする。

## 7.4 block sparse tensor

```python
@dataclass
class BlockSparseTensor:
    keys: list[SectorKey]
    offsets: torch.Tensor
    shapes: list[tuple[int, ...]]
    data: torch.Tensor
    norm_upper: float
    error_radius: str
```

巨大な場合はshard storeへ分割し、必要shardだけGPUへloadする。

## 7.5 bounded tensor

巨大tensorにcomponentwise intervalを持たせない。

```python
@dataclass
class BoundedTensor:
    center: BlockSparseTensor
    norm_kind: str
    radius: str
    provenance: list[str]
```

小さい最終行列だけcomponentwise intervalで保持する。

---

# 8. 数値精度

## 8.1 exploration

- FP64を標準。
- pilotだけFP32を許可。
- TF32を無効化。
- seed、hardware、versionを保存。

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

## 8.2 certification

GPU float結果だけでは証明にしない。

1. character/Peter–Weyl analytic tail。
2. input radiusのmultilinear伝播。
3. deterministic RSVD residual upper bound。
4. 必要な小行列のCPU多倍長再計算。
5. entrywise interval influence matrix。
6. Collatz–Wielandt上界。

`python-flint`等が利用できない場合はcertificate phaseを`BLOCKED`にし、通常floatへ黙ってfallbackしない。

---

# 9. 初期Wilson tensor

\[
\bar w_\beta(U)=e^{\beta(\cos\theta-1)}
\]

とし、

\[
e^{\beta\cos\theta}
=
\sum_{n\ge1}a_n(\beta)\chi_n(U),
\qquad
a_n(\beta)=\frac{2n}{\beta}I_n(\beta)
\]

を使う。係数はCPU多倍長でlower/upper boundを持たせる。

既存prototypeから次を移植する。

- `normalized_character_tail`
- `normalized_character_derivative_tail_upper`
- positive Taylor series enclosure

初期core tensorはarmillary reductionを使い、matrix indexを外部bondへ残さない。

---

# 10. armillary reduction

目的:

\[
\text{matrix indices}
\longrightarrow
\text{representation/fusion indices}.
\]

実装順:

1. \(j_{\max}=1/2\) で明示的matrix-index tensorを作る。
2. 同じtensorをfusion basisで作る。
3. 高精度で一致を検証する。
4. 一致後、matrix-index版をtest専用にする。
5. 4次元armillary generatorへ進む。

Wigner 3j/6j/9j cacheにはsymbol convention、phase convention、library version、input hashを保存する。

acceptance tests:

- orthogonality。
- tetrahedral symmetry。
- fusion parity。
- singlet projection。
- dense版との一致。
- orientation reversal consistency。

---

# 11. Triad-ATRG backend

backend優先順位:

1. cuTensorNet / cuQuantum。
2. `opt_einsum` + torch CUDA。
3. torch CUDA。
4. CPUはtest/validation用。

```python
class ContractionBackend(Protocol):
    def contract(self, expression, operands, *, path=None): ...
    def estimate_path(self, expression, shapes, memory_limit): ...
    def apply_linear(self, operator, x): ...
    def apply_adjoint(self, operator, y): ...
```

巨大中間行列を生成せず、\(x\mapsto Ax\)、\(y\mapsto A^*y\)をtensor contractionとして実装する。

contraction pathは次をkeyにcacheする。

- graph hash。
- block shapes。
- dtype。
- GPU model。
- memory limit。
- algorithm version。

GPU memoryは毎task前に確認し、通常25%以上、checkpoint前35%以上のheadroomを残す。OOM時はshard sizeを半分にして再queueし、同じitemで3回OOMなら`BLOCKED_RESOURCE`にする。

---

# 12. RSVD

explorationでは固定seedでrange finderを実行する。

parameters:

- target rank \(\chi\)
- oversampling \(p\)
- power iteration \(q\)
- internal overspanning rank
- residual tolerance

probabilistic errorだけを証明には使わない。GPUで得た\(Q\)を固定し、

\[
\|Q^*Q-I\|,
\qquad
\|(I-QQ^*)A\|
\]

をdeterministicに囲う。

段階:

1. Frobenius upper bound。
2. sectorwise deterministic bound。
3. 必要ならCPU高精度projected contraction。
4. 証明marginより十分小さいことを確認。

残差が大きい場合はrankを上げ、同phaseを再実行する。

---

# 13. forward-mode source derivatives

sourceごとにRG全体を再実行しない。

```python
@dataclass
class DualTensor:
    primal: BoundedTensor
    tangent: dict[SourceClass, BoundedTensor]
```

初版source class:

- temporal link source。
- spatial link source。
- electric-like channel。
- magnetic/plaquette-like channel。
- 必要な低representation channel。

初版validationではSVD basisをprimalから固定し、basis variationを別residualへ入れる。SVD derivativeを黙って無視しない。

---

# 14. 誤差伝播

multilinear contraction \(C=\mathcal C(A_1,\dots,A_k)\) に対し、

\[
\|A_i-\widetilde A_i\|\le r_i
\]

なら、submultiplicative normで

\[
\|C-\mathcal C(\widetilde A_1,\dots,\widetilde A_k)\|
\le
\|\mathcal C\|_{\mathrm{struct}}
\left[
\prod_i(\|\widetilde A_i\|+r_i)
-
\prod_i\|\widetilde A_i\|
\right]
+r_{\mathrm{round}}+r_{\mathrm{trunc}}.
\]

provenanceを項別に保存する。

- input radius。
- representation tail。
- omitted fusion/channel。
- RSVD residual。
- rounding/backward error。
- normalization error。
- derivative error。

同じ誤差を二重計上しないためerror DAGを持つ。

---

# 15. influence matrixとcertificate

finite kernel \(\widetilde K(x;\eta)\) と誤差から

\[
Z(\eta)=\int K(x;\eta)\,dx
\]

の厳密lower bound \(z_{\min}>0\) を得る。得られなければcertificateを停止する。

\[
\|K-\widetilde K\|_1\le\varepsilon_0,
\qquad
\|\partial_{\eta_j}K-\partial_{\eta_j}\widetilde K\|_1
\le\varepsilon_{1,j}
\]

からinfluence entryを上から囲う。

weighted matrix:

\[
(\overline B_m)_{ab}
=
\sum_z e^{m|z_0|}\overline c_{ab}(z).
\]

GPU近似からPerron vector候補を得た後、positive rational/interval vectorへ変換する。

\[
q_{\mathrm{CW}}
=
\max_a\frac{(\overline B_m w)_a}{w_a}.
\]

```python
if all_validation_passed and q_CW + analytic_tail < 1:
    status = "CERTIFIED"
else:
    status = "NOT_CERTIFIED"
```

reportに必ず出すもの:

- \(q_{\mathrm{CW}}\)。
- tail総和。
- margin。
- cutoff、bond dimension、RG steps。
- source classes。
- checkpoint/config/code hash。
- hardware/software versions。
- 未検証項目。

---

# 16. state machine

```text
BOOTSTRAP
 -> REPRESENTATION_CACHE
 -> FUSION_ENUMERATION
 -> ARMILLARY_BUILD
 -> PILOT_CONTRACTION
 -> TRIAD_DECOMPOSITION
 -> RG_CONTRACTION
 -> FORWARD_AD
 -> RESIDUAL_VALIDATION
 -> INFLUENCE_BUILD
 -> CERTIFICATE_CHECK
 -> REPORT
 -> COMPLETE
```

各phaseはidempotentにする。global mutable stateだけに依存しない。

---

# 17. adaptive schedule

最初から高cutoffを使わない。

```text
pilot:      j2_max = 1, 2, 3
screening:  j2_max = 5, 7
candidate:  j2_max = 9, 11
validation: marginに必要なcutoffまで増加
```

早期停止:

- approximate \(\rho(B_m)>1.2\): schemeを不採用。
- truncation residualがmarginの50%超: rank増加。
- normalization lower boundが非正: 停止。
- storage quota 80%超: compact phase。
- 3回連続でmargin改善なし: `STALLED`。

---

# 18. notebook cell構成

1. タイトルと数学目標。
2. persistent storage mount。
3. runtime/GPU検出。
4. package bootstrap。
5. config。
6. module生成。
7. unit tests。
8. create/resume run。
9. representation/fusion cache。
10. armillary build。
11. pilot run。
12. `run_until_checkpoint()`。
13. validation。
14. influence/certificate。
15. report。
16. checkpoint inspection。
17. next-session instructions。

長時間実行entry pointは一つにする。

```python
result = orchestrator.run_until_checkpoint()
```

---

# 19. tests

## software tests

- Irrep arithmetic。
- triangle/parity。
- cache hash。
- atomic checkpoint。
- corrupt checkpoint fallback。
- RNG restore。
- work item recovery。
- timer drain。
- tensor shard round-trip。
- radius propagation。
- Collatz–Wielandt bound。

## mathematical regression

- 2D \(SU(2)\): \(r_n'=r_n^4\)。
- \(j_{\max}=1/2\): dense vs armillary。
- source derivative finite difference。
- zero source tangent。
- symmetry-related influence entries。
- cutoff増加でanalytic tail単調減少。
- residual増加時にfail closed。

## interruption test

dummy workload中に例外を発生させ、fresh processで再開する。完了済みitemを再計算せず、未完了itemだけを再開する。

---

# 20. loggingと再現性

JSON Lines logを使用する。

```json
{
  "timestamp": "...",
  "run_id": "...",
  "checkpoint": 12,
  "phase": "RG_CONTRACTION",
  "item_id": "...",
  "event": "item_completed",
  "elapsed_s": 384.2,
  "gpu_peak_bytes": 123456789,
  "error_radius": "1.2e-8"
}
```

15分ごとに表示:

- elapsed / remaining。
- phase。
- done / pending。
- last checkpoint。
- GPU memory。
- residual。
- approximate spectral radius。

再現用保存:

- exact config。
- package versions。
- CUDA/driver/GPU。
- source/notebook hash。
- RNG states。
- contraction path。
- sector ordering。
- normalization/Wigner convention。
- dtype。

secret/tokenは保存しない。

---

# 21. milestone

## M0 persistence shell

- shutdown/restart後にdummy queueをresume。
- short timer test。
- corrupt latest fallback。

## M1 2D exact benchmark

- tail enclosure再現。
- \(r_n'=r_n^4\) interval計算。

## M2 4D low-cutoff armillary

- \(j_{\max}=1/2\) dense版と一致。
- 一段kernelをsave/resume。

## M3 GPU Triad pilot

- matrix-free contraction。
- RSVD。
- path cache。
- adaptive sharding。

## M4 source derivative

- forward tangent。
- finite difference regression。
- symmetry reduction。

## M5 one-step validation

- residual upper bound。
- interval influence matrix。
- fail-closed certificate。

## M6 multi-step

- 3〜5 RG steps。
- sessionを跨いでresume。
- final report。

---

# 22. Codexへの禁止事項

1. 未実装項をzero errorにしない。
2. TODOを`pass`のままcomplete扱いしない。
3. float comparisonだけで`CERTIFIED`を出さない。
4. checkpointをephemeral storageだけに置かない。
5. 5時間以降に20分超taskを開始しない。
6. notebook cell順序へ暗黙依存させない。
7. global RAMだけに状態を保持しない。
8. 巨大fileへin-place上書きしない。
9. Wigner conventionを無記録で変更しない。
10. OOM、NaN、Infを握りつぶさない。

---

# 23. 最初のCodex task

最初はRG本体を実装しない。M0だけを実装する。

1. persistence root確認。
2. immutable configとhash。
3. `SessionGuard`。
4. atomic `CheckpointManager`。
5. sharded tensor round-trip。
6. persistent `WorkQueue`。
7. dummy CPU/GPU workload。
8. interruption/resume test。
9. 5時間30分runner。
10. next-session instructions。

M0のtestsがすべて通るまでM1へ進まない。


---

# 24. M1–M6の詳細計画と数学的certification

本設計書のM1–M6は、次の二文書を規範とする。

- `M1_M6_VALIDATED_RG_ROADMAP.md`
- `MATHEMATICAL_CERTIFICATION_SPEC.md`

実装担当は、各milestone開始前に対応するCodex promptを
`CODEX_PROMPTS_M1_M6.md`から使用する。

優先順位は次である。

1. milestone acceptance condition。
2. mathematical proof obligation。
3. restartabilityとartifact integrity。
4. performance optimization。

性能改善のために証明条件を弱めてはならない。
