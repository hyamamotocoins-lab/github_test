# 4次元 SU(2) validated RG enclosure
## M1–M6 実装・検証ロードマップ

**版:** 0.2.0  
**前提:** M0のcheckpoint・resume・work queue・5時間30分session guardが完成していること。  
**主実行面:** `validated_4d_su2_rg_gpu_driver.ipynb`  
**計画管理:** `validated_4d_su2_rg_M1_M6_tracker.ipynb`

---

# 1. 最終目標と成果の段階

最終的な計算対象は、有限回RGによって得られる局所kernelまたは局所tensorを

\[
T_r=\widetilde T_r+E_r,
\qquad
\|E_r\|\le \varepsilon_r
\]

として囲い、時間方向weighted influence matrixのentrywise上界
\(\overline B_m\) を構成することである。

最終判定は、正ベクトル \(w>0\) に対して

\[
q_{\mathrm{CW}}
=
\max_a
\frac{(\overline B_m w)_a}{w_a}
\]

を厳密に上から評価し、解析tailと全数値残差を加えた

\[
q_{\mathrm{cert}}
=
q_{\mathrm{CW}}
+
\varepsilon_{\mathrm{analytic}}
+
\varepsilon_{\mathrm{numerical}}
<1
\]

を証明することである。

成果を次の段階に区別する。

| 状態 | 意味 |
|---|---|
| `EXPLORATORY` | GPU近似計算のみ。証明上の主張なし。 |
| `CORE_REPRODUCED` | 低cutoffで独立実装との一致を確認。 |
| `TAIL_BOUNDED` | representation・polymer・channel tailを解析的に囲った。 |
| `STEP_ENCLOSED` | 一段RGについて有限coreと誤差半径を厳密に囲った。 |
| `ONE_STEP_CERTIFIED` | 一段後のinfluence criterionを厳密に証明した。 |
| `MULTISTEP_ENCLOSED` | 複数段RGの誤差合成を厳密に囲った。 |
| `CERTIFIED` | 最終的に \(q_{\mathrm{cert}}<1\) を証明した。 |
| `NOT_CERTIFIED` | 計算は完了したが不等式を証明できない。 |
| `BLOCKED_MATH` | 必要な数学的評価が未実装または未証明。 |
| `BLOCKED_RESOURCE` | GPU memory・storage・時間制約で実行不能。 |
| `STALLED` | cutoffやrankを増やしてもmarginが改善しない。 |

`CERTIFIED` はM6の全acceptance conditionを満たした場合にだけ使用する。

---

# 2. 共通の5時間30分session単位

各milestoneは、再開可能な20分以下のwork itemへ分割する。

```text
0:00–0:20   mount・resume・checkpoint検証・GPU診断
0:20–4:50   work itemを順次処理
4:50–5:00   新規large shardを開始しない
5:00–5:15   short itemのみ、report更新
5:15–5:20   drain、GPU tensorをCPUへ移動
5:20–5:25   final atomic checkpoint
5:25–5:30   hash再検証、次回指示を出力してreturn
```

各session終了時に必ず生成する。

- `session_summary.json`
- `next_session_plan.md`
- `latest_metrics.json`
- committed checkpoint
- 未完了work item一覧
- 数学的statusと未証明項目

---

# 3. M1 — 2次元exact benchmarkと解析tail基盤

## 3.1 目的

4次元計算で使うrepresentation係数・tail・区間演算・checkpointed numerical pipelineを、exact RGが既知の2次元pure \(SU(2)\) で検証する。

2次元 \(2\times2\) blockingについて、normalized Fourier ratio

\[
r_n=\frac{a_n}{n a_1}
\]

が

\[
r_n' = r_n^4
\]

を満たすことを基準とする。

## 3.2 実装対象

- `representations.py`
- `tail_bounds.py`
- `interval_backend.py`
- `benchmarks_2d.py`
- `tests/test_2d_exact.py`
- notebook内M1 execution cell

## 3.3 work packages

### M1-A: representation convention

- `Irrep(j2: int)` を実装。
- dimension、Casimir、tensor productを整数演算で実装。
- 半整数をfloatで保持しない。
- orientation reversalとdual representationのconventionを固定する。

### M1-B: Wilson character coefficients

\[
a_n(\beta)=\frac{2n}{\beta}I_n(\beta)
\]

を多倍長・区間で計算する。

- 正項級数によるlower/upper enclosure。
- library Bessel値はcross-checkにのみ使用。
- \(\beta\)、cutoff、precisionをcache keyに含める。

### M1-C: analytic tails

少なくとも次を実装する。

\[
\tau_N^{(0)}
\ge
\left\|\bar w_\beta-P_N\bar w_\beta\right\|_\infty,
\]

\[
\tau_N^{(1)}
\ge
\left\|\nabla(\bar w_\beta-P_N\bar w_\beta)\right\|_\infty.
\]

- cutoff増加に伴う単調減少をtest。
- 既存prototypeの数値をregression fixtureにする。

### M1-D: exact 2D RG

- coefficientwise exact recurrence。
- 1〜5段のinterval trajectory。
- checkpoint/resume後にbitwise同じmetadataと同じ区間を得る。

### M1-E: independent cross-check

二つの独立経路を用意する。

1. coefficient recurrence。
2. 小さいgroup quadratureまたはexplicit convolution。

低cutoffで両者が誤差区間内に一致することを確認する。

## 3.4 数学的成果

M1完了時に証明されるのは次である。

> 指定した \(\beta,N\) について、初期Wilson weightのrepresentation tailと一次微分tailが明示的区間で囲われ、2次元の有限回exact RGが同じ区間演算基盤上で再現される。

4次元RGに関する主張はまだ行わない。

## 3.5 acceptance conditions

- 2D recurrenceの全testが通る。
- tailが高精度参照値を含む。
- cutoff増加でtail upper boundが非増加。
- fresh process resume testに成功。
- M1 reportに全conventionと式を記載。
- `certification_status` は `NOT_CERTIFIED` のまま。

## 3.6 失敗時の分岐

- Bessel enclosureが広い: precisionまたはtail majorantを改善。
- quadratureと不一致: Haar measure・character normalizationを再確認。
- checkpoint再現性不一致: sector orderingとRNG依存を除去。

## 3.7 想定session数

2〜4 session。M0のmodule化が不十分なら追加1〜2 session。

---

# 4. M2 — 4次元low-cutoff armillary tensorの厳密同定

## 4.1 目的

4次元の局所tensorについて、matrix-index formulationとarmillary/fusion formulationが低cutoffで同じtensorを表すことを検証する。

ここではRG近似よりも、**基底変換・orientation・normalizationの正しさ**を確立することが主目的である。

## 4.2 初期範囲

- \(j_{\max}=1/2\) から開始。
- 通過後に \(j_{\max}=1\)、必要なら \(3/2\)。
- 一つのlink starまたは最小4D local cellを対象にする。

## 4.3 実装対象

- `fusion.py`
- `wigner_cache.py`
- `armillary.py`
- `dense_reference.py`
- `sector_canonicalization.py`
- `tests/test_armillary_equivalence.py`

## 4.4 work packages

### M2-A: Wigner/fusion conventions

- Clebsch–Gordan係数のphase conventionを固定。
- orthogonalityとcompleteness test。
- \(3j,6j\) symmetry test。
- versionとconvention hashをcacheへ保存。

### M2-B: dense reference

低cutoffに限ってmatrix indicesを保持したtensorを生成する。

- tensor dimensionとorderingをmanifestに保存。
- exact zeroとnumerical small valueを区別する。
- CPU高精度で生成可能な範囲に限定する。

### M2-C: armillary generator

- link周囲のrepresentation labels。
- fusion tree。
- singlet projection。
- orientation sign。
- lattice direction permutation。

を明示的な`SectorKey`へ変換する。

### M2-D: basis equivalence

armillary tensorをdense basisへ戻し、reference tensorとの差を評価する。

\[
\|T_{\mathrm{dense}}-U T_{\mathrm{arm}}U^*\|
\le \varepsilon_{\mathrm{basis}}.
\]

低cutoffではCPU多倍長で \(\varepsilon_{\mathrm{basis}}\) を極小にする。

### M2-E: symmetry reduction

- cubic rotations。
- reflection/orientation reversal。
- gauge singlet。
- equivalent sector canonicalization。

により保存sector数を削減する。

## 4.5 数学的成果

M2完了時に証明されるのは次である。

> 指定した低cutoffで、armillary representationは通常のPeter–Weyl matrix-index tensorと同じ局所gauge tensorを表し、全phase・normalization・orientation conventionが固定されている。

## 4.6 acceptance conditions

- dense vs armillary差が指定tolerance内で厳密に囲われる。
- gauge-noninvariant sectorが消える。
- symmetry orbitの代表選択が決定的。
- cacheを削除して再生成しても同じhash。
- tensor shardのcheckpoint round-trip成功。

## 4.7 失敗時の分岐

- phase mismatch: Wigner conventionを一つに限定しcacheをinvalidate。
- sector explosion: canonicalizationを先に適用。
- dense referenceが大きすぎる: \(j_{\max}=1/2\) の局所部分に縮小し、局所identityを積み上げる。

## 4.8 想定session数

5〜12 session。

---

# 5. M3 — GPU matrix-free Triad-ATRG pilot

## 5.1 目的

巨大tensorを明示的に生成せず、armillary sector blockを用いて

\[
x\mapsto Ax,
\qquad
y\mapsto A^*y
\]

をGPU contractionとして実装し、Triad系factorizationとRSVDを行う。

M3は探索段階であり、最終certificateはまだ出さない。

## 5.2 実装対象

- `contraction_backend.py`
- `linear_operator.py`
- `triad_atrg.py`
- `rsvd.py`
- `gpu_sharding.py`
- `path_cache.py`
- `tests/test_matrix_free.py`

## 5.3 work packages

### M3-A: backend abstraction

優先順位:

1. cuTensorNet/cuQuantum。
2. torch CUDA + opt_einsum。
3. torch CUDA。
4. CPU reference。

全backendで同じoperator interfaceを使う。

### M3-B: matrix-free contraction

- input sector shard。
- output sector shard。
- contraction graph。
- path cache。
- slicing。

をwork item化する。

### M3-C: deterministic ordering

GPU reduction orderが変わっても数学的metadataは同一になるようにする。
近似値のbitwise一致は要求しないが、sector ordering・seed・pathを固定する。

### M3-D: RSVD探索

- target rank \(\chi\)。
- oversampling \(p\)。
- power iteration \(q\)。
- fixed random seed。
- singular value decay report。

を実装する。

### M3-E: Triad factorization

各directionのcoarse grainingを小さなfactorへ分け、中間rankとmemory peakを記録する。

### M3-F: resource adaptation

- OOM時にshard sizeを半分。
- 3回OOMで`BLOCKED_RESOURCE`。
- remaining timeに応じてwork itemを開始しない。
- 5時間20分以前にGPU stateをCPUへ移す。

## 5.4 数学的成果

M3単独では厳密なRG enclosureを主張しない。得られるのは、

> 指定cutoffにおけるfinite coreの候補低rank部分空間、特異値減衰、contraction graph、および計算資源の実測。

である。

## 5.5 acceptance conditions

- matrix-free `matvec` が小規模explicit matrixと一致。
- adjoint consistency:
  \[
  \langle Ax,y\rangle\approx\langle x,A^*y\rangle.
  \]
- RSVD再構成誤差がexplicit SVDと一致する低cutoff test。
- checkpoint後に同じbasis候補を復元可能。
- GPU OOM recovery test。
- path cacheが再利用される。

## 5.6 screening rule

近似的influence spectral radiusまたはproxyを計算し、

- \(>1.2\): 現schemeを早期終了。
- \(1.0\)付近: cutoff/rank依存を調査。
- \(<0.8\): M4/M5へ優先的に進む。

この値は証明ではない。

## 5.7 想定session数

8〜20 session。GPU modelとcutoffに強く依存する。

---

# 6. M4 — source derivativeと完全な誤差台帳

## 6.1 目的

境界sourceに対する一次応答をprimal tensorと同時に伝播し、各RG操作の誤差をprovenance付きで合成する。

## 6.2 実装対象

- `forward_ad.py`
- `source_channels.py`
- `error_ledger.py`
- `normalization.py`
- `tests/test_forward_ad.py`

## 6.3 source classes

初版では格子対称性を使い、独立sourceを限定する。

- temporal link source。
- spatial link source。
- electric-like channel。
- plaquette/magnetic-like channel。
- 低representation boundary mode。

後で必要に応じて拡張する。

## 6.4 work packages

### M4-A: dual tensor

\[
T(s)=T+s\dot T+O(s^2)
\]

を`DualTensor(primal, tangent)`として保持する。

### M4-B: multilinear differentiation

contraction、normalization、basis projection、coarse regroupingの微分を実装する。

### M4-C: basis dependence

SVD basis \(Q(T)\) の微分を無視しない。初版は安全側に、

- primalから得た固定basisをtangentにも適用。
- basis variationの影響をprojection residualへ含める。

方式を採用する。

### M4-D: derivative regression

小規模ケースでcentered finite differenceと比較する。

\[
\frac{T(h)-T(-h)}{2h}
\]

ただしfinite differenceは検算用であり、証明誤差には使わない。

### M4-E: error ledger

誤差項をDAGで管理する。

- initial representation tail。
- basis equivalence error。
- input radius propagation。
- GPU rounding/backward error。
- RSVD projection residual。
- omitted fusion/channel tail。
- normalization error。
- tangent error。

全項にsource checkpointと計算式を記録する。

## 6.5 数学的成果

M4完了時に、各一段RG outputについて

\[
\|T_1-\widetilde T_1\|\le\varepsilon_1,
\qquad
\|\dot T_1-\widetilde{\dot T}_1\|\le\varepsilon_1^{(1)}
\]

を構成するための全誤差成分が定義される。

この段階ではRSVD残差など一部がまだ近似上界でもよいが、その場合は`BLOCKED_MATH`と明示する。

## 6.6 acceptance conditions

- source derivativeのsymmetry relationが成立。
- zero sourceで不要なtangentが0。
- finite difference comparisonが収束。
- 全output radiusが非負・finite。
- provenanceのないradius項が存在しない。
- basis variationをzeroとして扱っていない。

## 6.7 想定session数

6〜15 session。

---

# 7. M5 — 一段RGの数学的certification

## 7.1 目的

一段RGについて、finite core、全tail、derivative、normalizationを厳密に囲い、interval influence matrixを構成する。

M5は最初の本格的な数学的certificateである。

## 7.2 必要な定理・補題

### C1: initial tail theorem

指定 \(\beta,N\) について

\[
\|T_0-P_NT_0\|\le\varepsilon_{\mathrm{rep},0},
\qquad
\|\dot T_0-P_N\dot T_0\|\le\varepsilon_{\mathrm{rep},1}.
\]

### C2: armillary equivalence theorem

有限cutoff coreが元のPeter–Weyl truncated tensorと同値である。

### C3: contraction stability lemma

multilinear contractionに対してinput radiiを明示的に伝播できる。

### C4: deterministic low-rank residual theorem

GPUで得たbasis \(Q\) を固定した後、

\[
\|(I-QQ^*)A\|
\le\varepsilon_{\mathrm{svd}}
\]

をprobabilityに依存しない方法で上から評価する。

許可する方法:

1. sectorwise explicit Frobenius residual。
2. deterministic block norm bound。
3. CPU高精度projected contraction。
4. 解析的discarded-channel bound。

randomized failure probabilityだけでは不十分。

### C5: rounding/backward error theorem

各GPU contractionについて、使用algorithmに対応した後方誤差または再計算残差を上から評価する。

初版では、最終証明対象となる縮約をCPU高精度でsectorwise再計算する方式を優先する。

### C6: normalization lower bound

局所kernel \(K\) に対し

\[
Z(\eta)=\int K(x;\eta)\,dx\ge z_{\min}>0
\]

を全許容境界dataで証明する。

### C7: influence lemma

\[
\|K-\widetilde K\|_1\le\varepsilon_0,
\quad
\|\partial_jK-\partial_j\widetilde K\|_1\le\varepsilon_{1,j}
\]

および \(\widetilde Z_{\min}-\varepsilon_0>0\) から

\[
c_{ij}
\le
\operatorname{diam}(E_i)
\frac{
\|\partial_j\widetilde K\|_1+\varepsilon_{1,j}
}{
\widetilde Z_{\min}-\varepsilon_0
}
\]

型のentrywise上界を作る。

### C8: finite Perron certificate

正ベクトル \(w\) に対し

\[
\rho(B)
\le
\max_a\frac{(\overline B w)_a}{w_a}.
\]

## 7.3 実装対象

- `residual_validation.py`
- `interval_kernel.py`
- `influence.py`
- `certificate.py`
- `proof_manifest.py`
- `tests/test_one_step_certificate.py`

## 7.4 one-step certificate package

```text
one_step_certificate/
├── theorem_statement.md
├── config.json
├── code_hashes.json
├── conventions.json
├── initial_tail.json
├── basis_equivalence.json
├── contraction_residuals.json
├── derivative_residuals.json
├── normalization_bounds.json
├── influence_matrix_intervals.json
├── perron_vector.json
├── collatz_bound.json
├── proof_dependencies.json
└── verdict.json
```

## 7.5 acceptance conditions

- C1–C8が全て`PASS`。
- interval denominatorが正。
- NaN/Infなし。
- 全誤差の合計が重複計上されていない。
- independent verifier notebookでcertificate packageを再検証可能。
- `ONE_STEP_CERTIFIED`は一段の不等式が成立した場合のみ。
- 不等式が成立しなくても計算が正しければ`NOT_CERTIFIED`として完了可能。

## 7.6 失敗時の分岐

- residualが大きい: bond dimensionまたはcutoff増加。
- \(z_{\min}\le0\): kernel parameterization・normalization enclosureを改善。
- influenceが1以上: block size、source norm、RG schemeを変更。
- proof marginがroundingに埋もれる: final sectorのみ高精度再計算。

## 7.7 想定session数

10〜25 session。

---

# 8. M6 — 複数段validated RGと最終certificate

## 8.1 目的

M5で確立した一段enclosureを3〜5段に合成し、最終coarse specificationについて

\[
q_{\mathrm{cert}}<1
\]

を厳密に検証する。

## 8.2 重要原則

一段certificateを単純にコピーしない。各段でtensor family、normalization、source derivative、tailが変化するため、段ごとの状態を個別に保存する。

## 8.3 family enclosure

単一tensorだけでなく、入力誤差ball全体

\[
\mathcal K_r
=
\{T:\|T-\widetilde T_r\|\le\varepsilon_r\}
\]

を次段へ写す。

\[
\mathcal R(\mathcal K_r)
\subset
\mathcal K_{r+1}.
\]

一段mapの局所Lipschitz boundまたはmultilinear radius propagationを使う。

## 8.4 work packages

### M6-A: step composition

- step-specific core。
- step-specific basis。
- input radius。
- source tangent radius。
- normalization scale。
- analytic tails。

をcheckpointごとに保存する。

### M6-B: adaptive cutoff/rank

各段で別の \(j_{\max},\chi\) を許可する。

- tailがmarginの25%超: cutoff増加。
- low-rank residualがmarginの25%超: rank増加。
- 合計がmarginの50%超: 次段へ進まない。

### M6-C: normalization control

各段でoverall scalarを分離し、overflow/underflowを避ける。
physical influenceに影響しないnormalizationと、kernel denominatorに必要なnormalizationを区別する。

### M6-D: multi-step derivative

chain ruleでsource tangentを伝播する。
各段でsource supportがどうcoarse blockへ写るかを記録する。

### M6-E: final influence matrix

最終stepで、有限個のblock type・displacement・channelについてentrywise intervalを作る。

\[
(\overline B_m)_{ab}
=
\sum_z e^{m|z_0|}\overline c_{ab}(z).
\]

spatial/time symmetryで重複を除去するが、除去の証明をmanifestへ記録する。

### M6-F: Collatz–Wielandt optimization

GPU/floatで候補 \(w\) を求め、positive rational vectorへ外向き丸めする。
CPU多倍長で

\[
q_{\mathrm{CW}}
=
\max_a\frac{(\overline B_m w)_a}{w_a}
\]

を再計算する。

### M6-G: independent verification

main notebookとコード共有を最小化した`certificate_verifier.ipynb`を作る。
それは保存済みinterval・hash・formulaだけを読み、最終判定を再計算する。

## 8.5 最終acceptance conditions

全て必要。

1. M1–M5のcertificate dependencyが有効。
2. 各RG stepが`STEP_ENCLOSED`。
3. family inclusionが全段で証明済み。
4. derivative enclosureが全段で有効。
5. final normalization lower boundが正。
6. final influence matrixがentrywise上界。
7. positive rational Perron vectorを保存。
8. independent verifierが同じ \(q_{\mathrm{cert}}\) を再現。
9. \(q_{\mathrm{cert}}<1\)。
10. 全artifactのhashとenvironment metadataが保存済み。

## 8.6 最終出力

```text
final_certificate/
├── README.md
├── theorem_scope.md
├── assumptions.md
├── run_config.json
├── environment.json
├── source_hashes.json
├── checkpoint_chain.json
├── rg_step_00/
├── rg_step_01/
├── ...
├── final_influence_matrix.json
├── perron_vector.json
├── final_bound.json
├── independent_verifier_report.json
├── limitations.md
└── verdict.json
```

`verdict.json`例:

```json
{
  "status": "CERTIFIED",
  "scope": "finite-cutoff, finite-step 4D SU(2) RG influence certificate",
  "q_collatz_upper": "0.8731...",
  "analytic_tail_upper": "0.0012...",
  "numerical_residual_upper": "0.0034...",
  "q_cert_upper": "0.8777...",
  "margin_lower": "0.1222..."
}
```

## 8.7 想定session数

15〜50 session。cutoff・bond dimension・marginに強く依存する。

---

# 9. M6後に数学的に言えること

M6だけで直接証明するscopeは慎重に限定する。

## 9.1 計算だけで証明するもの

- 指定された4D \(SU(2)\) local tensor/RG scheme。
- 指定されたbare parametersまたはinterval family。
- 指定された有限回RG。
- 指定されたsource classとnorm。
- 最終weighted influence matrixの収縮。

## 9.2 外部定理を接続して得るもの

別途、以下を明示的に証明または引用する必要がある。

1. tensor/RG representationが元の格子measureとexactに対応する。
2. influence contractionから無限体積の指数clusterが従う。
3. reflection positivityとtransfer matrix。
4. punctured RGによるfine observableの保持。
5. continuum scaling familyへの一様性。
6. OS continuum limitの存在と非自明性。

M6の`theorem_scope.md`では、計算certificateとこれらの外部bridgeを混同しない。

---

# 10. 依存関係

```text
M0
 └─ M1
     └─ M2
         └─ M3
             └─ M4
                 └─ M5
                     └─ M6
```

部分的並列化可能:

- M1 tail改善とM2 Wigner cache。
- M3 backendとM4 error-ledger skeleton。
- M5 independent verifier skeletonはM4後半から開始可能。

ただしmilestone acceptanceを飛ばして次を`complete`にしてはならない。

---

# 11. Codex作業規約

各milestone開始時にCodexへ渡すもの:

- このroadmap。
- `MATHEMATICAL_CERTIFICATION_SPEC.md`。
- 前milestone report。
- 最新checkpoint summary。
- 該当milestone prompt。

各milestone終了時にCodexが返すもの:

1. 変更file一覧。
2. 実装した数学式とcode objectの対応。
3. 実行したtests。
4. restart test結果。
5. GPU peak memoryと処理時間。
6. 厳密に証明済みの項目。
7. heuristicのままの項目。
8. 次milestoneへ進めるかの判定。

---

# 12. 最初に実行する順序

1. M0の正式testを完了。
2. M1をCPU中心で完成。
3. M2を \(j_{\max}=1/2\) で完成。
4. M3をsmall GPU runでprofiling。
5. M4でsource derivativeとerror ledgerを完成。
6. M5で一段certificateを完成。
7. 一段のapproximate contractionが弱い場合、M6へ進まずschemeを変更。
8. marginが十分ならM6の3段から開始し、必要な場合だけ5段へ増やす。
