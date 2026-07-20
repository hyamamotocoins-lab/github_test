# validated 4D SU(2) RG — Paperspace 共通プロジェクト実行ガイド

この bundle は M0–M6 を**同じ Paperspace project と永続ストレージ**で管理し、段階ごとの
ノートブックを fresh kernel で実行する構成です。旧 `validated_4d_su2_rg_gpu_driver.ipynb` は
案内専用です。実行入口は次の順序です。

1. `notebooks/00_project_setup.ipynb`
2. `notebooks/10_m0_accepted_audit.ipynb`
3. `notebooks/20_m1_exact_2d.ipynb`
4. `notebooks/30_m2_armillary.ipynb`
5. `notebooks/40_m3_gpu_triad_atrg.ipynb`
6. `notebooks/50_m4_derivatives.ipynb`
7. `notebooks/60_m5_one_step_certificate.ipynb`
8. `notebooks/70_m6_multistep_certificate.ipynb`

最初の setup notebook と各段階 notebook は、必要な依存を確認してから実行します。M1/M2 の
証明経路は CPU exact arithmetic、M3 の探索経路は CUDA FP64 を使います。M3 は
cuTensorNet が利用できれば優先し、今回の実測環境では `torch_cuda_opt_einsum` を使用しました。
TF32 は無効化します。

現在は M1 の二次元 exact benchmark、M2 の low-cutoff 4D armillary tensor、M3 の
GPU matrix-free Triad-ATRG pilot まで実行済みです。人間向けの判定と参照確認は次を開いてください。

- `notebooks/35_m2_execution_reference_report.ipynb`
- `notebooks/45_m3_execution_reference_report.ipynb`

すべての M1–M3 経路で

```text
certification_status = NOT_CERTIFIED
```

を維持します。M1/M2 の厳密結果も、M3 の再現可能な探索結果も、4次元 RG や mass gap の
certification ではありません。

## 1. 現在の milestone 状態

| milestone | 状態 | 意味 |
|---|---|---|
| M0 | 受理済み・凍結 | persistence、checkpoint、resume、CPU/GPU test が外部実行で完了 |
| M1 | 受理済み・凍結 | 全受理ゲート PASS、`ckpt_000014`、`NOT_CERTIFIED` |
| M2 | 実行・受理済み | 全14ゲート PASS、exact dense/armillary equivalence、`NOT_CERTIFIED` |
| M3 | 実行完了・独立レビュー待ち | 全18ゲート PASS、`CORE_REPRODUCED`、`NOT_CERTIFIED` |
| M4 | 受理ゲートのみ・未実装 | M3 受理後の source derivative と error ledger |
| M5 | 受理ゲートのみ・未実装 | M4 受理後の deterministic one-step validation |
| M6 | 受理ゲートのみ・未実装 | M5 受理後の multi-step certificate と独立 final verifier |

M4 は M3 report の独立レビューと受理記録が完了するまで開始しません。

現在の完了 run は次です。いずれも外部永続ストレージ `/storage` に保存されています。

```text
run ID: M1-20260719T235423Z-a7cacde2ead9
phase: M1_COMPLETE
checkpoint: /storage/validated_4d_su2_rg/runs/M1-20260719T235423Z-a7cacde2ead9/checkpoints/ckpt_000014
report: /storage/validated_4d_su2_rg/runs/M1-20260719T235423Z-a7cacde2ead9/reports/M1_report.json
acceptance: /storage/validated_4d_su2_rg/runs/M1-20260719T235423Z-a7cacde2ead9/reports/M1_acceptance.json
certification status: NOT_CERTIFIED

run ID: M2-20260720T005145Z-dd3e385d0a61
phase: M2_COMPLETE
checkpoint: /storage/validated_4d_su2_rg/runs/M2-20260720T005145Z-dd3e385d0a61/checkpoints/ckpt_000014
report: /storage/validated_4d_su2_rg/runs/M2-20260720T005145Z-dd3e385d0a61/reports/M2_report.json
acceptance: /storage/validated_4d_su2_rg/runs/M2-20260720T005145Z-dd3e385d0a61/reports/M2_acceptance.json
certification status: NOT_CERTIFIED

run ID: M3-20260720T013551Z-ae995e91e861
phase: M3_COMPLETE
milestone status: CORE_REPRODUCED
checkpoint: /storage/validated_4d_su2_rg/runs/M3-20260720T013551Z-ae995e91e861/checkpoints/ckpt_000014
report: /storage/validated_4d_su2_rg/runs/M3-20260720T013551Z-ae995e91e861/reports/M3_report.json
acceptance: /storage/validated_4d_su2_rg/runs/M3-20260720T013551Z-ae995e91e861/reports/M3_acceptance.json
certification status: NOT_CERTIFIED
```

M3 の実測要約：RTX A4000、FP64、rank 16、operator dimension 729、matrix-free/explicit
最大絶対誤差 `1.56e-17`、adjoint 相対誤差 `3.55e-16`、RSVD 相対残差 `0.23999`、GPU peak
約35.3 MiB、checkpoint 398,702 bytes（save 0.092 s）です。RSVD 残差、特異値、Triad 残差、
influence proxy はすべて探索値で、rigorous bound はありません。

## 2. 受理済み M0 親成果物

M1 は次の M0 run を immutable な親成果物として扱います。

```text
parent milestone: M0
parent run ID: 20260719T120406Z-731966c8fd28
parent checkpoint: ckpt_000014
parent checkpoint path:
/storage/validated_4d_su2_rg/runs/20260719T120406Z-731966c8fd28/checkpoints/ckpt_000014
M0 phase: M0_COMPLETE
M0 certification status: NOT_CERTIFIED
```

受理範囲は [audit/m0_accepted_parent.json](audit/m0_accepted_parent.json) に固定しています。これは提示
された M0 report に基づく受理であり、第三者が checkpoint 全体を独立再実行したという記録
ではありません。その制限も JSON に明記しています。

M1 作成時には親 checkpoint を read-only で検査します。

1. directory 名が `ckpt_000014` であること。
2. `COMMITTED` と `hashes.json` が存在すること。
3. symlink がないこと。
4. `hashes.json` の file-set が実 file-set と一致すること。
5. 全 file の SHA-256 が一致すること。
6. state が `M0_COMPLETE`、checkpoint index 14、`NOT_CERTIFIED` であること。
7. queue が `done=6, pending=running=failed=0` であること。
8. 各 done item の result artifact、SHA-256、`.done` marker が一致すること。

検査した `hashes.json` 自体の SHA-256 と M0 受理記録の SHA-256 を、新しい M1
`run_manifest.json` に保存します。M0 checkpoint を変更、移動、上書きしません。

## 3. Governing documents

次の文書を M1 run identity に含め、SHA-256 を固定します。

- `validated_4d_su2_rg_codex_design.md`
- `AGENTS.md`
- `validated_4d_su2_rg_full_plan_bundle/validated_4d_su2_rg_codex_design_v0_2.md`
- `validated_4d_su2_rg_full_plan_bundle/M1_M6_VALIDATED_RG_ROADMAP.md`
- `validated_4d_su2_rg_full_plan_bundle/MATHEMATICAL_CERTIFICATION_SPEC.md`
- `validated_4d_su2_rg_full_plan_bundle/CODEX_PROMPTS_M1_M6.md`
- `validated_4d_su2_rg_full_plan_bundle/AGENTS_validated_4d_su2_rg_v0_2.md`

次の3点は証明済み入力ではなく、回帰参照として別の hash 区分に保存します。

- `validated_su2_rg_prototype.py`
- `validated_su2_rg_report.md`
- `validated_4d_su2_rg_M1_M6_tracker.ipynb`

prototype の数値を M1 output にコピーしません。今回の正項有理級数で再計算した enclosure が、
prototype report の外向き丸め区間に含まれることだけを test します。

## 4. M1 で実装する数学

### 4.1 SU(2) representation 規約

半整数 representation を float で保存しません。

```python
@dataclass(frozen=True, order=True)
class Irrep:
    j2: int
```

規約は

\[
j=\frac{j2}{2},\qquad d_j=j2+1,\qquad
C_2(j)=j(j+1)=\frac{j2(j2+2)}4
\]

です。SU(2) irrep は self-dual とし、orientation reversal は同じ `Irrep` を返します。
tensor product は `j2` の整数列として構成します。

class angle は

\[
\operatorname{Tr}U=2\cos\theta
\]

で規格化します。

### 4.2 normalized Wilson weight と character coefficient

使用する plaquette weight は

\[
\bar w_\beta(U)=e^{\beta(\cos\theta-1)}
\]

です。unnormalized expansion を

\[
e^{\beta\cos\theta}=\sum_{n\ge1}a_n(\beta)\chi_n(U)
\]

とすると、SU(2) class Haar measure と
\(\chi_n(\theta)=\sin(n\theta)/\sin\theta\) から

\[
a_n(\beta)
=\frac2\pi\int_0^\pi e^{\beta\cos\theta}\sin\theta\sin(n\theta)\,d\theta
=I_{n-1}(\beta)-I_{n+1}(\beta)
=\frac{2n}{\beta}I_n(\beta)
\]

を得ます。最後の等式は modified Bessel recurrence であり、下記の正項級数から項別に
確認できます。実装では library の Bessel 値を証明に使わず、次の正項級数を直接囲います。

\[
I_n(\beta)=\sum_{k=0}^{\infty}
\frac{(\beta/2)^{2k+n}}{k!(n+k)!}.
\]

有限和の最後の項を \(t_K\) とすると、次項比は

\[
q_{K+1}=\frac{(\beta/2)^2}{(K+1)(n+K+1)}.
\]

以後の項比は単調に減少します。\(q_{K+1}<1\) を確認した場合だけ

\[
0\le\sum_{k>K}t_k\le \frac{t_Kq_{K+1}}{1-q_{K+1}}
\]

を使用します。比が1以上なら精度不足として停止します。

### 4.3 representation value tail

identity では \(\chi_n(I)=n\) なので、係数の正値性から

\[
e^\beta=\sum_{n\ge1}n a_n(\beta)
\]

です。また \(|\chi_n(U)|\le n\) より、dimension cutoff \(N\) に対し

\[
\left\|\bar w_\beta-P_N\bar w_\beta\right\|_\infty
\le e^{-\beta}\sum_{n>N}n a_n(\beta)
=e^{-\beta}\sum_{n>N}\frac{2n^2}{\beta}I_n(\beta)
=:\tau_N^{(0)}.
\]

`exp(beta)`、有限係数和、除算をすべて有理区間で行います。tail を未知量0として扱いません。
4次元 \(2^4\) cell の全216 plaquette を置換する粗い telescoping bound として
\(216\tau_N^{(0)}\) も記録します。

### 4.4 Casimir-gradient tail

bi-invariant metric は \(C_2(j)=j(j+1)\) で規格化します。実装が使用するのは、より鋭い
metric-dependent estimate ではなく、明示的で保守的な

\[
\|\nabla\chi_j\|_\infty\le\frac{n^2}{2}
\]

です。class-angle 上で
\(\chi_j(\theta)=\sum_{m=-j}^{j}e^{2im\theta}\) と展開し、微分後に三角不等式を使うと
\(\sum_m|2m|\le n^2/2\) なので、この上界が得られます。

さらに、正項級数の各分母を下から評価すると

\[
I_n(\beta)
\le \frac{(\beta/2)^n}{n!}
\exp\!\left(\frac{\beta^2}{4(n+1)}\right).
\]

したがって

\[
\tau_N^{(1)}
\le e^{-\beta}\sum_{n>N}\frac{n^3}{\beta}I_n(\beta)
\]

を、\(n^3(\beta/2)^n/n!\) の単調減少する項比を使って上から囲います。

- `6 × tail`: 一つの4次元 fine link に接する6 plaquette に対する bound。
- `216 × tail`: cell 全体に variation が作用する場合だけの意図的に粗い比較値。

source derivative に対して216を無条件に使用しません。value tail と gradient tail は別
artifact、別 provenance です。

### 4.5 二次元 exact 2×2 RG

normalized Fourier ratio を

\[
r_n=\frac{a_n}{n a_1}
\]

とします。central class function の Haar convolution では Schur orthogonality により

\[
\chi_n*\chi_m=\delta_{nm}\frac{\chi_n}{n}.
\]

4 plaquette の convolution coefficient は \(a_n^4/n^3\) となるので、再規格化後には

\[
r_n'=r_n^4
\]

です。M1 は

```text
beta = 11/5
n = 2, 3, 4
steps = 0, 1, 2, 3
```

の全 endpoint を exact rational interval として保存します。

### 4.6 独立 verifier

独立経路は primary recurrence 関数を呼びません。

1. Bessel 級数の各項を factorial から直接再構成する。
2. primary より少ない72項と rigorous remainder を用い、意図的に広い係数区間を作る。
3. finite representation basis 上の diagonal convolution operator を構成する。
4. 各 step で係数を \(a_n^4/n^3\) として更新する。
5. primary interval が independent interval に含まれ、かつ両者が交差することを確認する。

`python-flint`/Arb がない場合は `NOT_AVAILABLE_NOT_REQUIRED` と記録します。Arb がなくても、
二つの独立した有理演算経路を必須 verifier として維持します。

## 5. 生成される module と test

M1 module：

```text
src/
├── exact_arithmetic.py
├── su2_representations.py
├── su2_special_functions.py
├── tail_bounds.py
├── exact_2d_rg.py
├── m1_verifier.py
├── m1_config.py
├── m1_reporting.py
└── m1_orchestrator.py
```

M0 の `common.py`、`runtime.py`、`session_guard.py`、`work_queue.py`、`checkpoint.py` を
再利用します。`checkpoint.py` は M0 API を維持しつつ、M1 state も同じ atomic commit、
SHA-256 検証、corrupt-latest fallback で保存できるよう一般化しています。

M1 test：

```text
tests/test_m1_exact_arithmetic.py
tests/test_m1_coefficients.py
tests/test_m1_tails.py
tests/test_m1_exact_2d_rg.py
tests/test_m1_independent_verifier.py
tests/test_m1_restart.py
tests/test_m1_fail_closed.py
```

## 6. Paperspace の準備

Paperspace の公式構成では `/notebooks` と `/storage` が永続領域です。この driver は source
を `/notebooks`、run/checkpoint/artifact を `/storage` に置きます。

- [Paperspace Storage Architecture](https://docs.digitalocean.com/products/paperspace/notebooks/details/storage-architecture/)
- [Paperspace Storage Types](https://docs.digitalocean.com/products/paperspace/notebooks/details/storage-types/)

### 6.1 runtime

Python 3.11 以上の PyTorch/CUDA runtime を選びます。runtime 名は更新され得るため、起動後に
実体を確認します。

```bash
python --version
python -c "import numpy; print(numpy.__version__)"
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

この project では pytest だけが不足している前提です。`00_project_setup.ipynb` の最初の
コードセルが、未導入の場合だけ次と同等の処理を行います。

```bash
cd /notebooks/validated_4d_su2_rg_codex_bundle
python -m pip install -r requirements/paperspace.txt
```

PyTorch は CUDA runtime と整合させる必要があるため `requirements/paperspace.txt` から
導入しません。M1 の proof path は CPU 有理演算です。GPU test は環境 smoke test であり、
GPU float を rigorous interval として使いません。

### 6.2 bundle 配置

完全な directory を次へ置きます。

```text
/notebooks/validated_4d_su2_rg_codex_bundle/
├── validated_4d_su2_rg_gpu_driver.ipynb       # 旧パス互換の案内のみ
├── notebooks/
│   ├── 00_project_setup.ipynb
│   ├── 10_m0_accepted_audit.ipynb
│   ├── 20_m1_exact_2d.ipynb
│   ├── 30_m2_armillary.ipynb
│   ├── 40_m3_gpu_triad_atrg.ipynb
│   ├── 50_m4_derivatives.ipynb
│   ├── 60_m5_one_step_certificate.ipynb
│   └── 70_m6_multistep_certificate.ipynb
├── audit/
│   └── m0_accepted_parent.json
├── requirements/
│   └── paperspace.txt
├── validated_4d_su2_rg_codex_design.md
├── AGENTS.md
├── README.md
└── validated_4d_su2_rg_full_plan_bundle/
```

notebook だけ、または full-plan directory だけを upload してはいけません。bootstrap は必須
file と M0 受理記録をすべて検査します。

### 6.3 永続保存先

Paperspace では次を自動設定します。

```text
VALIDATED_RG_PROJECT_ROOT=/notebooks/validated_4d_su2_rg_codex_bundle
VALIDATED_RG_PERSIST_ROOT=/storage/validated_4d_su2_rg
VALIDATED_RG_PERSIST_ACK=I_CONFIRM_THIS_PATH_IS_PERSISTENT
```

team 内で保存先を分ける場合は、bootstrap より前の一時 cell で変更します。

```python
import os
os.environ['VALIDATED_RG_PERSIST_ROOT'] = \
    '/storage/validated_4d_su2_rg/USER_OR_EXPERIMENT'
os.environ['VALIDATED_RG_PERSIST_ACK'] = \
    'I_CONFIRM_THIS_PATH_IS_PERSISTENT'
```

`/storage` は team 共有です。この M1 driver は分散 lock manager を持たないため、同じ
`VALIDATED_RG_PERSIST_ROOT` に複数 kernel から同時に書き込まないでください。

親 M0 checkpoint の path を変更した場合、既定の M1 config は fail-closed で停止します。
M0 を別 storage subdirectory で実行していた場合は、受理記録と M1 config を監査付きで
一致させる必要があります。path を推測して変更しないでください。

## 7. M1 の初回実行と再開

1. Paperspace で受理済み M0 と同じ storage region を使用する。
2. fresh kernel で `notebooks/00_project_setup.ipynb` を上から実行する。
3. fresh kernel で `notebooks/10_m0_accepted_audit.ipynb` を実行し、`M0_AUDIT.status=PASS` を確認する。
4. fresh kernel で `notebooks/20_m1_exact_2d.ipynb` を開く。
5. 上から順に全 cell を実行する。
6. M0 regression suite と M1 required CPU suite が `PASS` したことを確認する。
7. CUDA がある場合は M0/M1 GPU smoke suite も `PASS` したことを確認する。
8. `m1_orchestrator` が作成されたことを確認する。
9. 唯一の長時間入口を実行する。

```python
result = m1_orchestrator.run_until_checkpoint()
```

M1 run ID は M0 と混同しないよう、必ず `M1-` prefix を持ちます。

```text
M1-20260719T130000Z-0123456789ab
```

`VALIDATED_RG_RUN_ID` は M0 用なので M1 では使用しません。M1 を明示的に再開する場合は

```python
import os
os.environ['VALIDATED_RG_M1_RUN_ID'] = 'M1-表示された-run-id'
```

を設定します。未設定なら `LATEST_M1_RUN.json` の compatible run を候補にします。

### 7.1 6時間停止をまたぐ人手再開

通常は `run_until_checkpoint()` に任せます。15分ごとと各 work item の前後に保存し、5時間15分で
drain、遅くとも5時間20分で final checkpoint を開始し、5時間30分までに戻ります。戻り値と
`reports/session_summary.json` が表示されてから runtime を停止してください。

手動で早く止める必要がある場合は、Jupyter の interrupt を一度だけ実行します。
`KeyboardInterrupt` を受けた item は `pending` に戻され、可能な時間帯なら checkpoint が作られます。
`M1 checkpoint ... committed and verified` と session summary を確認してから shutdown します。
checkpoint 保存中に強制停止された場合も、次回は破損した最新候補を捨て、直前の valid checkpoint
へ戻ります。

次の session では、同じ Paperspace machine/runtime と同じ `/storage` を mount し、fresh kernel で
`notebooks/20_m1_exact_2d.ipynb` を上から実行します。特定 run を明示する場合は、
orchestrator 作成セルより前に次を実行します。

```python
import os
os.environ['VALIDATED_RG_PROJECT_ROOT'] = '/notebooks/validated_4d_su2_rg_codex_bundle'
os.environ['VALIDATED_RG_PERSIST_ROOT'] = '/storage/validated_4d_su2_rg'
os.environ['VALIDATED_RG_PERSIST_ACK'] = 'I_CONFIRM_THIS_PATH_IS_PERSISTENT'
os.environ['VALIDATED_RG_M1_RUN_ID'] = 'M1-20260719T235423Z-a7cacde2ead9'
```

その後、同じ入口を再実行します。

```python
result = m1_orchestrator.run_until_checkpoint()
```

完了済みの現在 run では `M1 already complete; no work or checkpoint was started` と返ります。
未完了 run では hash 検証済みの最新 checkpoint から queue を復元し、`running` item を `pending`
に戻して続行します。runtime signature が変わった場合は fail closed で停止するため、別 GPU image
や異なる PyTorch/CUDA 環境へ切り替えて同じ run ID を続けないでください。

## 8. M1 work item

queue は次の順で処理します。

```text
M1_COEFFICIENT_BATCH
M1_VALUE_TAIL
M1_GRADIENT_TAIL
M1_RG_TRAJECTORY
M1_INDEPENDENT_VERIFY
M1_REPORT
```

各 item は config hash と parameter から content-addressed ID を持ちます。item の前後で
checkpoint し、result JSON と `.done` marker を atomic commit します。runtime が item
完了直後に停止しても、次回 resume が result hash と marker を検査して `done` に修復します。

## 9. session 時間規約

| 経過時間 | 動作 |
|---|---|
| 0:00–5:00 | 20分以下の M1 work item を実行 |
| 5:00 | 新しい長時間 item を開始しない |
| 5:15 | drain と checkpoint を開始 |
| 5:20 | final checkpoint 開始の絶対上限 |
| 5:30 | hard return |

checkpoint trigger：

- 15分ごと。
- work item の開始前と完了後。
- drain/final-save 移行時。
- M1 acceptance 完了時。
- `KeyboardInterrupt` または例外時。ただし hard-return 後に遅い保存を開始しない。

各正常 session で次を更新します。

```text
reports/session_summary.json
reports/latest_metrics.json
reports/next_session_plan.md
```

## 10. checkpoint と再開

M0 と同じ方式を使います。

1. `checkpoints/.tmp-UUID/` に全 file を書く。
2. file と directory を可能な範囲で `fsync` する。
3. file-set と全 SHA-256 を `hashes.json` に保存する。
4. `ckpt_XXXXXX` へ atomic rename する。
5. `COMMITTED` marker を作る。
6. `LATEST.json` を atomic replace する。
7. commit 直後に再検証する。

load 時は新しい checkpoint から検査し、破損していれば以前の valid checkpoint へ fallback
します。config、source、governing document、reference artifact、parent M0、runtime signature
のいずれかが変わった場合は fallback ではなく compatibility error で停止します。

再開時は Python、NumPy、PyTorch、CUDA、cuDNN、GPU model/capability/device count を
`run_manifest.json` と RNG checkpoint で照合します。Paperspace の machine/runtime を
変更して signature が変わった場合、同じ run を黙って続けません。

## 11. M1 output

M1 notebook の identity は nbformat 4 の cell type、cell source、実行制御 tag を canonical JSON
として SHA-256 固定します。Jupyter が保存する output、execution count、cell ID は計算定義では
ないため identity から除外します。したがって自動保存後も同じ run を再開できますが、cell source
または実行制御 tag の変更は従来どおり compatibility error で fail closed になります。使用した
policy 名は `run_manifest.json` の `notebook_hash_policy` に保存します。

完了時に次を生成します。

```text
PERSIST_ROOT/runs/M1-RUN_ID/reports/
├── M1_report.json
├── M1_report.md
├── M1_acceptance.json
├── session_summary.json
├── latest_metrics.json
└── next_session_plan.md
```

`M1_report.json` には次を保存します。

- accepted M0 parent identity と checkpoint manifest hash。
- M0 acceptance record hash。
- M1 config、config hash、source hash、convention hash。
- governing/reference artifact hash。
- β、cutoff、series term 数、rounding policy。
- coefficient interval。
- value-tail interval と216倍 bound。
- gradient-tail interval、fine-link 6倍、coarse 216倍 bound。
- exact 2D trajectory。
- independent verifier method と全 containment check。
- 各 proof artifact の SHA-256。
- test、restart、GPU smoke status。
- checkpoint size/save/verify time と memory。
- rigorous result、heuristic result、unresolved issue。
- `certification_status=NOT_CERTIFIED`。

exact rational endpoint は numerator/denominator を16進文字列で保存します。非常に小さい step
でも情報を失いません。decimal endpoint は人間向けの外向き丸め表示です。

## 12. M1 acceptance conditions

次をすべて満たした場合だけ `phase=M1_COMPLETE` になります。

- M0 regression CPU suite が PASS。
- M1 required CPU suite が PASS。
- CUDA がある場合は GPU smoke suite が PASS。
- coefficient enclosure が正項有理級数に基づく。
- value tail が rigorous で cutoff に対して非増加。
- gradient tail が rigorous で cutoff に対して非増加。
- precision を増やした区間が低精度区間から外れない。
- exact 2D trajectory が rigorous。
- independent verifier が全12組の containment check に PASS。
- fresh-process resume と session drain/restart が PASS。
- queue に pending/running/failed item がない。
- final checkpoint が commit 後に再検証済み。
- status が `NOT_CERTIFIED` のまま。

tail artifact、independent verifier、test report のどれかが欠ける場合、M1 completion を拒否
します。未知量を0として穴埋めしません。

## 13. 既知の非主張と残る課題

M1 で rigorous とするのは、指定した β、cutoff、dimension、step に対する次だけです。

- Wilson character coefficient enclosure。
- representation value tail enclosure。
- Casimir-gradient tail upper bound。
- 二次元 exact RG interval trajectory。
- finite diagonal convolution による独立 containment。

M2 では low-cutoff の4次元 armillary tensor と dense/armillary exact equivalence を実装済みです。
M3 では GPU matrix-free operator、固定 seed RSVD、Triad factorization、明示行列との low-cutoff
比較を実装済みですが、これらは厳密 bound ではありません。次は未実装です。

- 4次元 RG step の a posteriori enclosure。
- low-rank residual、GPU rounding error、normalization lower bound の厳密評価。
- source tangent、influence matrix、Collatz–Wielandt bound。
- P0 の最終的な transitive checkpoint hash-chain certificate package。
- thermodynamic/continuum bridge B1–B7。

したがって M1–M3 report は `CERTIFIED`、`TAIL_BOUNDED` を4次元全体の verdict として出しません。
milestone acceptance と最終数学的 certification を区別します。

## 14. よくあるエラー

### `Python 3.11+ is required for M1`

active runtime が古いため、module を生成する前に停止しています。Python 3.11 以上の別
Paperspace runtime または custom container を選び、fresh kernel から再実行してください。

### `Accepted M0 checkpoint path is unavailable or unsafe`

受理済み M0 checkpoint が既定 `/storage` path にありません。storage region、mount、run ID、
checkpoint directory を確認してください。空 directory を作って回避してはいけません。

### `Accepted M0 checkpoint hash mismatch`

凍結すべき M0 file が変更または破損しています。M1 を開始せず、M0 audit artifact を保全して
原因を調査してください。

### `M1 run ID must use the M1- namespace`

M0 run ID を M1 に渡しています。`VALIDATED_RG_RUN_ID` を使用せず、必要なら
`VALIDATED_RG_M1_RUN_ID=M1-...` を設定してください。

### `M1 manifest/source/parent/runtime identity changed`

既存 M1 run の source、文書、親 checkpoint、runtime のいずれかが変わっています。元の
環境へ戻すか、変更理由を監査して新しい M1 run ID を作ってください。

### `M1 acceptance gates failed closed`

error に列挙された tail、verifier、test、queue 条件を確認します。判定を得るために bound を
緩めたり、missing result を PASS に書き換えたりしないでください。

### Paperspace で `cuda_available: false`

CPU machine または CUDA 非対応 runtime です。M1 の rigorous CPU path は実行できますが、
GPU machine を選んだのに false の場合は `nvidia-smi` と PyTorch/CUDA の整合を確認します。

### `/storage` 書込み probe が失敗する

storage mount、quota、team permission を確認します。`/tmp` や ephemeral local disk に変更して
続行しないでください。

## 15. 終了時の人間チェック

各 session 後：

- `certification_status` が `NOT_CERTIFIED`。
- run ID が `M1-` で始まり、M0 run ID と異なる。
- checkpoint path が M1 run directory 配下。
- `M1 checkpoint ... committed and verified` が表示された。
- `session_summary.json` と `next_session_plan.md` が更新された。
- `failed` item が0。
- M0 parent checkpoint の timestamp/content を変更していない。

M1 完了後：

- `phase=M1_COMPLETE`。
- queue が全件 `done`。
- `M1_acceptance.json` の全 gate が true。
- `M1_report.json` と `M1_report.md` が存在する。
- value/gradient tail が空でも0でもない。
- independent verifier の12 check がすべて containment PASS。
- rigorous/heuristic/unresolved の区分が report にある。
- M2 を自動開始していない。

この確認後に M1 を受理するかを別途判断してください。

## 16. M2–M6 の段階別ノートブック

M2–M6 のノートブックは、直前段階の `reports/M*_acceptance.json` を `/storage` から読み、
milestone、phase、`status=PASS`、`certification_status=NOT_CERTIFIED` を照合します。直前段階の
run ID は次の環境変数で明示します。

| 開く notebook | 必須の直前 run ID |
|---|---|
| `30_m2_armillary.ipynb` | `VALIDATED_RG_M1_RUN_ID=M1-...` |
| `40_m3_gpu_triad_atrg.ipynb` | `VALIDATED_RG_M2_RUN_ID=M2-...` |
| `50_m4_derivatives.ipynb` | `VALIDATED_RG_M3_RUN_ID=M3-...` |
| `60_m5_one_step_certificate.ipynb` | `VALIDATED_RG_M4_RUN_ID=M4-...` |
| `70_m6_multistep_certificate.ipynb` | `VALIDATED_RG_M5_RUN_ID=M5-...` |

M1 と M2 は実環境 report を人間が確認して受理済みです。M3 は GPU 実行まで完了し、全18ゲートが
PASS しましたが、`CORE_REPRODUCED` は certification ではありません。`notebooks/45_m3_execution_reference_report.ipynb`
で保存済み artifact と非主張を独立確認し、受理記録を作成するまで M4 を開始しません。後続段階でも、
直前 milestone の実環境 report を人間が受理した後にだけ、対応する notebook へ実装、test、
checkpoint/restart、独立 verifier、acceptance report を追加します。

## 17. 整理後の管理原則

- 編集対象の notebook は `notebooks/` に限定する。
- 受理根拠は `audit/`、追加依存は `requirements/` に分ける。
- governing documents と full-plan reference artifacts は hash/path を壊さないため移動しない。
- notebook 実行時に生成される `src/` と `tests/` は project root に置き、全段階から共有する。
- run、checkpoint、report、work item artifact は project directory ではなく `/storage` に置く。
- 段階を跨いで同じ kernel を使わない。同じ project root と persist root を fresh kernel で再設定する。
