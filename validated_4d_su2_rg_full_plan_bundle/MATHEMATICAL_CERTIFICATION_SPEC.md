# Mathematical Certification Specification
## 4D SU(2) finite-step validated RG

**版:** 0.2.0  
**目的:** 実装が `CERTIFIED` を出すために必要な数学的proof obligationsを定義する。  
**注意:** 本文はcontinuum Yang–Millsの存在・mass gap全体を自動的に証明するものではない。

---

# 1. 基本対象

各RG段階で、有限core tensorと誤差半径を

\[
\mathbf T_r=(\widetilde T_r,\varepsilon_r)
\]

として保持する。意味は

\[
T_r\in\mathcal B_r,
\qquad
\mathcal B_r
=
\{T:\|T-\widetilde T_r\|_r\le\varepsilon_r\}.
\]

norm \(\|\cdot\|_r\) は各段で明示し、contractionに対してsubmultiplicativeまたは明示的structure constantを持つものを使用する。

normを途中で変更する場合、comparison constant

\[
\|X\|_{r+1}\le C_{r\to r+1}\|X\|_r
\]

を証明してerror ledgerへ加える。

---

# 2. Proof obligation P0 — 再現性と状態完全性

数学的certificateは、計算状態の再現性がない場合は無効とする。

必要条件:

- config hash。
- source hash。
- convention hash。
- checkpoint hash chain。
- sector ordering。
- RNG state。
- contraction path。
- precisionとrounding policy。
-全proof artifactのSHA-256。

`P0=PASS`は数学的定理ではないが、certificateの同一性に必要である。

---

# 3. P1 — 初期representation tail

normalized Wilson plaquette weightを

\[
\bar w_\beta(U)
=e^{\beta(\cos\theta-1)}
\]

とし、Peter–Weyl projectionを \(P_N\) とする。

少なくとも

\[
\|\bar w_\beta-P_N\bar w_\beta\|_\infty
\le\tau_N^{(0)},
\]

\[
\|\nabla(\bar w_\beta-P_N\bar w_\beta)\|_\infty
\le\tau_N^{(1)}
\]

を証明する。

一つのblockに \(N_p\) plaquetteがあり、各factorが \(0\le\bar w\le1\) を満たすなら、telescopingにより

\[
\left\|
\prod_{p=1}^{N_p}\bar w_p
-
\prod_{p=1}^{N_p}P_N\bar w_p
\right\|_\infty
\le
N_p\tau_N^{(0)}.
\]

微分tailでは、一つのsourceが実際に接触するplaquette数を使い、全216枚を無条件に掛けない。

**PASS条件:** tail formula、定数、metric normalizationがproof artifactに保存されている。

---

# 4. P2 — Armillary equivalence

finite Peter–Weyl truncated local tensorを \(T_N^{\mathrm{PW}}\)、armillary tensorを \(T_N^{\mathrm{arm}}\) とする。

明示的なunitaryまたはisometric basis map \(U_N\) を構成し、

\[
T_N^{\mathrm{PW}}
=
U_NT_N^{\mathrm{arm}}U_N^*
\]

を理論上証明する。

実装検証では低cutoffで

\[
\|T_N^{\mathrm{PW}}-U_NT_N^{\mathrm{arm}}U_N^*\|
\le\varepsilon_{\mathrm{basis}}
\]

を多倍長で確認する。

一般cutoffでexact identityを使う場合、codeがそのidentityを実装していることをstructural testsで保証する。

**禁止:** 低cutoff数値一致だけから任意cutoffの同値性を主張しない。

---

# 5. P3 — Multilinear contraction enclosure

RG contractionを有限個のmultilinear map

\[
\mathcal C(A_1,\ldots,A_k)
\]

の合成として表す。

各inputに

\[
\|A_i-\widetilde A_i\|\le r_i
\]

がある場合、structure constant \(L_{\mathcal C}\) を用いて

\[
\|\mathcal C(A_1,\ldots,A_k)
-
\mathcal C(\widetilde A_1,\ldots,\widetilde A_k)\|
\]

を

\[
L_{\mathcal C}
\left(
\prod_i(\|\widetilde A_i\|+r_i)
-
\prod_i\|\widetilde A_i\|
\right)
\]

で囲う。

contractionのindex summationにより追加dimension factorが必要なら、\(L_{\mathcal C}\)へ明示する。

---

# 6. P4 — Low-rank projection residual

GPUで得たorthonormal候補basisを \(Q\) とする。

まず

\[
\|Q^*Q-I\|\le\varepsilon_Q
\]

を検証する。

次にprojection residual

\[
R=(I-QQ^*)A
\]

について、決定論的な上界

\[
\|R\|\le\varepsilon_{\mathrm{proj}}
\]

を得る。

許可するcertificate:

1. 全sectorのFrobenius normを厳密加算。
2. block row/column normからspectral normを上から評価。
3. 高精度CPU contractionで \(R\) の小行列表示を構成。
4. 解析的channel tail。

probabilistic RSVD theoremはbasis発見に使用できるが、最終上界の唯一の根拠にはしない。

\(Q\) が完全にはorthonormalでない場合、\(QQ^*\)を正射影とみなしてはならない。QRの高精度再直交化または補正項が必要である。

---

# 7. P5 — Floating-point contraction error

GPU contraction outputを \(\widehat C\) とする。

次のいずれかを採用する。

## Route A: proof-critical sectorのCPU高精度再計算

最終influenceに寄与するsectorだけCPU多倍長で再計算し、GPUは探索専用とする。

## Route B: verified residual

GPU outputを固定し、元のcontraction equationのresidualを高精度で評価する。

## Route C: directed rounding kernel

各演算を外向き丸めしたcustom kernelで囲う。

初版はRoute Aを推奨する。

単にmachine epsilonとFLOP数を掛けた粗い値を使う場合、condition numberとsummation treeを含む証明が必要である。

---

# 8. P6 — RG step enclosure

一段近似mapを \(\widetilde{\mathcal R}\) とする。

入力ball \(\mathcal B_r\) 全体について

\[
\mathcal R(\mathcal B_r)
\subset
\mathcal B_{r+1}
\]

を示す。

出力半径は少なくとも

\[
\varepsilon_{r+1}
=
\varepsilon_{\mathrm{input-prop}}
+
\varepsilon_{\mathrm{rep}}
+
\varepsilon_{\mathrm{channel}}
+
\varepsilon_{\mathrm{proj}}
+
\varepsilon_{\mathrm{round}}
+
\varepsilon_{\mathrm{norm}}
\]

を含む。

各項にprovenanceを持たせ、同じdiscarded sectorをrepresentation tailとprojection residualの両方へ入れない。

---

# 9. P7 — Source derivative enclosure

source parameterを \(s\) とし、

\[
\dot T_r=\left.\frac{d}{ds}T_r(s)\right|_{s=0}
\]

を考える。

primalとtangentのball

\[
\|T_r-\widetilde T_r\|\le\varepsilon_r,
\qquad
\|\dot T_r-\widetilde{\dot T}_r\|
\le\varepsilon_r^{(1)}
\]

を同時に伝播する。

固定basis projectionでは

\[
\dot T\mapsto Q^*\dot TQ
\]

を使い、basis variationによる不足をprojection residualに含める。

source付きnormalization

\[
\widehat T(s)=\frac{T(s)}{\lambda(s)}
\]

を使う場合、

\[
\dot{\widehat T}
=
\frac{\dot T}{\lambda}
-
\frac{T\dot\lambda}{\lambda^2}
\]

の全項を囲う。

---

# 10. P8 — Positive kernel and normalization lower bound

influence評価に用いるkernel \(K(x;\eta)\) は非負である必要がある。

truncated core \(\widetilde K\) 自体が点wise非負でない場合でも、真のkernelの非負性を用いることはできるが、分母下界を別途証明する。

\[
Z(\eta)=\int K(x;\eta)\,dx.
\]

coreから

\[
\widetilde Z(\eta)=\int\widetilde K(x;\eta)\,dx
\]

を計算し、

\[
\inf_\eta \widetilde Z(\eta)-\varepsilon_0
\ge z_{\min}>0
\]

を示す。

\(z_{\min}\le0\)ならcertificateは失敗する。

---

# 11. P9 — Influence entry bound

条件付き密度

\[
p_\eta(x)=\frac{K(x;\eta)}{Z(\eta)}
\]

について

\[
\partial_j p
=
\frac{\partial_jK}{Z}
-
\frac{K\,\partial_jZ}{Z^2}.
\]

\(K\ge0\)より

\[
\|\partial_jp\|_1
\le
\frac{2\|\partial_jK\|_1}{Z}.
\]

直径 \(D_i=\operatorname{diam}(E_i)\) のmetric space上で、1-Lipschitz関数のoscillationは \(D_i\) 以下なので

\[
c_{ij}
\le
\frac{D_i}{2}\|\partial_jp\|_1
\le
D_i\frac{\|\partial_jK\|_1}{Z}.
\]

したがってenclosureから

\[
\overline c_{ij}
=
D_i
\frac{
\|\partial_j\widetilde K\|_1+
\varepsilon_{1,j}
}{z_{\min}}
\]

を使用できる。

source parameterizationの速度と境界metricの単位が一致していることを証明する。

---

# 12. P10 — Weighted matrix enclosure

block typeを \(a,b\)、displacementを \(z\) とする。

\[
(\overline B_m)_{ab}
=
\sum_z e^{m|z_0|}\overline c_{ab}(z)
+
\varepsilon_{ab}^{\mathrm{spatial-tail}}.
\]

有限range切断を使う場合、除外したdisplacement tailを必ず加える。

translation/cubic symmetryで項をまとめる場合、orbit multiplicityを外向きに正しく掛ける。

---

# 13. P11 — Collatz–Wielandt certificate

\(\overline B_m\) はentrywise非負の上界行列とする。

正ベクトル \(w>0\) に対し

\[
\rho(B_m)
\le
\rho(\overline B_m)
\le
\max_a\frac{(\overline B_mw)_a}{w_a}.
\]

候補 \(w\) はfloatで探索してよいが、最終的にはpositive rationalまたはoutward intervalへ変換する。

最終値

\[
q_{\mathrm{cert}}
=
q_{\mathrm{CW}}
+
\varepsilon_{\mathrm{outside-matrix}}
\]

が1未満ならcontractive certificateが成立する。

matrix entriesへ既に全tailを入れた場合は、outside tailを再度加えない。

---

# 14. P12 — Multi-step composition

各段で

\[
\mathcal R_r(\mathcal B_r)\subset\mathcal B_{r+1}
\]

が必要である。

単一center trajectoryだけを追い、input radiusに対する感度を無視してはならない。

source tangentについても

\[
D\mathcal R_r(\mathcal B_r,\dot{\mathcal B}_r)
\subset
\dot{\mathcal B}_{r+1}
\]

を証明する。

最終certificateはcheckpoint chain全体のproof dependencyを持つ。

---

# 15. P13 — Independent verification

独立verifierは巨大tensorを再計算しなくてもよいが、少なくとも次を再検証する。

- artifact hash。
- 全radiusの非負性。
- error-ledger sum。
- normalization denominator。
- influence entry formula。
- weighted matrix construction。
- rational Perron vector。
- Collatz bound。
- final margin。

verifierがmain codeと同じ`certificate.py`をimportするだけでは独立性が弱い。最終算術部分は別moduleまたはnotebookに再実装する。

---

# 16. Certificate verdict

`CERTIFIED`に必要なboolean condition:

```text
P0 && P1 && P2 && P3 && P4 && P5 && P6
&& P7 && P8 && P9 && P10 && P11 && P12 && P13
&& q_cert_upper < 1
```

一つでも不明なら`CERTIFIED`を出さない。

---

# 17. 計算certificateからOS mass gapまでの外部bridge

計算完了後、次を別定理として接続する。

## B1: exact representation bridge

計算対象のtensor network/RGが元の格子Yang–Mills measureのexact rewritingである。

## B2: influence-to-clustering bridge

weighted influence contractionから、局所gauge-invariant observableの時間方向指数相関減衰が従う。

## B3: punctured RG bridge

fine local observableを保持したままbulk RG certificateを適用できる。

## B4: reflection positivity/transfer bridge

Euclidean時間相関がOS Hilbert空間のsemigroup matrix elementに対応する。

## B5: thermodynamic uniformity

certificateの定数が体積と境界条件に一様である。

## B6: continuum uniformity

scaling trajectoryに沿ってphysical block lengthとcontraction exponentの正の下界が保たれる。

## B7: continuum existence

Schwinger functionsが非自明なOS-positive continuum limitを持つ。

M6の計算だけでB5–B7を自動的に主張しない。
