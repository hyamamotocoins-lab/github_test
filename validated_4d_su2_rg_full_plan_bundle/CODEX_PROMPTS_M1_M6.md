# Codex prompts for M1–M6

各promptを使用する前に、Codexへ次を読ませる。

1. `validated_4d_su2_rg_codex_design_v0_2.md`
2. `M1_M6_VALIDATED_RG_ROADMAP.md`
3. `MATHEMATICAL_CERTIFICATION_SPEC.md`
4. `AGENTS_validated_4d_su2_rg_v0_2.md`
5. 前milestoneのreportとtest結果

各promptの終了条件を満たしたら、そのmilestoneで一度停止させる。

---

# M1 prompt

Implement milestone M1 only: the exact two-dimensional SU(2) benchmark and the analytic representation-tail foundation.

Required deliverables:

- integer-valued `Irrep(j2)` conventions;
- rigorous Wilson character coefficient enclosures;
- zeroth- and first-derivative Peter–Weyl tail bounds;
- exact 2D recurrence `r_n -> r_n**4` with interval arithmetic;
- independent low-cutoff convolution cross-check;
- CPU-only tests;
- checkpoint/resume regression;
- `reports/M1_report.md` with formulas, conventions, and remaining heuristics.

Do not implement the 4D armillary tensor yet. Do not change certification status from `NOT_CERTIFIED`.

Stop after all M1 acceptance tests pass and report:

- files changed;
- tests run;
- reproduced numerical intervals;
- mathematical statements actually validated;
- unresolved issues.

---

# M2 prompt

Implement milestone M2 only: low-cutoff four-dimensional SU(2) armillary equivalence.

Start with `j2_max=1`. Build both:

1. a dense matrix-index reference tensor small enough for CPU high precision;
2. the armillary/fusion representation.

Fix and document all Wigner, orientation, duality, and normalization conventions. Construct the explicit basis map and verify dense-versus-armillary equivalence. Add structural tests that extend beyond numerical coincidence.

Required deliverables:

- Wigner/fusion cache with convention hash;
- deterministic sector canonicalization;
- dense reference generator;
- armillary generator;
- equivalence and symmetry tests;
- checkpointable tensor shards;
- `reports/M2_report.md`.

Do not proceed to GPU Triad-ATRG until M2 acceptance conditions pass.

---

# M3 prompt

Implement milestone M3 only: GPU matrix-free contraction and Triad-ATRG pilot.

Requirements:

- backend abstraction with cuTensorNet first and torch CUDA fallback;
- matrix-free `matvec` and `rmatvec` over armillary sector shards;
- contraction-path cache;
- memory-aware slicing and OOM recovery;
- fixed-seed RSVD exploration;
- low-cutoff comparison with explicit matrices;
- adjoint-consistency tests;
- checkpoint/resume across GPU sessions;
- profiling report with peak memory, item times, singular-value decay, and approximate influence proxy.

This milestone remains exploratory. Do not claim deterministic RSVD certification. Keep status `EXPLORATORY` or `CORE_REPRODUCED`.

---

# M4 prompt

Implement milestone M4 only: source derivatives and the complete error ledger.

Requirements:

- `DualTensor` with primal and symmetry-reduced tangent channels;
- forward differentiation through contraction, normalization, regrouping, and fixed-basis projection;
- explicit treatment of basis dependence through residual bounds;
- finite-difference regression at low cutoff;
- error DAG with provenance and no double counting;
- checkpoint serialization of all derivative and error state;
- `reports/M4_report.md` listing every rigorous and heuristic error term.

Do not mark a step as enclosed while any error term has no deterministic upper bound.

---

# M5 prompt

Implement milestone M5 only: a rigorous one-step certificate.

Read proof obligations P1–P11 in `MATHEMATICAL_CERTIFICATION_SPEC.md` and implement them as explicit validation gates.

Requirements:

- deterministic low-rank residual bound;
- proof-critical contraction rounding validation, preferably CPU multiprecision recomputation;
- positive normalization lower bound;
- interval source-derivative norms;
- entrywise interval influence matrix;
- rational positive Perron-vector certificate;
- independent arithmetic verifier;
- complete one-step certificate package;
- fail-closed verdict.

The result may be `ONE_STEP_CERTIFIED` or `NOT_CERTIFIED`. A mathematically correct failure is acceptable. Never loosen a bound solely to obtain contraction.

Stop after the independent verifier reproduces the verdict.

---

# M6 prompt

Implement milestone M6 only: three-to-five-step validated RG and the final finite-step contraction certificate.

Requirements:

- propagate input balls, not only center tensors;
- propagate source tangent balls;
- allow step-dependent cutoff and bond dimension;
- maintain a proof dependency chain across checkpoints;
- adapt ranks based on certified error budget;
- construct the final weighted interval influence matrix;
- optimize a float Perron vector, convert it to a positive rational vector, and verify the Collatz–Wielandt bound in multiprecision;
- create an independent final verifier notebook;
- output `final_certificate/` exactly as specified in the roadmap;
- write `limitations.md` separating finite-step lattice certification from thermodynamic and continuum bridges.

Emit `CERTIFIED` only if every proof obligation P0–P13 passes and the final upper bound is strictly below one.
