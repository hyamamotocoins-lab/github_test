# AGENTS.md — validated 4D SU(2) RG

## Mission

Implement the restartable, fail-closed GPU Jupyter workflow specified in
`validated_4d_su2_rg_codex_design.md`.

## Hard requirements

- The user-facing entry point is an `.ipynb`.
- A GPU session lasts at most six hours.
- Begin final checkpointing no later than 5 h 20 min.
- Return no later than 5 h 30 min.
- Durable state must be stored outside ephemeral runtime storage.
- Every phase and work item must be restartable.
- Never emit `CERTIFIED` from approximate floating-point calculations alone.
- Missing bounds, NaN, Inf, nonpositive normalization bounds, or unvalidated
  residuals make certification fail closed.
- Do not silently change representation, orientation, phase, or normalization
  conventions.

## Development order

1. M0 persistence and recovery.
2. M1 2D exact benchmark.
3. M2 low-cutoff 4D armillary tensor.
4. M3 GPU matrix-free Triad-ATRG pilot.
5. M4 forward derivatives.
6. M5 one-step a posteriori validation.
7. M6 multi-step certificate.

Do not begin a later milestone before the previous milestone's tests pass.

## Code quality

- Python 3.11+.
- Full public type hints.
- Dataclasses for config and state.
- Explicit exceptions; no silent fallback.
- Content-addressed work item IDs.
- Atomic directory commits for checkpoints.
- CPU-only unit tests; tagged optional GPU tests.
- No hidden state in notebook globals.
- No monolithic long-running cell except the orchestrator entry point.

## Numerical rules

- Disable TF32.
- Use FP64 for serious exploratory runs.
- Use analytic or CPU multiprecision bounds for certification.
- Record seeds, versions, paths, and sector ordering.
- Verify hashes before loading checkpoints.
- Never treat a heuristic residual as rigorous.

## Checkpoint rules

- Save every 15 minutes and at every phase boundary.
- Save before and after costly basis changes.
- Use temporary directories and atomic rename.
- Store `COMMITTED` and SHA-256 manifests.
- Fall back to the previous valid checkpoint if the newest is corrupt.
- A `running` work item without a valid result returns to `pending` on resume.

## Required milestone report

After each milestone, report:

- files changed;
- tests run and results;
- restart test status;
- remaining TODOs;
- every bound that remains heuristic;
- peak CPU/GPU memory;
- checkpoint size and save time.
