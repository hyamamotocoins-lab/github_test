# Codex master prompt

Read first:

1. `validated_4d_su2_rg_codex_design.md`
2. `AGENTS.md`
3. `README.md`
4. `notebooks/00_project_setup.ipynb`
5. 現在の段階に対応する `notebooks/10_...`–`notebooks/70_...`

Implement the project milestone by milestone. M0 is accepted and frozen; M1 is the current implemented
milestone. Do not start M2 implementation until the Paperspace M1 acceptance artifact is reviewed.

## Historical M0 baseline (already accepted; do not rerun as a new result)

Implement and test:

- environment detection;
- explicit persistent-root validation;
- immutable run configuration and config hashing;
- `SessionGuard` with 5 h / 5 h 15 min / 5 h 20 min / 5 h 30 min states;
- atomic checkpoint directories;
- SHA-256 manifests and `COMMITTED` markers;
- fallback to the newest valid checkpoint;
- Python, NumPy, PyTorch CPU and CUDA RNG checkpointing;
- sharded tensor save/load;
- persistent pending/running/done work queue;
- interrupted-item recovery;
- dummy CPU/GPU workload split into bounded work items;
- runner that returns cleanly by the time budget;
- short simulated-session and fresh-process resume tests.

Do not implement the 4D RG algorithm until all M0 tests pass.

## Notebook requirements

The notebook must:

- execute top to bottom;
- generate modular files with `%%writefile` cells or equivalent;
- work in CPU-only mode for persistence tests;
- detect CUDA and use it for the dummy workload when available;
- require a persistent storage root;
- expose one main call:

```python
orchestrator.run_until_checkpoint()
```

- print exact next-session instructions on exit.

## Certification safety

During M0:

```python
certification_status = "NOT_CERTIFIED"
```

No code path may change this value.

After M0, show tests and stop for review.
