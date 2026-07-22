"""Campaign B persist pause: zip for download, then purge Paperspace disk.

Implementation
--------------
- Library: ``src/campaign_b/persist_export.py``
- CLI: ``scripts/persist_export_pause.py``
- Notebook: ``notebooks/100_persist_export_pause.ipynb``

Why
---
Paperspace ``/storage`` fills with Campaign B ``runs/`` + ``campaign_b/``.
To pause safely: archive the persist root, download the zip locally, then
delete the on-machine tree so the next session starts clean (or restore
later by unzipping into an empty ``VALIDATED_RG_PERSIST_ROOT``).

Safety (fail-closed)
--------------------
1. Default is **dry-run** (size / free-space / lease check only).
2. Archive path must be **outside** the persist root (default
   ``/storage/exports/validated_4d_su2_rg_<stamp>.zip``).
3. Free disk must cover source size + margin (default **2 GiB**).
4. Live GPU lane lease blocks export unless ``--allow-live-gpu-lease``.
5. ``--purge`` requires ``--i-understand-purge PURGE_PERSIST_ROOT`` and
   runs only after zip CRC + SHA-256 verification.
6. Purge never deletes the zip / ``.sha256`` / ``.manifest.json``.
7. After purge, writes ``/storage/exports/<name>_PURGED.json`` pointing at
   the archive.

Paperspace
----------
::

  cd /notebooks/validated_4d_su2_rg_codex_bundle
  git pull
  export VALIDATED_RG_PERSIST_ROOT=/storage/validated_4d_su2_rg

  # Stop 96/97 first (or release lease)
  python scripts/release_gpu_lane_lease.py --status

  # Plan
  python scripts/persist_export_pause.py

  # Zip only (keep persist until download finishes)
  python scripts/persist_export_pause.py --execute

  # Easier Jupyter download target:
  python scripts/persist_export_pause.py --execute \\
      --export-dir /notebooks/persist_exports

  # After local download + sha256 check, free Paperspace:
  python scripts/persist_export_pause.py --execute --purge \\
      --i-understand-purge PURGE_PERSIST_ROOT \\
      --archive-path /storage/exports/validated_4d_su2_rg_....zip

Restore later
-------------
::

  mkdir -p /storage/validated_4d_su2_rg
  cd /storage/validated_4d_su2_rg
  unzip /path/to/validated_4d_su2_rg_....zip

Notes
-----
- Default compression is **STORE** (fast). ``--compress`` rarely helps
  tensor blobs and is much slower.
- A ~55 GiB tree needs ~57+ GiB free on the same volume for a same-disk zip.
  If free space is short, attach another volume / download incrementally, or
  reclaim M3 first (``docs/campaign_b_m3_storage_reclaim.md``) then export.
- Screening-only: export does not invent ``CERTIFIED``.
"""
