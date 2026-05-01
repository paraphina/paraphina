# Phase 5 Worktree Audit - 2026-05-01

This audit records the repository hygiene decision before freezing the promoted
Phase 5 closeout baseline.

## Repository State

- Branch: `mm-pnl-harness-clean`
- Pre-freeze HEAD: `4b9a1f4d086366328403f11e9285f06d6b302264`
- Remote: `origin https://github.com/paraphina/paraphina.git`
- Upstream: `origin/mm-pnl-harness-clean`
- GitHub repository: `paraphina/paraphina`
- GitHub connector permission: read-only; branch/tag publication must use local
  Git credentials.

## Dirty Worktree Summary

- Tracked diff before freeze curation: `59` files, `50309` insertions, `6260`
  deletions.
- Untracked Phase 5 state before curation: `6319` files under `phase5/` across
  `272` run directories.
- Largest untracked Phase 5 files were duplicated queue/orchestration snapshots
  inside lane overlays, not source files or compact closeout evidence.

## Kept In The Closeout Baseline

- Source and runtime changes already present in the accepted Phase 5 surface,
  including live runner, connector, order-management, telemetry, and config
  changes.
- Phase 5 tooling and tests required to validate the closeout from a fresh
  clone.
- Documentation that defines the current repo/source-of-truth structure and the
  promoted Phase 5 closeout.
- Curated Phase 5 state:
  - `phase5/queue.yaml`
  - `phase5/status.md`
  - `phase5/control_pack.yaml`
  - `phase5/runs/README.md`
  - `phase5/runs/phase5_reopened_final_closeout/**`
  - `phase5/runs/phase5_reopened_wallet_econ_full_cost_opportunity_adjusted_gate_requal/**`

## Excluded From The Closeout Baseline

- `phase5/orchestration.yaml`, because it is ephemeral lane/session state.
- `phase5/runs/**/lanes/**`, copied overlays, queue previews, and tranche-card
  previews, because they are worktree scratch and duplicate repo state.
- Raw live telemetry, logs, stdout/stderr captures, systemd dumps, and full
  promotion-run directories.
- Local historical Phase 5 investigation notes that are not required to
  reproduce the final accepted closeout.
- Build artifacts under `target/`.

## Cloneability Standard

A fresh GitHub clone of this branch should contain all source, tools, focused
tests, curated Phase 5 queue/status/control state, and compact final closeout
evidence needed to validate the Phase 5 promoted checkpoint without relying on
local `/home/ubuntu/promotion_runs` artifacts.

## Validation Boundary

- `python3 tools/phase5_tranche.py validate`: pass.
- `python3 tools/phase5_tranche.py audit-state-sync --tranche-id phase5_reopened_final_closeout`: pass.
- `python3 tools/phase5_tranche.py audit-gate-contract --tranche-id phase5_reopened_wallet_econ_full_cost_opportunity_adjusted_gate_requal`: pass.
- `python3 -m py_compile tools/phase5_tranche.py tools/paraphina_watch.py tools/telemetry_analyzer.py`: pass.
- `python3 tools/check_docs_integrity.py`: pass.
- `python3 -m unittest tests.test_phase5_tranche_system tests.test_phase5_balance_snapshot tests.test_phase5_econ_attribution tests.test_telemetry_analyzer tests.test_paraphina_watch tests.test_ws_soak_report tests.test_extended_trade`: pass.
- `cargo test --all-features --lib --bins --tests`: pass.
- `cargo test --all-features`: all library, binary, and integration tests pass; local doctest runner is blocked by the EC2 Rust toolchain reporting `rustdoc` not applicable to `stable-aarch64-unknown-linux-gnu`.
