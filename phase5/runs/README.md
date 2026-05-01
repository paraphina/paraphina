# Phase 5 Run Registry

This directory stores **small tranche-local manifests and cards**, not the heavy
telemetry/log payloads.

Heavy runtime artifacts stay in `/home/ubuntu/promotion_runs/`.

For each tranche, the automation layer writes:

- `phase5/runs/<tranche_id>/tranche_card.yaml`
- `phase5/runs/<tranche_id>/latest_run.yaml`
- optional command transcripts / metadata

The goal is to keep the repo-side workflow state auditable without copying large
runtime evidence into git-tracked locations.
