# ROADMAP_B: Live Venue Registry & Rollout

## Canonical venues (single source of truth)
Roadmap-B targets exactly five venues, in canonical order:

1. Extended
2. Hyperliquid
3. Aster
4. Lighter
5. Paradex

This ordering is codified in `paraphina/src/live/venues.rs` and is the reference
for per-venue telemetry arrays, connector selection, and readiness gates.

## Intended rollout
1. **Shadow**: validate market data and account snapshots without execution.
2. **Paper/Testnet**: enable execution in paper/testnet environments, with live keys still disabled.
3. **Live**: enable real execution only after hard gates pass and allowlists are explicit.

## Safety gates (explicit)
- **Execution enable gate**: execution requires `PARAPHINA_LIVE_EXEC_ENABLE=1`.
- **Trade mode gate**: execution requires `trade_mode != shadow`.
- **Per-venue allowlist**: connectors must be explicitly selectable in `paraphina_live` for each
  Roadmap-B venue; stub connectors are allowed while implementations are pending.
- **Roadmap-B gate**: when `PARAPHINA_ROADMAP_B=1`, the registry must expose all five
  canonical venues or the process/test must fail.
