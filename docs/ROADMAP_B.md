# Roadmap B (Live Venues)

## Canonical venues (single source of truth)

Roadmap-B targets exactly five venues, in stable order:

1. Extended
2. Hyperliquid
3. Aster
4. Lighter
5. Paradex

Canonical registry: `paraphina/src/live/venues.rs`

## Intended rollout

1. **Shadow**: shadow-only traffic, no live execution.
2. **Paper/Testnet**: paper and testnet execution, keep live execution disabled.
3. **Live**: enable live execution only after the safety gates below pass.

For ETH market making, the intended live rollout is all-5 venue connectivity
with a micro-canary profile. Scale live by increasing inventory/order caps, not
by reducing venue count, because the MM strategy relies on multi-venue fair
value.

## Explicit safety gates

- `PARAPHINA_LIVE_EXEC_ENABLE=1` is required for any live execution.
- `trade_mode != shadow` is required for account polling/execution.
- Per-venue allowlist must be explicitly approved before live execution on that venue.

These gates are additive and must remain intact as Roadmap-B progresses.

Legacy config and telemetry fields ending in `_tao` still denote base-asset
inventory units in the current ETH deployment, not TAO market selection.
