# Funding Live Readiness (Read-Only Audit)

This report is **read-only** and contains **no Rust code changes**. It answers:
“Are funding mechanics implemented? To what extent? Would they work in live? What must be implemented for shadow-safe + live-correct funding?”

## TL;DR
- Funding is **partially implemented**: MM/exit/hedge/treasury consume `funding_8h`, but ingestion is account-snapshot-only and defaults to `0.0` when missing. (`paraphina/src/mm.rs:L90-L123`, `paraphina/src/exit.rs:L110-L114`, `paraphina/src/hedge.rs:L366-L370`, `paraphina/src/treasury.rs:L46-L54`, `paraphina/src/state.rs:L629-L661`)
- In **shadow**, funding will stay at zero: account snapshots are disabled and market-data funding events are ignored. (`paraphina/src/bin/paraphina_live.rs:L1843-L1871`, `paraphina/src/bin/paraphina_live.rs:L1958-L1984`, `paraphina/src/live/runner.rs:L2060-L2094`)
- In **live**, funding updates only for venues where account snapshots are enabled **and** the snapshot includes `funding_8h`. (`paraphina/src/live/runner.rs:L2313-L2339`, `paraphina/src/live/connectors/hyperliquid.rs:L1620-L1633`, `paraphina/src/live/connectors/lighter.rs:L2192-L2219`, `paraphina/src/live/connectors/paradex.rs:L849-L860`)
- Aster and Extended snapshots hardcode `funding_8h: None`, so funding never updates there even in live. (`paraphina/src/live/connectors/aster.rs:L1039-L1047`, `paraphina/src/live/connectors/extended.rs:L1042-L1050`)
- `MarketDataEvent::FundingUpdate` exists but is **ignored**, so public funding feeds do not reach state. (`paraphina/src/live/types.rs:L73-L88`, `paraphina/src/live/runner.rs:L2060-L2094`)
- Telemetry and per-intent stats **coerce missing funding to 0.0**, masking unknowns. (`paraphina/src/telemetry.rs:L1388-L1403`, `paraphina/src/telemetry.rs:L1465-L1491`)
- Minimal work needed: wire FundingUpdate into state, add null semantics + telemetry fields, add public funding ingestion per venue, add staleness + boundary window gates.

## Funding implemented?

**Yes, but only in decision layers; ingestion is incomplete.**

Algorithms using funding:
- **Market maker**: `funding_8h` feeds reservation pricing. (`paraphina/src/mm.rs:L90-L123`)
- **Exit**: `funding_8h` contributes to ranking. (`paraphina/src/exit.rs:L110-L114`)
- **Hedge**: `funding_8h` affects hedge cost. (`paraphina/src/hedge.rs:L366-L370`)
- **Treasury guidance**: aggregates `funding_8h` and funding cost metrics. (`paraphina/src/treasury.rs:L46-L54`)

**Implication:** The logic is wired, but the data path is not shadow-safe and only partially live-ready.

## Funding ingestion today (paths that can set funding in state)

### Path A — Account snapshots (only active ingestion)
**Connector parse → AccountSnapshot.funding_8h → VenueAccountCache → apply_account_snapshot_to_state → GlobalState.venues[*].funding_8h**

Evidence:
- Account snapshot stores funding in cache: `VenueAccountCache.funding_8h = snapshot.funding_8h`. (`paraphina/src/live/state_cache.rs:L164-L228`)
- Runner applies funding to state **only if Some**. (`paraphina/src/live/runner.rs:L2313-L2339`)
- Snapshot parses funding (per venue):
  - Hyperliquid: parses `funding8h`. (`paraphina/src/live/connectors/hyperliquid.rs:L1620-L1633`)
  - Lighter: parses `funding_8h`. (`paraphina/src/live/connectors/lighter.rs:L2192-L2219`)
  - Paradex: parses `funding_8h`. (`paraphina/src/live/connectors/paradex.rs:L849-L860`)
  - Aster: `funding_8h: None`. (`paraphina/src/live/connectors/aster.rs:L1039-L1047`)
  - Extended: `funding_8h: None`. (`paraphina/src/live/connectors/extended.rs:L1042-L1050`)

### Path B — Market data funding events (exists, but ignored)
**MarketDataEvent::FundingUpdate → apply_market_event → (ignored)**

Evidence:
- Event exists in type: `MarketDataEvent::FundingUpdate`. (`paraphina/src/live/types.rs:L73-L88`)
- Runner ignores it: `MarketDataEvent::FundingUpdate(_) => {}`. (`paraphina/src/live/runner.rs:L2060-L2094`)

**Conclusion:** Only account snapshots can update funding today. Market-data funding is unused.

## Shadow vs Live behavior (per venue)

### Hyperliquid
**Shadow:** funding **does not update**. Account snapshots are disabled in shadow.  
Evidence: `account_snapshots_disabled=true reason=trade_mode_shadow`. (`paraphina/src/bin/paraphina_live.rs:L1843-L1871`)

**Live:** funding **updates if** `vault_address` is present (account snapshots enabled).  
Evidence: account snapshots enabled only when `vault_address` exists; otherwise disabled. (`paraphina/src/bin/paraphina_live.rs:L1843-L1865`)  
Funding parse path exists. (`paraphina/src/live/connectors/hyperliquid.rs:L1620-L1633`)

### Lighter
**Shadow:** funding **does not update**. Account snapshots disabled in shadow.  
Evidence: `account_snapshots_disabled=true reason=trade_mode_shadow`. (`paraphina/src/bin/paraphina_live.rs:L1958-L1984`)

**Live:** funding **updates if** auth is present and paper mode is off (or fixture).  
Evidence: snapshots disabled when `paper_mode` or missing auth; otherwise enabled. (`paraphina/src/bin/paraphina_live.rs:L1966-L1976`)  
Funding parse path exists. (`paraphina/src/live/connectors/lighter.rs:L2192-L2219`)

### Paradex
**Shadow:** funding **does not update** if auth is missing; account snapshots disabled.  
Evidence: `account_snapshots_disabled=true reason=missing_paradex_auth`. (`paraphina/src/bin/paraphina_live.rs:L2340-L2360`)

**Live:** funding **updates if** Paradex auth present (account polling enabled).  
Evidence: account polling starts only when auth present; otherwise disabled. (`paraphina/src/bin/paraphina_live.rs:L2340-L2360`)  
Funding parse path exists. (`paraphina/src/live/connectors/paradex.rs:L849-L860`)

### Extended
**Shadow:** funding **does not update** if API keys are missing; account snapshots disabled.  
Evidence: `account_snapshots_disabled=true reason=missing_extended_api_keys`. (`paraphina/src/bin/paraphina_live.rs:L2142-L2151`)

**Live:** funding **would still not update** even with snapshots because connector sets `funding_8h: None`.  
Evidence: snapshot `funding_8h: None`. (`paraphina/src/live/connectors/extended.rs:L1042-L1050`)

### Aster
**Shadow:** funding **does not update** if API keys are missing; account snapshots disabled.  
Evidence: `account_snapshots_disabled=true reason=missing_aster_api_keys`. (`paraphina/src/bin/paraphina_live.rs:L2248-L2258`)

**Live:** funding **would still not update** even with snapshots because connector sets `funding_8h: None`.  
Evidence: snapshot `funding_8h: None`. (`paraphina/src/live/connectors/aster.rs:L1039-L1047`)

## Venue matrix

| venue | funding source in code | works in shadow? | works in live? | requirements | blockers |
|---|---|---|---|---|---|
| hyperliquid | Account snapshot (`funding8h`) | No | Yes if vault address | `vault_address` + account polling | Shadow disables snapshots; no market-data wiring |
| lighter | Account snapshot (`funding_8h`) | No | Yes if auth and not paper | auth + non-paper (or fixture) | Shadow disables snapshots; no market-data wiring |
| paradex | Account snapshot (`funding_8h`) | No if no auth | Yes if auth | Paradex auth | Shadow lacks auth; no market-data wiring |
| extended | Account snapshot (always `None`) | No | No | n/a | Connector sets `funding_8h: None`; no market-data wiring |
| aster | Account snapshot (always `None`) | No | No | n/a | Connector sets `funding_8h: None`; no market-data wiring |

Evidence:
- Snapshot parsing and `None` values: (`paraphina/src/live/connectors/hyperliquid.rs:L1620-L1633`, `paraphina/src/live/connectors/lighter.rs:L2192-L2219`, `paraphina/src/live/connectors/paradex.rs:L849-L860`, `paraphina/src/live/connectors/extended.rs:L1042-L1050`, `paraphina/src/live/connectors/aster.rs:L1039-L1047`)
- Account snapshot enable/disable logs: (`paraphina/src/bin/paraphina_live.rs:L1843-L1871`, `paraphina/src/bin/paraphina_live.rs:L1958-L1984`, `paraphina/src/bin/paraphina_live.rs:L2340-L2360`, `paraphina/src/bin/paraphina_live.rs:L2142-L2151`, `paraphina/src/bin/paraphina_live.rs:L2248-L2258`)
- Market-data funding ignored: (`paraphina/src/live/runner.rs:L2060-L2094`)

## Exact breakpoint(s) preventing funding from working in shadow
1) **Account snapshots disabled in shadow** (Hyperliquid, Lighter). (`paraphina/src/bin/paraphina_live.rs:L1843-L1871`, `paraphina/src/bin/paraphina_live.rs:L1958-L1984`)
2) **Missing auth disables account snapshots** (Paradex/Extended/Aster). (`paraphina/src/bin/paraphina_live.rs:L2340-L2360`, `paraphina/src/bin/paraphina_live.rs:L2142-L2151`, `paraphina/src/bin/paraphina_live.rs:L2248-L2258`)
3) **Market-data funding ignored**, so public feeds never reach state. (`paraphina/src/live/runner.rs:L2060-L2094`)
4) **Funding defaults to zero**, masking missing data. (`paraphina/src/state.rs:L629-L661`)

## Minimal implementation plan (design only)

1) **Wire FundingUpdate into state**
   - Add handling in `apply_market_event` to update canonical funding state on `MarketDataEvent::FundingUpdate`.  
   Evidence of current ignore: (`paraphina/src/live/runner.rs:L2060-L2094`)

2) **Null semantics + telemetry**
   - Replace silent zeros with explicit Unknown/Null for funding in telemetry.  
   Evidence of current zero coercion: (`paraphina/src/telemetry.rs:L1388-L1403`, `paraphina/src/telemetry.rs:L1465-L1491`)

3) **Public funding ingestion per venue**
   - Add market-data funding parsing per connector and emit `FundingUpdate` with timestamps and interval metadata.  
   Evidence of existing FundingUpdate type: (`paraphina/src/live/types.rs:L73-L88`)

4) **Staleness + boundary window**
   - Gate funding usage by age and avoid windows near `next_funding_ms`.  
   Evidence of current direct usage: (`paraphina/src/mm.rs:L90-L123`, `paraphina/src/exit.rs:L110-L114`, `paraphina/src/hedge.rs:L366-L370`)

## Risks
- **Silent zero**: default `funding_8h = 0.0` masks missing data and biases MM/exit/hedge. (`paraphina/src/state.rs:L629-L661`)
- **Wrong units**: no source interval captured; `funding_8h` assumes 8h rate even if native rate differs. (Data model limitation)
- **Sign conventions**: no explicit sign validation in code; wrong sign would invert incentives. (No explicit guard)
- **Price basis**: funding payments may require oracle/mark/index price; not represented in state. (No explicit field)

## Commands Run
```
pwd && git status -sb && git checkout -b research/funding-live-readiness
rg -n "funding_8h|FundingUpdate|MarketDataEvent::FundingUpdate|venue_funding|account_snapshots_disabled" paraphina/src tools docs
rg -n "funding_weight|funding_horizon|funding" paraphina/src/mm.rs paraphina/src/exit.rs paraphina/src/hedge.rs paraphina/src/treasury.rs
```
