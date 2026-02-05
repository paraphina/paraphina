# Funding Repo Audit (Shadow-Only, Read-Only)

## TL;DR
- Runtime telemetry shows `venue_funding_8h` present but all zeros in shadow. Account snapshots are disabled in shadow mode, so funding never updates from live connectors. `venue_funding_8h` is emitted directly from `GlobalState::venues[*].funding_8h`.  
```1402:1543:paraphina/src/telemetry.rs
                "funding_8h": venue.map(|v| v.funding_8h).unwrap_or(0.0),
...
        venue_funding_8h.push(venue.funding_8h);
...
            "venue_funding_8h".to_string(),
            serde_json::json!(venue_funding_8h),
```
- Funding ingestion exists only via account snapshots (Hyperliquid, Paradex, Lighter). Aster/Extended set `funding_8h: None`, so even with account snapshots they will not populate funding.  
```1039:1047:paraphina/src/live/connectors/aster.rs
    Some(AccountSnapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        positions,
        balances,
        funding_8h: None,
        margin,
        liquidation,
    })
```
```1042:1050:paraphina/src/live/connectors/extended.rs
    Some(AccountSnapshot {
        venue_index,
        venue_id: venue_id.to_string(),
        seq,
        timestamp_ms,
        positions,
        balances,
        funding_8h: None,
        margin,
        liquidation,
    })
```
- Shadow mode explicitly disables account snapshots for Hyperliquid and Lighter (and other connectors gated by missing auth), so funding updates do not flow in shadow.  
```1843:1873:paraphina/src/bin/paraphina_live.rs
            ConnectorArg::Hyperliquid => {
                #[cfg(feature = "live_hyperliquid")]
                {
                    ...
                    if trade_mode.trade_mode != TradeMode::Shadow {
                        if hl_cfg.vault_address.is_some() {
                            let account_tx = channels.account_tx.clone();
                            hl = hl.with_account_tx(account_tx);
                        } else {
                            eprintln!(
                                "paraphina_live | account_snapshots_disabled=true reason=missing_hl_vault_address connector=hyperliquid"
                            );
                            ...
                        }
                    } else {
                        eprintln!(
                            "paraphina_live | account_snapshots_disabled=true reason=trade_mode_shadow connector=hyperliquid"
                        );
                        ...
                    }
```
- Funding affects MM/exit/hedge calculations in code (so all-zero funding means funding-aware logic is effectively disabled in shadow).  
```90:124:paraphina/src/mm.rs
    let funding_horizon_frac = tau / (8.0 * 60.0 * 60.0);
    ...
    let funding_pnl_per_unit = vstate.funding_8h * funding_horizon_frac * fair;
    let funding_adj = cfg.mm.funding_weight * funding_pnl_per_unit;
```
```110:114:paraphina/src/exit.rs
    let horizon_frac = cfg.exit.funding_horizon_sec / (8.0 * 60.0 * 60.0);
    let funding_benefit_per_tao = match intent.side {
        Side::Sell => v.funding_8h * horizon_frac * fair,
        Side::Buy => -v.funding_8h * horizon_frac * fair,
    };
```
```366:370:paraphina/src/hedge.rs
    let horizon_frac = cfg.hedge.funding_horizon_sec / (8.0 * 60.0 * 60.0);
    let funding_benefit = match intent.side {
        Side::Sell => v.funding_8h * horizon_frac * fair,
        Side::Buy => -v.funding_8h * horizon_frac * fair,
    };
```

## Observed Runtime Symptom (telemetry/watch)
From the current shadow telemetry, `venue_funding_8h` exists and is all zeros across venues. Example excerpt:
```json
{
  "t": 217,
  "venue_status": [
    "Healthy",
    "Healthy",
    "Healthy",
    "Healthy",
    "Healthy"
  ],
  "venue_funding_8h": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
  ]
}
```

The watch tool does not display funding fields; its table renders only mid/spread/age/position/health and global metrics.  
```205:285:tools/paraphina_watch.py
    q_global = record.get("q_global_tao")
    delta_usd = record.get("dollar_delta_usd")
    basis = record.get("basis_usd")
    basis_gross = record.get("basis_gross_usd")
    ...
    rows.append(
        [
            venue_id,
            format_num(mid_val, 10).strip(),
            format_num(spread_val, 8).strip(),
            format_ms(age_val),
            format_num(pos_val, 8).strip(),
            str(open_orders),
            last_fill_age,
            f"{health} tox={tox_str}",
        ]
    )
```

## Repo Touchpoints Inventory (where funding appears)
**Types / structs**
- `GlobalState::VenueState` includes `funding_8h` and defaults it to 0.0.  
```141:661:paraphina/src/state.rs
    // ----- Position & funding -----
    /// Net TAO-equivalent perp position.
    pub position_tao: f64,
    /// Current 8h funding rate (dimensionless).
    pub funding_8h: f64,
...
                funding_8h: 0.0,
```
- `FundingUpdate` execution/event types exist.  
```161:166:paraphina/src/types.rs
pub struct FundingUpdate {
    pub venue_index: usize,
    pub venue_id: Arc<str>,
    pub funding_8h: f64,
}
```
```73:88:paraphina/src/live/types.rs
pub struct FundingUpdate {
    pub venue_index: usize,
    pub venue_id: String,
    pub seq: u64,
    pub timestamp_ms: TimestampMs,
    pub funding_rate_8h: f64,
}
```

**State fields / caches**
- `VenueAccountCache` stores `funding_8h` from account snapshots.  
```164:228:paraphina/src/live/state_cache.rs
pub struct VenueAccountCache {
    ...
    pub funding_8h: Option<f64>,
    ...
}
...
        self.funding_8h = snapshot.funding_8h;
```
- Account snapshots apply funding to `GlobalState` when present.  
```2313:2339:paraphina/src/live/runner.rs
        v.price_liq = acct.price_liq;
        if let Some(funding) = acct.funding_8h {
            v.funding_8h = funding;
        }
```

**Telemetry emission**
- `venue_funding_8h` is emitted directly from `GlobalState`.  
```1402:1543:paraphina/src/telemetry.rs
                "funding_8h": venue.map(|v| v.funding_8h).unwrap_or(0.0),
...
        venue_funding_8h.push(venue.funding_8h);
...
            "venue_funding_8h".to_string(),
            serde_json::json!(venue_funding_8h),
```

**Connector parsing / account snapshots**
- Hyperliquid account snapshot parses `funding8h` into `funding_8h`.  
```1620:1633:paraphina/src/live/connectors/hyperliquid.rs
    let funding_8h = data
        .get("funding8h")
        .and_then(|v| v.as_str())
        .and_then(|v| v.parse::<f64>().ok());
...
        funding_8h,
```
- Lighter account snapshot parses `funding_8h`.  
```2192:2219:paraphina/src/live/connectors/lighter.rs
    let funding_8h = data.get("funding_8h").and_then(|v| v.as_f64());
...
        funding_8h,
```
- Paradex account snapshot parses `funding_8h`.  
```849:860:paraphina/src/live/connectors/paradex.rs
        return Some(AccountSnapshot {
            ...
            funding_8h: value.get("funding_8h").and_then(|v| v.as_f64()),
            margin,
            liquidation,
        });
```
- Aster and Extended account snapshots set `funding_8h: None`.  
```1039:1047:paraphina/src/live/connectors/aster.rs
    Some(AccountSnapshot {
        ...
        funding_8h: None,
        margin,
        liquidation,
    })
```
```1042:1050:paraphina/src/live/connectors/extended.rs
    Some(AccountSnapshot {
        ...
        funding_8h: None,
        margin,
        liquidation,
    })
```

**MM / quoting / hedge logic**
- Funding enters reservation price and inventory targets.  
```90:214:paraphina/src/mm.rs
    let funding_horizon_frac = tau / (8.0 * 60.0 * 60.0);
    ...
    let funding_pnl_per_unit = vstate.funding_8h * funding_horizon_frac * fair;
    let funding_adj = cfg.mm.funding_weight * funding_pnl_per_unit;
...
///   phi(funding) = clip(funding_8h / FUNDING_TARGET_RATE_SCALE, -1, 1) * FUNDING_TARGET_MAX_TAO
```
- Funding enters exit and hedge edge calculations.  
```110:114:paraphina/src/exit.rs
    let horizon_frac = cfg.exit.funding_horizon_sec / (8.0 * 60.0 * 60.0);
    let funding_benefit_per_tao = match intent.side {
        Side::Sell => v.funding_8h * horizon_frac * fair,
        Side::Buy => -v.funding_8h * horizon_frac * fair,
    };
```
```366:370:paraphina/src/hedge.rs
    let horizon_frac = cfg.hedge.funding_horizon_sec / (8.0 * 60.0 * 60.0);
    let funding_benefit = match intent.side {
        Side::Sell => v.funding_8h * horizon_frac * fair,
        Side::Buy => -v.funding_8h * horizon_frac * fair,
    };
```

**Treasury guidance**
- Funding used for telemetry-only guidance (not trading).  
```46:83:paraphina/src/treasury.rs
            acc.sum_funding_8h += venue.funding_8h;
            acc.sum_funding_cost_usd_8h += venue.funding_8h * abs_pos * fair;
...
            let avg_funding_8h = acc.sum_funding_8h / samples;
            let avg_funding_cost = acc.sum_funding_cost_usd_8h / samples;
```

## Funding Pipeline Trace (source → state → telemetry)
**Observed pipeline:**
1) **Connector parses account snapshot** (funding optional) → `AccountSnapshot.funding_8h`.  
```1620:1633:paraphina/src/live/connectors/hyperliquid.rs
    let funding_8h = data
        .get("funding8h")
        .and_then(|v| v.as_str())
        .and_then(|v| v.parse::<f64>().ok());
...
        funding_8h,
```
2) **Live state cache** stores funding in `VenueAccountCache`.  
```216:228:paraphina/src/live/state_cache.rs
        self.funding_8h = snapshot.funding_8h;
```
3) **Runner applies account snapshot** to `GlobalState`, only if `funding_8h` is `Some`.  
```2313:2339:paraphina/src/live/runner.rs
        if let Some(funding) = acct.funding_8h {
            v.funding_8h = funding;
        }
```
4) **Telemetry** reads `GlobalState.venues[*].funding_8h` into `venue_funding_8h`.  
```1454:1543:paraphina/src/telemetry.rs
        venue_funding_8h.push(venue.funding_8h);
...
            "venue_funding_8h".to_string(),
            serde_json::json!(venue_funding_8h),
```

**Breakpoint (why funding is 0 in shadow):**
- In shadow mode, account snapshots are disabled for Hyperliquid and Lighter (and account auth is missing for other connectors). With no account snapshots, funding stays at the default 0.0.  
```1843:1873:paraphina/src/bin/paraphina_live.rs
                    if trade_mode.trade_mode != TradeMode::Shadow {
                        if hl_cfg.vault_address.is_some() {
                            let account_tx = channels.account_tx.clone();
                            hl = hl.with_account_tx(account_tx);
                        } else {
                            eprintln!(
                                "paraphina_live | account_snapshots_disabled=true reason=missing_hl_vault_address connector=hyperliquid"
                            );
                            ...
                        }
                    } else {
                        eprintln!(
                            "paraphina_live | account_snapshots_disabled=true reason=trade_mode_shadow connector=hyperliquid"
                        );
                        ...
                    }
```
- Funding updates from market data are not applied to state; `MarketDataEvent::FundingUpdate` is ignored in `apply_market_event`.  
```2066:2094:paraphina/src/live/runner.rs
    match event {
        super::types::MarketDataEvent::L2Snapshot(snapshot) => { ... }
        super::types::MarketDataEvent::L2Delta(delta) => { ... }
        super::types::MarketDataEvent::Trade(_)
        | super::types::MarketDataEvent::FundingUpdate(_) => {}
    }
```

## Venue Capability Matrix (live connectors)
venue | funding source present in code? | parsed? | stored in state? | telemetry emits? | used in MM? | timestamps/staleness tracked?
---|---|---|---|---|---|---
hyperliquid | Account REST/WS (account snapshot) | Yes (`funding8h`) | Yes (via account snapshot) | Yes (`venue_funding_8h`) | Yes | Only account snapshot timestamp; no funding-specific age
paradex | Account REST snapshot | Yes (`funding_8h`) | Yes (via account snapshot) | Yes | Yes | Only account snapshot timestamp; no funding-specific age
lighter | Account REST snapshot | Yes (`funding_8h`) | Yes (via account snapshot) | Yes | Yes | Only account snapshot timestamp; no funding-specific age
extended | Account REST snapshot | **No** (explicit `None`) | No | Yes (default 0.0) | Yes | No
aster | Account REST snapshot | **No** (explicit `None`) | No | Yes (default 0.0) | Yes | No

Evidence:
- Hyperliquid funding parse:  
```1620:1633:paraphina/src/live/connectors/hyperliquid.rs
    let funding_8h = data
        .get("funding8h")
...
        funding_8h,
```
- Paradex funding parse:  
```849:860:paraphina/src/live/connectors/paradex.rs
            funding_8h: value.get("funding_8h").and_then(|v| v.as_f64()),
```
- Lighter funding parse:  
```2192:2219:paraphina/src/live/connectors/lighter.rs
    let funding_8h = data.get("funding_8h").and_then(|v| v.as_f64());
...
        funding_8h,
```
- Aster/Extended funding disabled (None):  
```1039:1047:paraphina/src/live/connectors/aster.rs
        funding_8h: None,
```
```1042:1050:paraphina/src/live/connectors/extended.rs
        funding_8h: None,
```

## Market-Making / Bias Hooks Audit (present/absent)
**Exists.** Funding is explicitly used in MM reservation price, per-venue inventory targeting, exit edge, and hedge cost.
- MM reservation components include funding adjustment.  
```90:124:paraphina/src/mm.rs
    let funding_pnl_per_unit = vstate.funding_8h * funding_horizon_frac * fair;
    let funding_adj = cfg.mm.funding_weight * funding_pnl_per_unit;
```
- Per-venue target inventory depends on funding.  
```209:214:paraphina/src/mm.rs
///   phi(funding) = clip(funding_8h / FUNDING_TARGET_RATE_SCALE, -1, 1) * FUNDING_TARGET_MAX_TAO
```
- Exit and hedge logic apply funding benefit.  
```110:114:paraphina/src/exit.rs
    let funding_benefit_per_tao = match intent.side {
        Side::Sell => v.funding_8h * horizon_frac * fair,
        Side::Buy => -v.funding_8h * horizon_frac * fair,
    };
```
```366:370:paraphina/src/hedge.rs
    let funding_benefit = match intent.side {
        Side::Sell => v.funding_8h * horizon_frac * fair,
        Side::Buy => -v.funding_8h * horizon_frac * fair,
    };
```

## Funding PnL Attribution Audit (present/absent)
**Absent as an explicit PnL stream.** There is no funding payment/accrual accounting, only funding used as an input to expected edge and to telemetry-only guidance.
- Treasury guidance computes an average funding cost for reporting only.  
```46:83:paraphina/src/treasury.rs
            acc.sum_funding_8h += venue.funding_8h;
            acc.sum_funding_cost_usd_8h += venue.funding_8h * abs_pos * fair;
...
            let avg_funding_cost = acc.sum_funding_cost_usd_8h / samples;
```
- No explicit funding PnL ledger or accrual path is present in state or execution events (only `funding_8h` values are stored/propagated).

## Tests + Docs Audit
**Tests/fixtures**
- Funding appears in account snapshot fixtures, but there are no explicit tests that validate funding propagation into telemetry or MM decisions.  
```1:14:tests/fixtures/hyperliquid/rest_account_snapshot.json
  "funding8h": "0.001"
```
```1:13:tests/fixtures/lighter/rest_account_snapshot.json
  "funding_8h": 0.001
```
```1:11:tests/fixtures/roadmap_b/extended/account_snapshot.json
  "funding_8h": 0.0,
```

**Docs**
- `RUNBOOK.md` instructs verifying funding/margin/liquidation fields, but does not describe missing funding in shadow mode or venue differences.  
```380:390:docs/RUNBOOK.md
### Reconcile Mismatch
...
4. Verify funding/margin/liquidation fields are populated.
```
- `TELEMETRY_SCHEMA_V1.md` does not document `venue_funding_8h` (no matches found in schema).  

## Gaps + Recommendations (ranked)
**Critical**
1) **Shadow mode funding is always zero**, yet MM/exit/hedge logic consumes `funding_8h` as a live signal. This creates a silent discrepancy between design and shadow behavior. Evidence: default funding value is 0.0, account snapshots disabled in shadow, telemetry shows zeros.  
```659:661:paraphina/src/state.rs
                funding_8h: 0.0,
```
```1843:1873:paraphina/src/bin/paraphina_live.rs
                        eprintln!(
                            "paraphina_live | account_snapshots_disabled=true reason=trade_mode_shadow connector=hyperliquid"
                        );
```

**High**
2) **Extended/Aster never populate funding** even with account snapshots (hard-coded `None`). Funding-aware logic will silently treat these venues as zero funding.  
```1039:1047:paraphina/src/live/connectors/aster.rs
        funding_8h: None,
```
```1042:1050:paraphina/src/live/connectors/extended.rs
        funding_8h: None,
```
3) **Funding update path via market data is unused.** `MarketDataEvent::FundingUpdate` exists but is ignored in `apply_market_event`, so even if funding feed is added later, it won’t reach state unless wired.  
```2066:2094:paraphina/src/live/runner.rs
        super::types::MarketDataEvent::Trade(_)
        | super::types::MarketDataEvent::FundingUpdate(_) => {}
```

**Medium**
4) **No funding staleness/age tracking.** There is no `funding_age_ms` or similar; telemetry only publishes a value.  
```1454:1543:paraphina/src/telemetry.rs
        venue_funding_8h.push(venue.funding_8h);
```
5) **Telemetry schema/doc gap.** `venue_funding_8h` is not in `TELEMETRY_SCHEMA_V1.md`, which complicates downstream validation.

**Low**
6) **No explicit funding PnL accounting.** Funding is used for expected edge and guidance but not accrued or booked as PnL.

## Design Requirements for Funding-Aware Module (requirements only)
1) **Canonical funding data model**
   - Single struct with fields: `rate_8h`, `interval_sec`, `as_of_ms`, `source` (exchange vs computed), `venue_id`.
   - Clear sign convention (positive funding = longs pay or receive? must be explicit).
2) **Timestamp semantics**
   - Store both exchange timestamp (if present) and receipt timestamp.
   - Track `funding_age_ms` and `funding_delay_ms` in telemetry.
3) **Telemetry fields**
   - Add `venue_funding_8h`, `venue_funding_age_ms`, `venue_funding_source` to schema and telemetry schema docs.
4) **Safety gates**
   - Funding should be explicit: `None`/`Unknown` must not silently map to 0.0.
   - Enforce per-venue stale gating for funding if used in MM/exit decisions.
5) **Connector coverage**
   - Each venue connector must declare its funding source (account snapshot, WS, REST).
   - For venues that cannot provide funding, telemetry should expose `null` and a reason.
6) **Testing strategy**
   - Deterministic fixtures for each connector with non-zero funding.
   - Unit tests for funding propagation: connector → cache → state → telemetry.
   - Contract tests asserting telemetry schema includes funding fields.

## Commands Run
```
pwd && git status -sb
git checkout -b research/funding-repo-audit
git checkout research/funding-repo-audit
python3 - <<'PY'
import json
from pathlib import Path
path = Path('/tmp/paraphina_live_shadow/telemetry.jsonl')
lines = path.read_text().splitlines()
for line in lines[-5:]:
    ...
PY
python3 - <<'PY'
import json
from pathlib import Path
path = Path('/tmp/paraphina_live_shadow/telemetry.jsonl')
lines = path.read_text().splitlines()
last = lines[-1]
...
PY
rg -n "funding|Funding|fundingRate|nextFunding|premium|carry|mark[_ ]price|index[_ ]price|rate_8h|funding_8h" paraphina/src tools docs
rg -n "venue_funding|funding_age|funding_delay" paraphina/src
rg -n "funding_8h" paraphina/src
rg -n "funding_pnl|funding cost|funding_cost|funding payment|funding accrual|funding_payout" paraphina/src
rg -n "account_snapshots_disabled" paraphina/src
rg -n "funding|funding_8h|rate_8h|premium|mark|index" paraphina/src -S
rg -n "funding|funding_8h|rate_8h|premium|mark|index" tests
rg -n "funding|funding_8h|rate_8h|premium|mark|index" docs
rg -n "funding_8h|funding" docs/TELEMETRY_SCHEMA_V1.md
```
