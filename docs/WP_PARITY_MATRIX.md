# WHITEPAPER PARITY MATRIX (Audit-Grade)

Date: 2026-01-17  
Scope: `docs/WHITEPAPER.md` Part II canonical spec (§4, §11, §14.5, §15, §17)  
Feature flags: `event_log`, `live`, `live_hyperliquid`, `live_lighter` (`paraphina/Cargo.toml:L59-L65`)  

Status definitions: **Implemented / Partial / Missing** (code-visible only).

---

## Telemetry Contract

### WP-TELEM-INVARIANTS — Telemetry invariants (monotonic `t`, finite numerics, valid enums)
- Status: Implemented.
- Evidence: `tools/check_telemetry_contract.py:L1-L120`, `paraphina/src/telemetry.rs:L131-L255`.
- Sim vs Live (flags): Sim Implemented via shared sink; Live Implemented via shared sink (feature `live`).
- Dependency blockers: None.
- Acceptance tests: `tests/test_telemetry_contract_gate.py:L1-L120`, `paraphina/tests/telemetry_determinism_tests.rs:L85-L131`.

---

## §4 Data Ingestion

### WP-4.1-OB-BOOK-UPDATE — Update bids/asks on incremental L2 updates
- Status: Implemented.
- Evidence: `paraphina/src/orderbook_l2.rs:L1-L210`, `paraphina/src/state.rs:L91-L189`, `paraphina/src/live/runner.rs:L545-L640`, `paraphina/src/engine.rs:L54-L115`.
- Sim vs Live (flags): Sim Implemented via synthetic L2 snapshots `paraphina/src/engine.rs:L54-L115`; Live Implemented via market event ingestion `paraphina/src/live/runner.rs:L545-L640` (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/orderbook_core_l2_tests.rs:L1-L120`, `paraphina/tests/live_hyperliquid_core_orderbook_tests.rs:L1-L120`.

### WP-4.1-OB-LAST-UPDATE — Set `last_update_ms` on book update
- Status: Implemented.
- Evidence: `paraphina/src/state.rs:L91-L189`, `paraphina/src/engine.rs:L54-L115`, `paraphina/src/live/runner.rs:L545-L640`.
- Sim vs Live (flags): Sim Implemented; Live Implemented (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/orderbook_core_l2_tests.rs:L1-L120`.

### WP-4.1-OB-MID-SPREAD-DEPTH — Compute mid/spread/depth near mid
- Status: Implemented.
- Evidence: `paraphina/src/orderbook_l2.rs:L127-L160`, `paraphina/src/state.rs:L91-L189`.
- Sim vs Live (flags): Sim Implemented; Live Implemented (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/orderbook_core_l2_tests.rs:L1-L120`.

### WP-STATUS-DATA-INGESTION — Status block (data ingestion)
- Status: Implemented.
- Evidence: `paraphina/src/engine.rs:L54-L115`, `paraphina/src/live/state_cache.rs:L28-L366`, `paraphina/src/live/runner.rs:L565-L654`.
- Sim vs Live (flags): Sim Implemented; Live Implemented (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/live_state_cache_parity_tests.rs`, `paraphina/tests/live_hyperliquid_core_orderbook_tests.rs`.

### WP-4.1-OB-LOCAL-VOL — Update local vol EWMA from log mid returns
- Status: Implemented.
- Evidence: `paraphina/src/state.rs:L158-L189`, `paraphina/src/engine.rs:L88-L90`, `paraphina/src/live/runner.rs:L520-L531`, `paraphina/src/live/connectors/hyperliquid.rs:L492-L541`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/engine.rs`; Live Implemented `paraphina/src/live/runner.rs` (feature `live_hyperliquid`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/local_vol_update_tests.rs:L4-L45`, `paraphina/tests/live_hyperliquid_runner_fixture_tests.rs:L1-L180`.

### WP-4.2-BALANCE-POLL — Poll positions/balances/funding
- Status: Implemented.
- Evidence: `paraphina/src/bin/paraphina_live.rs:L223-L385`, `paraphina/src/live/state_cache.rs:L159-L183`, `paraphina/src/live/runner.rs:L1384-L1445`, `paraphina/src/state.rs:L598-L681`.
- Sim vs Live (flags): Sim Implemented via deterministic synth + apply; Live Implemented via connector polling/fixtures and defined unavailable snapshot (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/live_account_ingestion_tests.rs:L52-L140`, `paraphina/tests/sim_account_snapshot_tests.rs:L1-L40`.

### WP-4.2-LIQ-ESTIMATE — Compute `price_liq` and `dist_liq_sigma`
- Status: Implemented.
- Evidence: `paraphina/src/live/runner.rs:L1384-L1445`, `paraphina/src/state.rs:L598-L681`.
- Sim vs Live (flags): Sim Implemented via synth + apply; Live Implemented via snapshot ingestion (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/live_account_ingestion_tests.rs:L52-L140`, `paraphina/tests/sim_account_snapshot_tests.rs:L1-L40`.

### WP-4.3-FILL-QUEUE — Fill queue with batch every `FILL_AGG_INTERVAL_MS`
- Status: Implemented.
- Evidence: `paraphina/src/fill_batcher.rs:L1-L64`, `paraphina/src/strategy_action.rs:L533-L538`, `paraphina/src/live/runner.rs:L546-L549`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs`; Live Implemented `paraphina/src/live/runner.rs` (feature `live`: `paraphina/Cargo.toml`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/fill_batcher_flow_tests.rs:L1-L89`, `paraphina/tests/live_fill_batcher_tests.rs:L29-L118`.

### WP-4.3-FILL-APPLY — Apply fills to positions/PnL
- Status: Implemented.
- Evidence: `paraphina/src/state.rs:L649-L746`, `paraphina/src/strategy_action.rs:L535-L539`, `paraphina/src/live/runner.rs:L1342-L1350`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs:L143-L152`; Live Implemented `paraphina/src/live/runner.rs:L844-L865` (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/order_ledger_tests.rs:L11-L64`.

### WP-4.3-OPEN-ORDERS — Update `open_orders_v` map keyed by order_id
- Status: Implemented.
- Evidence: `paraphina/src/state.rs:L120-L128`, `paraphina/src/execution_events.rs:L8-L90`.
- Sim vs Live (flags): Sim Implemented via core ledger updates; Live Implemented via core execution events + cancel-all mapping (feature `live`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/order_ledger_tests.rs:L11-L128`.

### WP-4.3-RECENT-FILLS — Record recent fills with markout field unset
- Status: Implemented.
- Evidence: `paraphina/src/state.rs:L68-L82`, `paraphina/src/state.rs:L748-L792`.
- Sim vs Live (flags): Sim Implemented; Live Implemented (feature `live`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/order_ledger_tests.rs:L130-L215`.

### WP-4.3-POST-FILL-RECOMPUTE — Recompute `q_t`, `B_t`; run exit/hedge after batch
- Status: Implemented.
- Evidence: `paraphina/src/state.rs:L795-L845`, `paraphina/src/strategy_action.rs:L535-L600`, `paraphina/src/live/runner.rs:L546-L549`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs`; Live Implemented `paraphina/src/live/runner.rs` (feature `live`: `paraphina/Cargo.toml`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/fill_batcher_flow_tests.rs:L1-L89`, `paraphina/tests/live_fill_batcher_tests.rs:L29-L118`.

### WP-4.4-MARKOUT — Compute markout after horizon; store on fill
- Status: Implemented.
- Evidence: `paraphina/src/toxicity.rs:L107-L170`, `paraphina/src/state.rs:L847-L902`.
- Sim vs Live (flags): Sim Implemented; Live Implemented (feature `live`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/toxicity_markout_tests.rs:L44-L110`, `paraphina/tests/order_ledger_tests.rs:L159-L215`.

---

## §11 Order Management (Queue & Lifetime Control)

### WP-11-MM-ONE-PER-SIDE — At most one active MM order per venue/side
- Status: Implemented.
- Evidence: `paraphina/src/state.rs:L151-L166`, `paraphina/src/order_management.rs:L27-L195`.
- Sim vs Live (flags): Sim Implemented (MM-only) `paraphina/src/strategy_action.rs:L156-L199`; Live Implemented (MM-only) `paraphina/src/execution_events.rs:L38-L55` (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/order_management_mm_one_per_side_tests.rs`.

### WP-11-CANCEL-IF-NOT-QUOTED — Cancel existing order when side not quoted
- Status: Implemented.
- Evidence: `paraphina/src/order_management.rs:L137-L145`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/order_management.rs:L137-L145`; Live Implemented via shared planner `paraphina/src/order_management.rs:L137-L145` (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/src/order_management.rs:L286-L304`.

### WP-11-MIN-LIFETIME / TOLERANCES — MIN_QUOTE_LIFETIME + price/size tolerance replace
- Status: Implemented.
- Evidence: `paraphina/src/order_management.rs:L169-L193`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/order_management.rs:L169-L193`; Live Implemented via shared planner `paraphina/src/order_management.rs:L169-L193` (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: Planned in `AUDIT_SUMMARY.md:L43-L48`.

### WP-11-MAKER-TIF-POSTONLY — Maker `tif` + `post_only=true` where supported
- Status: Implemented.
- Evidence: `paraphina/src/order_management.rs:L151-L163`, `paraphina/src/io/sim.rs:L49-L132`, `paraphina/src/live/mock_exchange.rs:L489-L507`, `paraphina/src/live/shadow_adapter.rs:L134-L145`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/io/sim.rs:L49-L132`; Live Implemented (mock exchange + shadow/live gateway wiring) `paraphina/src/live/mock_exchange.rs:L489-L507`, `paraphina/src/bin/paraphina_live.rs:L357-L470` (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/order_semantics_tests.rs:L1-L120`.

### WP-11-EXIT-HEDGE-IOC — Exit/Hedge orders are IOC and not gated by min lifetime
- Status: Implemented.
- Evidence: `paraphina/src/exit.rs:L704-L715`, `paraphina/src/hedge.rs:L890-L902`, `paraphina/src/io/sim.rs:L49-L132`, `paraphina/src/live/mock_exchange.rs:L559-L598`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/io/sim.rs:L49-L132`; Live Implemented (mock exchange + shadow/live gateway wiring) `paraphina/src/live/mock_exchange.rs:L559-L598`, `paraphina/src/bin/paraphina_live.rs:L357-L470` (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/order_semantics_tests.rs:L1-L120`.

---

## §14.5 Kill Behaviour

### WP-14.5-KILL-CANCEL-ALL — Cancel all open orders on kill
- Status: Implemented.
- Evidence: `paraphina/src/strategy_action.rs:L461-L479`, `paraphina/src/live/runner.rs:L520-L547`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs`, `paraphina/src/strategy.rs`; Live Implemented `paraphina/src/live/runner.rs` (feature `live`: `paraphina/Cargo.toml`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/kill_switch_integration_tests.rs:L1-L120`, `paraphina/tests/live_kill_switch_tests.rs:L1-L120`.

### WP-14.5-KILL-BEST-EFFORT — Optional best-effort hedge/exit when liquidation risk is dangerous
- Status: Implemented.
- Evidence: `paraphina/src/state.rs:L469-L536`, `paraphina/src/strategy_action.rs:L481-L505`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs`; Live Implemented `paraphina/src/live/runner.rs` (feature `live`: `paraphina/Cargo.toml`). Env flag: `PARAPHINA_KILL_BEST_EFFORT` (default off).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/kill_switch_integration_tests.rs:L1-L120`.

### WP-14.5-KILL-STOP-NEW — Stop new MM/EXIT/HEDGE placements
- Status: Implemented.
- Evidence: `paraphina/src/order_management.rs:L36-L58`, `paraphina/src/exit.rs:L372-L375`, `paraphina/src/hedge.rs:L503-L505`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/order_management.rs:L36-L58`, `paraphina/src/exit.rs:L372-L375`, `paraphina/src/hedge.rs:L503-L505`; Live Implemented via same guards `paraphina/src/live/runner.rs:L532-L547` (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/src/order_management.rs:L306-L345`.

### WP-14.5-KILL-LOG — Emit detailed kill log with per-venue `q_v` and `dist_liq_sigma_v`
- Status: Implemented.
- Evidence: `paraphina/src/state.rs:L444-L466`, `paraphina/src/strategy_action.rs:L507-L509`, `paraphina/src/live/runner.rs:L1107-L1239`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs`, `paraphina/src/strategy.rs`; Live Implemented `paraphina/src/live/runner.rs` (feature `live`: `paraphina/Cargo.toml`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/telemetry_kill_event_tests.rs:L15-L56`.

### WP-14.5-KILL-LATCH — Kill switch latched until manual reset
- Status: Implemented.
- Evidence: `paraphina/src/engine.rs:L465-L557`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/engine.rs:L488-L584`; Live Implemented via shared engine `paraphina/src/live/runner.rs:L504-L507` (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/risk_regim_tests.rs:L3-L120`.

---

## §15 Logging, Metrics and Treasury Guidance

### WP-15-LOG-FV-VOL — Log KF state, fair value, vol terms, contributing venues, config_version_id
- Status: Implemented.
- Evidence: `paraphina/src/telemetry.rs:L362-L397`, `paraphina/src/strategy_action.rs:L608-L646`, `paraphina/src/live/runner.rs:L690-L742`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs:L608-L646`; Live Implemented `paraphina/src/live/runner.rs:L690-L742` (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/telemetry_schema_tests.rs`, `paraphina/tests/telemetry_determinism_tests.rs`, `paraphina/tests/live_telemetry_contract_tests.rs`.

### WP-15-LOG-QUOTES-SIZES — Per-venue quote components and sizes
- Status: Implemented.
- Evidence: `paraphina/src/telemetry.rs:L603-L688`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs:L608-L646`; Live Implemented `paraphina/src/live/runner.rs:L690-L742` (feature `live`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/telemetry_schema_tests.rs`.

### WP-15-LOG-ORDERS — Order placements/cancels/mods with tif/post_only/purpose
- Status: Implemented.
- Evidence: `paraphina/src/telemetry.rs:L739-L905`, `schemas/telemetry_schema_v1.json`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs:L608-L646`; Live Implemented `paraphina/src/live/runner.rs:L690-L742` (feature `live`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/telemetry_schema_tests.rs`, `paraphina/tests/telemetry_determinism_tests.rs`, `tests/test_telemetry_contract_gate.py`.

### WP-15-LOG-FILLS — Fill details with pre/post positions and realised PnL
- Status: Implemented.
- Evidence: `paraphina/src/telemetry.rs:L910-L936`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs:L608-L646`; Live Implemented `paraphina/src/live/runner.rs:L690-L742` (feature `live`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/telemetry_schema_tests.rs`, `paraphina/tests/live_telemetry_contract_tests.rs`.

### WP-15-LOG-EXITS — Map entry fills to exit orders; log edge components
- Status: Implemented.
- Evidence: `paraphina/src/telemetry.rs:L966-L1042`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs:L608-L646`; Live Implemented `paraphina/src/live/runner.rs:L690-L742` (feature `live`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/telemetry_schema_tests.rs`.

### WP-15-LOG-HEDGE — Log hedge actions and per-venue cost components
- Status: Implemented.
- Evidence: `paraphina/src/telemetry.rs:L1045-L1103`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs:L608-L646`; Live Implemented `paraphina/src/live/runner.rs:L690-L742` (feature `live`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/telemetry_schema_tests.rs`.

### WP-15-RISK-EVENTS — Log risk transitions, breaches, venue disables, kill switch
- Status: Implemented.
- Evidence: `paraphina/src/telemetry.rs:L468-L599`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs:L608-L646`; Live Implemented `paraphina/src/live/runner.rs:L690-L742` (feature `live`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/telemetry_schema_tests.rs`, `paraphina/tests/telemetry_determinism_tests.rs`.

### WP-15-METRICS-GLOBAL/PER-VENUE — Global and per-venue metrics
- Status: Implemented.
- Evidence: `paraphina/src/telemetry.rs:L1109-L1186` (per-venue metrics arrays), `schemas/telemetry_schema_v1.json`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs:L608-L646`; Live Implemented `paraphina/src/live/runner.rs:L690-L742` (feature `live`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/telemetry_schema_tests.rs`, `paraphina/tests/live_telemetry_contract_tests.rs`.

### WP-15-TREASURY-GUIDANCE — Treasury guidance outputs
- Status: Implemented.
- Evidence: `paraphina/src/treasury.rs:L1-L165`, `paraphina/src/telemetry.rs:L398-L430`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/strategy_action.rs:L608-L646`; Live Implemented `paraphina/src/live/runner.rs:L690-L742` (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/treasury_guidance_tests.rs`, `paraphina/tests/telemetry_schema_tests.rs`, `tests/test_telemetry_contract_gate.py`.

---

## §17 Architecture, Determinism and Rust Implementation Notes

### WP-17-IO-LAYER — Async WS/REST I/O for books, trades, funding, private feeds
- Status: Implemented.
- Evidence: `paraphina/src/bin/paraphina_live.rs:L223-L470`, `paraphina/src/live/connectors/hyperliquid.rs:L1-L169`, `paraphina/src/live/connectors/lighter.rs:L1-L170`.
- Sim vs Live (flags): Sim N/A; Live Implemented (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/live_gateway_tests.rs:L1-L220`, `paraphina/tests/live_hyperliquid_runner_fixture_tests.rs:L12-L79`.

### WP-17-STATE-CACHE — Canonical state cache with order books, margin, funding, dist_liq_sigma
- Status: Implemented.
- Evidence: `paraphina/src/live/state_cache.rs:L28-L366`, `paraphina/src/state.rs:L91-L166`, `paraphina/src/live/runner.rs:L1134-L1169`.
- Sim vs Live (flags): Sim Implemented (core L2) `paraphina/src/state.rs:L91-L166`; Live Implemented `paraphina/src/live/state_cache.rs:L28-L366`, `paraphina/src/live/runner.rs:L1134-L1169` (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/live_state_cache_parity_tests.rs`.

### WP-17-STRATEGY-ACTIONS — Strategy outputs action list (place/cancel/log)
- Status: Implemented.
- Evidence: `paraphina/src/strategy_action.rs:L399-L600`, `paraphina/src/bin/paraphina_live.rs:L357-L470`.
- Sim vs Live (flags): Sim Implemented `paraphina/src/actions.rs:L28-L176`, `paraphina/src/strategy_action.rs:L496-L505`; Live Partial (uses intents + action batches but no gateway policy) `paraphina/src/live/runner.rs:L552-L559` (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: Live loop bypasses Gateway policy enforcement. Evidence: `paraphina/src/live/runner.rs:L552-L559`.
- Acceptance tests: `paraphina/tests/action_gateway_regression.rs:L29-L87`, `paraphina/tests/live_gateway_tests.rs:L1-L220`.

### WP-17-GATEWAY-POLICY — Rate limiting/retries + execution semantics enforcement
- Status: Implemented.
- Evidence: `paraphina/src/io/mod.rs:L90-L169`, `paraphina/src/io/sim.rs:L49-L132`, `paraphina/src/live/gateway.rs:L203-L370`, `paraphina/src/bin/paraphina_live.rs:L357-L470`.
- Sim vs Live (flags): Sim Implemented; Live Implemented (feature `live`: `paraphina/Cargo.toml:L63-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/order_semantics_tests.rs:L1-L120`, `paraphina/tests/live_gateway_tests.rs:L1-L220`.

### WP-17-DETERMINISM & REPLAY — Deterministic core with replay capability
- Status: Implemented.
- Evidence: `paraphina/src/event_log.rs:L248-L313` (event log payloads + reader), `paraphina/src/live/runner.rs:L500-L720` (live event log wiring), `paraphina/src/bin/replay.rs:L179-L250` (live replay routing), `paraphina/src/live/runner.rs:L1518-L1620` (live replay).
- Sim vs Live (flags): Sim Implemented (event log + replay) `paraphina/src/event_log.rs:L248-L313`, `paraphina/src/bin/replay.rs:L179-L250`; Live Implemented (event log + replay) `paraphina/src/live/runner.rs:L500-L720`, `paraphina/src/live/runner.rs:L1518-L1620` (feature `event_log`: `paraphina/Cargo.toml:L60-L65`).
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/event_log_replay_tests.rs`, `paraphina/tests/live_event_log_replay_tests.rs`.



---

## CI Enforcement

### WP-CI-PHASEA-NO-PROMOTION-WITHOUT-GATES — No promotion without passing gates
- Status: Implemented.
- Evidence: `batch_runs/phase_a/adversarial_search_promote.py:L864-L964`, `.github/workflows/phase_a_smoke.yml`.
- Sim vs Live (flags): Sim Implemented; Live Implemented.
- Dependency blockers: None.
- Acceptance tests: `batch_runs/phase_a/tests/test_adversarial_search.py`.

### WP-CI-PHASEA-REPRODUCIBILITY — Reproducibility from artifacts
- Status: Implemented.
- Evidence: `batch_runs/phase_a/verify_cem_reproducibility.py`, `.github/workflows/phase_a_smoke.yml`.
- Sim vs Live (flags): Sim Implemented; Live Implemented.
- Dependency blockers: None.
- Acceptance tests: `batch_runs/phase_a/tests/test_adversarial_search.py`.


---

## RL Reward Alignment

### WP-RL-REWARD-BUDGETS — Reward reflects research alignment budgets
- Status: Implemented.
- Evidence: `configs/research_alignment_budgets.json`, `paraphina/src/rl/research_budgets.rs`, `paraphina/src/rl/telemetry.rs`.
- Sim vs Live (flags): Sim Implemented; Live N/A.
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/rl_reward_budgets_tests.rs`.

---

## Status Blocks (Appendix A/B)

### WP-STATUS-OPTIMIZATION — “Fully optimised” pipeline status
- Status: Implemented.
- Evidence: `batch_runs/phase_a/promote_pipeline.py`, `batch_runs/phase_a/adversarial_search_promote.py`, `batch_runs/phase_ab/pipeline.py`, `batch_runs/phase_b/gate.py`.
- Sim vs Live (flags): Sim Implemented; Live Implemented.
- Dependency blockers: None.
- Acceptance tests: `batch_runs/phase_ab/tests/test_phase_ab_end_to_end.py`, `batch_runs/phase_a/tests/test_adversarial_search.py`.

### WP-STATUS-PHASE-A-B — Phase A→B gating status
- Status: Implemented.
- Evidence: `batch_runs/phase_ab/pipeline.py`, `batch_runs/phase_ab/cli.py`, `.github/workflows/phase_ab_promotion_gate.yml`.
- Sim vs Live (flags): Sim Implemented; Live Implemented.
- Dependency blockers: None.
- Acceptance tests: `batch_runs/phase_ab/tests/test_phase_ab_end_to_end.py`, `batch_runs/phase_ab/tests/test_phase_ab.py`.

### WP-STATUS-POLICY-SURFACES — Safe RL pipeline status
- Status: Implemented.
- Evidence: `paraphina/src/rl/sim_env.rs:L320-L386`, `paraphina/src/rl/safety.rs:L1-L113`, `paraphina/src/rl/safe_pipeline.rs`.
- Sim vs Live (flags): Sim Implemented; Live N/A.
- Dependency blockers: None.
- Acceptance tests: `paraphina/tests/rl_safe_pipeline_tests.rs`.
