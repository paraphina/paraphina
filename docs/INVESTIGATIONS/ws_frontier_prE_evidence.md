# WS Frontier PR E Evidence (Before/After)

## Status
- **Blocked in Phase 1 (BEFORE soak)**
- Stop condition triggered: unable to run the required 10m soak due to host build/runtime constraint.
- No tuning code changes were applied.

## Phase 0 Checks
- `pgrep -fa 'paraphina_live.*--trade-mode shadow'` -> no running shadow process.
- `git switch main && git pull --ff-only origin main` -> up to date.
- `git status -sb` -> clean.
- Branch created: `connectivity/ws-frontier-tuning-prE`.
- Frontier primitives verified via `rg` on:
  - `venue_age_ms` / `venue_age_event_ms`
  - `tools/ws_soak_report.py` and `docs/INVESTIGATIONS/ws_frontier_status.md`
  - `PARAPHINA_LIGHTER_PING_INTERVAL_MS` / `PARAPHINA_EXTENDED_WS_READ_TIMEOUT_MS`
  - `WS_AUDIT` lines in connectors + market publisher

## BEFORE Attempt (10m Baseline Soak)

### Exact command used
```bash
OUT_DIR=/tmp/ws_frontier_prE_before_20260211T124924Z
mkdir -p "$OUT_DIR"
PARAPHINA_LIVE_MODE=1 \
PARAPHINA_LIVE_PREFLIGHT_OK=1 \
PARAPHINA_WS_AUDIT=1 \
PARAPHINA_MARKET_RX_STATS=1 \
PARAPHINA_MARKET_RX_STATS_PATH="$OUT_DIR/market_rx_stats.log" \
PARAPHINA_TELEMETRY_MODE=jsonl \
PARAPHINA_TELEMETRY_PATH="$OUT_DIR/telemetry.jsonl" \
PARAPHINA_LIGHTER_PING_INTERVAL_MS=10000 \
PARAPHINA_EXTENDED_WS_READ_TIMEOUT_MS=45000 \
timeout 10m cargo run --release -p paraphina --bin paraphina_live \
  --features "live_hyperliquid live_lighter live_extended live_aster live_paradex" -- \
  --trade-mode shadow \
  --connectors hyperliquid,lighter,extended,aster,paradex \
  --out-dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/run.log"
```

### Artifacts produced
- `OUT_DIR=/tmp/ws_frontier_prE_before_20260211T124924Z`
- Present: `run.log`
- Missing (not produced): `telemetry.jsonl`, `market_rx_stats.log`

### Key failure signal (from `run.log`)
```text
error: failed to write /home/developer/code/paraphina/target/release/deps/libparaphina-79dcb65cdb42974b.rmeta: Invalid cross-device link (os error 18)
error: could not compile `paraphina` (lib) due to 1 previous error; 18 warnings emitted
```

## Phase 1B Report Generation Attempt

### Command
```bash
python3 tools/ws_soak_report.py --out-dir /tmp/ws_frontier_prE_before_20260211T124924Z
```

### Result
```text
error: missing required file: /tmp/ws_frontier_prE_before_20260211T124924Z/telemetry.jsonl
```

## BEFORE Metrics Availability
- `venue_age_ms` p50/p95/p99: unavailable (no telemetry).
- `venue_age_event_ms` p50/p95/p99: unavailable (no telemetry).
- stale plateau durations: unavailable (no telemetry).
- reconnect reason counts: unavailable from soak report (telemetry/report generation blocked).
- MarketPublisher pressure counters: unavailable from soak report.
- runner `cap_hits`: unavailable (no `market_rx_stats.log`).

## Outcome
- Per instructions, workflow stops after Phase 1C when 10m soak cannot be produced.
- No frontier tuning guesses or code behavior changes were made in this attempt.

---

## PR E0: EXDEV Unblock Diagnostics (2026-02-11)

### Phase 0 (process + branch)
- `pgrep -fa '[p]araphina_live.*--trade-mode shadow'` -> empty.
- Current branch: `connectivity/ws-frontier-tuning-prE`.
- Working tree before E0 diagnostics: only this evidence doc untracked.

### Phase 1A: filesystem identity
Command:
```bash
pwd
df -T /home/developer/code/paraphina /home/developer/code/paraphina/target /tmp
stat -c '%d %n' /home/developer/code/paraphina /home/developer/code/paraphina/target /tmp
```

Observed:
```text
/home/developer/code/paraphina
Filesystem     Type 1K-blocks     Used Available Use% Mounted on
/dev/sda1      ext4  78425224 57995280  17185204  78% /
/dev/sda1      ext4  78425224 57995280  17185204  78% /
/dev/sda1      ext4  78425224 57995280  17185204  78% /
2049 /home/developer/code/paraphina
2049 /home/developer/code/paraphina/target
2049 /tmp
```

### Phase 1B: hardlink probes
Commands executed:
```bash
mkdir -p /tmp/ws_build_fs_probe && cd /tmp/ws_build_fs_probe
echo hi > a && ln a b && ls -la a b

cd /home/developer/code/paraphina
mkdir -p target/ws_build_fs_probe && cd target/ws_build_fs_probe
echo hi > a && (ln a b && ls -la a b) || true
```

Observed:
```text
-rw-rw-r-- 2 developer developer 3 ... a
-rw-rw-r-- 2 developer developer 3 ... b
-rw-rw-r-- 2 developer developer 3 ... a
-rw-rw-r-- 2 developer developer 3 ... b
```

### Phase 1C: temp-dir environment visibility
```text
TMPDIR=None
RUSTC_TMPDIR=None
tempfile.gettempdir()=/tmp
```

### Phase 2 Strategy 1 (build under `/tmp`) — FAILED
Environment used:
```bash
BUILD_ROOT=/tmp/paraphina_build_20260211T131748Z
export CARGO_TARGET_DIR="$BUILD_ROOT/target"
export TMPDIR="$BUILD_ROOT/tmp"
export RUSTC_TMPDIR="$BUILD_ROOT/tmp"
export CARGO_NET_OFFLINE=true
```

Failure:
```text
error: failed to write /tmp/paraphina_build_20260211T131748Z/target/debug/deps/libunicode_ident-3c2fd7f55134ed81.rmeta: Invalid cross-device link (os error 18)
error: could not compile `unicode-ident` (lib) due to 1 previous error
error: failed to write /tmp/paraphina_build_20260211T131748Z/target/release/deps/libunicode_ident-9fb3fd2828c906ec.rmeta: Invalid cross-device link (os error 18)
error: could not compile `unicode-ident` (lib) due to 1 previous error
```

### Phase 2 Strategy 2 (build under repo) — FAILED
Environment used:
```bash
BUILD_ROOT=/home/developer/code/paraphina/target/ws_build_20260211T131800Z
export CARGO_TARGET_DIR="$BUILD_ROOT/target"
export TMPDIR="$BUILD_ROOT/tmp"
export RUSTC_TMPDIR="$BUILD_ROOT/tmp"
export CARGO_NET_OFFLINE=true
```

Failure:
```text
error: failed to write /home/developer/code/paraphina/target/ws_build_20260211T131800Z/target/debug/deps/libunicode_ident-3c2fd7f55134ed81.rmeta: Invalid cross-device link (os error 18)
error: could not compile `unicode-ident` (lib) due to 1 previous error
error: failed to write /home/developer/code/paraphina/target/ws_build_20260211T131800Z/target/release/deps/libunicode_ident-9fb3fd2828c906ec.rmeta: Invalid cross-device link (os error 18)
error: could not compile `unicode-ident` (lib) due to 1 previous error
```

### strace step (required after both failures)
- Initial sandboxed `strace` failed with ptrace permission errors.
- Re-ran with escalation and wrote log to `/tmp/ws_exdev_strace.log`.
- `rg -n "EXDEV|Invalid cross-device link" /tmp/ws_exdev_strace.log` returned no matches in captured window.
- The traced run showed many successful `rename(...) = 0` operations before manual stop; no failing syscall was captured before interruption.

### E0 conclusion
- EXDEV is reproducible in both prescribed build-root strategies with explicit `TMPDIR`/`RUSTC_TMPDIR` co-location.
- No working local build-root pair was found in this host/session.
- Recommended next execution environment pair: source + build root in a different unsandboxed workspace on the same native filesystem (for example, `/tmp/paraphina_src` with `CARGO_TARGET_DIR=/tmp/paraphina_src/target` and `TMPDIR=/tmp/paraphina_src/tmp`) and rerun Phase 2 there.
- Per stop rule, no soak rerun and no tuning changes were attempted in E0.
