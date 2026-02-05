# Snapshot Decode Silent Failure Audit

**Date**: 2026-02-03  
**Branch**: `research/snapshot-decode-silent-fail-audit`  
**Mode**: AUDIT ONLY (read-only)

## TL;DR

The silent decode failure pattern discovered in Hyperliquid (where `decode_l2book_snapshot` can return `None` without logging, causing permanent staleness) **exists in ALL other connectors** to varying degrees. Lighter is the most concerning because it has **NO freshness tracking or watchdog mechanism at all**.

## Summary Table

| Venue | Full-Book Decode? | Strict Parse Risk? | Silent Failure Risk? | Bounded Logging? | Freshness Semantics Safe? | Recommendation |
|-------|-------------------|-------------------|---------------------|------------------|---------------------------|----------------|
| Hyperliquid | Yes | **HIGH** - `parse_levels` fails on ANY malformed level | **HIGH** - `decode_l2book_snapshot` returns None silently | Partial (only top decode) | **YES** - only on publishable events | **Must fix now** (already fixed in PR) |
| Paradex | Yes | **HIGH** - `parse_levels_from_value` fails on ANY malformed level | **MEDIUM** - has some logging but snapshot decode can fail silently | Partial (only top decode) | **YES** - only on publishable events | Should fix |
| Extended | Yes | **HIGH** - `parse_levels_from_value` fails on ANY malformed level | **MEDIUM** - uses typed deser but fallback paths can fail silently | Partial (only top decode) | **PARTIAL** - updated before publish confirmation | Should fix |
| Aster | Yes | **HIGH** - `parse_levels_from_value` fails on ANY malformed level | **MEDIUM** - uses typed deser for updates, snapshot has logging | Partial (only top decode) | **YES** - only on publishable events | Should fix |
| Lighter | Yes | **MEDIUM** - `parse_levels_from_objects` skips some invalid entries but fails on others | **CRITICAL** - NO logging on decode failures | NO (only top decode with field check) | **N/A - NO FRESHNESS TRACKING** | **Must fix now** |

## Connector-by-Connector Analysis

---

### 1. Hyperliquid

**Status**: Being fixed in `fix/hl-snapshot-decode-resilience` PR

#### Decode Functions

| Function | File:Line | Returns | Risk |
|----------|-----------|---------|------|
| `decode_l2book_snapshot` | `hyperliquid.rs:1044` | `Option<MarketDataEvent>` | **HIGH** - silent failure |
| `decode_l2book_top` | `hyperliquid.rs:1017` | `Option<TopOfBook>` | MEDIUM - bounded logging exists |
| `parse_levels` | `hyperliquid.rs:970` | `Option<Vec<BookLevel>>` | **HIGH** - fails on ANY malformed level |

#### Failure Paths

```rust
// hyperliquid.rs:379-393 - SILENT FAILURE PATH
if let Some(snapshot) = decode_l2book_snapshot(...) {
    freshness.last_parsed_ns.store(mono_now_ns(), Ordering::Relaxed);  // Line 385-387
    // ... publish
}
// NO else branch - silently dropped!
```

#### Strict Parse Pattern

```rust
// hyperliquid.rs:970-977
fn parse_levels(levels: &serde_json::Value) -> Option<Vec<BookLevel>> {
    let mut out = Vec::new();
    for level in levels.as_array()? {
        let level = parse_level_entry(level)?;  // Returns None on ANY failure
        out.push(level);
    }
    Some(out)
}
```

#### Freshness Semantics
- `last_parsed_ns` updated at line 385-387 **only when snapshot decode succeeds** ✓ Safe
- `last_published_ns` updated at line 277-278 by forwarder task ✓ Safe

---

### 2. Paradex

**Status**: Should fix

#### Decode Functions

| Function | File:Line | Returns | Risk |
|----------|-----------|---------|------|
| `decode_bbo_top_and_snapshot` | `paradex.rs:1039` | `Option<(TopOfBook, MarketDataEvent)>` | **MEDIUM** - silent failure |
| `decode_top_of_book_value` | `paradex.rs:1024` | `Option<TopOfBook>` | MEDIUM - bounded logging for misses |
| `parse_levels_from_value` | `paradex.rs:1180` | `Option<Vec<BookLevel>>` | **HIGH** - fails on ANY malformed level |
| `parse_levels_any` | `paradex.rs:1190` | `Option<Vec<BookLevel>>` | MEDIUM - skips some invalid but can still fail |

#### Failure Paths

```rust
// paradex.rs:352-375 - PARTIAL LOGGING
if let Some((top, snapshot)) = decode_bbo_top_and_snapshot(...) {
    // FIRST_DECODED_TOP logging (lines 358-364)
    self.freshness.last_parsed_ns.store(...);  // Line 369-371
    self.publish_market(snapshot).await;
}
// No else - silently dropped, but bounded logging exists for top decode
```

#### Strict Parse Pattern

```rust
// paradex.rs:1180-1188
fn parse_levels_from_value(value: &Value) -> Option<Vec<BookLevel>> {
    let entries = value.as_array()?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let (price, size) = parse_level_pair(entry)?;  // Returns None on ANY failure
        out.push(BookLevel { price, size });
    }
    Some(out)
}
```

Note: `parse_levels_any` (line 1190) is more resilient but still fails if object parse fails.

#### Freshness Semantics
- `last_parsed_ns` updated at line 369-371 and 404-406 **only on successful decode** ✓ Safe

---

### 3. Extended

**Status**: Should fix

#### Decode Functions

| Function | File:Line | Returns | Risk |
|----------|-----------|---------|------|
| `decode_top_from_value` | `extended.rs:1278` | `Option<TopOfBook>` | MEDIUM - bounded logging for misses |
| `parse_depth_snapshot_from_ws` | `extended.rs:1305` | `Option<MarketDataEvent>` | **MEDIUM** - silent failure |
| `parse_levels_from_value` | `extended.rs:1393` | `Option<Vec<BookLevel>>` | **HIGH** - fails on ANY malformed level |

#### Failure Paths

Extended uses typed deserialization (`ExtendedDepthUpdate`) for primary path but has fallback JSON parsing:

```rust
// extended.rs:460-462 - Freshness updated BEFORE publish confirmation
self.freshness.last_parsed_ns.store(mono_now_ns(), Ordering::Relaxed);
// Then later:
if let Some(event) = outcome {
    self.publish_market(event).await;
}
```

#### Strict Parse Pattern

```rust
// extended.rs:1393-1401
fn parse_levels_from_value(value: &Value) -> Option<Vec<BookLevel>> {
    let entries = value.as_array()?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let (price, size) = parse_level_pair(entry)?;  // Returns None on ANY failure
        out.push(BookLevel { price, size });
    }
    Some(out)
}
```

#### Freshness Semantics
- `last_parsed_ns` updated at lines 438-440, 460-462 **before** checking if publishable event was produced
- **PARTIAL RISK**: Freshness can be updated even if no event is published

---

### 4. Aster

**Status**: Should fix

#### Decode Functions

| Function | File:Line | Returns | Risk |
|----------|-----------|---------|------|
| `decode_top_of_book_with_raw` | `aster.rs:1242` | `Option<(TopOfBook, String, String)>` | MEDIUM - bounded logging exists |
| `parse_depth_snapshot` | `aster.rs:1231` | `Option<AsterDepthSnapshot>` | **MEDIUM** - used in REST fallback |
| `parse_depth_update` | `aster.rs:1273` | `Option<AsterDepthUpdate>` | **MEDIUM** - silent failure |
| `parse_levels_from_value` | `aster.rs:1301` | `Option<Vec<BookLevel>>` | **HIGH** - fails on ANY malformed level |

#### Failure Paths

```rust
// aster.rs:308-341 - Snapshot decode with logging
if let Ok(value) = serde_json::from_str::<Value>(&snapshot_raw) {
    if let Some((top, bid_raw, ask_raw)) = decode_top_of_book_with_raw(&value) {
        // FIRST_DECODED_TOP logging
    } else if decode_miss_count < 3 {
        log_decode_miss(...);  // Bounded logging exists
    }
}
// Snapshot event created directly from typed deser (line 343-351)
```

#### Strict Parse Pattern

```rust
// aster.rs:1301-1309
fn parse_levels_from_value(value: &Value) -> Option<Vec<BookLevel>> {
    let entries = value.as_array()?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let (price, size) = parse_level_pair(entry)?;  // Returns None on ANY failure
        out.push(BookLevel { price, size });
    }
    Some(out)
}
```

#### Freshness Semantics
- `last_parsed_ns` updated at lines 352-354, 366-368 **only on successful publish** ✓ Safe

---

### 5. Lighter

**Status**: **MUST FIX NOW** - Critical gap

#### Decode Functions

| Function | File:Line | Returns | Risk |
|----------|-----------|---------|------|
| `decode_order_book_snapshot` | `lighter.rs:862` | `Option<ParsedL2Message>` | **HIGH** - silent failure |
| `decode_order_book_channel_message` | `lighter.rs:946` | `Option<ParsedL2Message>` | **HIGH** - silent failure |
| `decode_order_book_top` | `lighter.rs:907` | `Option<TopOfBook>` | MEDIUM - bounded logging exists |
| `parse_levels_from_objects` | `lighter.rs:1138` | `Option<Vec<BookLevel>>` | **MEDIUM** - skips short arrays but fails on object parse |

#### Failure Paths

```rust
// lighter.rs:529-544 - SILENT FAILURE (channel message)
if let Some(parsed) = decode_order_book_channel_message(...) {
    let outcome = tracker.on_message(parsed);
    if let Some(event) = outcome.event {
        self.publish_market(event).await;
    }
    continue;
}
// NO logging on failure!

// lighter.rs:547-563 - SILENT FAILURE (snapshot)
if let Some(parsed) = decode_order_book_snapshot(...) {
    let outcome = tracker.on_message(parsed);
    if let Some(event) = outcome.event {
        self.publish_market(event).await;
    }
}
// NO logging on failure!
```

#### Mixed Parse Pattern

```rust
// lighter.rs:1138-1161 - Partially resilient
fn parse_levels_from_objects(value: &serde_json::Value) -> Option<Vec<BookLevel>> {
    let entries = value.as_array()?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        if let Some(obj) = entry.as_object() {
            let price = obj.get("price").or_else(|| obj.get("px"))?;  // Can fail
            let size = obj.get("size").or_else(|| obj.get("sz"))?;    // Can fail
            out.push(BookLevel {
                price: parse_f64_value(price)?,  // Can fail
                size: parse_f64_value(size)?,    // Can fail
            });
            continue;
        }
        if let Some(items) = entry.as_array() {
            if items.len() < 2 {
                continue;  // Skips invalid - GOOD
            }
            // ...
        }
    }
    Some(out)
}
```

Note: Empty arrays `[]` are explicitly allowed for bids/asks (lines 873-880).

#### Freshness Semantics
- **NO FRESHNESS TRACKING WHATSOEVER**
- No `Freshness` struct
- No `last_parsed_ns` or `last_published_ns`
- No watchdog mechanism
- **If Lighter goes silent, there is NO automatic recovery mechanism**

---

## Ranked Recommendations

### MUST FIX NOW (High risk of permanent staleness)

1. **Lighter: Add Freshness tracking + watchdog**
   - File: `lighter.rs`
   - Issue: No freshness tracking means no automatic recovery if WS goes silent
   - Fix: Add `Freshness` struct with `last_parsed_ns`/`last_published_ns`, watchdog task, and `reset_for_new_connection()`

2. **Lighter: Add bounded logging for decode failures**
   - File: `lighter.rs:529-563`
   - Issue: `decode_order_book_channel_message` and `decode_order_book_snapshot` fail silently
   - Fix: Add rate-limited logging similar to Hyperliquid fix

### SHOULD FIX (Diagnostic improvements)

3. **All connectors: Make level parsing resilient**
   - Files: `paradex.rs:1180`, `extended.rs:1393`, `aster.rs:1301`, `lighter.rs:826`
   - Issue: `parse_levels_*` functions fail if ANY level is malformed
   - Fix: Skip invalid levels, require min 1 valid level per side, log skip counts

4. **Paradex: Add bounded logging for snapshot decode failures**
   - File: `paradex.rs:352-375`
   - Issue: `decode_bbo_top_and_snapshot` returns None silently
   - Fix: Add rate-limited logging on failure path

5. **Extended: Fix freshness update timing**
   - File: `extended.rs:460-462`
   - Issue: `last_parsed_ns` updated before confirming publishable event
   - Fix: Move freshness update to after `outcome.event` is confirmed

6. **Aster: Add bounded logging for `parse_depth_update` failures**
   - File: `aster.rs:1273`
   - Issue: WS depth updates can fail silently
   - Fix: Add rate-limited logging

### NICE TO HAVE

7. **Standardize decode result types across connectors**
   - Create a shared `DecodeResult` struct with diagnostic fields (similar to `SnapshotDecodeResult` in HL fix)
   - Enables consistent logging and monitoring

8. **Add telemetry field for decode failure rate**
   - Track per-venue decode success/failure rates in telemetry
   - Enable alerting on elevated failure rates

---

## Commands Run

```bash
# Static search for decode/parse patterns
rg -n "decode_.*snapshot|decode_.*book|parse_levels|levels\(|l2Book|order_book|BBO|TopOfBook" paraphina/src/live/connectors

# Search for decode conditionals
rg -n "if let Some\(.*decode|match .*decode" paraphina/src/live/connectors

# Search for continue statements (potential silent drops)
rg -n "continue;\s*$" paraphina/src/live/connectors

# Search for freshness patterns
rg -n "last_parsed_ns|last_published_ns|freshness" paraphina/src/live/connectors

# Lighter-specific: confirm no freshness
rg -n "watchdog|stale|STALE" paraphina/src/live/connectors/lighter.rs
```

---

## Appendix: File References

| Connector | Main File | Key Decode Functions | Key Parse Functions |
|-----------|-----------|---------------------|---------------------|
| Hyperliquid | `paraphina/src/live/connectors/hyperliquid.rs` | Lines 1017, 1044 | Line 970 |
| Paradex | `paraphina/src/live/connectors/paradex.rs` | Lines 1010, 1024, 1039 | Lines 1180, 1190 |
| Extended | `paraphina/src/live/connectors/extended.rs` | Lines 1278, 1305 | Line 1393 |
| Aster | `paraphina/src/live/connectors/aster.rs` | Lines 1231, 1242, 1273 | Line 1301 |
| Lighter | `paraphina/src/live/connectors/lighter.rs` | Lines 862, 907, 946 | Lines 826, 1138 |
