# Telemetry Schema v1

This document defines the **Telemetry Contract** for Paraphina per-tick telemetry JSONL output. It is the authoritative reference for what telemetry consumers (analytics, Phase B world-model training, research tooling) can depend on.

## Purpose

The Telemetry Contract Gate prevents **telemetry schema drift** from breaking downstream consumers silently. Any schema change (field additions, removals, type changes) must be explicit, versioned, and backwards-compatible.

## Record Format

**Format:** JSONL (JSON Lines) — one valid JSON object per line, newline-delimited.

**File extension:** `.jsonl`

**Encoding:** UTF-8 (no BOM)

**Line terminator:** `\n` (LF)

---

## Schema Version

Every telemetry record **MUST** include:

```json
{"schema_version": 1, ...}
```

The `schema_version` field is a required integer that identifies the schema contract in use. Consumers MUST check this field and fail fast if they encounter an unexpected version.

---

## Required Fields

All telemetry records MUST include the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | integer | Schema version identifier. MUST be `1` for this schema. |
| `t` | integer | Tick index (monotonically increasing within a run, starting from 0). |
| `pnl_realised` | number | Daily realised PnL in USD. |
| `pnl_unrealised` | number | Daily unrealised PnL in USD. |
| `pnl_total` | number | Daily total PnL in USD (`pnl_realised + pnl_unrealised`). |
| `risk_regime` | string | Current risk regime. One of: `"Normal"`, `"Warning"`, `"HardLimit"`. |
| `kill_switch` | boolean | Whether the kill switch has been triggered. |
| `kill_reason` | string | Kill switch reason (e.g., `"None"`, `"DeltaLimit"`, `"LossLimit"`, etc.). |
| `q_global_tao` | number | Global inventory in TAO. |
| `dollar_delta_usd` | number | Dollar delta exposure in USD. |
| `basis_usd` | number | Net basis exposure in USD. |

---

## Optional Fields

The following fields MAY be present. If present, they MUST conform to the specified type:

| Field | Type | Description |
|-------|------|-------------|
| `fv_available` | boolean | Whether fair value is available (sufficient healthy venues). |
| `fair_value` | number \| null | Current fair value estimate (null if unavailable). |
| `sigma_eff` | number | Effective volatility (after floor). |
| `healthy_venues_used_count` | integer | Number of healthy venues used for FV update. |
| `healthy_venues_used` | array[integer] | Indices of venues used in last FV update. |
| `basis_gross_usd` | number | Gross basis exposure in USD. |

---

## Invariants

1. **Monotonic tick**: The `t` field MUST be monotonically increasing within a run (i.e., `t[i+1] > t[i]`).

2. **Non-NaN/Inf numerics**: All numeric fields (`number` type) MUST be finite (not NaN, not +Inf, not -Inf).

3. **Schema version consistency**: All records in a single JSONL file MUST have the same `schema_version`.

4. **Risk regime enum**: `risk_regime` MUST be one of the defined enum values.

5. **Boolean fields**: `kill_switch` and `fv_available` (if present) MUST be JSON booleans (`true`/`false`), not strings.

6. **Integer fields**: `schema_version`, `t`, `healthy_venues_used_count`, and elements of `healthy_venues_used` MUST be integers (no decimal point).

---

## Type Definitions

| Type | JSON Representation | Notes |
|------|---------------------|-------|
| integer | JSON number with no decimal | e.g., `1`, `42`, `-5` |
| number | JSON number | May have decimal. MUST be finite. |
| string | JSON string | UTF-8, properly escaped |
| boolean | JSON boolean | `true` or `false` |
| array[T] | JSON array | Elements must conform to type T |
| null | JSON null | Explicit null value |

---

## Backwards Compatibility Rules

1. **New optional fields may be added** in minor versions without changing `schema_version`.
   - Consumers MUST ignore unrecognized fields.

2. **Required fields may NOT be removed** without incrementing `schema_version`.

3. **Field types may NOT change** without incrementing `schema_version`.

4. **Field semantics may NOT change** (e.g., changing units from USD to TAO) without incrementing `schema_version`.

5. **New required fields** require incrementing `schema_version`.

---

## Validation

Telemetry files can be validated using:

```bash
python3 tools/check_telemetry_contract.py path/to/telemetry.jsonl
```

Exit codes:
- `0` — All records valid
- `1` — Contract violation (schema error)
- `2` — File not found or unreadable
- `3` — Internal validator error

---

## Machine-Readable Schema

The canonical machine-readable schema is at:

```
schemas/telemetry_schema_v1.json
```

This JSON file is used by the validator and MUST be kept in sync with this document.

---

## Example Record

```json
{
  "schema_version": 1,
  "t": 0,
  "pnl_realised": 0.0,
  "pnl_unrealised": 0.0,
  "pnl_total": 0.0,
  "risk_regime": "Normal",
  "kill_switch": false,
  "kill_reason": "None",
  "q_global_tao": 0.0,
  "dollar_delta_usd": 0.0,
  "basis_usd": 0.0,
  "fv_available": true,
  "fair_value": 250.5,
  "sigma_eff": 0.02,
  "healthy_venues_used_count": 3,
  "healthy_venues_used": [0, 1, 2]
}
```

---

## Related Documentation

- `docs/SIM_OUTPUT_SCHEMA.md` — Simulation output schema (run_summary.json, metrics.jsonl)
- `docs/PHASE_A_PROMOTION_PIPELINE.md` — Promotion pipeline telemetry requirements
- `docs/EVIDENCE_PACK.md` — Evidence pack specification

---

## Changelog

### v1 (2026-01-05)
- Initial schema version
- Defined required fields: `schema_version`, `t`, `pnl_realised`, `pnl_unrealised`, `pnl_total`, `risk_regime`, `kill_switch`, `kill_reason`, `q_global_tao`, `dollar_delta_usd`, `basis_usd`
- Defined optional fields: `fv_available`, `fair_value`, `sigma_eff`, `healthy_venues_used_count`, `healthy_venues_used`, `basis_gross_usd`
- Established invariants and backwards compatibility rules

