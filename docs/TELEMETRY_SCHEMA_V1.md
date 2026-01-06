# Telemetry Schema v1

This document defines the **Telemetry Contract** for Paraphina's JSONL output files. It is the authoritative reference for what telemetry consumers (analytics, Phase B world-model training, research tooling) can depend on.

## Purpose

The Telemetry Contract Gate prevents **telemetry schema drift** from breaking downstream consumers silently. Any schema change (field additions, removals, type changes) must be explicit, versioned, and backwards-compatible.

## File Types and Schema Mapping

The telemetry system supports multiple file types, each with its own schema:

| File | Schema | Description |
|------|--------|-------------|
| `telemetry.jsonl` | `schemas/telemetry_schema_v1.json` | Per-tick simulation telemetry |
| `mc_runs.jsonl` | `schemas/mc_runs_schema_v1.json` | Monte Carlo per-run summary records |

**Note:** Files like `metrics.jsonl` and `trajectories.jsonl` are recognized but currently unmapped. The validator will fail loudly with instructions on how to add schema support.

---

## Common Record Format

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

# Per-Tick Telemetry Schema (`telemetry.jsonl`)

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

## Invariants

1. **Monotonic tick**: The `t` field MUST be monotonically increasing within a run (i.e., `t[i+1] > t[i]`).

2. **Non-NaN/Inf numerics**: All numeric fields (`number` type) MUST be finite (not NaN, not +Inf, not -Inf).

3. **Schema version consistency**: All records in a single JSONL file MUST have the same `schema_version`.

4. **Risk regime enum**: `risk_regime` MUST be one of the defined enum values.

5. **Boolean fields**: `kill_switch` and `fv_available` (if present) MUST be JSON booleans (`true`/`false`), not strings.

6. **Integer fields**: `schema_version`, `t`, `healthy_venues_used_count`, and elements of `healthy_venues_used` MUST be integers (no decimal point).

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

# Monte Carlo Runs Schema (`mc_runs.jsonl`)

## Required Fields

All MC run records MUST include the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | integer | Schema version identifier. MUST be `1` for this schema. |
| `run_index` | integer | Global run index (0-based, monotonically increasing across shards). |
| `seed` | integer | RNG seed used for this run (should equal `base_seed + run_index`). |
| `pnl_total` | number | Final PnL at end of run in USD. |
| `max_drawdown` | number | Maximum drawdown observed during run in USD. |
| `kill_switch` | boolean | Whether kill switch was triggered. |
| `kill_reason` | string | Kill switch reason (e.g., `"None"`, `"LossLimitBreached"`, etc.). |
| `ticks_executed` | integer | Number of ticks executed in the run. |

## Optional Fields

The following fields MAY be present. If present, they MUST conform to the specified type:

| Field | Type | Description |
|-------|------|-------------|
| `kill_tick` | integer \| null | Tick at which kill switch was triggered (null if not triggered). |
| `max_abs_delta_usd` | number | Maximum absolute dollar delta in USD. |
| `max_abs_basis_usd` | number | Maximum absolute basis exposure in USD. |
| `max_abs_q_tao` | number | Maximum absolute inventory in TAO. |
| `max_venue_toxicity` | number | Maximum venue toxicity observed. |

## Invariants

1. **Monotonic run_index**: The `run_index` field MUST be monotonically increasing within a file (supports sharded aggregation).

2. **Non-NaN/Inf numerics**: All numeric fields (`number` type) MUST be finite (not NaN, not +Inf, not -Inf).

3. **Schema version consistency**: All records in a single JSONL file MUST have the same `schema_version`.

4. **Extra fields allowed**: Additional fields beyond those listed are permitted for forward compatibility. Consumers MUST ignore unrecognized fields.

## Example Record

```json
{
  "schema_version": 1,
  "run_index": 0,
  "seed": 12345,
  "pnl_total": -50.23,
  "max_drawdown": 120.50,
  "kill_switch": false,
  "kill_tick": null,
  "kill_reason": "None",
  "ticks_executed": 1000,
  "max_abs_delta_usd": 5000.0,
  "max_abs_basis_usd": 2500.0,
  "max_abs_q_tao": 15.0,
  "max_venue_toxicity": 0.3
}
```

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
# Validate a specific file
python3 tools/check_telemetry_contract.py path/to/telemetry.jsonl
python3 tools/check_telemetry_contract.py path/to/mc_runs.jsonl

# Validate all telemetry files in a directory
python3 tools/check_telemetry_contract.py path/to/run_directory/

# Show help
python3 tools/check_telemetry_contract.py --help
```

Exit codes:
- `0` — All records valid (or no telemetry files found)
- `1` — Contract violation (schema error)
- `2` — File not found, CLI usage error, or unmapped file type

---

## Machine-Readable Schemas

The canonical machine-readable schemas are at:

```
schemas/telemetry_schema_v1.json    # Per-tick telemetry
schemas/mc_runs_schema_v1.json      # Monte Carlo runs
```

These JSON files are used by the validator and MUST be kept in sync with this document.

---

## Adding Schema Support for New File Types

To add schema support for a new JSONL file type (e.g., `metrics.jsonl`):

1. Create the schema file: `schemas/metrics_schema_v1.json`
2. Add the mapping to `FILE_SCHEMA_MAP` in `tools/check_telemetry_contract.py`
3. Remove the filename from `UNMAPPED_FILES` in the same file
4. Document the schema in this file
5. Add unit tests in `tests/test_telemetry_contract_gate.py`

---

## Related Documentation

- `docs/SIM_OUTPUT_SCHEMA.md` — Simulation output schema (run_summary.json, metrics.jsonl)
- `docs/PHASE_A_PROMOTION_PIPELINE.md` — Promotion pipeline telemetry requirements
- `docs/EVIDENCE_PACK.md` — Evidence pack specification

---

## Changelog

### v1 (2026-01-06)
- Added `mc_runs.jsonl` schema for Monte Carlo per-run records
- Added multi-schema support with per-file-type validation
- Updated validation tool to support multiple file types
- Added `schema_version` field to `mc_runs.jsonl` (Rust emitter updated)

### v1 (2026-01-05)
- Initial schema version for `telemetry.jsonl`
- Defined required fields: `schema_version`, `t`, `pnl_realised`, `pnl_unrealised`, `pnl_total`, `risk_regime`, `kill_switch`, `kill_reason`, `q_global_tao`, `dollar_delta_usd`, `basis_usd`
- Defined optional fields: `fv_available`, `fair_value`, `sigma_eff`, `healthy_venues_used_count`, `healthy_venues_used`, `basis_gross_usd`
- Established invariants and backwards compatibility rules
