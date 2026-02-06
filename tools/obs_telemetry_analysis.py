#!/usr/bin/env python3
"""
Observability analysis for paraphina shadow telemetry.
Computes metrics from a JSONL telemetry window.
stdlib only — no external dependencies.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VENUE_NAMES = ["extended", "hyperliquid", "aster", "lighter", "paradex"]


def safe_float(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v):
        return float(v)
    return None


def safe_int(v: Any) -> int | None:
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer() and not math.isnan(v):
        return int(v)
    return None


def ts_to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def percentile(data: list[float], pct: float) -> float:
    if not data:
        return float("nan")
    k = (len(data) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] * (c - k) + data[c] * (k - f)


def load_records(path: Path, max_lines: int = 0) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    records.append(rec)
            except json.JSONDecodeError:
                pass
    return records


def compute_core_metrics(records: list[dict]) -> dict:
    total = len(records)
    if total == 0:
        return {"total_lines": 0}

    ticks = [safe_int(r.get("t")) for r in records]
    ticks_valid = [t for t in ticks if t is not None]

    kf_ts = [safe_int(r.get("kf_last_update_ms")) for r in records]
    kf_valid = [t for t in kf_ts if t is not None]

    first_ts = kf_valid[0] if kf_valid else None
    last_ts = kf_valid[-1] if kf_valid else None
    duration_sec = (last_ts - first_ts) / 1000.0 if first_ts and last_ts else 0
    lines_per_sec = total / duration_sec if duration_sec > 0 else 0

    # Monotonicity check
    non_monotonic = 0
    if kf_valid:
        for i in range(1, len(kf_valid)):
            if kf_valid[i] < kf_valid[i - 1]:
                non_monotonic += 1

    # Gap detection (>2s between consecutive kf_last_update_ms)
    gaps = []
    if kf_valid:
        for i in range(1, len(kf_valid)):
            delta = kf_valid[i] - kf_valid[i - 1]
            if delta > 2000:
                gaps.append({
                    "tick_index": i,
                    "tick_a": ticks[i - 1],
                    "tick_b": ticks[i],
                    "delta_ms": delta,
                    "ts_a": ts_to_utc(kf_valid[i - 1]),
                    "ts_b": ts_to_utc(kf_valid[i]),
                })

    # Inter-record deltas
    deltas_ms = []
    if kf_valid:
        for i in range(1, len(kf_valid)):
            deltas_ms.append(kf_valid[i] - kf_valid[i - 1])

    deltas_sorted = sorted(deltas_ms) if deltas_ms else []
    delta_stats = {}
    if deltas_sorted:
        delta_stats = {
            "p50": percentile(deltas_sorted, 50),
            "p95": percentile(deltas_sorted, 95),
            "p99": percentile(deltas_sorted, 99),
            "max": deltas_sorted[-1],
            "min": deltas_sorted[0],
            "mean": statistics.mean(deltas_sorted),
        }

    return {
        "total_lines": total,
        "first_tick": ticks_valid[0] if ticks_valid else None,
        "last_tick": ticks_valid[-1] if ticks_valid else None,
        "first_ts_utc": ts_to_utc(first_ts) if first_ts else None,
        "last_ts_utc": ts_to_utc(last_ts) if last_ts else None,
        "first_ts_ms": first_ts,
        "last_ts_ms": last_ts,
        "duration_sec": round(duration_sec, 2),
        "lines_per_sec": round(lines_per_sec, 4),
        "monotonic_violations": non_monotonic,
        "gaps_gt_2s": len(gaps),
        "gap_details": gaps[:20],
        "inter_record_delta_stats_ms": delta_stats,
    }


def compute_venue_metrics(records: list[dict], num_venues: int = 5) -> dict:
    venue_data: dict[int, dict] = {i: {
        "status_dist": Counter(),
        "funding_status_dist": Counter(),
        "funding_rate_8h_values": [],
        "funding_age_ms_values": [],
        "funding_source_dist": Counter(),
        "funding_settlement_dist": Counter(),
        "age_ms_values": [],
        "mid_values": [],
        "spread_values": [],
        "stale_ticks": 0,
        "healthy_ticks": 0,
        "total_ticks": 0,
        "funding_status_transitions": [],
        "status_transitions": [],
        "depth_usd_values": [],
        "funding_interval_sec_values": [],
        "next_funding_ms_values": [],
    } for i in range(num_venues)}

    prev_funding_status = [None] * num_venues
    prev_venue_status = [None] * num_venues

    for rec_idx, rec in enumerate(records):
        tick = safe_int(rec.get("t"))
        venue_status = rec.get("venue_status", [])
        funding_status = rec.get("venue_funding_status", [])
        funding_rate = rec.get("venue_funding_rate_8h", [])
        funding_age = rec.get("venue_funding_age_ms", [])
        funding_source = rec.get("venue_funding_source", [])
        funding_settlement = rec.get("venue_funding_settlement_price_kind", [])
        venue_age = rec.get("venue_age_ms", [])
        venue_mid = rec.get("venue_mid_usd", [])
        venue_spread = rec.get("venue_spread_usd", [])
        venue_depth = rec.get("venue_depth_near_mid_usd", [])
        funding_interval = rec.get("venue_funding_interval_sec", [])
        next_funding = rec.get("venue_next_funding_ms", [])

        for i in range(num_venues):
            vd = venue_data[i]
            vd["total_ticks"] += 1

            # Status
            vs = venue_status[i] if isinstance(venue_status, list) and i < len(venue_status) else None
            if vs:
                vd["status_dist"][vs] += 1
                if vs == "Stale":
                    vd["stale_ticks"] += 1
                elif vs == "Healthy":
                    vd["healthy_ticks"] += 1
                if prev_venue_status[i] is not None and vs != prev_venue_status[i]:
                    vd["status_transitions"].append({
                        "tick": tick,
                        "rec_idx": rec_idx,
                        "from": prev_venue_status[i],
                        "to": vs,
                    })
                prev_venue_status[i] = vs

            # Funding status
            fs = funding_status[i] if isinstance(funding_status, list) and i < len(funding_status) else None
            if fs:
                vd["funding_status_dist"][fs] += 1
                if prev_funding_status[i] is not None and fs != prev_funding_status[i]:
                    vd["funding_status_transitions"].append({
                        "tick": tick,
                        "rec_idx": rec_idx,
                        "from": prev_funding_status[i],
                        "to": fs,
                    })
                prev_funding_status[i] = fs

            # Funding rate
            fr = None
            if isinstance(funding_rate, list) and i < len(funding_rate):
                fr = safe_float(funding_rate[i])
            if fr is not None:
                vd["funding_rate_8h_values"].append(fr)

            # Funding age
            fa = None
            if isinstance(funding_age, list) and i < len(funding_age):
                fa = safe_float(funding_age[i])
            if fa is not None:
                vd["funding_age_ms_values"].append(fa)

            # Funding source
            fsr = None
            if isinstance(funding_source, list) and i < len(funding_source):
                fsr = funding_source[i]
            if fsr:
                vd["funding_source_dist"][fsr] += 1

            # Settlement
            fse = None
            if isinstance(funding_settlement, list) and i < len(funding_settlement):
                fse = funding_settlement[i]
            if fse:
                vd["funding_settlement_dist"][fse] += 1

            # Age
            va = None
            if isinstance(venue_age, list) and i < len(venue_age):
                va = safe_float(venue_age[i])
            if va is not None:
                vd["age_ms_values"].append(va)

            # Mid / Spread
            vm = None
            if isinstance(venue_mid, list) and i < len(venue_mid):
                vm = safe_float(venue_mid[i])
            if vm is not None:
                vd["mid_values"].append(vm)

            vsp = None
            if isinstance(venue_spread, list) and i < len(venue_spread):
                vsp = safe_float(venue_spread[i])
            if vsp is not None:
                vd["spread_values"].append(vsp)

            # Depth
            vdp = None
            if isinstance(venue_depth, list) and i < len(venue_depth):
                vdp = safe_float(venue_depth[i])
            if vdp is not None:
                vd["depth_usd_values"].append(vdp)

            # Funding interval
            fi = None
            if isinstance(funding_interval, list) and i < len(funding_interval):
                fi = safe_float(funding_interval[i])
            if fi is not None:
                vd["funding_interval_sec_values"].append(fi)

    # Summarize
    result = {}
    for i in range(num_venues):
        vd = venue_data[i]
        name = VENUE_NAMES[i] if i < len(VENUE_NAMES) else f"venue_{i}"

        age_sorted = sorted(vd["age_ms_values"])
        fage_sorted = sorted(vd["funding_age_ms_values"])
        spread_sorted = sorted(vd["spread_values"])
        depth_sorted = sorted(vd["depth_usd_values"])

        result[name] = {
            "total_ticks": vd["total_ticks"],
            "venue_status_distribution": dict(vd["status_dist"]),
            "healthy_pct": round(100 * vd["healthy_ticks"] / max(1, vd["total_ticks"]), 2),
            "stale_pct": round(100 * vd["stale_ticks"] / max(1, vd["total_ticks"]), 2),
            "status_transitions": len(vd["status_transitions"]),
            "status_transition_details": vd["status_transitions"][:30],
            "funding_status_distribution": dict(vd["funding_status_dist"]),
            "funding_status_transitions": len(vd["funding_status_transitions"]),
            "funding_transition_details": vd["funding_status_transitions"][:30],
            "funding_rate_8h": {
                "count": len(vd["funding_rate_8h_values"]),
                "mean": round(statistics.mean(vd["funding_rate_8h_values"]), 8) if vd["funding_rate_8h_values"] else None,
                "min": round(min(vd["funding_rate_8h_values"]), 8) if vd["funding_rate_8h_values"] else None,
                "max": round(max(vd["funding_rate_8h_values"]), 8) if vd["funding_rate_8h_values"] else None,
            },
            "funding_age_ms": {
                "count": len(fage_sorted),
                "p50": round(percentile(fage_sorted, 50), 1) if fage_sorted else None,
                "p95": round(percentile(fage_sorted, 95), 1) if fage_sorted else None,
                "max": round(fage_sorted[-1], 1) if fage_sorted else None,
                "mean": round(statistics.mean(fage_sorted), 1) if fage_sorted else None,
            },
            "funding_source_distribution": dict(vd["funding_source_dist"]),
            "funding_settlement_distribution": dict(vd["funding_settlement_dist"]),
            "venue_age_ms": {
                "p50": round(percentile(age_sorted, 50), 1) if age_sorted else None,
                "p95": round(percentile(age_sorted, 95), 1) if age_sorted else None,
                "max": round(age_sorted[-1], 1) if age_sorted else None,
                "mean": round(statistics.mean(age_sorted), 1) if age_sorted else None,
            },
            "spread_usd": {
                "p50": round(percentile(spread_sorted, 50), 4) if spread_sorted else None,
                "p95": round(percentile(spread_sorted, 95), 4) if spread_sorted else None,
                "max": round(spread_sorted[-1], 4) if spread_sorted else None,
            },
            "depth_near_mid_usd": {
                "p50": round(percentile(depth_sorted, 50), 2) if depth_sorted else None,
                "p95": round(percentile(depth_sorted, 95), 2) if depth_sorted else None,
                "min": round(depth_sorted[0], 2) if depth_sorted else None,
            },
        }

    return result


def detect_anomalies(records: list[dict], venue_metrics: dict) -> list[dict]:
    anomalies = []

    # 1. Funding status flaps per venue
    for vname, vm in venue_metrics.items():
        ft = vm.get("funding_status_transitions", 0)
        if ft > 5:
            anomalies.append({
                "severity": 3,
                "type": "funding_status_flap",
                "venue": vname,
                "count": ft,
                "description": f"{vname}: {ft} funding status transitions detected",
                "details": vm.get("funding_transition_details", [])[:5],
            })

    # 2. Venue status flaps
    for vname, vm in venue_metrics.items():
        st = vm.get("status_transitions", 0)
        if st > 5:
            anomalies.append({
                "severity": 2,
                "type": "venue_status_flap",
                "venue": vname,
                "count": st,
                "description": f"{vname}: {st} venue status transitions (reconnect/stale flaps)",
                "details": vm.get("status_transition_details", [])[:5],
            })

    # 3. High funding age spikes
    for vname, vm in venue_metrics.items():
        fage = vm.get("funding_age_ms", {})
        if fage.get("max") and fage["max"] > 60000:
            anomalies.append({
                "severity": 3,
                "type": "funding_age_spike",
                "venue": vname,
                "max_ms": fage["max"],
                "p95_ms": fage.get("p95"),
                "description": f"{vname}: funding_age_ms max={fage['max']}ms (p95={fage.get('p95')}ms)",
            })

    # 4. High stale percentage
    for vname, vm in venue_metrics.items():
        sp = vm.get("stale_pct", 0)
        if sp > 1.0:
            anomalies.append({
                "severity": 4,
                "type": "high_stale_pct",
                "venue": vname,
                "stale_pct": sp,
                "description": f"{vname}: {sp}% of ticks in Stale status",
            })

    # 5. Funding settlement Unknown
    for vname, vm in venue_metrics.items():
        sd = vm.get("funding_settlement_distribution", {})
        unknown_count = sd.get("Unknown", 0)
        total = sum(sd.values())
        if total > 0 and unknown_count > 0:
            pct = round(100 * unknown_count / total, 2)
            if pct > 50:
                anomalies.append({
                    "severity": 2,
                    "type": "funding_settlement_unknown",
                    "venue": vname,
                    "unknown_pct": pct,
                    "description": f"{vname}: {pct}% funding settlement price kind = Unknown",
                })

    # 6. Funding rate nulls
    for vname, vm in venue_metrics.items():
        fr = vm.get("funding_rate_8h", {})
        total = vm.get("total_ticks", 0)
        count = fr.get("count", 0)
        if total > 0 and count < total * 0.9:
            anomalies.append({
                "severity": 3,
                "type": "funding_rate_null_freq",
                "venue": vname,
                "null_pct": round(100 * (total - count) / total, 2),
                "description": f"{vname}: {round(100 * (total - count) / total, 2)}% funding_rate_8h null/missing",
            })

    # 7. Wide spreads
    for vname, vm in venue_metrics.items():
        sp = vm.get("spread_usd", {})
        if sp.get("p95") and sp["p95"] > 1.0:
            anomalies.append({
                "severity": 2,
                "type": "wide_spread",
                "venue": vname,
                "p95_usd": sp["p95"],
                "max_usd": sp.get("max"),
                "description": f"{vname}: p95 spread={sp['p95']}USD (max={sp.get('max')}USD)",
            })

    # 8. Low depth
    for vname, vm in venue_metrics.items():
        dp = vm.get("depth_near_mid_usd", {})
        if dp.get("min") is not None and dp["min"] < 1000:
            anomalies.append({
                "severity": 2,
                "type": "low_depth",
                "venue": vname,
                "min_usd": dp["min"],
                "p50_usd": dp.get("p50"),
                "description": f"{vname}: min depth_near_mid={dp['min']}USD",
            })

    # 9. Telemetry write pauses from gap data (computed externally)
    # Will be populated by caller

    # 10. fv_available false
    fv_false_count = sum(1 for r in records if not r.get("fv_available", True))
    if fv_false_count > 0:
        pct = round(100 * fv_false_count / max(1, len(records)), 2)
        anomalies.append({
            "severity": 4 if pct > 5 else 2,
            "type": "fv_unavailable",
            "count": fv_false_count,
            "pct": pct,
            "description": f"fv_available=false in {fv_false_count} ticks ({pct}%)",
        })

    # 11. Healthy venues count drops
    healthy_counts = [safe_int(r.get("healthy_venues_used_count")) for r in records]
    healthy_valid = [c for c in healthy_counts if c is not None]
    if healthy_valid:
        min_healthy = min(healthy_valid)
        low_count = sum(1 for c in healthy_valid if c < 3)
        if min_healthy < 3 or low_count > 10:
            anomalies.append({
                "severity": 3,
                "type": "low_healthy_venue_count",
                "min_count": min_healthy,
                "ticks_below_3": low_count,
                "description": f"healthy_venues_used_count dropped to {min_healthy} (below 3 in {low_count} ticks)",
            })

    # 12. Kill switch activations
    kill_count = sum(1 for r in records if r.get("kill_switch"))
    if kill_count > 0:
        anomalies.append({
            "severity": 5,
            "type": "kill_switch_activated",
            "count": kill_count,
            "description": f"kill_switch=true in {kill_count} ticks",
        })

    # Sort by severity descending
    anomalies.sort(key=lambda a: -a.get("severity", 0))
    return anomalies


def compute_funding_interval_analysis(records: list[dict], num_venues: int = 5) -> dict:
    """Analyze funding interval null patterns."""
    result = {}
    for i in range(num_venues):
        name = VENUE_NAMES[i] if i < len(VENUE_NAMES) else f"venue_{i}"
        null_count = 0
        non_null_count = 0
        values = Counter()
        for rec in records:
            fi = rec.get("venue_funding_interval_sec", [])
            if isinstance(fi, list) and i < len(fi):
                v = fi[i]
                if v is None:
                    null_count += 1
                else:
                    non_null_count += 1
                    values[v] += 1
        result[name] = {
            "null_count": null_count,
            "non_null_count": non_null_count,
            "value_distribution": dict(values),
        }
    return result


def analyze(path: Path, max_lines: int = 0) -> dict:
    print(f"Loading records from {path}...", file=sys.stderr)
    records = load_records(path, max_lines)
    if not records:
        print("ERROR: No records found", file=sys.stderr)
        return {"error": "no records"}

    print(f"Loaded {len(records)} records. Computing metrics...", file=sys.stderr)

    core = compute_core_metrics(records)
    venues = compute_venue_metrics(records)
    anomalies = detect_anomalies(records, venues)
    funding_intervals = compute_funding_interval_analysis(records)

    return {
        "core": core,
        "venues": venues,
        "anomalies": anomalies,
        "funding_intervals": funding_intervals,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze paraphina telemetry window")
    parser.add_argument("--input", required=True, help="Path to telemetry JSONL")
    parser.add_argument("--max-lines", type=int, default=0, help="Max lines to process (0=all)")
    parser.add_argument("--output", default="-", help="Output JSON path (- for stdout)")
    args = parser.parse_args()

    result = analyze(Path(args.input), args.max_lines)

    if args.output == "-":
        json.dump(result, sys.stdout, indent=2, default=str)
        print()
    else:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Wrote results to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
