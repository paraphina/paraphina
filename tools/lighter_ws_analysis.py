#!/usr/bin/env python3
"""Lighter WS stability analysis — computes metrics from a telemetry window.

Usage:
    python3 tools/lighter_ws_analysis.py <telemetry.jsonl> [--venue-index 3] [--json-out metrics.json]
"""

import argparse
import json
import sys
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Lighter WS stability analysis")
    parser.add_argument("telemetry", help="Path to telemetry JSONL file")
    parser.add_argument("--venue-index", type=int, default=3, help="Venue index for Lighter (default: 3)")
    parser.add_argument("--json-out", help="Optional path to write JSON metrics")
    args = parser.parse_args()

    vi = args.venue_index
    ticks = []
    for line_no, line in enumerate(open(args.telemetry), 1):
        line = line.strip()
        if not line:
            continue
        try:
            ticks.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not ticks:
        print("ERROR: No ticks found", file=sys.stderr)
        sys.exit(1)

    total = len(ticks)
    print(f"Total ticks: {total}")

    # --- Venue status distribution ---
    statuses = Counter()
    ages = []
    depths = []
    spreads = []
    mids = []
    transitions = []
    prev_status = None
    disabled_runs = []
    current_disabled_run = 0

    for i, t in enumerate(ticks):
        vs = t.get("venue_status", [])
        va = t.get("venue_age_ms", [])
        vd = t.get("venue_depth_near_mid_usd", [])
        vsp = t.get("venue_spread_usd", [])
        vm = t.get("venue_mid_usd", [])

        st = vs[vi] if len(vs) > vi else "N/A"
        age = va[vi] if len(va) > vi else -1
        depth = vd[vi] if len(vd) > vi else 0.0
        spread = vsp[vi] if len(vsp) > vi else 0.0
        mid = vm[vi] if len(vm) > vi else 0.0

        statuses[st] += 1
        if age >= 0:
            ages.append(age)
        depths.append(depth)
        if spread > 0:
            spreads.append(spread)
        if mid > 0:
            mids.append(mid)

        if prev_status is not None and st != prev_status:
            transitions.append({
                "tick": i,
                "from": prev_status,
                "to": st,
                "age_ms": age,
                "depth_usd": depth,
                "spread_usd": spread,
            })
        prev_status = st

        if st == "Disabled":
            current_disabled_run += 1
        else:
            if current_disabled_run > 0:
                disabled_runs.append(current_disabled_run)
            current_disabled_run = 0
    if current_disabled_run > 0:
        disabled_runs.append(current_disabled_run)

    # --- Percentiles ---
    def pct(data, p):
        if not data:
            return None
        data_s = sorted(data)
        idx = int(len(data_s) * p / 100.0)
        idx = min(idx, len(data_s) - 1)
        return data_s[idx]

    # --- Depth zero analysis ---
    depth_zero_count = sum(1 for d in depths if d <= 0.0)
    depth_zero_pct = 100.0 * depth_zero_count / total if total > 0 else 0.0
    # Depth zero outside first 10 ticks
    depth_zero_steady = sum(1 for d in depths[10:] if d <= 0.0)

    # --- Status percentages ---
    disabled_count = statuses.get("Disabled", 0)
    stale_count = statuses.get("Stale", 0)
    healthy_count = statuses.get("Healthy", 0)
    disabled_pct = 100.0 * disabled_count / total if total > 0 else 0.0
    stale_pct = 100.0 * stale_count / total if total > 0 else 0.0
    healthy_pct = 100.0 * healthy_count / total if total > 0 else 0.0

    # --- Transition rate ---
    total_transitions = len(transitions)
    to_disabled = sum(1 for t in transitions if t["to"] == "Disabled")
    from_disabled = sum(1 for t in transitions if t["from"] == "Disabled")
    transitions_per_hour = total_transitions * (3600.0 / total) if total > 0 else 0.0

    # --- Other venues check ---
    venue_names = ["extended", "hyperliquid", "aster", "lighter", "paradex"]
    other_venues = {}
    for ovi, vname in enumerate(venue_names):
        if ovi == vi:
            continue
        ov_statuses = Counter()
        for t in ticks:
            vs = t.get("venue_status", [])
            if len(vs) > ovi:
                ov_statuses[vs[ovi]] += 1
        ov_total = sum(ov_statuses.values())
        other_venues[vname] = {
            "healthy_pct": 100.0 * ov_statuses.get("Healthy", 0) / ov_total if ov_total > 0 else 0.0,
            "disabled_pct": 100.0 * ov_statuses.get("Disabled", 0) / ov_total if ov_total > 0 else 0.0,
            "stale_pct": 100.0 * ov_statuses.get("Stale", 0) / ov_total if ov_total > 0 else 0.0,
        }

    metrics = {
        "total_ticks": total,
        "lighter_status": {
            "healthy_pct": round(healthy_pct, 4),
            "disabled_pct": round(disabled_pct, 4),
            "stale_pct": round(stale_pct, 4),
            "healthy_count": healthy_count,
            "disabled_count": disabled_count,
            "stale_count": stale_count,
        },
        "lighter_transitions": {
            "total": total_transitions,
            "to_disabled": to_disabled,
            "from_disabled": from_disabled,
            "transitions_per_hour": round(transitions_per_hour, 2),
        },
        "lighter_age_ms": {
            "p50": pct(ages, 50),
            "p95": pct(ages, 95),
            "p99": pct(ages, 99),
            "max": max(ages) if ages else None,
            "min": min(ages) if ages else None,
        },
        "lighter_depth_usd": {
            "p50": round(pct(depths, 50), 2) if depths else None,
            "p95": round(pct(depths, 95), 2) if depths else None,
            "zero_count": depth_zero_count,
            "zero_pct": round(depth_zero_pct, 4),
            "zero_steady_state": depth_zero_steady,
        },
        "lighter_spread_usd": {
            "p50": round(pct(spreads, 50), 4) if spreads else None,
            "p95": round(pct(spreads, 95), 4) if spreads else None,
            "max": round(max(spreads), 4) if spreads else None,
        },
        "disabled_runs": {
            "count": len(disabled_runs),
            "avg_ticks": round(sum(disabled_runs) / len(disabled_runs), 2) if disabled_runs else 0,
            "max_ticks": max(disabled_runs) if disabled_runs else 0,
        },
        "other_venues": other_venues,
    }

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"LIGHTER WS STABILITY METRICS")
    print(f"{'='*60}")
    print(f"Status: Healthy={healthy_pct:.2f}%  Disabled={disabled_pct:.2f}%  Stale={stale_pct:.2f}%")
    print(f"Transitions: {total_transitions} total ({transitions_per_hour:.1f}/hr)  →Disabled={to_disabled}")
    print(f"Age (ms): p50={pct(ages,50)}  p95={pct(ages,95)}  p99={pct(ages,99)}  max={max(ages) if ages else 'N/A'}")
    print(f"Depth (USD): p50={pct(depths,50):.0f}  zero_count={depth_zero_count} ({depth_zero_pct:.2f}%)  zero_steady={depth_zero_steady}")
    print(f"Spread (USD): p50={pct(spreads,50):.4f}  p95={pct(spreads,95):.4f}  max={max(spreads):.4f}" if spreads else "Spread: N/A")
    print(f"Disabled runs: {len(disabled_runs)} runs, avg={metrics['disabled_runs']['avg_ticks']:.1f} ticks, max={metrics['disabled_runs']['max_ticks']} ticks")
    print(f"\nOther venues:")
    for vname, vstats in other_venues.items():
        print(f"  {vname}: Healthy={vstats['healthy_pct']:.1f}%  Disabled={vstats['disabled_pct']:.1f}%  Stale={vstats['stale_pct']:.1f}%")

    # --- Sample transitions ---
    if transitions:
        print(f"\nFirst 10 transitions:")
        for t in transitions[:10]:
            print(f"  tick {t['tick']}: {t['from']}→{t['to']} age={t['age_ms']}ms depth=${t['depth_usd']:.0f} spread=${t['spread_usd']:.4f}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics written to {args.json_out}")


if __name__ == "__main__":
    main()
