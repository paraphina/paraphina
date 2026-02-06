#!/usr/bin/env python3
"""
obs_capture_window.py — Capture exactly N telemetry ticks from a live JSONL file.

Seeks to EOF, then captures newly appended lines. Writes rolling progress
summaries every --progress-every ticks.

stdlib only. No external dependencies.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def safe_float(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if not math.isnan(v):
            return float(v)
    return None


def safe_int(v: Any) -> int | None:
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer() and not math.isnan(v):
        return int(v)
    return None


def percentile_approx(values: list[float], pct: float) -> float:
    """Approximate percentile from an unsorted list (sorts internally)."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    f = math.floor(k)
    c = min(math.ceil(k), len(s) - 1)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


VENUE_NAMES = ["extended", "hyperliquid", "aster", "lighter", "paradex"]


# ---------------------------------------------------------------------------
# progress summary builder
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Accumulates per-venue stats for a rolling window."""

    def __init__(self, num_venues: int = 5):
        self.num_venues = num_venues
        self.venue_status_counts: dict[int, Counter] = {i: Counter() for i in range(num_venues)}
        self.venue_status_prev: dict[int, str | None] = {i: None for i in range(num_venues)}
        self.venue_status_flips: dict[int, int] = {i: 0 for i in range(num_venues)}
        self.venue_age_values: dict[int, list[float]] = {i: [] for i in range(num_venues)}
        self.funding_status_counts: dict[int, Counter] = {i: Counter() for i in range(num_venues)}
        self.funding_rate_nonnull: dict[int, int] = {i: 0 for i in range(num_venues)}
        self.ticks = 0

    def record(self, rec: dict) -> None:
        self.ticks += 1
        venue_status = rec.get("venue_status", [])
        venue_age = rec.get("venue_age_ms", [])
        funding_status = rec.get("venue_funding_status", [])
        funding_rate = rec.get("venue_funding_rate_8h", [])

        for i in range(self.num_venues):
            # venue status
            vs = venue_status[i] if isinstance(venue_status, list) and i < len(venue_status) else None
            if isinstance(vs, str):
                self.venue_status_counts[i][vs] += 1
                prev = self.venue_status_prev[i]
                if prev is not None and prev != vs:
                    self.venue_status_flips[i] += 1
                self.venue_status_prev[i] = vs

            # venue age
            if isinstance(venue_age, list) and i < len(venue_age):
                va = safe_float(venue_age[i])
                if va is not None:
                    self.venue_age_values[i].append(va)

            # funding status
            fs = funding_status[i] if isinstance(funding_status, list) and i < len(funding_status) else None
            if isinstance(fs, str):
                self.funding_status_counts[i][fs] += 1

            # funding rate non-null
            if isinstance(funding_rate, list) and i < len(funding_rate):
                fr = safe_float(funding_rate[i])
                if fr is not None:
                    self.funding_rate_nonnull[i] += 1

    def summary(self, elapsed_s: float) -> dict:
        per_venue_status = {}
        per_venue_flips = {}
        per_venue_age_p50 = {}
        per_venue_age_p95 = {}
        funding_sc = {}
        funding_rate_nn = {}

        for i in range(self.num_venues):
            name = VENUE_NAMES[i] if i < len(VENUE_NAMES) else f"venue_{i}"
            per_venue_status[name] = dict(self.venue_status_counts[i])
            per_venue_flips[name] = self.venue_status_flips[i]
            vals = self.venue_age_values[i]
            per_venue_age_p50[name] = round(percentile_approx(vals, 50), 1) if vals else None
            per_venue_age_p95[name] = round(percentile_approx(vals, 95), 1) if vals else None
            funding_sc[name] = dict(self.funding_status_counts[i])
            funding_rate_nn[name] = self.funding_rate_nonnull[i]

        return {
            "ticks_captured": self.ticks,
            "elapsed_s": round(elapsed_s, 2),
            "ticks_per_s": round(self.ticks / max(elapsed_s, 0.001), 4),
            "per_venue_status_counts": per_venue_status,
            "per_venue_status_flips": per_venue_flips,
            "per_venue_age_p50_approx": per_venue_age_p50,
            "per_venue_age_p95_approx": per_venue_age_p95,
            "funding_status_counts": funding_sc,
            "funding_rate_nonnull_counts": funding_rate_nn,
        }


# ---------------------------------------------------------------------------
# main capture loop
# ---------------------------------------------------------------------------

def get_inode(path: str) -> int:
    try:
        return os.stat(path).st_ino
    except OSError:
        return -1


def capture(source: str, outdir: str, n: int, progress_every: int) -> bool:
    source_path = Path(source)
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    window_path = outdir_path / f"telemetry_window_{n}.jsonl"
    progress_path = outdir_path / "progress.jsonl"

    # Seek to current EOF
    initial_size = source_path.stat().st_size
    initial_inode = get_inode(source)
    print(f"[capture] source={source}", file=sys.stderr)
    print(f"[capture] initial_size={initial_size} inode={initial_inode}", file=sys.stderr)
    print(f"[capture] target={n} ticks, progress every {progress_every}", file=sys.stderr)
    print(f"[capture] output={window_path}", file=sys.stderr)
    print(f"[capture] start={datetime.now(timezone.utc).isoformat()}", file=sys.stderr)

    tracker = ProgressTracker()
    ticks_written = 0
    start_time = time.monotonic()
    read_offset = initial_size
    partial_line = ""

    with open(window_path, "w", encoding="utf-8") as wf, \
         open(progress_path, "a", encoding="utf-8") as pf:

        while ticks_written < n:
            # Check for rotation/truncation
            current_inode = get_inode(source)
            try:
                current_size = source_path.stat().st_size
            except OSError:
                current_size = 0

            if current_inode != initial_inode:
                fail_rec = {"capture_failed": True, "reason": "inode_changed",
                            "ticks_captured": ticks_written,
                            "timestamp": datetime.now(timezone.utc).isoformat()}
                pf.write(json.dumps(fail_rec) + "\n")
                pf.flush()
                print(f"[capture] FAIL: inode changed ({initial_inode} -> {current_inode})", file=sys.stderr)
                return False

            if current_size < read_offset:
                fail_rec = {"capture_failed": True, "reason": "file_truncated",
                            "ticks_captured": ticks_written,
                            "timestamp": datetime.now(timezone.utc).isoformat()}
                pf.write(json.dumps(fail_rec) + "\n")
                pf.flush()
                print(f"[capture] FAIL: file truncated ({read_offset} -> {current_size})", file=sys.stderr)
                return False

            # Read new data
            if current_size <= read_offset:
                time.sleep(0.5)
                continue

            try:
                with open(source, "r", encoding="utf-8") as sf:
                    sf.seek(read_offset)
                    chunk = sf.read(current_size - read_offset)
                    read_offset = sf.tell()
            except OSError:
                time.sleep(1)
                continue

            # Process lines
            data = partial_line + chunk
            lines = data.split("\n")
            # Last element may be partial if no trailing newline
            partial_line = lines[-1]
            complete_lines = lines[:-1]

            for line in complete_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue

                wf.write(json.dumps(rec, separators=(",", ":")) + "\n")
                ticks_written += 1
                tracker.record(rec)

                # Progress
                if ticks_written % progress_every == 0:
                    elapsed = time.monotonic() - start_time
                    summary = tracker.summary(elapsed)
                    summary["timestamp"] = datetime.now(timezone.utc).isoformat()
                    pf.write(json.dumps(summary) + "\n")
                    pf.flush()
                    rate = summary["ticks_per_s"]
                    remaining = (n - ticks_written) / max(rate, 0.01)
                    print(
                        f"[capture] {ticks_written}/{n} ticks | "
                        f"{rate:.2f} t/s | "
                        f"elapsed={elapsed:.0f}s | "
                        f"eta={remaining:.0f}s",
                        file=sys.stderr,
                    )

                if ticks_written >= n:
                    break

            wf.flush()

    elapsed = time.monotonic() - start_time
    end_ts = datetime.now(timezone.utc).isoformat()

    # Final progress
    summary = tracker.summary(elapsed)
    summary["timestamp"] = end_ts
    summary["capture_complete"] = True
    with open(progress_path, "a", encoding="utf-8") as pf:
        pf.write(json.dumps(summary) + "\n")

    print(f"[capture] COMPLETE: {ticks_written} ticks in {elapsed:.1f}s", file=sys.stderr)
    print(f"[capture] end={end_ts}", file=sys.stderr)
    return True


def main():
    parser = argparse.ArgumentParser(description="Capture N telemetry ticks from live JSONL")
    parser.add_argument("--source", required=True, help="Path to active telemetry JSONL")
    parser.add_argument("--outdir", required=True, help="Output directory for run artifacts")
    parser.add_argument("--n", type=int, default=25000, help="Number of ticks to capture")
    parser.add_argument("--progress-every", type=int, default=250, help="Progress interval")
    args = parser.parse_args()

    ok = capture(args.source, args.outdir, args.n, args.progress_every)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
