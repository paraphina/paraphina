#!/usr/bin/env python3
"""
telemetry_analyzer.py — Comprehensive 15-dimension telemetry analysis for paraphina shadow runs.

Streaming single-pass architecture: reads JSONL line-by-line, accumulates statistics
via online algorithms, and outputs a structured text report.

Usage:
    python3 tools/telemetry_analyzer.py --telemetry /tmp/shadow_eth_post_fix/telemetry.jsonl
    python3 tools/telemetry_analyzer.py --telemetry /path/to/telemetry.jsonl --max-ticks 10000
    python3 tools/telemetry_analyzer.py --telemetry /path/to/telemetry.jsonl --checkpoint-json out/cp_10k.json

stdlib only — no external dependencies.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VENUE_NAMES = ["extended", "hyperliquid", "aster", "lighter", "paradex"]
NUM_VENUES = 5
WINDOW_SIZE = 10_000  # ticks per trend window

# Staleness thresholds (from config)
STALE_MS = {
    "extended": 1000,
    "hyperliquid": 2000,
    "aster": 1000,
    "lighter": 1000,
    "paradex": 3000,
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def safe_float(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    return None


def safe_int(v: Any) -> int | None:
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    if isinstance(v, float) and v.is_integer() and not (math.isnan(v) or math.isinf(v)):
        return int(v)
    return None


def pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{100.0 * num / denom:.2f}%"


def fmt_f(v: float | None, decimals: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{decimals}f}"


def fmt_i(v: int | None) -> str:
    if v is None:
        return "n/a"
    return str(v)


def ts_str(ms: int | None) -> str:
    if ms is None:
        return "n/a"
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Online statistics accumulator
# ---------------------------------------------------------------------------

class OnlineStats:
    """Welford's online algorithm for mean/variance + min/max/count + stored values for percentiles."""

    __slots__ = ("n", "mean", "M2", "_min", "_max", "_vals", "_store_vals")

    def __init__(self, store_vals: bool = True) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self._min = float("inf")
        self._max = float("-inf")
        self._store_vals = store_vals
        self._vals: list[float] = [] if store_vals else []

    def push(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x
        if self._store_vals:
            self._vals.append(x)

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def percentile(self, p: float) -> float:
        if not self._vals:
            return float("nan")
        self._vals.sort()  # lazy sort
        k = (len(self._vals) - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return self._vals[int(k)]
        return self._vals[f] * (c - k) + self._vals[c] * (k - f)

    def summary(self) -> dict:
        if self.n == 0:
            return {"count": 0, "mean": "n/a", "p50": "n/a", "p95": "n/a", "p99": "n/a", "min": "n/a", "max": "n/a"}
        return {
            "count": self.n,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self._min, 6),
            "p50": round(self.percentile(50), 6),
            "p95": round(self.percentile(95), 6),
            "p99": round(self.percentile(99), 6),
            "max": round(self._max, 6),
        }

    def summary_line(self, label: str, unit: str = "") -> str:
        s = self.summary()
        if self.n == 0:
            return f"  {label}: (no data)"
        u = f" {unit}" if unit else ""
        return (
            f"  {label}: mean={s['mean']}{u} p50={s['p50']}{u} "
            f"p95={s['p95']}{u} p99={s['p99']}{u} max={s['max']}{u} (n={s['count']})"
        )


# ---------------------------------------------------------------------------
# Window accumulator (for trend analysis)
# ---------------------------------------------------------------------------

class WindowAccumulator:
    """Collects per-window statistics for a single metric."""

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        self.window_size = window_size
        self.windows: list[OnlineStats] = [OnlineStats(store_vals=True)]
        self._count = 0

    def push(self, x: float) -> None:
        if self._count >= self.window_size and self._count % self.window_size == 0:
            self.windows.append(OnlineStats(store_vals=True))
        self.windows[-1].push(x)
        self._count += 1

    @property
    def current_window(self) -> OnlineStats:
        return self.windows[-1]

    def trend_summary(self) -> list[dict]:
        result = []
        for i, w in enumerate(self.windows):
            s = w.summary()
            s["window"] = i
            s["tick_start"] = i * self.window_size
            s["tick_end"] = min((i + 1) * self.window_size - 1, i * self.window_size + w.n - 1)
            result.append(s)
        return result


# ---------------------------------------------------------------------------
# Anomaly buffer
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    tick: int
    category: str
    severity: str  # Critical / Warning / Info
    description: str
    evidence: dict = field(default_factory=dict)


class AnomalyCollector:
    def __init__(self, max_items: int = 5000) -> None:
        self.items: list[Anomaly] = []
        self.max_items = max_items
        self.counts: Counter = Counter()

    def add(self, tick: int, category: str, severity: str, description: str, evidence: dict | None = None) -> None:
        self.counts[category] += 1
        if len(self.items) < self.max_items:
            self.items.append(Anomaly(tick=tick, category=category, severity=severity, description=description, evidence=evidence or {}))

    def by_severity(self, severity: str) -> list[Anomaly]:
        return [a for a in self.items if a.severity == severity]


# ---------------------------------------------------------------------------
# Main Accumulator — all 15 dimensions
# ---------------------------------------------------------------------------

class TelemetryAccumulator:
    def __init__(self) -> None:
        self.tick_count = 0
        self.first_tick: int | None = None
        self.last_tick: int | None = None
        self.first_ts_ms: int | None = None
        self.last_ts_ms: int | None = None

        # --- Dimension 1: Venue Health ---
        self.venue_status_counts: list[Counter] = [Counter() for _ in range(NUM_VENUES)]
        self.venue_age_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_age_windows: list[WindowAccumulator] = [WindowAccumulator() for _ in range(NUM_VENUES)]
        self.venue_status_flips: list[int] = [0] * NUM_VENUES
        self.venue_prev_status: list[str | None] = [None] * NUM_VENUES
        self.venue_consecutive_stale: list[int] = [0] * NUM_VENUES
        self.venue_max_consecutive_stale: list[int] = [0] * NUM_VENUES
        self.venue_stale_runs: list[list[int]] = [[] for _ in range(NUM_VENUES)]  # lengths of stale runs

        # --- Dimension 2: Tick Timing ---
        self.timing_total_us = OnlineStats()
        self.timing_drain_us = OnlineStats()
        self.timing_engine_us = OnlineStats()
        self.timing_submit_us = OnlineStats()
        self.timing_reconcile_us = OnlineStats()
        self.timing_order_pending = OnlineStats()
        self.timing_total_windows = WindowAccumulator()
        self.timing_budget_exceed = Counter()  # keys: ">250us", ">1ms", ">5ms", ">50ms"

        # --- Dimension 3: Market Data Quality ---
        self.venue_mid_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_spread_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_depth_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_mid_delta_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_prev_mid: list[float | None] = [None] * NUM_VENUES
        self.fv_unavailable_count = 0
        self.fv_unavailable_runs: list[int] = []  # lengths of consecutive unavailable runs
        self.fv_unavailable_current_run = 0
        self.mid_jump_count: list[int] = [0] * NUM_VENUES  # >50bps jumps

        # --- Dimension 4: Cross-Venue Coherence ---
        self.cross_venue_dispersion = OnlineStats()
        self.cross_venue_dispersion_windows = WindowAccumulator()
        self.venue_fv_deviation: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]

        # --- Dimension 5: Kalman Filter ---
        self.kf_p_stats = OnlineStats()
        self.kf_p_windows = WindowAccumulator()
        self.kf_xhat_stats = OnlineStats()
        self.fv_stats = OnlineStats()
        self.fv_delta_stats = OnlineStats()
        self.fv_prev: float | None = None
        self.fv_jump_count = 0  # >20bps
        self.healthy_venues_count_stats = OnlineStats()
        self.fv_vs_median_residual = OnlineStats()

        # --- Dimension 6: Markout (shadow: limited data) ---
        self.venue_markout_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_toxicity_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_toxicity_windows: list[WindowAccumulator] = [WindowAccumulator() for _ in range(NUM_VENUES)]
        self.fills_count = 0

        # --- Dimension 7: Quote-Level Edge ---
        self.venue_edge_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_delta_final_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_basis_adj_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_funding_adj_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_inventory_term_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_spread_mult_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_size_mult_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_size_raw_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_size_final_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_margin_cap_binding: list[int] = [0] * NUM_VENUES
        self.venue_liq_factor_binding: list[int] = [0] * NUM_VENUES
        self.venue_any_constraint_binding: list[int] = [0] * NUM_VENUES
        self.venue_quote_count: list[int] = [0] * NUM_VENUES

        # --- Dimension 8: Exits ---
        self.exits_tick_count = 0
        self.exit_edge_stats = OnlineStats()

        # --- Dimension 9: Hedges ---
        self.hedges_tick_count = 0
        self.hedge_delta_stats = OnlineStats()

        # --- Dimension 10: Margin ---
        self.venue_margin_util_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_margin_avail_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_dist_liq_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]

        # --- Dimension 11: Volatility Model ---
        self.fv_short_vol_stats = OnlineStats()
        self.fv_long_vol_stats = OnlineStats()
        self.sigma_eff_stats = OnlineStats()
        self.regime_ratio_stats = OnlineStats()
        self.regime_ratio_windows = WindowAccumulator()
        self.venue_local_vol_short: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_local_vol_long: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        # Realized vol: collect FV returns to compute actual vol
        self.fv_returns: list[float] = []

        # --- Dimension 12: Reconcile Drift ---
        self.reconcile_drift_tick_count = 0
        self.drift_by_venue: list[int] = [0] * NUM_VENUES

        # --- Dimension 13: Risk & Order Flow ---
        self.risk_regime_counts: Counter = Counter()
        self.risk_regime_prev: str | None = None
        self.risk_regime_transitions: int = 0
        self.kill_switch_count = 0
        self.would_send_count_stats = OnlineStats()
        self.would_send_zero_ticks = 0
        self.would_send_consecutive_zero = 0
        self.would_send_max_consecutive_zero = 0
        self.dollar_delta_stats = OnlineStats()
        self.q_global_stats = OnlineStats()
        self.risk_event_counts: Counter = Counter()
        self.order_action_counts: Counter = Counter()
        self.orders_per_venue: list[int] = [0] * NUM_VENUES

        # --- Dimension 14: Anomalies ---
        self.anomalies = AnomalyCollector()

        # --- Dimension 15: Funding ---
        self.venue_funding_rate_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_funding_status_counts: list[Counter] = [Counter() for _ in range(NUM_VENUES)]
        self.venue_funding_age_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]

        # --- PnL ---
        self.pnl_total_stats = OnlineStats()
        self.pnl_realised_stats = OnlineStats()

        # --- Correlated staleness detection ---
        self.multi_stale_ticks = 0  # ticks where 2+ venues are stale simultaneously

    def process(self, rec: dict) -> None:
        tick = safe_int(rec.get("t"))
        if tick is None:
            return

        self.tick_count += 1
        if self.first_tick is None:
            self.first_tick = tick
        self.last_tick = tick

        tg = rec.get("treasury_guidance")
        ts_ms = None
        if isinstance(tg, dict):
            ts_ms = safe_int(tg.get("as_of_ms"))
        if ts_ms is not None:
            if self.first_ts_ms is None:
                self.first_ts_ms = ts_ms
            self.last_ts_ms = ts_ms

        # === Dimension 1: Venue Health ===
        venue_status = rec.get("venue_status", [])
        venue_age = rec.get("venue_age_ms", [])
        stale_this_tick = 0

        for i in range(NUM_VENUES):
            status = venue_status[i] if isinstance(venue_status, list) and i < len(venue_status) else None
            age = venue_age[i] if isinstance(venue_age, list) and i < len(venue_age) else None

            if isinstance(status, str):
                self.venue_status_counts[i][status] += 1
                if self.venue_prev_status[i] is not None and status != self.venue_prev_status[i]:
                    self.venue_status_flips[i] += 1
                self.venue_prev_status[i] = status

                if status != "Healthy":
                    stale_this_tick += 1
                    self.venue_consecutive_stale[i] += 1
                else:
                    if self.venue_consecutive_stale[i] > 0:
                        run_len = self.venue_consecutive_stale[i]
                        if len(self.venue_stale_runs[i]) < 1000:
                            self.venue_stale_runs[i].append(run_len)
                        self.venue_max_consecutive_stale[i] = max(self.venue_max_consecutive_stale[i], run_len)
                    self.venue_consecutive_stale[i] = 0

            age_f = safe_float(age)
            if age_f is not None and age_f >= 0:
                self.venue_age_stats[i].push(age_f)
                self.venue_age_windows[i].push(age_f)

        if stale_this_tick >= 2:
            self.multi_stale_ticks += 1

        # === Dimension 2: Tick Timing ===
        tt = rec.get("tick_timing", {})
        if isinstance(tt, dict):
            for val, acc in [
                (tt.get("total_us"), self.timing_total_us),
                (tt.get("event_drain_us"), self.timing_drain_us),
                (tt.get("engine_us"), self.timing_engine_us),
                (tt.get("submit_us"), self.timing_submit_us),
                (tt.get("reconcile_us"), self.timing_reconcile_us),
            ]:
                f = safe_float(val)
                if f is not None:
                    acc.push(f)

            total_us = safe_float(tt.get("total_us"))
            if total_us is not None:
                self.timing_total_windows.push(total_us)
                if total_us > 250:
                    self.timing_budget_exceed[">250us"] += 1
                if total_us > 1000:
                    self.timing_budget_exceed[">1ms"] += 1
                if total_us > 5000:
                    self.timing_budget_exceed[">5ms"] += 1
                if total_us > 50000:
                    self.timing_budget_exceed[">50ms"] += 1

            pending = safe_float(tt.get("order_tx_pending"))
            if pending is not None:
                self.timing_order_pending.push(pending)

        # === Dimension 3: Market Data Quality ===
        venue_mid = rec.get("venue_mid_usd", [])
        venue_spread = rec.get("venue_spread_usd", [])
        venue_depth = rec.get("venue_depth_near_mid_usd", [])
        fv_available = rec.get("fv_available", False)

        for i in range(NUM_VENUES):
            mid = safe_float(venue_mid[i] if isinstance(venue_mid, list) and i < len(venue_mid) else None)
            spread = safe_float(venue_spread[i] if isinstance(venue_spread, list) and i < len(venue_spread) else None)
            depth = safe_float(venue_depth[i] if isinstance(venue_depth, list) and i < len(venue_depth) else None)

            if mid is not None and mid > 0:
                self.venue_mid_stats[i].push(mid)
                # Tick-to-tick mid delta (bps)
                if self.venue_prev_mid[i] is not None and self.venue_prev_mid[i] > 0:
                    delta_bps = abs(mid - self.venue_prev_mid[i]) / self.venue_prev_mid[i] * 10000
                    self.venue_mid_delta_stats[i].push(delta_bps)
                    if delta_bps > 50:
                        self.mid_jump_count[i] += 1
                        self.anomalies.add(tick, "mid_jump", "Warning",
                            f"{VENUE_NAMES[i]} mid jumped {delta_bps:.1f}bps",
                            {"venue": VENUE_NAMES[i], "delta_bps": round(delta_bps, 2), "mid": mid, "prev_mid": self.venue_prev_mid[i]})
                self.venue_prev_mid[i] = mid

            if spread is not None and spread >= 0:
                self.venue_spread_stats[i].push(spread)
            if depth is not None and depth >= 0:
                self.venue_depth_stats[i].push(depth)

        # FV availability
        if not fv_available:
            self.fv_unavailable_count += 1
            self.fv_unavailable_current_run += 1
        else:
            if self.fv_unavailable_current_run > 0:
                self.fv_unavailable_runs.append(self.fv_unavailable_current_run)
            self.fv_unavailable_current_run = 0

        # === Dimension 4: Cross-Venue Coherence ===
        healthy_mids = []
        for i in range(NUM_VENUES):
            status_i = venue_status[i] if isinstance(venue_status, list) and i < len(venue_status) else None
            mid_i = safe_float(venue_mid[i] if isinstance(venue_mid, list) and i < len(venue_mid) else None)
            if status_i == "Healthy" and mid_i is not None and mid_i > 0:
                healthy_mids.append((i, mid_i))

        if len(healthy_mids) >= 2:
            mids_only = [m for _, m in healthy_mids]
            dispersion = max(mids_only) - min(mids_only)
            self.cross_venue_dispersion.push(dispersion)
            self.cross_venue_dispersion_windows.push(dispersion)

        # Per-venue deviation from FV
        fv = safe_float(rec.get("fair_value"))
        if fv is not None and fv > 0 and fv_available:
            for i, mid_i in healthy_mids:
                dev = abs(mid_i - fv)
                self.venue_fv_deviation[i].push(dev)

        # === Dimension 5: Kalman Filter ===
        kf_p = safe_float(rec.get("kf_p"))
        kf_xhat = safe_float(rec.get("kf_x_hat"))
        if kf_p is not None:
            self.kf_p_stats.push(kf_p)
            self.kf_p_windows.push(kf_p)
        if kf_xhat is not None:
            self.kf_xhat_stats.push(kf_xhat)

        if fv is not None and fv > 0:
            self.fv_stats.push(fv)
            if self.fv_prev is not None and self.fv_prev > 0:
                fv_delta_bps = abs(fv - self.fv_prev) / self.fv_prev * 10000
                self.fv_delta_stats.push(fv_delta_bps)
                if fv_delta_bps > 20:
                    self.fv_jump_count += 1
                # Realized vol: log return
                ret = math.log(fv / self.fv_prev)
                self.fv_returns.append(ret)
            self.fv_prev = fv

        hv_count = safe_int(rec.get("healthy_venues_used_count"))
        if hv_count is not None:
            self.healthy_venues_count_stats.push(float(hv_count))

        # FV vs median of healthy mids
        if fv is not None and fv > 0 and fv_available and len(healthy_mids) >= 2:
            mids_only = sorted([m for _, m in healthy_mids])
            median_mid = mids_only[len(mids_only) // 2]
            residual = abs(fv - median_mid)
            self.fv_vs_median_residual.push(residual)

        # === Dimension 6: Markout ===
        venue_markout = rec.get("venue_markout_ewma_usd_per_tao", [])
        venue_toxicity = rec.get("venue_toxicity", [])
        for i in range(NUM_VENUES):
            mk = safe_float(venue_markout[i] if isinstance(venue_markout, list) and i < len(venue_markout) else None)
            tox = safe_float(venue_toxicity[i] if isinstance(venue_toxicity, list) and i < len(venue_toxicity) else None)
            if mk is not None:
                self.venue_markout_stats[i].push(mk)
            if tox is not None:
                self.venue_toxicity_stats[i].push(tox)
                self.venue_toxicity_windows[i].push(tox)

        fills = rec.get("fills", [])
        if isinstance(fills, list):
            self.fills_count += len(fills)

        # === Dimension 7: Quote-Level Edge ===
        quote_levels = rec.get("quote_levels", [])
        if isinstance(quote_levels, list):
            for ql in quote_levels:
                if not isinstance(ql, dict):
                    continue
                vi = safe_int(ql.get("venue_index"))
                if vi is None or vi < 0 or vi >= NUM_VENUES:
                    continue
                self.venue_quote_count[vi] += 1

                edge = safe_float(ql.get("edge_local"))
                if edge is not None:
                    self.venue_edge_stats[vi].push(edge)

                delta_f = safe_float(ql.get("delta_final"))
                if delta_f is not None:
                    self.venue_delta_final_stats[vi].push(delta_f)

                basis_adj = safe_float(ql.get("basis_adj_usd"))
                if basis_adj is not None:
                    self.venue_basis_adj_stats[vi].push(basis_adj)

                funding_adj = safe_float(ql.get("funding_adj_usd"))
                if funding_adj is not None:
                    self.venue_funding_adj_stats[vi].push(funding_adj)

                inv_term = safe_float(ql.get("inventory_term_usd"))
                if inv_term is not None:
                    self.venue_inventory_term_stats[vi].push(inv_term)

                sm = safe_float(ql.get("spread_mult"))
                if sm is not None:
                    self.venue_spread_mult_stats[vi].push(sm)

                szm = safe_float(ql.get("size_mult"))
                if szm is not None:
                    self.venue_size_mult_stats[vi].push(szm)

                size_raw = safe_float(ql.get("size_raw"))
                size_final = safe_float(ql.get("size_final"))
                size_margin_cap = safe_float(ql.get("size_margin_cap"))
                liq_factor = safe_float(ql.get("size_liq_factor"))

                if size_raw is not None:
                    self.venue_size_raw_stats[vi].push(size_raw)
                if size_final is not None:
                    self.venue_size_final_stats[vi].push(size_final)

                if size_margin_cap is not None and size_raw is not None and size_margin_cap < size_raw:
                    self.venue_margin_cap_binding[vi] += 1
                if liq_factor is not None and liq_factor < 1.0:
                    self.venue_liq_factor_binding[vi] += 1
                if size_final is not None and size_raw is not None and size_final < size_raw - 0.001:
                    self.venue_any_constraint_binding[vi] += 1

        # === Dimension 8: Exits ===
        exits = rec.get("exits", [])
        if isinstance(exits, list) and exits:
            self.exits_tick_count += 1
            for ex in exits:
                if isinstance(ex, dict):
                    ef = safe_float(ex.get("edge_final"))
                    if ef is not None:
                        self.exit_edge_stats.push(ef)

        # === Dimension 9: Hedges ===
        hedges = rec.get("hedges", [])
        if isinstance(hedges, list) and hedges:
            self.hedges_tick_count += 1

        hedge_delta = safe_float(rec.get("hedge_delta_h_t"))
        if hedge_delta is not None:
            self.hedge_delta_stats.push(hedge_delta)

        # === Dimension 10: Margin ===
        margin_balance = rec.get("venue_margin_balance_usd", [])
        margin_avail = rec.get("venue_margin_available_usd", [])
        margin_used = rec.get("venue_margin_used_usd", [])
        dist_liq = rec.get("venue_dist_liq_sigma", [])

        for i in range(NUM_VENUES):
            bal = safe_float(margin_balance[i] if isinstance(margin_balance, list) and i < len(margin_balance) else None)
            avail = safe_float(margin_avail[i] if isinstance(margin_avail, list) and i < len(margin_avail) else None)
            used = safe_float(margin_used[i] if isinstance(margin_used, list) and i < len(margin_used) else None)
            dl = safe_float(dist_liq[i] if isinstance(dist_liq, list) and i < len(dist_liq) else None)

            if bal is not None and used is not None and bal > 0:
                util = used / bal
                self.venue_margin_util_stats[i].push(util)
            if avail is not None:
                self.venue_margin_avail_stats[i].push(avail)
            if dl is not None:
                self.venue_dist_liq_stats[i].push(dl)

        # === Dimension 11: Volatility Model ===
        fv_sv = safe_float(rec.get("fv_short_vol"))
        fv_lv = safe_float(rec.get("fv_long_vol"))
        sigma = safe_float(rec.get("sigma_eff"))
        rr = safe_float(rec.get("regime_ratio"))

        if fv_sv is not None:
            self.fv_short_vol_stats.push(fv_sv)
        if fv_lv is not None:
            self.fv_long_vol_stats.push(fv_lv)
        if sigma is not None:
            self.sigma_eff_stats.push(sigma)
        if rr is not None:
            self.regime_ratio_stats.push(rr)
            self.regime_ratio_windows.push(rr)

        venue_lvs = rec.get("venue_local_vol_short", [])
        venue_lvl = rec.get("venue_local_vol_long", [])
        for i in range(NUM_VENUES):
            vs = safe_float(venue_lvs[i] if isinstance(venue_lvs, list) and i < len(venue_lvs) else None)
            vl = safe_float(venue_lvl[i] if isinstance(venue_lvl, list) and i < len(venue_lvl) else None)
            if vs is not None:
                self.venue_local_vol_short[i].push(vs)
            if vl is not None:
                self.venue_local_vol_long[i].push(vl)

        # === Dimension 12: Reconcile Drift ===
        drift = rec.get("reconcile_drift")
        if isinstance(drift, list) and drift:
            self.reconcile_drift_tick_count += 1
            for d in drift:
                if isinstance(d, dict):
                    vi = safe_int(d.get("venue_index"))
                    if vi is not None and 0 <= vi < NUM_VENUES:
                        self.drift_by_venue[vi] += 1

        # === Dimension 13: Risk & Order Flow ===
        risk_regime = rec.get("risk_regime")
        if isinstance(risk_regime, str):
            self.risk_regime_counts[risk_regime] += 1
            if self.risk_regime_prev is not None and risk_regime != self.risk_regime_prev:
                self.risk_regime_transitions += 1
            self.risk_regime_prev = risk_regime

        kill = rec.get("kill_switch", False)
        if kill:
            self.kill_switch_count += 1
            self.anomalies.add(tick, "kill_switch", "Critical",
                f"Kill switch activated: {rec.get('kill_reason', 'unknown')}",
                {"reason": rec.get("kill_reason")})

        wso_count = safe_int(rec.get("would_send_orders_count"))
        if wso_count is not None:
            self.would_send_count_stats.push(float(wso_count))
            if wso_count == 0:
                self.would_send_zero_ticks += 1
                self.would_send_consecutive_zero += 1
                if self.would_send_consecutive_zero > self.would_send_max_consecutive_zero:
                    self.would_send_max_consecutive_zero = self.would_send_consecutive_zero
            else:
                self.would_send_consecutive_zero = 0

        dd = safe_float(rec.get("dollar_delta_usd"))
        if dd is not None:
            self.dollar_delta_stats.push(dd)

        qg = safe_float(rec.get("q_global_tao"))
        if qg is not None:
            self.q_global_stats.push(qg)

        risk_events = rec.get("risk_events", [])
        if isinstance(risk_events, list):
            for re_item in risk_events:
                if isinstance(re_item, dict):
                    et = re_item.get("event_type", "unknown")
                    self.risk_event_counts[et] += 1

        orders = rec.get("orders", [])
        if isinstance(orders, list):
            for o in orders:
                if isinstance(o, dict):
                    action = o.get("action", "unknown")
                    self.order_action_counts[action] += 1
                    vi = safe_int(o.get("venue_index"))
                    if vi is not None and 0 <= vi < NUM_VENUES:
                        self.orders_per_venue[vi] += 1

        # === Dimension 14: Anomaly Detection ===
        # Age spikes
        for i in range(NUM_VENUES):
            age_f = safe_float(venue_age[i] if isinstance(venue_age, list) and i < len(venue_age) else None)
            if age_f is not None and age_f > 3000:
                threshold = STALE_MS.get(VENUE_NAMES[i], 1000)
                if age_f > threshold * 3:
                    self.anomalies.add(tick, "age_spike", "Warning",
                        f"{VENUE_NAMES[i]} age={age_f:.0f}ms (>{threshold*3}ms)",
                        {"venue": VENUE_NAMES[i], "age_ms": age_f})

        # Spread blowout detection (relative to running median)
        for i in range(NUM_VENUES):
            spread_f = safe_float(venue_spread[i] if isinstance(venue_spread, list) and i < len(venue_spread) else None)
            if spread_f is not None and self.venue_spread_stats[i].n > 100:
                median_spread = self.venue_spread_stats[i].percentile(50)
                if median_spread > 0 and spread_f > median_spread * 10:
                    self.anomalies.add(tick, "spread_blowout", "Warning",
                        f"{VENUE_NAMES[i]} spread={spread_f:.4f} ({spread_f/median_spread:.1f}x median)",
                        {"venue": VENUE_NAMES[i], "spread": spread_f, "median": median_spread})

        # Quoting gap detection (>20 consecutive zero would_send)
        if self.would_send_consecutive_zero == 20:
            self.anomalies.add(tick, "quoting_gap", "Warning",
                f"20+ consecutive ticks with 0 would_send_orders",
                {"consecutive_zero": self.would_send_consecutive_zero})

        # === Dimension 15: Funding ===
        funding_rate = rec.get("venue_funding_rate_8h", [])
        funding_status = rec.get("venue_funding_status", [])
        funding_age = rec.get("venue_funding_age_ms", [])

        for i in range(NUM_VENUES):
            fr = safe_float(funding_rate[i] if isinstance(funding_rate, list) and i < len(funding_rate) else None)
            fs = funding_status[i] if isinstance(funding_status, list) and i < len(funding_status) else None
            fa = safe_float(funding_age[i] if isinstance(funding_age, list) and i < len(funding_age) else None)

            if fr is not None:
                self.venue_funding_rate_stats[i].push(fr)
            if isinstance(fs, str):
                self.venue_funding_status_counts[i][fs] += 1
            if fa is not None and fa >= 0:
                self.venue_funding_age_stats[i].push(fa)

        # PnL
        pnl_t = safe_float(rec.get("pnl_total"))
        pnl_r = safe_float(rec.get("pnl_realised"))
        if pnl_t is not None:
            self.pnl_total_stats.push(pnl_t)
        if pnl_r is not None:
            self.pnl_realised_stats.push(pnl_r)


# ---------------------------------------------------------------------------
# Report Formatter
# ---------------------------------------------------------------------------

def format_table(headers: list[str], rows: list[list[str]], indent: int = 2) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            if idx < len(widths):
                widths[idx] = max(widths[idx], len(cell))
    prefix = " " * indent
    lines = []
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    lines.append(prefix + header_line)
    lines.append(prefix + sep_line)
    for row in rows:
        lines.append(prefix + " | ".join(
            (row[i] if i < len(row) else "").ljust(widths[i]) for i in range(len(headers))
        ))
    return "\n".join(lines)


def generate_report(acc: TelemetryAccumulator) -> str:
    lines: list[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"  {title}")
        lines.append("=" * 80)

    def subsection(title: str) -> None:
        lines.append("")
        lines.append(f"  --- {title} ---")

    # Header
    elapsed_s = 0.0
    if acc.first_ts_ms and acc.last_ts_ms:
        elapsed_s = (acc.last_ts_ms - acc.first_ts_ms) / 1000.0
    tick_rate = acc.tick_count / elapsed_s if elapsed_s > 0 else 0

    lines.append("=" * 80)
    lines.append("  PARAPHINA TELEMETRY ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"  Ticks analyzed: {acc.tick_count} (tick {acc.first_tick} -> {acc.last_tick})")
    lines.append(f"  Time range: {ts_str(acc.first_ts_ms)} -> {ts_str(acc.last_ts_ms)} ({elapsed_s:.0f}s / {elapsed_s/3600:.2f}h)")
    lines.append(f"  Tick rate: {tick_rate:.2f} ticks/sec")
    lines.append(f"  Venues: {', '.join(VENUE_NAMES)}")
    lines.append(f"  Anomalies detected: {len(acc.anomalies.items)} "
                 f"(Critical={len(acc.anomalies.by_severity('Critical'))}, "
                 f"Warning={len(acc.anomalies.by_severity('Warning'))}, "
                 f"Info={len(acc.anomalies.by_severity('Info'))})")

    # =====================================================================
    # CATEGORY A: Infrastructure & Connectivity
    # =====================================================================
    section("CATEGORY A: INFRASTRUCTURE & CONNECTIVITY")

    # Dimension 1: Venue Health
    subsection("Dimension 1: Venue Health & Connectivity")
    headers = ["Venue", "Healthy%", "Stale%", "Disabled%", "Flips", "MaxStaleRun", "StaleRuns"]
    rows = []
    for i in range(NUM_VENUES):
        total = sum(acc.venue_status_counts[i].values())
        healthy = acc.venue_status_counts[i].get("Healthy", 0)
        stale = acc.venue_status_counts[i].get("Stale", 0)
        disabled = acc.venue_status_counts[i].get("Disabled", 0)
        max_run = acc.venue_max_consecutive_stale[i]
        # Check if currently in a stale run
        if acc.venue_consecutive_stale[i] > max_run:
            max_run = acc.venue_consecutive_stale[i]
        num_runs = len(acc.venue_stale_runs[i])
        if acc.venue_consecutive_stale[i] > 0:
            num_runs += 1  # count ongoing run
        rows.append([
            VENUE_NAMES[i],
            pct(healthy, total),
            pct(stale, total),
            pct(disabled, total),
            str(acc.venue_status_flips[i]),
            str(max_run),
            str(num_runs),
        ])
    lines.append(format_table(headers, rows))

    subsection("Venue Age (ms) Statistics")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_age_stats[i].summary_line(VENUE_NAMES[i], "ms"))

    lines.append(f"\n  Correlated staleness (2+ venues stale simultaneously): {acc.multi_stale_ticks} ticks ({pct(acc.multi_stale_ticks, acc.tick_count)})")

    # Dimension 2: Tick Timing
    subsection("Dimension 2: Tick Performance / Timing Budget")
    lines.append(acc.timing_total_us.summary_line("total_us", "us"))
    lines.append(acc.timing_drain_us.summary_line("event_drain_us", "us"))
    lines.append(acc.timing_engine_us.summary_line("engine_us", "us"))
    lines.append(acc.timing_submit_us.summary_line("submit_us", "us"))
    lines.append(acc.timing_reconcile_us.summary_line("reconcile_us", "us"))
    lines.append(acc.timing_order_pending.summary_line("order_tx_pending"))

    lines.append("")
    lines.append("  Budget exceedances:")
    for label in [">250us", ">1ms", ">5ms", ">50ms"]:
        count = acc.timing_budget_exceed.get(label, 0)
        lines.append(f"    {label}: {count} ticks ({pct(count, acc.tick_count)})")

    # Timing trend
    subsection("Tick Timing Trend (p95 total_us per 10K-tick window)")
    trend = acc.timing_total_windows.trend_summary()
    if trend:
        headers_t = ["Window", "Ticks", "Mean_us", "P95_us", "P99_us", "Max_us"]
        rows_t = []
        for w in trend:
            rows_t.append([
                f"{w['tick_start']}-{w['tick_end']}",
                str(w["count"]),
                fmt_f(w.get("mean"), 1),
                fmt_f(w.get("p95"), 1),
                fmt_f(w.get("p99"), 1),
                fmt_f(w.get("max"), 1),
            ])
        lines.append(format_table(headers_t, rows_t))

    # Phase dominance
    subsection("Phase Dominance (% of total_us at p99)")
    total_p99 = acc.timing_total_us.percentile(99) if acc.timing_total_us.n > 0 else 1
    if total_p99 > 0:
        for label, stat in [("event_drain", acc.timing_drain_us), ("engine", acc.timing_engine_us),
                            ("submit", acc.timing_submit_us), ("reconcile", acc.timing_reconcile_us)]:
            p99 = stat.percentile(99) if stat.n > 0 else 0
            lines.append(f"    {label}: p99={fmt_f(p99, 1)}us ({pct(int(p99), int(total_p99))} of total p99)")

    # =====================================================================
    # CATEGORY B: Market Data & Pricing
    # =====================================================================
    section("CATEGORY B: MARKET DATA & PRICING")

    # Dimension 3
    subsection("Dimension 3: Market Data Quality")
    lines.append("  Mid Price Statistics:")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_mid_stats[i].summary_line(f"  {VENUE_NAMES[i]} mid", "USD"))

    lines.append("")
    lines.append("  Spread Statistics:")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_spread_stats[i].summary_line(f"  {VENUE_NAMES[i]} spread", "USD"))

    lines.append("")
    lines.append("  Depth Near Mid Statistics:")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_depth_stats[i].summary_line(f"  {VENUE_NAMES[i]} depth", "USD"))

    lines.append("")
    lines.append("  Mid Price Tick-to-Tick Delta (bps):")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_mid_delta_stats[i].summary_line(f"  {VENUE_NAMES[i]}", "bps"))
        lines.append(f"      Jumps >50bps: {acc.mid_jump_count[i]}")

    lines.append(f"\n  FV Unavailable: {acc.fv_unavailable_count} ticks ({pct(acc.fv_unavailable_count, acc.tick_count)})")
    if acc.fv_unavailable_runs:
        lines.append(f"  FV Unavailable runs: {len(acc.fv_unavailable_runs)}, "
                     f"max={max(acc.fv_unavailable_runs)} ticks, "
                     f"mean={sum(acc.fv_unavailable_runs)/len(acc.fv_unavailable_runs):.1f} ticks")

    # Dimension 4
    subsection("Dimension 4: Cross-Venue Pricing Coherence")
    lines.append(acc.cross_venue_dispersion.summary_line("Cross-venue dispersion (max-min of healthy mids)", "USD"))

    lines.append("")
    lines.append("  Per-Venue Deviation from Fair Value:")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_fv_deviation[i].summary_line(f"  {VENUE_NAMES[i]}", "USD"))

    # Dispersion trend
    subsection("Cross-Venue Dispersion Trend (per window)")
    trend_d = acc.cross_venue_dispersion_windows.trend_summary()
    if trend_d:
        headers_d = ["Window", "Mean_USD", "P95_USD", "Max_USD"]
        rows_d = []
        for w in trend_d:
            rows_d.append([
                f"{w['tick_start']}-{w['tick_end']}",
                fmt_f(w.get("mean"), 4),
                fmt_f(w.get("p95"), 4),
                fmt_f(w.get("max"), 4),
            ])
        lines.append(format_table(headers_d, rows_d))

    # Dimension 5
    subsection("Dimension 5: Kalman Filter / Fair Value Health")
    lines.append(acc.kf_p_stats.summary_line("kf_p (covariance)"))
    lines.append(acc.kf_xhat_stats.summary_line("kf_x_hat"))
    lines.append(acc.fv_stats.summary_line("fair_value", "USD"))
    lines.append(acc.fv_delta_stats.summary_line("FV tick-to-tick delta", "bps"))
    lines.append(f"  FV jumps >20bps: {acc.fv_jump_count}")
    lines.append(acc.healthy_venues_count_stats.summary_line("healthy_venues_used_count"))
    lines.append(acc.fv_vs_median_residual.summary_line("FV vs median(healthy mids) residual", "USD"))

    # KF covariance trend
    subsection("KF Covariance (kf_p) Trend")
    trend_kf = acc.kf_p_windows.trend_summary()
    if trend_kf:
        headers_kf = ["Window", "Mean", "P95", "Max"]
        rows_kf = []
        for w in trend_kf:
            rows_kf.append([
                f"{w['tick_start']}-{w['tick_end']}",
                f"{w.get('mean', 0):.8f}",
                f"{w.get('p95', 0):.8f}",
                f"{w.get('max', 0):.8f}",
            ])
        lines.append(format_table(headers_kf, rows_kf))

    # =====================================================================
    # CATEGORY C: PnL-Critical Strategy Analysis
    # =====================================================================
    section("CATEGORY C: PnL-CRITICAL STRATEGY ANALYSIS")

    # Dimension 6
    subsection("Dimension 6: Markout & Adverse Selection")
    lines.append("  Per-Venue Markout EWMA (USD/tao):")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_markout_stats[i].summary_line(f"  {VENUE_NAMES[i]}"))
    lines.append("")
    lines.append("  Per-Venue Toxicity:")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_toxicity_stats[i].summary_line(f"  {VENUE_NAMES[i]}"))
    lines.append(f"\n  Total fills: {acc.fills_count}")
    if acc.fills_count == 0:
        lines.append("  NOTE: Shadow mode -- no real fills. Markout/toxicity from EWMA estimates only.")

    # Toxicity trend
    subsection("Toxicity Trend (per venue, per window)")
    for i in range(NUM_VENUES):
        trend_tox = acc.venue_toxicity_windows[i].trend_summary()
        if trend_tox and any(w["count"] > 0 for w in trend_tox):
            lines.append(f"  {VENUE_NAMES[i]}:")
            for w in trend_tox:
                if w["count"] > 0:
                    lines.append(f"    [{w['tick_start']}-{w['tick_end']}] mean={fmt_f(w.get('mean'), 4)} p95={fmt_f(w.get('p95'), 4)}")

    # Dimension 7
    subsection("Dimension 7: Quote-Level Edge Decomposition")
    lines.append("  Per-Venue Edge (edge_local):")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_edge_stats[i].summary_line(f"  {VENUE_NAMES[i]}", "USD"))
        lines.append(f"      Quote samples: {acc.venue_quote_count[i]}")

    lines.append("")
    lines.append("  Per-Venue Half-Spread (delta_final):")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_delta_final_stats[i].summary_line(f"  {VENUE_NAMES[i]}", "USD"))

    lines.append("")
    lines.append("  Edge Component Breakdown (mean absolute USD):")
    headers_ec = ["Venue", "basis_adj", "funding_adj", "inventory_term"]
    rows_ec = []
    for i in range(NUM_VENUES):
        rows_ec.append([
            VENUE_NAMES[i],
            fmt_f(acc.venue_basis_adj_stats[i].mean if acc.venue_basis_adj_stats[i].n > 0 else None, 4),
            fmt_f(acc.venue_funding_adj_stats[i].mean if acc.venue_funding_adj_stats[i].n > 0 else None, 4),
            fmt_f(acc.venue_inventory_term_stats[i].mean if acc.venue_inventory_term_stats[i].n > 0 else None, 4),
        ])
    lines.append(format_table(headers_ec, rows_ec))

    lines.append("")
    lines.append("  Spread/Size Multipliers:")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_spread_mult_stats[i].summary_line(f"  {VENUE_NAMES[i]} spread_mult"))
        lines.append(acc.venue_size_mult_stats[i].summary_line(f"  {VENUE_NAMES[i]} size_mult"))

    lines.append("")
    lines.append("  Size Constraint Binding Frequency:")
    headers_sc = ["Venue", "Quotes", "MarginCap<Raw", "LiqFactor<1", "AnyBinding"]
    rows_sc = []
    for i in range(NUM_VENUES):
        qc = acc.venue_quote_count[i]
        rows_sc.append([
            VENUE_NAMES[i],
            str(qc),
            f"{acc.venue_margin_cap_binding[i]} ({pct(acc.venue_margin_cap_binding[i], qc)})",
            f"{acc.venue_liq_factor_binding[i]} ({pct(acc.venue_liq_factor_binding[i], qc)})",
            f"{acc.venue_any_constraint_binding[i]} ({pct(acc.venue_any_constraint_binding[i], qc)})",
        ])
    lines.append(format_table(headers_sc, rows_sc))

    # Dimension 8
    subsection("Dimension 8: Exit Engine Effectiveness")
    lines.append(f"  Ticks with exits: {acc.exits_tick_count} ({pct(acc.exits_tick_count, acc.tick_count)})")
    lines.append(acc.exit_edge_stats.summary_line("exit edge_final", "USD"))
    if acc.exits_tick_count == 0:
        lines.append("  NOTE: No exits detected (expected in shadow mode)")

    # Dimension 9
    subsection("Dimension 9: Hedge Controller Performance")
    lines.append(f"  Ticks with hedges: {acc.hedges_tick_count} ({pct(acc.hedges_tick_count, acc.tick_count)})")
    lines.append(acc.hedge_delta_stats.summary_line("hedge_delta_h_t"))
    if acc.hedges_tick_count == 0:
        lines.append("  NOTE: No hedges detected (expected in shadow mode)")

    # =====================================================================
    # CATEGORY D: Capital Efficiency
    # =====================================================================
    section("CATEGORY D: CAPITAL EFFICIENCY")

    # Dimension 10
    subsection("Dimension 10: Margin Utilization & Liquidation Distance")
    lines.append("  Margin Utilization (used/balance):")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_margin_util_stats[i].summary_line(f"  {VENUE_NAMES[i]}"))

    lines.append("")
    lines.append("  Margin Available (USD):")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_margin_avail_stats[i].summary_line(f"  {VENUE_NAMES[i]}", "USD"))

    lines.append("")
    lines.append("  Distance to Liquidation (sigma):")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_dist_liq_stats[i].summary_line(f"  {VENUE_NAMES[i]}", "σ"))

    # =====================================================================
    # CATEGORY E: Model Validation
    # =====================================================================
    section("CATEGORY E: MODEL VALIDATION")

    # Dimension 11
    subsection("Dimension 11: Volatility Model Validation")
    lines.append(acc.fv_short_vol_stats.summary_line("fv_short_vol"))
    lines.append(acc.fv_long_vol_stats.summary_line("fv_long_vol"))
    lines.append(acc.sigma_eff_stats.summary_line("sigma_eff"))
    lines.append(acc.regime_ratio_stats.summary_line("regime_ratio"))

    # Realized vol comparison
    if len(acc.fv_returns) > 100:
        import statistics as stmod
        realized_std = stmod.stdev(acc.fv_returns)
        # Annualize: assume 4 ticks/sec -> 252*24*3600*4 ticks/year
        ticks_per_year = 252 * 24 * 3600 * 4
        realized_annual = realized_std * math.sqrt(ticks_per_year)
        model_short_vol = acc.fv_short_vol_stats.mean if acc.fv_short_vol_stats.n > 0 else 0
        lines.append(f"\n  Realized FV vol (per-tick stdev of log returns): {realized_std:.8f}")
        lines.append(f"  Realized FV vol (annualized, assuming 4 ticks/s): {realized_annual:.4f}")
        lines.append(f"  Model fv_short_vol (mean): {model_short_vol:.8f}")
        if model_short_vol > 0:
            ratio = realized_std / model_short_vol
            lines.append(f"  Ratio (realized / model): {ratio:.4f}")

    lines.append("")
    lines.append("  Per-Venue Local Vol (short):")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_local_vol_short[i].summary_line(f"  {VENUE_NAMES[i]}"))

    # Regime ratio trend
    subsection("Regime Ratio Trend (per window)")
    trend_rr = acc.regime_ratio_windows.trend_summary()
    if trend_rr:
        headers_rr = ["Window", "Mean", "P5", "P95", "Max"]
        rows_rr = []
        for w in trend_rr:
            s = OnlineStats()
            # We need to get p5 from the window, but our OnlineStats doesn't expose p5 in summary
            rows_rr.append([
                f"{w['tick_start']}-{w['tick_end']}",
                fmt_f(w.get("mean"), 4),
                fmt_f(w.get("min"), 4),
                fmt_f(w.get("p95"), 4),
                fmt_f(w.get("max"), 4),
            ])
        lines.append(format_table(headers_rr, rows_rr))

    # Dimension 12
    subsection("Dimension 12: Reconcile Drift (Shadow Accuracy)")
    lines.append(f"  Ticks with reconcile drift: {acc.reconcile_drift_tick_count} ({pct(acc.reconcile_drift_tick_count, acc.tick_count)})")
    if acc.reconcile_drift_tick_count > 0:
        lines.append("  Drift events per venue:")
        for i in range(NUM_VENUES):
            lines.append(f"    {VENUE_NAMES[i]}: {acc.drift_by_venue[i]}")
    else:
        lines.append("  No reconcile drift events detected.")

    # =====================================================================
    # CATEGORY F: Risk & Anomaly Detection
    # =====================================================================
    section("CATEGORY F: RISK & ANOMALY DETECTION")

    # Dimension 13
    subsection("Dimension 13: Risk, Quoting, and Order Flow")
    lines.append("  Risk Regime Distribution:")
    for regime, count in acc.risk_regime_counts.most_common():
        lines.append(f"    {regime}: {count} ({pct(count, acc.tick_count)})")
    lines.append(f"  Regime transitions: {acc.risk_regime_transitions}")
    lines.append(f"  Kill switch activations: {acc.kill_switch_count}")

    lines.append("")
    lines.append(acc.would_send_count_stats.summary_line("would_send_orders_count"))
    lines.append(f"  Ticks with 0 would_send: {acc.would_send_zero_ticks} ({pct(acc.would_send_zero_ticks, acc.tick_count)})")
    lines.append(f"  Max consecutive 0 would_send: {acc.would_send_max_consecutive_zero}")

    lines.append("")
    lines.append(acc.dollar_delta_stats.summary_line("dollar_delta_usd", "USD"))
    lines.append(acc.q_global_stats.summary_line("q_global_tao", "tao"))

    lines.append("")
    lines.append("  Order Action Counts:")
    for action, count in acc.order_action_counts.most_common():
        lines.append(f"    {action}: {count}")

    lines.append("")
    lines.append("  Orders Per Venue:")
    for i in range(NUM_VENUES):
        lines.append(f"    {VENUE_NAMES[i]}: {acc.orders_per_venue[i]}")

    lines.append("")
    lines.append("  Risk Event Types:")
    if acc.risk_event_counts:
        for et, count in acc.risk_event_counts.most_common():
            lines.append(f"    {et}: {count}")
    else:
        lines.append("    (none)")

    # PnL
    subsection("PnL Summary")
    lines.append(acc.pnl_total_stats.summary_line("pnl_total", "USD"))
    lines.append(acc.pnl_realised_stats.summary_line("pnl_realised", "USD"))
    if acc.pnl_total_stats.n > 0:
        lines.append(f"  Final pnl_total: {fmt_f(acc.pnl_total_stats._vals[-1] if acc.pnl_total_stats._vals else None, 4)} USD")

    # Dimension 14
    subsection("Dimension 14: Anomaly Summary")
    lines.append(f"  Total anomalies detected: {len(acc.anomalies.items)}")
    lines.append("  By category:")
    for cat, count in acc.anomalies.counts.most_common():
        lines.append(f"    {cat}: {count}")

    lines.append("")
    lines.append("  Critical Anomalies:")
    critical = acc.anomalies.by_severity("Critical")
    if critical:
        for a in critical[:20]:
            lines.append(f"    tick={a.tick}: {a.description}")
    else:
        lines.append("    (none)")

    lines.append("")
    lines.append("  Warning Anomalies (first 30):")
    warnings = acc.anomalies.by_severity("Warning")
    if warnings:
        for a in warnings[:30]:
            lines.append(f"    tick={a.tick} [{a.category}]: {a.description}")
    else:
        lines.append("    (none)")

    # =====================================================================
    # CATEGORY G: Funding Data
    # =====================================================================
    section("CATEGORY G: FUNDING DATA")

    # Dimension 15
    subsection("Dimension 15: Funding Data")
    lines.append("  Funding Rate 8h:")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_funding_rate_stats[i].summary_line(f"  {VENUE_NAMES[i]}"))

    lines.append("")
    lines.append("  Funding Status Distribution:")
    headers_fs = ["Venue", "Healthy%", "Unknown%", "Stale%"]
    rows_fs = []
    for i in range(NUM_VENUES):
        total = sum(acc.venue_funding_status_counts[i].values())
        h = acc.venue_funding_status_counts[i].get("Healthy", 0)
        u = acc.venue_funding_status_counts[i].get("Unknown", 0)
        s = acc.venue_funding_status_counts[i].get("Stale", 0)
        rows_fs.append([VENUE_NAMES[i], pct(h, total), pct(u, total), pct(s, total)])
    lines.append(format_table(headers_fs, rows_fs))

    lines.append("")
    lines.append("  Funding Age (ms):")
    for i in range(NUM_VENUES):
        lines.append(acc.venue_funding_age_stats[i].summary_line(f"  {VENUE_NAMES[i]}", "ms"))

    # Cross-venue funding dispersion
    if all(acc.venue_funding_rate_stats[i].n > 0 for i in range(NUM_VENUES)):
        means = [acc.venue_funding_rate_stats[i].mean for i in range(NUM_VENUES)]
        valid_means = [m for m in means if m != 0]
        if len(valid_means) >= 2:
            disp = max(valid_means) - min(valid_means)
            lines.append(f"\n  Cross-venue funding rate dispersion (max-min of means): {disp:.6f}")

    # =====================================================================
    # Age Trend (per venue, per window)
    # =====================================================================
    section("APPENDIX: VENUE AGE TREND (per window)")
    for i in range(NUM_VENUES):
        trend_age = acc.venue_age_windows[i].trend_summary()
        if trend_age:
            lines.append(f"\n  {VENUE_NAMES[i]}:")
            headers_a = ["Window", "Mean_ms", "P95_ms", "P99_ms", "Max_ms"]
            rows_a = []
            for w in trend_age:
                rows_a.append([
                    f"{w['tick_start']}-{w['tick_end']}",
                    fmt_f(w.get("mean"), 1),
                    fmt_f(w.get("p95"), 1),
                    fmt_f(w.get("p99"), 1),
                    fmt_f(w.get("max"), 1),
                ])
            lines.append(format_table(headers_a, rows_a))

    lines.append("")
    lines.append("=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streaming reader
# ---------------------------------------------------------------------------

def stream_records(path: Path, max_ticks: int = 0) -> TelemetryAccumulator:
    acc = TelemetryAccumulator()
    count = 0
    t0 = time.monotonic()

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            acc.process(rec)
            count += 1
            if max_ticks > 0 and count >= max_ticks:
                break
            if count % 10000 == 0:
                elapsed = time.monotonic() - t0
                print(f"  ... processed {count} ticks ({elapsed:.1f}s)", file=sys.stderr)

    elapsed = time.monotonic() - t0
    print(f"  Processed {count} ticks in {elapsed:.1f}s ({count/elapsed:.0f} ticks/sec)", file=sys.stderr)
    return acc


# ---------------------------------------------------------------------------
# Checkpoint snapshot (JSON)
# ---------------------------------------------------------------------------

def save_checkpoint(acc: TelemetryAccumulator, path: Path) -> None:
    """Save a lightweight checkpoint JSON for cross-checkpoint comparison."""
    snapshot: dict[str, Any] = {
        "tick_count": acc.tick_count,
        "first_tick": acc.first_tick,
        "last_tick": acc.last_tick,
        "first_ts_ms": acc.first_ts_ms,
        "last_ts_ms": acc.last_ts_ms,
    }

    # Key metrics for regression scorecard
    snapshot["venue_health"] = {}
    for i in range(NUM_VENUES):
        total = sum(acc.venue_status_counts[i].values())
        healthy = acc.venue_status_counts[i].get("Healthy", 0)
        snapshot["venue_health"][VENUE_NAMES[i]] = {
            "healthy_pct": round(100.0 * healthy / total, 4) if total > 0 else 0,
            "age_p95": round(acc.venue_age_stats[i].percentile(95), 2) if acc.venue_age_stats[i].n > 0 else None,
            "flips": acc.venue_status_flips[i],
        }

    snapshot["timing"] = {
        "total_us_mean": round(acc.timing_total_us.mean, 2) if acc.timing_total_us.n > 0 else None,
        "total_us_p95": round(acc.timing_total_us.percentile(95), 2) if acc.timing_total_us.n > 0 else None,
        "total_us_p99": round(acc.timing_total_us.percentile(99), 2) if acc.timing_total_us.n > 0 else None,
        "budget_exceed_1ms": acc.timing_budget_exceed.get(">1ms", 0),
    }

    snapshot["pricing"] = {
        "cross_venue_dispersion_mean": round(acc.cross_venue_dispersion.mean, 6) if acc.cross_venue_dispersion.n > 0 else None,
        "cross_venue_dispersion_p95": round(acc.cross_venue_dispersion.percentile(95), 6) if acc.cross_venue_dispersion.n > 0 else None,
        "fv_unavailable_pct": round(100.0 * acc.fv_unavailable_count / acc.tick_count, 4) if acc.tick_count > 0 else 0,
        "fv_jump_count": acc.fv_jump_count,
    }

    snapshot["risk"] = {
        "kill_switch_count": acc.kill_switch_count,
        "regime_transitions": acc.risk_regime_transitions,
        "would_send_zero_pct": round(100.0 * acc.would_send_zero_ticks / acc.tick_count, 4) if acc.tick_count > 0 else 0,
    }

    snapshot["anomalies"] = {
        "total": len(acc.anomalies.items),
        "critical": len(acc.anomalies.by_severity("Critical")),
        "warning": len(acc.anomalies.by_severity("Warning")),
        "by_category": dict(acc.anomalies.counts),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2)
    print(f"  Checkpoint saved: {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Regression scorecard (compare two checkpoints)
# ---------------------------------------------------------------------------

def regression_scorecard(prev_path: Path, curr_snapshot: dict) -> str:
    if not prev_path.exists():
        return "  (no previous checkpoint for comparison)"

    with prev_path.open("r", encoding="utf-8") as fh:
        prev = json.load(fh)

    lines = []
    lines.append("  Regression Scorecard vs Previous Checkpoint:")
    lines.append(f"  Previous: {prev.get('tick_count', '?')} ticks, Current: {curr_snapshot.get('tick_count', '?')} ticks")
    lines.append("")

    def compare(label: str, prev_val: Any, curr_val: Any, lower_is_better: bool = True) -> str:
        if prev_val is None or curr_val is None:
            return f"    {label}: prev={prev_val} -> curr={curr_val} [N/A]"
        diff = curr_val - prev_val
        if abs(diff) < 0.001 * max(abs(prev_val), 1e-9):
            tag = "[STABLE]"
        elif (lower_is_better and diff < 0) or (not lower_is_better and diff > 0):
            tag = "[IMPROVING]"
        else:
            tag = "[DEGRADING]"
        return f"    {label}: {prev_val} -> {curr_val} (delta={diff:+.4f}) {tag}"

    # Timing
    pt = prev.get("timing", {})
    ct = curr_snapshot.get("timing", {})
    lines.append(compare("total_us p95", pt.get("total_us_p95"), ct.get("total_us_p95")))
    lines.append(compare("total_us p99", pt.get("total_us_p99"), ct.get("total_us_p99")))

    # Pricing
    pp = prev.get("pricing", {})
    cp = curr_snapshot.get("pricing", {})
    lines.append(compare("dispersion mean", pp.get("cross_venue_dispersion_mean"), cp.get("cross_venue_dispersion_mean")))
    lines.append(compare("fv_unavailable %", pp.get("fv_unavailable_pct"), cp.get("fv_unavailable_pct")))

    # Venue health
    for vn in VENUE_NAMES:
        pvh = prev.get("venue_health", {}).get(vn, {})
        cvh = curr_snapshot.get("venue_health", {}).get(vn, {})
        lines.append(compare(f"{vn} healthy%", pvh.get("healthy_pct"), cvh.get("healthy_pct"), lower_is_better=False))
        lines.append(compare(f"{vn} age p95", pvh.get("age_p95"), cvh.get("age_p95")))

    # Anomalies
    pa = prev.get("anomalies", {})
    ca = curr_snapshot.get("anomalies", {})
    lines.append(compare("anomaly count", pa.get("total"), ca.get("total")))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive 15-dimension telemetry analyzer.")
    parser.add_argument("--telemetry", required=True, help="Path to telemetry.jsonl")
    parser.add_argument("--max-ticks", type=int, default=0, help="Max ticks to process (0=all)")
    parser.add_argument("--checkpoint-json", type=str, default=None, help="Path to save checkpoint JSON")
    parser.add_argument("--prev-checkpoint", type=str, default=None, help="Path to previous checkpoint JSON for regression comparison")
    parser.add_argument("--output", type=str, default=None, help="Path to write report (default: stdout)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    telemetry_path = Path(args.telemetry)

    if not telemetry_path.exists():
        print(f"ERROR: telemetry file not found: {telemetry_path}", file=sys.stderr)
        return 1

    print(f"Analyzing {telemetry_path} ...", file=sys.stderr)
    acc = stream_records(telemetry_path, max_ticks=args.max_ticks)

    if acc.tick_count == 0:
        print("ERROR: no ticks found in telemetry", file=sys.stderr)
        return 1

    report = generate_report(acc)

    # Checkpoint
    if args.checkpoint_json:
        cp_path = Path(args.checkpoint_json)
        save_checkpoint(acc, cp_path)

        # Regression scorecard
        if args.prev_checkpoint:
            prev_path = Path(args.prev_checkpoint)
            scorecard = regression_scorecard(prev_path, json.loads(cp_path.read_text()))
            report += "\n\n" + "=" * 80 + "\n  REGRESSION SCORECARD\n" + "=" * 80 + "\n" + scorecard

    # Output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Report written to: {out_path}", file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
