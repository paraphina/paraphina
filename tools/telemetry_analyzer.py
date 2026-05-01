#!/usr/bin/env python3
"""
telemetry_analyzer.py — Comprehensive 15-dimension telemetry analysis.

Streaming single-pass architecture: reads JSONL line-by-line, accumulates statistics
via online algorithms, and outputs a structured text report. It also supports
preserved canary artifacts whose telemetry may contain multiple concatenated JSON
objects on a single physical line.

Usage:
    python3 tools/telemetry_analyzer.py --telemetry /tmp/shadow_eth_post_fix/telemetry.jsonl
    python3 tools/telemetry_analyzer.py --telemetry /path/to/telemetry.jsonl --max-ticks 10000
    python3 tools/telemetry_analyzer.py --telemetry /path/to/telemetry.jsonl --checkpoint-json out/cp_10k.json
    python3 tools/telemetry_analyzer.py --telemetry /path/to/telemetry.jsonl --execution-mode live
    python3 tools/telemetry_analyzer.py --telemetry /var/lib/paraphina/out/telemetry.jsonl --execution-mode live --last-segment --tail-bytes 134217728

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
DEFAULT_SAFE_TAIL_BYTES = 128 * 1024 * 1024
SAFE_PRODUCTION_SCAN_LIMIT_BYTES = 256 * 1024 * 1024
PRODUCTION_ROLLING_TELEMETRY = Path("/var/lib/paraphina/out/telemetry.jsonl")
PNL_BASELINE_MAX_DISPERSION_USD = 5.0
PNL_BASELINE_FINAL_PNL_TOL_USD = 5.0
PNL_BASELINE_FLAT_Q_TAO = 0.05
PNL_BASELINE_LARGE_UNREALISED_USD = 5.0
PNL_BASELINE_RECON_MISMATCH_TOL_USD = 0.05
OPPORTUNITY_FILL_REQUIRED_ACTIVE_SAMPLES = 200

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


def fmt_f(v: Any, decimals: int = 4) -> str:
    f = safe_float(v)
    if f is not None:
        return f"{f:.{decimals}f}"
    if isinstance(v, str) and v:
        return v
    return "n/a"


def fmt_i(v: int | None) -> str:
    if v is None:
        return "n/a"
    return str(v)


def ts_str(ms: int | None) -> str:
    if ms is None:
        return "n/a"
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%H:%M:%S")


def ts_iso(ms: int | None) -> str:
    if ms is None:
        return "n/a"
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def iter_json_objects(path: Path) -> Any:
    """Yield dict records from JSONL, tolerating concatenated objects per line."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            while line:
                try:
                    record, idx = decoder.raw_decode(line)
                except json.JSONDecodeError:
                    break
                if isinstance(record, dict):
                    yield record
                line = line[idx:].lstrip()


def iter_json_objects_from_tail(path: Path, tail_bytes: int) -> Any:
    """Yield dict records from the last tail_bytes of a JSONL file.

    This mode is intentionally approximate and meant for production-safe triage on
    very large rolling telemetry files. If the starting offset lands mid-line, the
    first partial physical line is discarded before parsing.
    """
    if tail_bytes <= 0:
        yield from iter_json_objects(path)
        return

    decoder = json.JSONDecoder()
    with path.open("rb") as raw_fh:
        raw_fh.seek(0, 2)
        file_size = raw_fh.tell()
        start = max(file_size - tail_bytes, 0)
        raw_fh.seek(start)
        if start > 0:
            raw_fh.readline()
        chunk = raw_fh.read().decode("utf-8", errors="ignore")

    for raw_line in chunk.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        while line:
            try:
                record, idx = decoder.raw_decode(line)
            except json.JSONDecodeError:
                break
            if isinstance(record, dict):
                yield record
            line = line[idx:].lstrip()


def last_execution_segment(path: Path, execution_mode: str) -> list[dict[str, Any]]:
    """Return the last contiguous segment for a given execution_mode."""
    last: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for record in iter_json_objects(path):
        if record.get("execution_mode") == execution_mode:
            current.append(record)
        elif current:
            last = current
            current = []
    if current:
        last = current
    return last


def last_execution_segment_from_iter(records: Any, execution_mode: str) -> list[dict[str, Any]]:
    """Return the last contiguous execution_mode segment from an arbitrary iterator."""
    last: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for record in records:
        if record.get("execution_mode") == execution_mode:
            current.append(record)
        elif current:
            last = current
            current = []
    if current:
        last = current
    return last


def guard_unsafe_live_scan(path: Path, tail_bytes: int, unsafe_allow_large_live_file: bool) -> None:
    """Refuse dangerous full scans of the rolling production telemetry file by default."""
    try:
        resolved = path.resolve()
        size_bytes = path.stat().st_size
    except OSError:
        return

    if resolved != PRODUCTION_ROLLING_TELEMETRY:
        return
    if unsafe_allow_large_live_file:
        return
    if tail_bytes > 0:
        return
    if size_bytes <= SAFE_PRODUCTION_SCAN_LIMIT_BYTES:
        return

    mib = size_bytes / (1024 * 1024)
    limit_mib = SAFE_PRODUCTION_SCAN_LIMIT_BYTES / (1024 * 1024)
    suggested_tail = DEFAULT_SAFE_TAIL_BYTES
    raise ValueError(
        "refusing full scan of rolling production telemetry "
        f"({mib:.1f} MiB > safe limit {limit_mib:.0f} MiB). "
        "Use a preserved artifact, copy the file elsewhere, or rerun with "
        f"--tail-bytes {suggested_tail} for bounded triage. "
        "If you truly intend the full scan, pass --unsafe-allow-large-live-file."
    )


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
        vals = sorted(self._vals)
        k = (len(vals) - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vals[int(k)]
        return vals[f] * (c - k) + vals[c] * (k - f)

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


@dataclass
class DecisionContribution:
    decision_id: str
    mm_venue_index: int | None = None
    mm_venue_id: str | None = None
    mm_fill_count: int = 0
    mm_fill_base: float = 0.0
    mm_fill_notional_usd: float = 0.0
    mm_gross_before_fee_usd: float = 0.0
    mm_fee_usd: float = 0.0
    mm_realised_net_usd: float = 0.0
    mm_markout_short_usd: float = 0.0
    hedge_record_count: int = 0
    hedge_fill_count: int = 0
    hedge_fill_notional_usd: float = 0.0
    hedge_fill_fee_usd: float = 0.0
    hedge_exec_cost_model_usd: float = 0.0
    hedge_total_cost_model_usd: float = 0.0

    @property
    def net_after_hedge_exec_model_usd(self) -> float:
        return self.mm_realised_net_usd - self.hedge_exec_cost_model_usd


# ---------------------------------------------------------------------------
# Main Accumulator — all 15 dimensions
# ---------------------------------------------------------------------------

class TelemetryAccumulator:
    def __init__(self) -> None:
        self.venue_names: list[str] = list(VENUE_NAMES)
        self.seen_venue_indices: set[int] = set()
        self.tick_count = 0
        self.first_tick: int | None = None
        self.last_tick: int | None = None
        self.first_ts_ms: int | None = None
        self.last_ts_ms: int | None = None

        # --- Dimension 1: Venue Health ---
        self.venue_status_counts: list[Counter] = [Counter() for _ in range(NUM_VENUES)]
        self.venue_age_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_age_windows: list[WindowAccumulator] = [WindowAccumulator() for _ in range(NUM_VENUES)]
        self.venue_age_event_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_age_event_windows: list[WindowAccumulator] = [WindowAccumulator() for _ in range(NUM_VENUES)]
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
        self.quote_gate_counts: dict[tuple[int, str], Counter] = defaultdict(Counter)
        self.quote_gate_suppression_counts: dict[tuple[int, str], Counter] = defaultdict(Counter)
        self.quote_gate_engine_reason_counts: dict[tuple[int, str], Counter] = defaultdict(Counter)
        self.quote_gate_edge_threshold_stats: dict[tuple[int, str], OnlineStats] = defaultdict(OnlineStats)
        self.quote_gate_edge_threshold_base_stats: dict[tuple[int, str], OnlineStats] = defaultdict(OnlineStats)
        self.quote_gate_hedge_cost_floor_stats: dict[tuple[int, str], OnlineStats] = defaultdict(OnlineStats)
        self.quote_gate_active_edge_stats: dict[tuple[int, str], OnlineStats] = defaultdict(OnlineStats)

        # --- Dimension 8: Exits ---
        self.exits_tick_count = 0
        self.exit_edge_stats = OnlineStats()

        # --- Dimension 9: Hedges ---
        self.hedges_tick_count = 0
        self.hedge_delta_stats = OnlineStats()
        self.hedge_record_total = 0
        self.hedge_source_attributed_count = 0
        self.hedge_source_fill_age_stats = OnlineStats()
        self.hedge_fill_attributed_count = 0
        self.hedge_fill_unattributed_count = 0

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
        self.kill_reason_counts: Counter = Counter()
        self.would_send_count_stats = OnlineStats()
        self.would_send_zero_ticks = 0
        self.would_send_consecutive_zero = 0
        self.would_send_max_consecutive_zero = 0
        self.dollar_delta_stats = OnlineStats()
        self.q_global_stats = OnlineStats()
        self.risk_event_counts: Counter = Counter()
        self.order_action_counts: Counter = Counter()
        self.orders_per_venue: list[int] = [0] * NUM_VENUES
        self.mm_keep_count_stats = OnlineStats()
        self.mm_replace_count_stats = OnlineStats()
        self.mm_place_count_stats = OnlineStats()
        self.mm_cancel_count_stats = OnlineStats()
        self.mm_keep_total = 0
        self.mm_replace_total = 0
        self.mm_place_total = 0
        self.mm_cancel_total = 0
        self.mm_keep_reason_counts: Counter = Counter()
        self.mm_replace_reason_counts: Counter = Counter()
        self.mm_keep_utility_tier_counts: Counter = Counter()
        self.mm_replace_utility_tier_counts: Counter = Counter()
        self.mm_keep_venue_role_counts: Counter = Counter()
        self.mm_replace_venue_role_counts: Counter = Counter()
        self.mm_decisions_by_venue: list[Counter] = [Counter() for _ in range(NUM_VENUES)]
        self.mm_inventory_reducing_counts: Counter = Counter()
        self.mm_decision_record_total = 0
        self.mm_decision_order_link_total = 0
        self.supported_replace_visibility_total = 0
        self.supported_replace_opportunity_counts: list[int] = [0] * NUM_VENUES
        self.supported_replace_visibility_miss_counts: list[int] = [0] * NUM_VENUES
        self.supported_replace_gap_grace_counts: list[int] = [0] * NUM_VENUES
        self.supported_replace_action_counts: list[Counter] = [Counter() for _ in range(NUM_VENUES)]
        self.supported_replace_blocker_counts: list[Counter] = [Counter() for _ in range(NUM_VENUES)]
        self.projected_mm_budget_total_records = 0
        self.projected_mm_budget_configured_ticks = 0
        self.projected_mm_budget_applied_ticks = 0
        self.projected_mm_budget_selected_counts: Counter = Counter()
        self.projected_mm_budget_suppressed_counts: Counter = Counter()
        self.projected_mm_budget_selected_venue_count_stats = OnlineStats()
        self.projected_mm_budget_suppressed_venue_count_stats = OnlineStats()
        self.projected_mm_budget_all5_selected_ticks = 0
        self.projected_mm_budget_last_limits: dict[str, float | None] = {}
        self.projected_mm_budget_q_global_after_stats = OnlineStats()
        self.projected_mm_budget_q_gross_after_stats = OnlineStats()
        self.projected_mm_budget_q_max_abs_venue_after_stats = OnlineStats()
        self.emergency_residual_fallback_total = 0
        self.emergency_residual_fallback_status_counts: Counter = Counter()
        self.emergency_residual_fallback_reason_counts: Counter = Counter()
        self.emergency_residual_fallback_class_counts: Counter = Counter()
        self.emergency_residual_fallback_venue_counts: Counter = Counter()
        self.emergency_residual_fallback_first_tick: int | None = None
        self.emergency_residual_fallback_last_tick: int | None = None
        self.aster_inventory_brake_fee_guard_enabled = False
        self.aster_inventory_brake_fee_guard_skipped_orders = 0
        self.aster_inventory_brake_fee_guard_skipped_base_tao = 0.0
        self.aster_inventory_brake_fee_guard_skipped_notional_usd = 0.0
        self.aster_residual_markout_guard_total = 0
        self.aster_residual_markout_guard_decision_counts: Counter = Counter()
        self.aster_residual_markout_guard_reason_counts: Counter = Counter()
        self.aster_residual_markout_guard_allowed_orders = 0
        self.aster_residual_markout_guard_suppressed_orders = 0
        self.aster_residual_markout_guard_age_stats = OnlineStats()
        self.aster_residual_markout_guard_adverse_stats = OnlineStats()
        self.aster_residual_markout_guard_unrealised_stats = OnlineStats()
        self.aster_residual_markout_guard_cleanup_fee_stats = OnlineStats()
        self.aster_residual_markout_guard_refresh_attempts = 0
        self.aster_residual_markout_guard_refresh_outcome_counts: Counter = Counter()
        self.aster_residual_markout_guard_refresh_suppressed_counts: Counter = Counter()
        self.aster_residual_markout_guard_refresh_latency_stats = OnlineStats()
        self.aster_residual_markout_guard_fresh_account_age_stats = OnlineStats()
        self.aster_residual_markout_guard_last: dict[str, Any] = {}
        self.canary_breach_response_total = 0
        self.canary_breach_active_ticks = 0
        self.canary_breach_candidate_target_ticks = 0
        self.canary_breach_observation_active_ticks = 0
        self.canary_breach_observation_uncovered_target_ticks = 0
        self.canary_breach_zero_target_hold_ticks = 0
        self.canary_breach_zero_target_hold_run = 0
        self.canary_breach_zero_target_hold_max_run = 0
        self.canary_breach_zero_target_hold_windows = 0
        self.canary_breach_zero_target_hold_first_tick: int | None = None
        self.canary_breach_zero_target_hold_last_tick: int | None = None
        self.canary_breach_response_mode_counts: Counter = Counter()
        self.mm_fill_attributed_count = 0
        self.mm_fill_unattributed_count = 0
        self.mm_fill_attributed_by_venue: list[int] = [0] * NUM_VENUES
        self.mm_gross_before_fee_attributed_by_venue: list[float] = [0.0] * NUM_VENUES
        self.mm_fee_attributed_by_venue: list[float] = [0.0] * NUM_VENUES
        self.mm_realised_net_attributed_by_venue: list[float] = [0.0] * NUM_VENUES
        self.mm_markout_short_attributed_by_venue: list[float] = [0.0] * NUM_VENUES
        self.mm_fill_decision_markout_short_stats = OnlineStats()
        self.mm_fill_decision_realised_pnl_stats = OnlineStats()
        self.mm_decision_fill_counts: Counter = Counter()
        self.decision_contributions: dict[str, DecisionContribution] = {}
        self.mm_gross_before_fee_attributed_usd = 0.0
        self.mm_fee_attributed_usd = 0.0
        self.mm_realised_net_attributed_usd = 0.0
        self.mm_realised_net_unattributed_usd = 0.0
        self.mm_fee_unattributed_usd = 0.0
        self.hedge_fill_fee_attributed_usd = 0.0
        self.hedge_fill_fee_unattributed_usd = 0.0
        self.hedge_exec_cost_model_attributed_usd = 0.0
        self.hedge_exec_cost_model_unattributed_usd = 0.0
        self.hedge_total_cost_model_attributed_usd = 0.0
        self.hedge_total_cost_model_unattributed_usd = 0.0

        # --- Dimension 14: Anomalies ---
        self.anomalies = AnomalyCollector()

        # --- Dimension 15: Funding ---
        self.venue_funding_rate_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]
        self.venue_funding_status_counts: list[Counter] = [Counter() for _ in range(NUM_VENUES)]
        self.venue_funding_age_stats: list[OnlineStats] = [OnlineStats() for _ in range(NUM_VENUES)]

        # --- PnL ---
        self.pnl_total_stats = OnlineStats()
        self.pnl_realised_stats = OnlineStats()
        self.final_pnl_total: float | None = None
        self.final_pnl_realised: float | None = None
        self.final_pnl_unrealised: float | None = None
        self.final_q_global_tao: float | None = None
        self.pnl_reconstruction_mismatch_stats = OnlineStats()
        self.flat_inventory_large_unrealised_ticks = 0

        # --- Correlated staleness detection ---
        self.multi_stale_ticks = 0  # ticks where 2+ venues are stale simultaneously
        self.execution_mode_counts: Counter = Counter()
        self.soft_governor_ticks = 0
        self.soft_governor_reason_counts: Counter = Counter()
        self.soft_governor_blocked_ticks: list[int] = [0] * NUM_VENUES
        self.order_status_action_counts: list[Counter] = [Counter() for _ in range(NUM_VENUES)]
        self.venue_fill_counts: list[int] = [0] * NUM_VENUES
        self.venue_fill_size_base: list[float] = [0.0] * NUM_VENUES
        self.venue_mm_fill_counts: list[int] = [0] * NUM_VENUES
        self.venue_mm_fill_size_base: list[float] = [0.0] * NUM_VENUES
        self.venue_hedge_fill_counts: list[int] = [0] * NUM_VENUES
        self.venue_hedge_fill_size_base: list[float] = [0.0] * NUM_VENUES
        self.drift_kind_counts: Counter = Counter()
        self.drift_kind_by_venue: list[Counter] = [Counter() for _ in range(NUM_VENUES)]

    def note_venue(self, venue_index: int | None, venue_id: str | None) -> None:
        if venue_index is None or venue_index < 0 or venue_index >= NUM_VENUES:
            return
        if not isinstance(venue_id, str) or not venue_id:
            return
        self.venue_names[venue_index] = venue_id
        self.seen_venue_indices.add(venue_index)

    def venue_name(self, venue_index: int) -> str:
        if 0 <= venue_index < len(self.venue_names):
            return self.venue_names[venue_index]
        return f"venue_{venue_index}"

    def report_venue_indices(self) -> list[int]:
        status_indices = [
            i for i, counts in enumerate(self.venue_status_counts) if sum(counts.values()) > 0
        ]
        if status_indices and len(status_indices) < NUM_VENUES:
            return status_indices
        if len(self.seen_venue_indices) > 1:
            return sorted(self.seen_venue_indices)
        return list(range(NUM_VENUES))

    def process(self, rec: dict) -> None:
        tick = safe_int(rec.get("t"))
        if tick is None:
            return

        self.tick_count += 1
        exec_mode = rec.get("execution_mode")
        if isinstance(exec_mode, str):
            self.execution_mode_counts[exec_mode] += 1
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

        inventory_attribution = rec.get("inventory_attribution")
        if isinstance(inventory_attribution, list):
            for item in inventory_attribution:
                if not isinstance(item, dict):
                    continue
                self.note_venue(
                    safe_int(item.get("venue_index")),
                    item.get("venue_id") if isinstance(item.get("venue_id"), str) else None,
                )

        # === Dimension 1: Venue Health ===
        venue_status = rec.get("venue_status", [])
        venue_age = rec.get("venue_age_ms", [])
        venue_age_event = rec.get("venue_age_event_ms", [])
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

            age_event_f = safe_float(
                venue_age_event[i]
                if isinstance(venue_age_event, list) and i < len(venue_age_event)
                else None
            )
            if age_event_f is not None and age_event_f >= 0:
                self.venue_age_event_stats[i].push(age_event_f)
                self.venue_age_event_windows[i].push(age_event_f)

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
                            f"{self.venue_name(i)} mid jumped {delta_bps:.1f}bps",
                            {"venue": self.venue_name(i), "delta_bps": round(delta_bps, 2), "mid": mid, "prev_mid": self.venue_prev_mid[i]})
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
                side = str(ql.get("side") or "unknown")
                quote_state = str(ql.get("quote_state") or "unknown")
                gate_key = (vi, side)
                self.quote_gate_counts[gate_key][quote_state] += 1
                if quote_state != "active":
                    reason = str(ql.get("suppression_reason") or "unspecified")
                    self.quote_gate_suppression_counts[gate_key][reason] += 1
                engine_reason = str(ql.get("engine_terminal_reason") or "unspecified")
                self.quote_gate_engine_reason_counts[gate_key][engine_reason] += 1

                edge_threshold = safe_float(ql.get("edge_threshold"))
                edge_threshold_base = safe_float(ql.get("edge_threshold_base"))
                hedge_cost_floor = safe_float(ql.get("hedge_cost_edge_floor"))
                if edge_threshold is not None:
                    self.quote_gate_edge_threshold_stats[gate_key].push(edge_threshold)
                if edge_threshold_base is not None:
                    self.quote_gate_edge_threshold_base_stats[gate_key].push(edge_threshold_base)
                if hedge_cost_floor is not None:
                    self.quote_gate_hedge_cost_floor_stats[gate_key].push(hedge_cost_floor)

                edge = safe_float(ql.get("edge_local"))
                if edge is not None:
                    self.venue_edge_stats[vi].push(edge)
                    if quote_state == "active":
                        self.quote_gate_active_edge_stats[gate_key].push(edge)

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
            for hedge in hedges:
                if not isinstance(hedge, dict):
                    continue
                self.note_venue(
                    safe_int(hedge.get("venue_index")),
                    hedge.get("venue_id") if isinstance(hedge.get("venue_id"), str) else None,
                )
                self.hedge_record_total += 1
                source_decision_id = hedge.get("source_decision_id")
                intended_size = abs(
                    safe_float(hedge.get("filled_size"))
                    or safe_float(hedge.get("intended_size"))
                    or safe_float(hedge.get("delta_h_v"))
                    or 0.0
                )
                cost_components = hedge.get("cost_components")
                hedge_exec_cost_model = None
                hedge_total_cost_model = None
                if isinstance(cost_components, dict):
                    exec_cost = safe_float(cost_components.get("exec_cost"))
                    total_cost = safe_float(cost_components.get("total_cost"))
                    if exec_cost is not None:
                        hedge_exec_cost_model = exec_cost * intended_size
                    if total_cost is not None:
                        hedge_total_cost_model = total_cost * intended_size
                if isinstance(source_decision_id, str) and source_decision_id:
                    self.hedge_source_attributed_count += 1
                    decision = self.decision_contributions.setdefault(
                        source_decision_id,
                        DecisionContribution(decision_id=source_decision_id),
                    )
                    if decision.mm_venue_index is None:
                        source_venue_index = safe_int(hedge.get("source_fill_venue_index"))
                        if source_venue_index is not None:
                            decision.mm_venue_index = source_venue_index
                    if decision.mm_venue_id is None:
                        source_venue_id = hedge.get("source_fill_venue_id")
                        hedge_venue_id = hedge.get("venue_id")
                        if isinstance(source_venue_id, str) and source_venue_id:
                            decision.mm_venue_id = source_venue_id
                        elif isinstance(hedge_venue_id, str) and hedge_venue_id:
                            decision.mm_venue_id = hedge_venue_id
                    decision.hedge_record_count += 1
                    if hedge_exec_cost_model is not None:
                        decision.hedge_exec_cost_model_usd += hedge_exec_cost_model
                        self.hedge_exec_cost_model_attributed_usd += hedge_exec_cost_model
                    if hedge_total_cost_model is not None:
                        decision.hedge_total_cost_model_usd += hedge_total_cost_model
                        self.hedge_total_cost_model_attributed_usd += hedge_total_cost_model
                else:
                    if hedge_exec_cost_model is not None:
                        self.hedge_exec_cost_model_unattributed_usd += hedge_exec_cost_model
                    if hedge_total_cost_model is not None:
                        self.hedge_total_cost_model_unattributed_usd += hedge_total_cost_model
                source_fill_age_ms = safe_float(hedge.get("source_fill_age_ms"))
                if source_fill_age_ms is not None:
                    self.hedge_source_fill_age_stats.push(source_fill_age_ms)

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
                    kind = d.get("kind", "unknown")
                    if isinstance(kind, str):
                        self.drift_kind_counts[kind] += 1
                    if vi is not None and 0 <= vi < NUM_VENUES:
                        self.drift_by_venue[vi] += 1
                        if isinstance(kind, str):
                            self.drift_kind_by_venue[vi][kind] += 1

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
            kill_reason = rec.get("kill_reason")
            if isinstance(kill_reason, str):
                self.kill_reason_counts[kill_reason] += 1
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
            self.final_q_global_tao = qg

        inv_soft = rec.get("inventory_soft_governor")
        if isinstance(inv_soft, dict) and inv_soft.get("triggered"):
            self.soft_governor_ticks += 1
            for reason in inv_soft.get("global_reasons", []):
                if isinstance(reason, str):
                    self.soft_governor_reason_counts[reason] += 1
            for blocked in inv_soft.get("blocked_venues", []):
                if isinstance(blocked, dict):
                    vi = safe_int(blocked.get("venue_index"))
                    if vi is not None and 0 <= vi < NUM_VENUES:
                        self.soft_governor_blocked_ticks[vi] += 1

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
                    self.note_venue(
                        vi,
                        o.get("venue_id") if isinstance(o.get("venue_id"), str) else None,
                    )
                    if vi is not None and 0 <= vi < NUM_VENUES:
                        self.orders_per_venue[vi] += 1
                        status = o.get("status", "unknown")
                        if isinstance(status, str):
                            self.order_status_action_counts[vi][f"{action}_{status}"] += 1

        mm_order_management = rec.get("mm_order_management")
        if isinstance(mm_order_management, dict):
            for key, stats_acc in [
                ("keep_count", self.mm_keep_count_stats),
                ("replace_count", self.mm_replace_count_stats),
                ("place_count", self.mm_place_count_stats),
                ("cancel_count", self.mm_cancel_count_stats),
            ]:
                value = safe_int(mm_order_management.get(key))
                if value is not None:
                    stats_acc.push(float(value))

            keep_count = safe_int(mm_order_management.get("keep_count"))
            replace_count = safe_int(mm_order_management.get("replace_count"))
            place_count = safe_int(mm_order_management.get("place_count"))
            cancel_count = safe_int(mm_order_management.get("cancel_count"))
            if keep_count is not None:
                self.mm_keep_total += keep_count
            if replace_count is not None:
                self.mm_replace_total += replace_count
            if place_count is not None:
                self.mm_place_total += place_count
            if cancel_count is not None:
                self.mm_cancel_total += cancel_count

            for raw_counts, target in [
                (mm_order_management.get("keep_by_reason"), self.mm_keep_reason_counts),
                (mm_order_management.get("replace_by_reason"), self.mm_replace_reason_counts),
                (
                    mm_order_management.get("keep_by_utility_tier"),
                    self.mm_keep_utility_tier_counts,
                ),
                (
                    mm_order_management.get("replace_by_utility_tier"),
                    self.mm_replace_utility_tier_counts,
                ),
                (
                    mm_order_management.get("keep_by_venue_role"),
                    self.mm_keep_venue_role_counts,
                ),
                (
                    mm_order_management.get("replace_by_venue_role"),
                    self.mm_replace_venue_role_counts,
                ),
            ]:
                if isinstance(raw_counts, dict):
                    for raw_key, raw_count in raw_counts.items():
                        count = safe_int(raw_count)
                        if isinstance(raw_key, str) and count is not None:
                            target[raw_key] += count

            replace_decisions = mm_order_management.get("replace_decisions")
            if isinstance(replace_decisions, list):
                for decision in replace_decisions:
                    if not isinstance(decision, dict):
                        continue
                    venue_index = safe_int(decision.get("venue_index"))
                    outcome = decision.get("outcome")
                    if (
                        venue_index is not None
                        and 0 <= venue_index < NUM_VENUES
                        and isinstance(outcome, str)
                    ):
                        self.mm_decisions_by_venue[venue_index][outcome] += 1
                        if decision.get("inventory_reducing") is True:
                            self.mm_inventory_reducing_counts[outcome] += 1

            decision_records = mm_order_management.get("decision_records")
            if isinstance(decision_records, list):
                for decision in decision_records:
                    if not isinstance(decision, dict):
                        continue
                    self.mm_decision_record_total += 1
                    client_order_id = decision.get("client_order_id")
                    if isinstance(client_order_id, str) and client_order_id:
                        self.mm_decision_order_link_total += 1

            replace_visibility_records = mm_order_management.get(
                "supported_replace_visibility_records"
            )
            if isinstance(replace_visibility_records, list):
                for decision in replace_visibility_records:
                    if not isinstance(decision, dict):
                        continue
                    venue_index = safe_int(decision.get("venue_index"))
                    if venue_index is None or not (0 <= venue_index < NUM_VENUES):
                        continue
                    self.supported_replace_visibility_total += 1
                    action = decision.get("action")
                    if isinstance(action, str):
                        self.supported_replace_action_counts[venue_index][action] += 1
                    blocked_by = decision.get("blocked_by")
                    if isinstance(blocked_by, str):
                        self.supported_replace_visibility_miss_counts[venue_index] += 1
                        self.supported_replace_blocker_counts[venue_index][blocked_by] += 1
                    desired_present = decision.get("desired_present") is True
                    current_present = decision.get("current_present") is True
                    if desired_present and current_present:
                        self.supported_replace_opportunity_counts[venue_index] += 1
                    if decision.get("current_source") == "gap_grace":
                        self.supported_replace_gap_grace_counts[venue_index] += 1

        projected_mm_budget = rec.get("projected_mm_budget")
        if isinstance(projected_mm_budget, dict):
            self.projected_mm_budget_total_records += 1
            if projected_mm_budget.get("configured"):
                self.projected_mm_budget_configured_ticks += 1
            if projected_mm_budget.get("applied"):
                self.projected_mm_budget_applied_ticks += 1

            selected_venues = {
                venue
                for venue in projected_mm_budget.get("selected_venues", []) or []
                if isinstance(venue, str) and venue
            }
            suppressed_venues = {
                venue
                for venue in projected_mm_budget.get("suppressed_venues", []) or []
                if isinstance(venue, str) and venue
            }
            for venue in selected_venues:
                self.projected_mm_budget_selected_counts[venue] += 1
            for venue in suppressed_venues:
                self.projected_mm_budget_suppressed_counts[venue] += 1
            self.projected_mm_budget_selected_venue_count_stats.push(float(len(selected_venues)))
            self.projected_mm_budget_suppressed_venue_count_stats.push(float(len(suppressed_venues)))
            if all(venue in selected_venues for venue in VENUE_NAMES):
                self.projected_mm_budget_all5_selected_ticks += 1

            for key in ("net_limit_tao", "gross_limit_tao", "venue_limit_tao"):
                if key in projected_mm_budget:
                    self.projected_mm_budget_last_limits[key] = safe_float(projected_mm_budget.get(key))
            for key, stats_acc in [
                ("projected_q_global_after_tao", self.projected_mm_budget_q_global_after_stats),
                ("projected_q_gross_after_tao", self.projected_mm_budget_q_gross_after_stats),
                (
                    "projected_q_max_abs_venue_after_tao",
                    self.projected_mm_budget_q_max_abs_venue_after_stats,
                ),
            ]:
                value = safe_float(projected_mm_budget.get(key))
                if value is not None:
                    stats_acc.push(value)

        emergency_residual_fallback = rec.get("emergency_residual_fallback")
        if isinstance(emergency_residual_fallback, dict):
            if bool(emergency_residual_fallback.get("aster_inventory_brake_fee_guard_enabled")):
                self.aster_inventory_brake_fee_guard_enabled = True
            skipped_orders = safe_int(
                emergency_residual_fallback.get("aster_inventory_brake_fee_guard_skipped_orders")
            )
            if skipped_orders is not None:
                self.aster_inventory_brake_fee_guard_skipped_orders = max(
                    self.aster_inventory_brake_fee_guard_skipped_orders,
                    skipped_orders,
                )
            skipped_base = safe_float(
                emergency_residual_fallback.get("aster_inventory_brake_fee_guard_skipped_base_tao")
            )
            if skipped_base is not None:
                self.aster_inventory_brake_fee_guard_skipped_base_tao = max(
                    self.aster_inventory_brake_fee_guard_skipped_base_tao,
                    skipped_base,
                )
            skipped_notional = safe_float(
                emergency_residual_fallback.get("aster_inventory_brake_fee_guard_skipped_notional_usd")
            )
            if skipped_notional is not None:
                self.aster_inventory_brake_fee_guard_skipped_notional_usd = max(
                    self.aster_inventory_brake_fee_guard_skipped_notional_usd,
                    skipped_notional,
                )
            fallback_records = emergency_residual_fallback.get("records")
            if isinstance(fallback_records, list):
                for record in fallback_records:
                    if not isinstance(record, dict):
                        continue
                    self.emergency_residual_fallback_total += 1
                    if self.emergency_residual_fallback_first_tick is None:
                        self.emergency_residual_fallback_first_tick = tick
                    self.emergency_residual_fallback_last_tick = tick
                    venue_index = safe_int(record.get("venue_index"))
                    venue_id = record.get("venue_id") if isinstance(record.get("venue_id"), str) else None
                    self.note_venue(venue_index, venue_id)
                    if isinstance(venue_id, str) and venue_id:
                        self.emergency_residual_fallback_venue_counts[venue_id] += 1
                    elif venue_index is not None and 0 <= venue_index < NUM_VENUES:
                        self.emergency_residual_fallback_venue_counts[self.venue_name(venue_index)] += 1
                    status = record.get("status")
                    reason = record.get("reason")
                    request_class = record.get("class")
                    if isinstance(status, str) and status:
                        self.emergency_residual_fallback_status_counts[status] += 1
                    if isinstance(reason, str) and reason:
                        self.emergency_residual_fallback_reason_counts[reason] += 1
                    if isinstance(request_class, str) and request_class:
                        self.emergency_residual_fallback_class_counts[request_class] += 1
            aster_markout_guard = emergency_residual_fallback.get("aster_residual_markout_guard")
            if isinstance(aster_markout_guard, dict) and aster_markout_guard.get("enabled"):
                self.aster_residual_markout_guard_total += 1
                self.aster_residual_markout_guard_last = dict(aster_markout_guard)
                decision = aster_markout_guard.get("decision")
                reason = aster_markout_guard.get("reason")
                if isinstance(decision, str) and decision:
                    self.aster_residual_markout_guard_decision_counts[decision] += 1
                if isinstance(reason, str) and reason:
                    self.aster_residual_markout_guard_reason_counts[reason] += 1
                self.aster_residual_markout_guard_allowed_orders += (
                    safe_int(aster_markout_guard.get("allowed_orders")) or 0
                )
                self.aster_residual_markout_guard_suppressed_orders += (
                    safe_int(aster_markout_guard.get("suppressed_orders")) or 0
                )
                guard_stats = [
                    (
                        "residual_age_ms",
                        self.aster_residual_markout_guard_age_stats,
                    ),
                    (
                        "adverse_markout_usd",
                        self.aster_residual_markout_guard_adverse_stats,
                    ),
                    (
                        "residual_unrealised_usd",
                        self.aster_residual_markout_guard_unrealised_stats,
                    ),
                    (
                        "cleanup_fee_estimate_usd",
                        self.aster_residual_markout_guard_cleanup_fee_stats,
                    ),
                ]
                for key, stats_acc in guard_stats:
                    value = safe_float(aster_markout_guard.get(key))
                    if value is not None:
                        stats_acc.push(value)
                if bool(aster_markout_guard.get("refresh_attempted")):
                    self.aster_residual_markout_guard_refresh_attempts += 1
                refresh_outcome = aster_markout_guard.get("refresh_outcome")
                if isinstance(refresh_outcome, str) and refresh_outcome:
                    self.aster_residual_markout_guard_refresh_outcome_counts[refresh_outcome] += 1
                refresh_suppressed = aster_markout_guard.get("refresh_suppressed_reason")
                if isinstance(refresh_suppressed, str) and refresh_suppressed:
                    self.aster_residual_markout_guard_refresh_suppressed_counts[refresh_suppressed] += 1
                refresh_latency = safe_float(aster_markout_guard.get("refresh_latency_ms"))
                if refresh_latency is not None:
                    self.aster_residual_markout_guard_refresh_latency_stats.push(refresh_latency)
                fresh_age = safe_float(aster_markout_guard.get("fresh_account_age_ms"))
                if fresh_age is not None:
                    self.aster_residual_markout_guard_fresh_account_age_stats.push(fresh_age)

        zero_target_hold_this_tick = False
        canary_breach_response = rec.get("canary_breach_response")
        if isinstance(canary_breach_response, dict):
            self.canary_breach_response_total += 1
            if canary_breach_response.get("active"):
                self.canary_breach_active_ticks += 1
            candidate_target_venues = [
                venue
                for venue in canary_breach_response.get("candidate_target_venues", []) or []
                if isinstance(venue, str) and venue
            ]
            if candidate_target_venues:
                self.canary_breach_candidate_target_ticks += 1
            if canary_breach_response.get("observation_active"):
                self.canary_breach_observation_active_ticks += 1
            observation_covers = canary_breach_response.get(
                "observation_covers_candidate_targets"
            )
            if (
                canary_breach_response.get("observation_active")
                and candidate_target_venues
                and observation_covers is False
            ):
                self.canary_breach_observation_uncovered_target_ticks += 1
            response_mode = canary_breach_response.get("response_mode")
            if isinstance(response_mode, str) and response_mode:
                self.canary_breach_response_mode_counts[response_mode] += 1
            zero_target_hold_this_tick = (
                canary_breach_response.get("zero_target_hold_this_tick") is True
            )
            if zero_target_hold_this_tick:
                self.canary_breach_zero_target_hold_ticks += 1
                if self.canary_breach_zero_target_hold_first_tick is None:
                    self.canary_breach_zero_target_hold_first_tick = tick
                self.canary_breach_zero_target_hold_last_tick = tick
                if self.canary_breach_zero_target_hold_run == 0:
                    self.canary_breach_zero_target_hold_windows += 1
                self.canary_breach_zero_target_hold_run += 1
                if (
                    self.canary_breach_zero_target_hold_run
                    > self.canary_breach_zero_target_hold_max_run
                ):
                    self.canary_breach_zero_target_hold_max_run = (
                        self.canary_breach_zero_target_hold_run
                    )
                self.anomalies.add(
                    tick,
                    "canary_zero_target_hold",
                    "Warning",
                    "Canary breach response stayed active without dispatch while candidate targets remained exposed",
                    {
                        "candidate_target_venues": candidate_target_venues,
                        "response_mode": response_mode,
                    },
                )
        if not zero_target_hold_this_tick:
            self.canary_breach_zero_target_hold_run = 0

        # === Dimension 14: Anomaly Detection ===
        # Age spikes
        for i in range(NUM_VENUES):
            age_f = safe_float(venue_age[i] if isinstance(venue_age, list) and i < len(venue_age) else None)
            if age_f is not None and age_f > 3000:
                threshold = STALE_MS.get(self.venue_name(i), 1000)
                if age_f > threshold * 3:
                    self.anomalies.add(tick, "age_spike", "Warning",
                        f"{self.venue_name(i)} age={age_f:.0f}ms (>{threshold*3}ms)",
                        {"venue": self.venue_name(i), "age_ms": age_f})

        # Spread blowout detection (relative to running median)
        for i in range(NUM_VENUES):
            spread_f = safe_float(venue_spread[i] if isinstance(venue_spread, list) and i < len(venue_spread) else None)
            if spread_f is not None and self.venue_spread_stats[i].n > 100:
                median_spread = self.venue_spread_stats[i].percentile(50)
                if median_spread > 0 and spread_f > median_spread * 10:
                    self.anomalies.add(tick, "spread_blowout", "Warning",
                        f"{self.venue_name(i)} spread={spread_f:.4f} ({spread_f/median_spread:.1f}x median)",
                        {"venue": self.venue_name(i), "spread": spread_f, "median": median_spread})

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
        pnl_u = safe_float(rec.get("pnl_unrealised"))
        if pnl_t is not None:
            self.pnl_total_stats.push(pnl_t)
            self.final_pnl_total = pnl_t
        if pnl_r is not None:
            self.pnl_realised_stats.push(pnl_r)
            self.final_pnl_realised = pnl_r
        if pnl_u is not None:
            self.final_pnl_unrealised = pnl_u
        if pnl_t is not None and pnl_r is not None and pnl_u is not None:
            mismatch = abs(pnl_t - (pnl_r + pnl_u))
            self.pnl_reconstruction_mismatch_stats.push(mismatch)
        if (
            qg is not None
            and pnl_u is not None
            and abs(qg) <= PNL_BASELINE_FLAT_Q_TAO
            and abs(pnl_u) > PNL_BASELINE_LARGE_UNREALISED_USD
        ):
            self.flat_inventory_large_unrealised_ticks += 1

        fills = rec.get("fills", [])
        if isinstance(fills, list):
            for fill in fills:
                if isinstance(fill, dict):
                    vi = safe_int(fill.get("venue_index"))
                    self.note_venue(
                        vi,
                        fill.get("venue_id") if isinstance(fill.get("venue_id"), str) else None,
                    )
                    price = safe_float(fill.get("price")) or 0.0
                    size = abs(safe_float(fill.get("size")) or 0.0)
                    fee_bps = safe_float(fill.get("fee_bps")) or 0.0
                    fee_usd = price * size * (fee_bps / 10_000.0)
                    if vi is not None and 0 <= vi < NUM_VENUES:
                        self.venue_fill_counts[vi] += 1
                        if size:
                            self.venue_fill_size_base[vi] += size
                    if fill.get("purpose") == "Mm":
                        if vi is not None and 0 <= vi < NUM_VENUES:
                            self.venue_mm_fill_counts[vi] += 1
                            if size:
                                self.venue_mm_fill_size_base[vi] += size
                        decision_id = fill.get("decision_id")
                        realised_pnl = safe_float(fill.get("realised_pnl_usd")) or 0.0
                        if isinstance(decision_id, str) and decision_id:
                            self.mm_fill_attributed_count += 1
                            self.mm_decision_fill_counts[decision_id] += 1
                            decision = self.decision_contributions.setdefault(
                                decision_id,
                                DecisionContribution(decision_id=decision_id),
                            )
                            decision.mm_fill_count += 1
                            decision.mm_fill_base += size
                            decision.mm_fill_notional_usd += price * size
                            decision.mm_fee_usd += fee_usd
                            decision.mm_realised_net_usd += realised_pnl
                            decision.mm_gross_before_fee_usd += realised_pnl + fee_usd
                            if decision.mm_venue_index is None:
                                decision.mm_venue_index = vi
                            if decision.mm_venue_id is None:
                                venue_id = fill.get("venue_id")
                                if isinstance(venue_id, str) and venue_id:
                                    decision.mm_venue_id = venue_id
                            self.mm_gross_before_fee_attributed_usd += realised_pnl + fee_usd
                            self.mm_fee_attributed_usd += fee_usd
                            self.mm_realised_net_attributed_usd += realised_pnl
                            if vi is not None and 0 <= vi < NUM_VENUES:
                                self.mm_fill_attributed_by_venue[vi] += 1
                                self.mm_gross_before_fee_attributed_by_venue[vi] += realised_pnl + fee_usd
                                self.mm_fee_attributed_by_venue[vi] += fee_usd
                                self.mm_realised_net_attributed_by_venue[vi] += realised_pnl
                            markout_short = safe_float(fill.get("markout_pnl_short"))
                            if markout_short is not None:
                                self.mm_fill_decision_markout_short_stats.push(markout_short)
                                decision.mm_markout_short_usd += markout_short
                                if vi is not None and 0 <= vi < NUM_VENUES:
                                    self.mm_markout_short_attributed_by_venue[vi] += markout_short
                            if safe_float(fill.get("realised_pnl_usd")) is not None:
                                self.mm_fill_decision_realised_pnl_stats.push(realised_pnl)
                        else:
                            self.mm_fill_unattributed_count += 1
                            self.mm_realised_net_unattributed_usd += realised_pnl
                            self.mm_fee_unattributed_usd += fee_usd
                    elif fill.get("purpose") == "Hedge":
                        if vi is not None and 0 <= vi < NUM_VENUES:
                            self.venue_hedge_fill_counts[vi] += 1
                            if size:
                                self.venue_hedge_fill_size_base[vi] += size
                        source_decision_id = fill.get("source_decision_id")
                        if isinstance(source_decision_id, str) and source_decision_id:
                            self.hedge_fill_attributed_count += 1
                            decision = self.decision_contributions.setdefault(
                                source_decision_id,
                                DecisionContribution(decision_id=source_decision_id),
                            )
                            if decision.mm_venue_index is None:
                                source_venue_index = safe_int(fill.get("source_fill_venue_index"))
                                if source_venue_index is not None:
                                    decision.mm_venue_index = source_venue_index
                            if decision.mm_venue_id is None:
                                source_venue_id = fill.get("source_fill_venue_id")
                                fill_venue_id = fill.get("venue_id")
                                if isinstance(source_venue_id, str) and source_venue_id:
                                    decision.mm_venue_id = source_venue_id
                                elif isinstance(fill_venue_id, str) and fill_venue_id:
                                    decision.mm_venue_id = fill_venue_id
                            decision.hedge_fill_count += 1
                            decision.hedge_fill_notional_usd += price * size
                            decision.hedge_fill_fee_usd += fee_usd
                            self.hedge_fill_fee_attributed_usd += fee_usd
                        else:
                            self.hedge_fill_unattributed_count += 1
                            self.hedge_fill_fee_unattributed_usd += fee_usd


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


def measurement_validity_lines(
    acc: TelemetryAccumulator,
    validation_profile: str,
) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    failures: list[str] = []

    lines.append(f"  Validation profile: {validation_profile}")
    lines.append(
        "  Cross-venue dispersion max: "
        f"{fmt_f(acc.cross_venue_dispersion._max if acc.cross_venue_dispersion.n > 0 else None, 4)} USD"
    )
    lines.append(
        "  PnL identity max abs(total-realised-unrealised): "
        f"{fmt_f(acc.pnl_reconstruction_mismatch_stats._max if acc.pnl_reconstruction_mismatch_stats.n > 0 else None, 4)} USD"
    )
    lines.append(f"  Final q_global_tao: {fmt_f(acc.final_q_global_tao, 6)}")
    lines.append(f"  Final pnl_total: {fmt_f(acc.final_pnl_total, 4)} USD")
    lines.append(f"  Final pnl_realised: {fmt_f(acc.final_pnl_realised, 4)} USD")
    lines.append(f"  Final pnl_unrealised: {fmt_f(acc.final_pnl_unrealised, 4)} USD")
    lines.append(
        "  Flat-inventory large-unrealised ticks: "
        f"{acc.flat_inventory_large_unrealised_ticks}"
    )
    if acc.kill_reason_counts:
        lines.append(
            "  Kill reasons: "
            + ", ".join(
                f"{reason}={count}" for reason, count in acc.kill_reason_counts.most_common()
            )
        )
    else:
        lines.append("  Kill reasons: none")
    lines.append(
        "  MM decision totals: "
        f"place={acc.mm_place_total} keep={acc.mm_keep_total} replace={acc.mm_replace_total} "
        f"cancel={acc.mm_cancel_total}"
    )

    if validation_profile == "pnl-baseline":
        if (
            acc.cross_venue_dispersion.n > 0
            and acc.cross_venue_dispersion._max > PNL_BASELINE_MAX_DISPERSION_USD
        ):
            failures.append(
                "cross-venue dispersion exceeded "
                f"{PNL_BASELINE_MAX_DISPERSION_USD:.2f} USD"
            )
        if acc.pnl_reconstruction_mismatch_stats.n > 0 and (
            acc.pnl_reconstruction_mismatch_stats._max > PNL_BASELINE_RECON_MISMATCH_TOL_USD
        ):
            failures.append(
                "PnL identity mismatch exceeded "
                f"{PNL_BASELINE_RECON_MISMATCH_TOL_USD:.2f} USD"
            )
        if acc.kill_reason_counts.get("BasisHardBreach", 0) > 0:
            failures.append("basis hard breach kill observed")
        if (
            acc.final_q_global_tao is not None
            and abs(acc.final_q_global_tao) <= PNL_BASELINE_FLAT_Q_TAO
            and acc.final_pnl_unrealised is not None
            and abs(acc.final_pnl_unrealised) > PNL_BASELINE_LARGE_UNREALISED_USD
        ):
            failures.append("large unrealised PnL observed while inventory stayed near flat")
        for label, value in [
            ("final pnl_total", acc.final_pnl_total),
            ("final pnl_realised", acc.final_pnl_realised),
            ("final pnl_unrealised", acc.final_pnl_unrealised),
        ]:
            if value is not None and abs(value) > PNL_BASELINE_FINAL_PNL_TOL_USD:
                failures.append(
                    f"{label} exceeded {PNL_BASELINE_FINAL_PNL_TOL_USD:.2f} USD"
                )
    elif validation_profile == "mm-churn-probe":
        if acc.mm_place_total <= 0:
            failures.append("no MM place decisions observed")
        if (acc.mm_keep_total + acc.mm_replace_total) <= 0:
            failures.append("no MM keep/replace decisions observed")
        if acc.kill_reason_counts.get("BasisHardBreach", 0) > 0:
            failures.append("basis hard breach kill observed")

    return lines, failures


def generate_report(acc: TelemetryAccumulator, validation_profile: str = "none") -> str:
    lines: list[str] = []
    report_venue_indices = acc.report_venue_indices()

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
    lines.append(f"  Segment UTC: {ts_iso(acc.first_ts_ms)} -> {ts_iso(acc.last_ts_ms)}")
    lines.append(f"  Venues: {', '.join(acc.venue_name(i) for i in report_venue_indices)}")
    if acc.execution_mode_counts:
        modes = ", ".join(
            f"{mode}={count}" for mode, count in sorted(acc.execution_mode_counts.items())
        )
        lines.append(f"  Execution modes: {modes}")
    lines.append(f"  Anomalies detected: {len(acc.anomalies.items)} "
                 f"(Critical={len(acc.anomalies.by_severity('Critical'))}, "
                 f"Warning={len(acc.anomalies.by_severity('Warning'))}, "
                 f"Info={len(acc.anomalies.by_severity('Info'))})")

    if validation_profile != "none":
        section("MEASUREMENT VALIDITY")
        validity_lines, validity_failures = measurement_validity_lines(
            acc,
            validation_profile,
        )
        lines.extend(validity_lines)
        if validity_failures:
            lines.append("  Status: FAIL")
            lines.append("  Failure reasons:")
            for failure in validity_failures:
                lines.append(f"    - {failure}")
        else:
            lines.append("  Status: PASS")

    # =====================================================================
    # CATEGORY A: Infrastructure & Connectivity
    # =====================================================================
    section("CATEGORY A: INFRASTRUCTURE & CONNECTIVITY")

    # Dimension 1: Venue Health
    subsection("Dimension 1: Venue Health & Connectivity")
    headers = ["Venue", "Healthy%", "Stale%", "Disabled%", "Flips", "MaxStaleRun", "StaleRuns"]
    rows = []
    for i in report_venue_indices:
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
            acc.venue_name(i),
            pct(healthy, total),
            pct(stale, total),
            pct(disabled, total),
            str(acc.venue_status_flips[i]),
            str(max_run),
            str(num_runs),
        ])
    lines.append(format_table(headers, rows))

    subsection("Venue Age (ms) Statistics")
    for i in report_venue_indices:
        lines.append(acc.venue_age_stats[i].summary_line(acc.venue_name(i), "ms"))

    if any(stat.n > 0 for stat in acc.venue_age_event_stats):
        subsection("Venue Event Age (ms) Statistics")
        for i in report_venue_indices:
            lines.append(acc.venue_age_event_stats[i].summary_line(acc.venue_name(i), "ms"))

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
    for i in report_venue_indices:
        lines.append(acc.venue_mid_stats[i].summary_line(f"  {acc.venue_name(i)} mid", "USD"))

    lines.append("")
    lines.append("  Spread Statistics:")
    for i in report_venue_indices:
        lines.append(acc.venue_spread_stats[i].summary_line(f"  {acc.venue_name(i)} spread", "USD"))

    lines.append("")
    lines.append("  Depth Near Mid Statistics:")
    for i in report_venue_indices:
        lines.append(acc.venue_depth_stats[i].summary_line(f"  {acc.venue_name(i)} depth", "USD"))

    lines.append("")
    lines.append("  Mid Price Tick-to-Tick Delta (bps):")
    for i in report_venue_indices:
        lines.append(acc.venue_mid_delta_stats[i].summary_line(f"  {acc.venue_name(i)}", "bps"))
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
    for i in report_venue_indices:
        lines.append(acc.venue_fv_deviation[i].summary_line(f"  {acc.venue_name(i)}", "USD"))

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
                fmt_f(w.get("mean"), 8),
                fmt_f(w.get("p95"), 8),
                fmt_f(w.get("max"), 8),
            ])
        lines.append(format_table(headers_kf, rows_kf))

    # =====================================================================
    # CATEGORY C: PnL-Critical Strategy Analysis
    # =====================================================================
    section("CATEGORY C: PnL-CRITICAL STRATEGY ANALYSIS")

    # Dimension 6
    subsection("Dimension 6: Markout & Adverse Selection")
    lines.append("  Per-Venue Markout EWMA (USD/tao):")
    for i in report_venue_indices:
        lines.append(acc.venue_markout_stats[i].summary_line(f"  {acc.venue_name(i)}"))
    lines.append("")
    lines.append("  Per-Venue Toxicity:")
    for i in report_venue_indices:
        lines.append(acc.venue_toxicity_stats[i].summary_line(f"  {acc.venue_name(i)}"))
    lines.append(f"\n  Total fills: {acc.fills_count}")
    if acc.fills_count == 0:
        lines.append("  NOTE: Shadow mode -- no real fills. Markout/toxicity from EWMA estimates only.")

    # Toxicity trend
    subsection("Toxicity Trend (per venue, per window)")
    for i in report_venue_indices:
        trend_tox = acc.venue_toxicity_windows[i].trend_summary()
        if trend_tox and any(w["count"] > 0 for w in trend_tox):
            lines.append(f"  {acc.venue_name(i)}:")
            for w in trend_tox:
                if w["count"] > 0:
                    lines.append(f"    [{w['tick_start']}-{w['tick_end']}] mean={fmt_f(w.get('mean'), 4)} p95={fmt_f(w.get('p95'), 4)}")

    # Dimension 7
    subsection("Dimension 7: Quote-Level Edge Decomposition")
    lines.append("  Per-Venue Edge (edge_local):")
    for i in report_venue_indices:
        lines.append(acc.venue_edge_stats[i].summary_line(f"  {acc.venue_name(i)}", "USD"))
        lines.append(f"      Quote samples: {acc.venue_quote_count[i]}")

    lines.append("")
    lines.append("  Quote Gate By Venue/Side:")
    headers_qg = ["Venue", "Side", "Samples", "Active", "Suppressed", "Top suppression", "Mean threshold", "Mean hedge floor"]
    rows_qg = []
    for i in report_venue_indices:
        for side in ("Bid", "Ask"):
            key = (i, side)
            state_counts = acc.quote_gate_counts.get(key, Counter())
            samples = sum(state_counts.values())
            suppression_counts = acc.quote_gate_suppression_counts.get(key, Counter())
            top_suppression = "-"
            if suppression_counts:
                reason, count = suppression_counts.most_common(1)[0]
                top_suppression = f"{reason}:{count}"
            threshold_stats = acc.quote_gate_edge_threshold_stats.get(key)
            hedge_floor_stats = acc.quote_gate_hedge_cost_floor_stats.get(key)
            rows_qg.append([
                acc.venue_name(i),
                side,
                str(samples),
                str(state_counts.get("active", 0)),
                str(state_counts.get("suppressed", 0)),
                top_suppression,
                fmt_f(threshold_stats.mean if threshold_stats and threshold_stats.n else None, 4),
                fmt_f(hedge_floor_stats.mean if hedge_floor_stats and hedge_floor_stats.n else None, 4),
            ])
    lines.append(format_table(headers_qg, rows_qg))

    lines.append("")
    lines.append("  Per-Venue Half-Spread (delta_final):")
    for i in report_venue_indices:
        lines.append(acc.venue_delta_final_stats[i].summary_line(f"  {acc.venue_name(i)}", "USD"))

    lines.append("")
    lines.append("  Edge Component Breakdown (mean absolute USD):")
    headers_ec = ["Venue", "basis_adj", "funding_adj", "inventory_term"]
    rows_ec = []
    for i in report_venue_indices:
        rows_ec.append([
            acc.venue_name(i),
            fmt_f(acc.venue_basis_adj_stats[i].mean if acc.venue_basis_adj_stats[i].n > 0 else None, 4),
            fmt_f(acc.venue_funding_adj_stats[i].mean if acc.venue_funding_adj_stats[i].n > 0 else None, 4),
            fmt_f(acc.venue_inventory_term_stats[i].mean if acc.venue_inventory_term_stats[i].n > 0 else None, 4),
        ])
    lines.append(format_table(headers_ec, rows_ec))

    lines.append("")
    lines.append("  Spread/Size Multipliers:")
    for i in report_venue_indices:
        lines.append(acc.venue_spread_mult_stats[i].summary_line(f"  {acc.venue_name(i)} spread_mult"))
        lines.append(acc.venue_size_mult_stats[i].summary_line(f"  {acc.venue_name(i)} size_mult"))

    lines.append("")
    lines.append("  Size Constraint Binding Frequency:")
    headers_sc = ["Venue", "Quotes", "MarginCap<Raw", "LiqFactor<1", "AnyBinding"]
    rows_sc = []
    for i in report_venue_indices:
        qc = acc.venue_quote_count[i]
        rows_sc.append([
            acc.venue_name(i),
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
    if acc.hedge_record_total > 0:
        lines.append(
            f"  Hedge source attribution: records={acc.hedge_record_total} "
            f"attributed={acc.hedge_source_attributed_count} "
            f"coverage={pct(acc.hedge_source_attributed_count, acc.hedge_record_total)}"
        )
        if acc.hedge_source_fill_age_stats.n > 0:
            lines.append(
                acc.hedge_source_fill_age_stats.summary_line("hedge source_fill_age_ms", "ms")
            )
    hedge_fill_total = acc.hedge_fill_attributed_count + acc.hedge_fill_unattributed_count
    if hedge_fill_total > 0:
        lines.append(
            f"  Hedge fill source attribution: attributed={acc.hedge_fill_attributed_count} "
            f"unattributed={acc.hedge_fill_unattributed_count} "
            f"coverage={pct(acc.hedge_fill_attributed_count, hedge_fill_total)}"
        )
    if acc.hedges_tick_count == 0:
        lines.append("  NOTE: No hedges detected (expected in shadow mode)")

    # =====================================================================
    # CATEGORY D: Capital Efficiency
    # =====================================================================
    section("CATEGORY D: CAPITAL EFFICIENCY")

    # Dimension 10
    subsection("Dimension 10: Margin Utilization & Liquidation Distance")
    lines.append("  Margin Utilization (used/balance):")
    for i in report_venue_indices:
        lines.append(acc.venue_margin_util_stats[i].summary_line(f"  {acc.venue_name(i)}"))

    lines.append("")
    lines.append("  Margin Available (USD):")
    for i in report_venue_indices:
        lines.append(acc.venue_margin_avail_stats[i].summary_line(f"  {acc.venue_name(i)}", "USD"))

    lines.append("")
    lines.append("  Distance to Liquidation (sigma):")
    for i in report_venue_indices:
        lines.append(acc.venue_dist_liq_stats[i].summary_line(f"  {acc.venue_name(i)}", "σ"))

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
    for i in report_venue_indices:
        lines.append(acc.venue_local_vol_short[i].summary_line(f"  {acc.venue_name(i)}"))

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
        for i in report_venue_indices:
            top_kind = ""
            if acc.drift_kind_by_venue[i]:
                kind, count = acc.drift_kind_by_venue[i].most_common(1)[0]
                top_kind = f" (top={kind}:{count})"
            lines.append(f"    {acc.venue_name(i)}: {acc.drift_by_venue[i]}{top_kind}")
        if acc.drift_kind_counts:
            lines.append("  Drift events by kind:")
            for kind, count in acc.drift_kind_counts.most_common():
                lines.append(f"    {kind}: {count}")
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
    lines.append(
        f"  Soft-governor triggered ticks: {acc.soft_governor_ticks} "
        f"({pct(acc.soft_governor_ticks, acc.tick_count)})"
    )
    if acc.soft_governor_reason_counts:
        lines.append("  Soft-governor reasons:")
        for reason, count in acc.soft_governor_reason_counts.most_common():
            lines.append(f"    {reason}: {count}")
        lines.append("  Soft-governor blocked venue ticks:")
        for i in report_venue_indices:
            lines.append(f"    {acc.venue_name(i)}: {acc.soft_governor_blocked_ticks[i]}")

    lines.append("")
    lines.append("  Order Action Counts:")
    for action, count in acc.order_action_counts.most_common():
        lines.append(f"    {action}: {count}")

    lines.append("")
    lines.append("  MM Order Management:")
    lines.append(acc.mm_keep_count_stats.summary_line("mm keep_count"))
    lines.append(acc.mm_replace_count_stats.summary_line("mm replace_count"))
    lines.append(acc.mm_place_count_stats.summary_line("mm place_count"))
    lines.append(acc.mm_cancel_count_stats.summary_line("mm cancel_count"))
    mm_decision_total = acc.mm_keep_total + acc.mm_replace_total
    lines.append(
        f"  MM decision totals: keep={acc.mm_keep_total} replace={acc.mm_replace_total} "
        f"replace_share={pct(acc.mm_replace_total, mm_decision_total)}"
    )
    if acc.mm_inventory_reducing_counts:
        lines.append(
            "  Inventory-reducing MM decisions: "
            + ", ".join(
                f"{outcome}={count}"
                for outcome, count in acc.mm_inventory_reducing_counts.most_common()
            )
        )
    if acc.mm_keep_reason_counts:
        lines.append("  Top keep reasons:")
        for reason, count in acc.mm_keep_reason_counts.most_common(5):
            lines.append(f"    {reason}: {count}")
    if acc.mm_replace_reason_counts:
        lines.append("  Top replace reasons:")
        for reason, count in acc.mm_replace_reason_counts.most_common(5):
            lines.append(f"    {reason}: {count}")
    if acc.mm_keep_utility_tier_counts or acc.mm_replace_utility_tier_counts:
        lines.append("  MM utility tiers:")
        tier_keys = sorted(
            set(acc.mm_keep_utility_tier_counts.keys())
            | set(acc.mm_replace_utility_tier_counts.keys())
        )
        for tier in tier_keys:
            lines.append(
                f"    {tier}: keep={acc.mm_keep_utility_tier_counts.get(tier, 0)} "
                f"replace={acc.mm_replace_utility_tier_counts.get(tier, 0)}"
            )
    if acc.mm_keep_venue_role_counts or acc.mm_replace_venue_role_counts:
        lines.append("  MM venue roles:")
        role_keys = sorted(
            set(acc.mm_keep_venue_role_counts.keys())
            | set(acc.mm_replace_venue_role_counts.keys())
        )
        for role in role_keys:
            lines.append(
                f"    {role}: keep={acc.mm_keep_venue_role_counts.get(role, 0)} "
                f"replace={acc.mm_replace_venue_role_counts.get(role, 0)}"
            )
    lines.append("  MM attribution coverage:")
    lines.append(
        f"    decision_records={acc.mm_decision_record_total} "
        f"with_order_lineage={acc.mm_decision_order_link_total} "
        f"({pct(acc.mm_decision_order_link_total, acc.mm_decision_record_total)})"
    )
    if acc.supported_replace_visibility_total > 0:
        lines.append("  Supported replace visibility:")
        for i in report_venue_indices:
            if (
                acc.supported_replace_opportunity_counts[i] <= 0
                and acc.supported_replace_visibility_miss_counts[i] <= 0
                and acc.supported_replace_gap_grace_counts[i] <= 0
            ):
                continue
            blockers = ", ".join(
                f"{reason}={count}"
                for reason, count in acc.supported_replace_blocker_counts[i].most_common(3)
            ) or "none"
            actions = ", ".join(
                f"{action}={count}"
                for action, count in acc.supported_replace_action_counts[i].most_common(3)
            ) or "none"
            lines.append(
                f"    {acc.venue_name(i)}: opportunities={acc.supported_replace_opportunity_counts[i]} "
                f"misses={acc.supported_replace_visibility_miss_counts[i]} "
                f"gap_grace={acc.supported_replace_gap_grace_counts[i]} "
                f"actions=[{actions}] blockers=[{blockers}]"
            )
    if acc.emergency_residual_fallback_total > 0:
        lines.append("  Emergency residual fallback:")
        lines.append(
            f"    total_records={acc.emergency_residual_fallback_total} "
            f"first_tick={fmt_i(acc.emergency_residual_fallback_first_tick)} "
            f"last_tick={fmt_i(acc.emergency_residual_fallback_last_tick)}"
        )
        lines.append(
            "    by_status="
            + (", ".join(f"{status}={count}" for status, count in acc.emergency_residual_fallback_status_counts.most_common()) or "none")
        )
        lines.append(
            "    by_reason="
            + (", ".join(f"{reason}={count}" for reason, count in acc.emergency_residual_fallback_reason_counts.most_common()) or "none")
        )
        lines.append(
            "    by_class="
            + (", ".join(f"{klass}={count}" for klass, count in acc.emergency_residual_fallback_class_counts.most_common()) or "none")
        )
        lines.append(
            "    by_venue="
            + (", ".join(f"{venue}={count}" for venue, count in acc.emergency_residual_fallback_venue_counts.most_common()) or "none")
        )
    if acc.aster_residual_markout_guard_total > 0:
        lines.append("  Aster residual markout guard:")
        lines.append(
            f"    snapshots={acc.aster_residual_markout_guard_total} "
            f"allowed_orders={acc.aster_residual_markout_guard_allowed_orders} "
            f"suppressed_orders={acc.aster_residual_markout_guard_suppressed_orders}"
        )
        lines.append(
            "    by_decision="
            + (", ".join(f"{decision}={count}" for decision, count in acc.aster_residual_markout_guard_decision_counts.most_common()) or "none")
        )
        lines.append(
            "    by_reason="
            + (", ".join(f"{reason}={count}" for reason, count in acc.aster_residual_markout_guard_reason_counts.most_common()) or "none")
        )
        lines.append(
            acc.aster_residual_markout_guard_age_stats.summary_line(
                "aster guard residual_age", "ms"
            )
        )
        lines.append(
            acc.aster_residual_markout_guard_adverse_stats.summary_line(
                "aster guard adverse_markout", "USD"
            )
        )
        lines.append(
            acc.aster_residual_markout_guard_unrealised_stats.summary_line(
                "aster guard residual_unrealised", "USD"
            )
        )
        lines.append(
            acc.aster_residual_markout_guard_cleanup_fee_stats.summary_line(
                "aster guard cleanup_fee_estimate", "USD"
            )
        )
        if acc.aster_residual_markout_guard_refresh_outcome_counts:
            lines.append(
                f"    refresh_attempts={acc.aster_residual_markout_guard_refresh_attempts} "
                + "outcomes="
                + ", ".join(
                    f"{outcome}={count}"
                    for outcome, count in acc.aster_residual_markout_guard_refresh_outcome_counts.most_common()
                )
            )
            if acc.aster_residual_markout_guard_refresh_suppressed_counts:
                lines.append(
                    "    refresh_suppressed="
                    + ", ".join(
                        f"{reason}={count}"
                        for reason, count in acc.aster_residual_markout_guard_refresh_suppressed_counts.most_common()
                    )
                )
            lines.append(
                acc.aster_residual_markout_guard_refresh_latency_stats.summary_line(
                    "aster guard refresh_latency", "ms"
                )
            )
            lines.append(
                acc.aster_residual_markout_guard_fresh_account_age_stats.summary_line(
                    "aster guard fresh_account_age", "ms"
                )
            )
    mm_fill_total = acc.mm_fill_attributed_count + acc.mm_fill_unattributed_count
    lines.append(
        f"    mm_fills_attributed={acc.mm_fill_attributed_count} "
        f"unattributed={acc.mm_fill_unattributed_count} "
        f"coverage={pct(acc.mm_fill_attributed_count, mm_fill_total)}"
    )
    if acc.mm_fill_decision_markout_short_stats.n > 0:
        lines.append(
            acc.mm_fill_decision_markout_short_stats.summary_line(
                "mm attributable markout_pnl_short", "USD"
            )
        )
    if acc.mm_fill_decision_realised_pnl_stats.n > 0:
        lines.append(
            acc.mm_fill_decision_realised_pnl_stats.summary_line(
                "mm attributable realised_pnl_usd", "USD"
            )
        )

    decision_contributions = [
        contribution
        for contribution in acc.decision_contributions.values()
        if contribution.mm_fill_count > 0
        or contribution.hedge_record_count > 0
        or contribution.hedge_fill_count > 0
    ]
    if decision_contributions:
        lines.append("")
        lines.append("  Decision-Level Contribution:")
        lines.append(
            "    mm_gross_before_fee_usd = mm_realised_net_usd + mm_fee_usd; "
            "hedge_exec_cost_model_usd uses hedge intent exec_cost * size"
        )
        headers_dc = [
            "Decision",
            "Venue",
            "MmFills",
            "MmGross",
            "MmFees",
            "MmNet",
            "HedgeFills",
            "HedgeExecModel",
            "NetAfterHedge",
        ]
        rows_dc = []
        ranked_decisions = sorted(
            decision_contributions,
            key=lambda contribution: (
                abs(contribution.net_after_hedge_exec_model_usd),
                abs(contribution.mm_realised_net_usd),
                contribution.decision_id,
            ),
            reverse=True,
        )
        for contribution in ranked_decisions[:10]:
            rows_dc.append([
                contribution.decision_id,
                contribution.mm_venue_id or "n/a",
                str(contribution.mm_fill_count),
                fmt_f(contribution.mm_gross_before_fee_usd, 4),
                fmt_f(contribution.mm_fee_usd, 4),
                fmt_f(contribution.mm_realised_net_usd, 4),
                str(contribution.hedge_fill_count),
                fmt_f(contribution.hedge_exec_cost_model_usd, 4),
                fmt_f(contribution.net_after_hedge_exec_model_usd, 4),
            ])
        lines.append(format_table(headers_dc, rows_dc))
        if len(ranked_decisions) > len(rows_dc):
            lines.append(
                f"    showing top {len(rows_dc)} of {len(ranked_decisions)} decisions by "
                "|net_after_hedge_exec_model_usd|"
            )

        lines.append("")
        lines.append("  Venue Contribution After Hedge:")
        headers_vc = [
            "Venue",
            "Decisions",
            "MmGross",
            "MmFees",
            "MmNet",
            "HedgeFee",
            "HedgeExecModel",
            "NetAfterHedge",
        ]
        venue_contrib: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "decisions": 0.0,
                "mm_gross": 0.0,
                "mm_fees": 0.0,
                "mm_net": 0.0,
                "hedge_fee": 0.0,
                "hedge_exec_model": 0.0,
                "net_after_hedge": 0.0,
            }
        )
        for contribution in decision_contributions:
            venue_key = contribution.mm_venue_id or "unassigned"
            bucket = venue_contrib[venue_key]
            bucket["decisions"] += 1.0
            bucket["mm_gross"] += contribution.mm_gross_before_fee_usd
            bucket["mm_fees"] += contribution.mm_fee_usd
            bucket["mm_net"] += contribution.mm_realised_net_usd
            bucket["hedge_fee"] += contribution.hedge_fill_fee_usd
            bucket["hedge_exec_model"] += contribution.hedge_exec_cost_model_usd
            bucket["net_after_hedge"] += contribution.net_after_hedge_exec_model_usd

        rows_vc = []
        for venue_key, bucket in sorted(
            venue_contrib.items(),
            key=lambda item: item[1]["net_after_hedge"],
            reverse=True,
        ):
            rows_vc.append([
                venue_key,
                str(int(bucket["decisions"])),
                fmt_f(bucket["mm_gross"], 4),
                fmt_f(bucket["mm_fees"], 4),
                fmt_f(bucket["mm_net"], 4),
                fmt_f(bucket["hedge_fee"], 4),
                fmt_f(bucket["hedge_exec_model"], 4),
                fmt_f(bucket["net_after_hedge"], 4),
            ])
        lines.append(format_table(headers_vc, rows_vc))

        lines.append("")
        lines.append("  Unattributed residual:")
        lines.append(
            f"    mm_realised_net_unattributed={fmt_f(acc.mm_realised_net_unattributed_usd, 4)} USD"
        )
        lines.append(f"    mm_fee_unattributed={fmt_f(acc.mm_fee_unattributed_usd, 4)} USD")
        lines.append(
            f"    hedge_fill_fee_unattributed={fmt_f(acc.hedge_fill_fee_unattributed_usd, 4)} USD"
        )
        lines.append(
            "    hedge_exec_cost_model_unattributed="
            f"{fmt_f(acc.hedge_exec_cost_model_unattributed_usd, 4)} USD"
        )

    lines.append("")
    lines.append("  Orders Per Venue:")
    for i in report_venue_indices:
        lines.append(f"    {acc.venue_name(i)}: {acc.orders_per_venue[i]}")

    lines.append("")
    lines.append("  Execution Scorecard:")
    headers_exec = [
        "Venue",
        "PlaceI",
        "ReplaceI",
        "CancelI",
        "PlaceAck",
        "CancelAck",
        "PlaceRej",
        "Fills",
        "FillBase",
        "MmFills",
        "HedgeFills",
    ]
    rows_exec = []
    for i in report_venue_indices:
        venue = acc.venue_name(i)
        counts = acc.order_status_action_counts[i]
        rows_exec.append([
            venue,
            str(counts.get("place_intent", 0)),
            str(counts.get("replace_intent", 0)),
            str(counts.get("cancel_intent", 0)),
            str(counts.get("place_ack", 0)),
            str(counts.get("cancel_ack", 0)),
            str(counts.get("place_reject", 0)),
            str(acc.venue_fill_counts[i]),
            fmt_f(acc.venue_fill_size_base[i], 4),
            str(acc.venue_mm_fill_counts[i]),
            str(acc.venue_hedge_fill_counts[i]),
        ])
    lines.append(format_table(headers_exec, rows_exec))

    if any(acc.mm_decisions_by_venue):
        lines.append("")
        lines.append("  MM Keep/Replace By Venue:")
        headers_mm = ["Venue", "KeepD", "ReplaceD"]
        rows_mm = []
        for i in report_venue_indices:
            venue = acc.venue_name(i)
            rows_mm.append([
                venue,
                str(acc.mm_decisions_by_venue[i].get("keep", 0)),
                str(acc.mm_decisions_by_venue[i].get("replace", 0)),
            ])
        lines.append(format_table(headers_mm, rows_mm))

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
        lines.append(f"  Peak pnl_total: {fmt_f(acc.pnl_total_stats._max, 4)} USD")
        lines.append(f"  Trough pnl_total: {fmt_f(acc.pnl_total_stats._min, 4)} USD")
        lines.append(f"  Final pnl_total: {fmt_f(acc.final_pnl_total, 4)} USD")
    if acc.final_pnl_realised is not None:
        lines.append(f"  Final pnl_realised: {fmt_f(acc.final_pnl_realised, 4)} USD")
    if acc.final_pnl_unrealised is not None:
        lines.append(f"  Final pnl_unrealised: {fmt_f(acc.final_pnl_unrealised, 4)} USD")

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
    for i in report_venue_indices:
        lines.append(acc.venue_funding_rate_stats[i].summary_line(f"  {acc.venue_name(i)}"))

    lines.append("")
    lines.append("  Funding Status Distribution:")
    headers_fs = ["Venue", "Healthy%", "Unknown%", "Stale%"]
    rows_fs = []
    for i in report_venue_indices:
        total = sum(acc.venue_funding_status_counts[i].values())
        h = acc.venue_funding_status_counts[i].get("Healthy", 0)
        u = acc.venue_funding_status_counts[i].get("Unknown", 0)
        s = acc.venue_funding_status_counts[i].get("Stale", 0)
        rows_fs.append([acc.venue_name(i), pct(h, total), pct(u, total), pct(s, total)])
    lines.append(format_table(headers_fs, rows_fs))

    lines.append("")
    lines.append("  Funding Age (ms):")
    for i in report_venue_indices:
        lines.append(acc.venue_funding_age_stats[i].summary_line(f"  {acc.venue_name(i)}", "ms"))

    # Cross-venue funding dispersion
    if report_venue_indices and all(acc.venue_funding_rate_stats[i].n > 0 for i in report_venue_indices):
        means = [acc.venue_funding_rate_stats[i].mean for i in report_venue_indices]
        valid_means = [m for m in means if m != 0]
        if len(valid_means) >= 2:
            disp = max(valid_means) - min(valid_means)
            lines.append(f"\n  Cross-venue funding rate dispersion (max-min of means): {disp:.6f}")

    # =====================================================================
    # Age Trend (per venue, per window)
    # =====================================================================
    section("APPENDIX: VENUE AGE TREND (per window)")
    for i in report_venue_indices:
        trend_age = acc.venue_age_windows[i].trend_summary()
        if trend_age:
            lines.append(f"\n  {acc.venue_name(i)}:")
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

    if any(w.current_window.n > 0 for w in acc.venue_age_event_windows):
        section("APPENDIX: VENUE EVENT AGE TREND (per window)")
        for i in report_venue_indices:
            trend_age_event = acc.venue_age_event_windows[i].trend_summary()
            if trend_age_event:
                lines.append(f"\n  {acc.venue_name(i)}:")
                headers_ae = ["Window", "Mean_ms", "P95_ms", "P99_ms", "Max_ms"]
                rows_ae = []
                for w in trend_age_event:
                    rows_ae.append([
                        f"{w['tick_start']}-{w['tick_end']}",
                        fmt_f(w.get("mean"), 1),
                        fmt_f(w.get("p95"), 1),
                        fmt_f(w.get("p99"), 1),
                        fmt_f(w.get("max"), 1),
                    ])
                lines.append(format_table(headers_ae, rows_ae))

    lines.append("")
    lines.append("=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streaming reader
# ---------------------------------------------------------------------------

def stream_records(
    path: Path,
    max_ticks: int = 0,
    execution_mode: str = "all",
    last_segment: bool = False,
    tail_bytes: int = 0,
) -> TelemetryAccumulator:
    acc = TelemetryAccumulator()
    count = 0
    t0 = time.monotonic()

    base_records = (
        iter_json_objects_from_tail(path, tail_bytes)
        if tail_bytes > 0
        else iter_json_objects(path)
    )

    if last_segment:
        if execution_mode == "all":
            raise ValueError("--last-segment requires --execution-mode to be set")
        records = last_execution_segment_from_iter(base_records, execution_mode)
    else:
        records = base_records

    for rec in records:
        if execution_mode != "all" and rec.get("execution_mode") != execution_mode:
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
        "venue_names": [acc.venue_name(i) for i in acc.report_venue_indices()],
    }

    # Key metrics for regression scorecard
    snapshot["venue_health"] = {}
    for i in acc.report_venue_indices():
        total = sum(acc.venue_status_counts[i].values())
        healthy = acc.venue_status_counts[i].get("Healthy", 0)
        snapshot["venue_health"][acc.venue_name(i)] = {
            "healthy_pct": round(100.0 * healthy / total, 4) if total > 0 else 0,
            "age_p95": round(acc.venue_age_stats[i].percentile(95), 2) if acc.venue_age_stats[i].n > 0 else None,
            "age_event_p95": round(acc.venue_age_event_stats[i].percentile(95), 2)
            if acc.venue_age_event_stats[i].n > 0
            else None,
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
        "kill_reasons": dict(acc.kill_reason_counts),
        "regime_transitions": acc.risk_regime_transitions,
        "would_send_zero_pct": round(100.0 * acc.would_send_zero_ticks / acc.tick_count, 4) if acc.tick_count > 0 else 0,
    }

    snapshot["pnl_validity"] = {
        "final_q_global_tao": round(acc.final_q_global_tao, 6) if acc.final_q_global_tao is not None else None,
        "final_pnl_total": round(acc.final_pnl_total, 6) if acc.final_pnl_total is not None else None,
        "final_pnl_realised": round(acc.final_pnl_realised, 6) if acc.final_pnl_realised is not None else None,
        "final_pnl_unrealised": round(acc.final_pnl_unrealised, 6) if acc.final_pnl_unrealised is not None else None,
        "pnl_reconstruction_mismatch_max": round(acc.pnl_reconstruction_mismatch_stats._max, 6)
        if acc.pnl_reconstruction_mismatch_stats.n > 0
        else None,
        "flat_inventory_large_unrealised_ticks": acc.flat_inventory_large_unrealised_ticks,
        "mm_place_total": acc.mm_place_total,
        "mm_keep_total": acc.mm_keep_total,
        "mm_replace_total": acc.mm_replace_total,
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


def economics_attribution_snapshot(
    acc: TelemetryAccumulator,
    report_indices: list[int],
) -> dict[str, Any]:
    per_venue: dict[str, Any] = {}
    for i in report_indices:
        per_venue[acc.venue_name(i)] = {
            "mm_fill_attributed_count": acc.mm_fill_attributed_by_venue[i],
            "mm_gross_before_fee_usd": round(acc.mm_gross_before_fee_attributed_by_venue[i], 6),
            "mm_fee_usd": round(acc.mm_fee_attributed_by_venue[i], 6),
            "mm_realised_net_usd": round(acc.mm_realised_net_attributed_by_venue[i], 6),
            "mm_markout_short_usd": round(acc.mm_markout_short_attributed_by_venue[i], 6),
            "venue_markout_stats": acc.venue_markout_stats[i].summary(),
            "venue_toxicity_stats": acc.venue_toxicity_stats[i].summary(),
        }

    venue_contribution_after_hedge: dict[str, Any] = {}
    for contribution in acc.decision_contributions.values():
        if (
            contribution.mm_fill_count == 0
            and contribution.hedge_record_count == 0
            and contribution.hedge_fill_count == 0
        ):
            continue
        venue_key = contribution.mm_venue_id or "unassigned"
        bucket = venue_contribution_after_hedge.setdefault(
            venue_key,
            {
                "decision_count": 0,
                "mm_fill_count": 0,
                "mm_gross_before_fee_usd": 0.0,
                "mm_fee_usd": 0.0,
                "mm_realised_net_usd": 0.0,
                "hedge_fill_count": 0,
                "hedge_fill_fee_usd": 0.0,
                "hedge_exec_cost_model_usd": 0.0,
                "hedge_total_cost_model_usd": 0.0,
                "net_after_hedge_exec_model_usd": 0.0,
            },
        )
        bucket["decision_count"] += 1
        bucket["mm_fill_count"] += contribution.mm_fill_count
        bucket["mm_gross_before_fee_usd"] += contribution.mm_gross_before_fee_usd
        bucket["mm_fee_usd"] += contribution.mm_fee_usd
        bucket["mm_realised_net_usd"] += contribution.mm_realised_net_usd
        bucket["hedge_fill_count"] += contribution.hedge_fill_count
        bucket["hedge_fill_fee_usd"] += contribution.hedge_fill_fee_usd
        bucket["hedge_exec_cost_model_usd"] += contribution.hedge_exec_cost_model_usd
        bucket["hedge_total_cost_model_usd"] += contribution.hedge_total_cost_model_usd
        bucket["net_after_hedge_exec_model_usd"] += contribution.net_after_hedge_exec_model_usd

    for bucket in venue_contribution_after_hedge.values():
        for key in (
            "mm_gross_before_fee_usd",
            "mm_fee_usd",
            "mm_realised_net_usd",
            "hedge_fill_fee_usd",
            "hedge_exec_cost_model_usd",
            "hedge_total_cost_model_usd",
            "net_after_hedge_exec_model_usd",
        ):
            bucket[key] = round(bucket[key], 6)

    attributed_net_after_hedge_exec_model_usd = (
        acc.mm_realised_net_attributed_usd - acc.hedge_exec_cost_model_attributed_usd
    )
    unattributed_net_after_hedge_exec_model_usd = (
        acc.mm_realised_net_unattributed_usd - acc.hedge_exec_cost_model_unattributed_usd
    )
    net_after_hedge_exec_model_usd = (
        attributed_net_after_hedge_exec_model_usd + unattributed_net_after_hedge_exec_model_usd
    )

    return {
        "mm_fill_attributed_count": acc.mm_fill_attributed_count,
        "mm_fill_unattributed_count": acc.mm_fill_unattributed_count,
        "mm_gross_before_fee_attributed_usd": round(acc.mm_gross_before_fee_attributed_usd, 6),
        "mm_fee_attributed_usd": round(acc.mm_fee_attributed_usd, 6),
        "mm_realised_net_attributed_usd": round(acc.mm_realised_net_attributed_usd, 6),
        "mm_realised_net_unattributed_usd": round(acc.mm_realised_net_unattributed_usd, 6),
        "mm_fee_unattributed_usd": round(acc.mm_fee_unattributed_usd, 6),
        "hedge_fill_attributed_count": acc.hedge_fill_attributed_count,
        "hedge_fill_unattributed_count": acc.hedge_fill_unattributed_count,
        "hedge_fill_fee_attributed_usd": round(acc.hedge_fill_fee_attributed_usd, 6),
        "hedge_fill_fee_unattributed_usd": round(acc.hedge_fill_fee_unattributed_usd, 6),
        "hedge_exec_cost_model_attributed_usd": round(acc.hedge_exec_cost_model_attributed_usd, 6),
        "hedge_exec_cost_model_unattributed_usd": round(acc.hedge_exec_cost_model_unattributed_usd, 6),
        "hedge_total_cost_model_attributed_usd": round(acc.hedge_total_cost_model_attributed_usd, 6),
        "hedge_total_cost_model_unattributed_usd": round(acc.hedge_total_cost_model_unattributed_usd, 6),
        "net_after_hedge_exec_model_attributed_usd": round(attributed_net_after_hedge_exec_model_usd, 6),
        "net_after_hedge_exec_model_unattributed_usd": round(unattributed_net_after_hedge_exec_model_usd, 6),
        "net_after_hedge_exec_model_usd": round(net_after_hedge_exec_model_usd, 6),
        "mm_fill_markout_short_stats": acc.mm_fill_decision_markout_short_stats.summary(),
        "mm_fill_realised_pnl_stats": acc.mm_fill_decision_realised_pnl_stats.summary(),
        "per_venue": per_venue,
        "venue_contribution_after_hedge": venue_contribution_after_hedge,
    }


def build_metrics_snapshot(acc: TelemetryAccumulator) -> dict[str, Any]:
    report_indices = acc.report_venue_indices()
    venue_names = [acc.venue_name(i) for i in report_indices]

    execution_scorecard: dict[str, Any] = {}
    mm_keep_replace_by_venue: dict[str, Any] = {}
    supported_replace_visibility: dict[str, Any] = {}
    venue_health: dict[str, Any] = {}
    fills_by_venue: dict[str, Any] = {}
    orders_per_venue: dict[str, int] = {}
    quote_gate_by_venue_side: dict[str, Any] = {}
    opportunity_adjusted_scorecard: dict[str, Any] = {}

    for i in report_indices:
        venue = acc.venue_name(i)
        counts = acc.order_status_action_counts[i]
        total = sum(acc.venue_status_counts[i].values())
        healthy = acc.venue_status_counts[i].get("Healthy", 0)
        execution_scorecard[venue] = {
            "place_i": counts.get("place_intent", 0),
            "replace_i": counts.get("replace_intent", 0),
            "cancel_i": counts.get("cancel_intent", 0),
            "place_ack": counts.get("place_ack", 0),
            "cancel_ack": counts.get("cancel_ack", 0),
            "place_reject": counts.get("place_reject", 0),
            "fills": acc.venue_fill_counts[i],
            "fill_base": round(acc.venue_fill_size_base[i], 6),
            "mm_fills": acc.venue_mm_fill_counts[i],
            "mm_fill_base": round(acc.venue_mm_fill_size_base[i], 6),
            "hedge_fills": acc.venue_hedge_fill_counts[i],
            "hedge_fill_base": round(acc.venue_hedge_fill_size_base[i], 6),
        }
        mm_keep_replace_by_venue[venue] = {
            "keep": acc.mm_decisions_by_venue[i].get("keep", 0),
            "replace": acc.mm_decisions_by_venue[i].get("replace", 0),
        }
        supported_replace_visibility[venue] = {
            "opportunities": acc.supported_replace_opportunity_counts[i],
            "misses": acc.supported_replace_visibility_miss_counts[i],
            "gap_grace": acc.supported_replace_gap_grace_counts[i],
            "actions": dict(acc.supported_replace_action_counts[i]),
            "blockers": dict(acc.supported_replace_blocker_counts[i]),
        }
        venue_health[venue] = {
            "healthy_pct": round(100.0 * healthy / total, 4) if total > 0 else 0.0,
            "age_p95": round(acc.venue_age_stats[i].percentile(95), 2) if acc.venue_age_stats[i].n > 0 else None,
            "age_event_p95": round(acc.venue_age_event_stats[i].percentile(95), 2)
            if acc.venue_age_event_stats[i].n > 0
            else None,
            "max_stale_run": acc.venue_max_consecutive_stale[i],
            "flips": acc.venue_status_flips[i],
        }
        fills_by_venue[venue] = {
            "fill_count": acc.venue_fill_counts[i],
            "fill_base": round(acc.venue_fill_size_base[i], 6),
            "mm_fill_count": acc.venue_mm_fill_counts[i],
            "mm_fill_base": round(acc.venue_mm_fill_size_base[i], 6),
            "hedge_fill_count": acc.venue_hedge_fill_counts[i],
            "hedge_fill_base": round(acc.venue_hedge_fill_size_base[i], 6),
        }
        orders_per_venue[venue] = acc.orders_per_venue[i]
        quote_gate_by_venue_side[venue] = {}
        active_quote_samples = 0
        total_quote_samples = 0
        for side in ("Bid", "Ask"):
            key = (i, side)
            state_counts = dict(acc.quote_gate_counts.get(key, Counter()))
            samples = sum(state_counts.values())
            active_quote_samples += state_counts.get("active", 0)
            total_quote_samples += samples
            quote_gate_by_venue_side[venue][side] = {
                "samples": samples,
                "active": state_counts.get("active", 0),
                "suppressed": state_counts.get("suppressed", 0),
                "states": state_counts,
                "suppression_reasons": dict(acc.quote_gate_suppression_counts.get(key, Counter())),
                "engine_terminal_reasons": dict(acc.quote_gate_engine_reason_counts.get(key, Counter())),
                "edge_threshold": acc.quote_gate_edge_threshold_stats[key].summary()
                if key in acc.quote_gate_edge_threshold_stats
                else OnlineStats().summary(),
                "edge_threshold_base": acc.quote_gate_edge_threshold_base_stats[key].summary()
                if key in acc.quote_gate_edge_threshold_base_stats
                else OnlineStats().summary(),
                "hedge_cost_edge_floor": acc.quote_gate_hedge_cost_floor_stats[key].summary()
                if key in acc.quote_gate_hedge_cost_floor_stats
                else OnlineStats().summary(),
                "active_edge_local": acc.quote_gate_active_edge_stats[key].summary()
                if key in acc.quote_gate_active_edge_stats
                else OnlineStats().summary(),
            }
        mm_fills = acc.venue_mm_fill_counts[i]
        fill_required = active_quote_samples >= OPPORTUNITY_FILL_REQUIRED_ACTIVE_SAMPLES
        pass_opportunity_adjusted = mm_fills > 0 or not fill_required
        if mm_fills > 0:
            reason = "mm_fill_evidence"
        elif fill_required:
            reason = "active_quote_underconversion"
        elif active_quote_samples > 0:
            reason = "insufficient_active_quote_sample"
        else:
            reason = "no_cost_positive_quote_opportunity"
        opportunity_adjusted_scorecard[venue] = {
            "passed": pass_opportunity_adjusted,
            "reason": reason,
            "mm_fills": mm_fills,
            "active_quote_samples": active_quote_samples,
            "total_quote_samples": total_quote_samples,
            "active_quote_pct": round(100.0 * active_quote_samples / total_quote_samples, 6)
            if total_quote_samples > 0
            else 0.0,
            "fill_required_active_samples": OPPORTUNITY_FILL_REQUIRED_ACTIVE_SAMPLES,
        }

    return {
        "schema_version": 1,
        "tick_count": acc.tick_count,
        "first_tick": acc.first_tick,
        "last_tick": acc.last_tick,
        "segment_start_utc": ts_iso(acc.first_ts_ms),
        "segment_end_utc": ts_iso(acc.last_ts_ms),
        "venue_names": venue_names,
        "execution_scorecard": execution_scorecard,
        "quote_gate_by_venue_side": quote_gate_by_venue_side,
        "opportunity_adjusted_scorecard": opportunity_adjusted_scorecard,
        "mm_keep_replace_by_venue": mm_keep_replace_by_venue,
        "supported_replace_visibility": supported_replace_visibility,
        "projected_mm_budget_summary": {
            "total_records": acc.projected_mm_budget_total_records,
            "configured_ticks": acc.projected_mm_budget_configured_ticks,
            "applied_ticks": acc.projected_mm_budget_applied_ticks,
            "selected_counts": {
                venue: acc.projected_mm_budget_selected_counts[venue]
                for venue in venue_names
                if acc.projected_mm_budget_selected_counts[venue] > 0
            },
            "suppressed_counts": {
                venue: acc.projected_mm_budget_suppressed_counts[venue]
                for venue in venue_names
                if acc.projected_mm_budget_suppressed_counts[venue] > 0
            },
            "all5_selected_ticks": acc.projected_mm_budget_all5_selected_ticks,
            "selected_venue_count": acc.projected_mm_budget_selected_venue_count_stats.summary(),
            "suppressed_venue_count": acc.projected_mm_budget_suppressed_venue_count_stats.summary(),
            "last_limits": dict(acc.projected_mm_budget_last_limits),
            "projected_after": {
                "q_global_tao": acc.projected_mm_budget_q_global_after_stats.summary(),
                "q_gross_tao": acc.projected_mm_budget_q_gross_after_stats.summary(),
                "q_max_abs_venue_tao": acc.projected_mm_budget_q_max_abs_venue_after_stats.summary(),
            },
        },
        "emergency_residual_fallback_summary": {
            "total_records": acc.emergency_residual_fallback_total,
            "by_status": dict(acc.emergency_residual_fallback_status_counts),
            "by_reason": dict(acc.emergency_residual_fallback_reason_counts),
            "by_class": dict(acc.emergency_residual_fallback_class_counts),
            "by_venue": dict(acc.emergency_residual_fallback_venue_counts),
            "first_tick": acc.emergency_residual_fallback_first_tick,
            "last_tick": acc.emergency_residual_fallback_last_tick,
            "aster_inventory_brake_fee_guard_enabled": acc.aster_inventory_brake_fee_guard_enabled,
            "aster_inventory_brake_fee_guard_skipped_orders": acc.aster_inventory_brake_fee_guard_skipped_orders,
            "aster_inventory_brake_fee_guard_skipped_base_tao": round(
                acc.aster_inventory_brake_fee_guard_skipped_base_tao,
                8,
            ),
            "aster_inventory_brake_fee_guard_skipped_notional_usd": round(
                acc.aster_inventory_brake_fee_guard_skipped_notional_usd,
                8,
            ),
            "aster_residual_markout_guard": {
                "snapshots": acc.aster_residual_markout_guard_total,
                "by_decision": dict(acc.aster_residual_markout_guard_decision_counts),
                "by_reason": dict(acc.aster_residual_markout_guard_reason_counts),
                "allowed_orders": acc.aster_residual_markout_guard_allowed_orders,
                "suppressed_orders": acc.aster_residual_markout_guard_suppressed_orders,
                "residual_age_ms": acc.aster_residual_markout_guard_age_stats.summary(),
                "adverse_markout_usd": acc.aster_residual_markout_guard_adverse_stats.summary(),
                "residual_unrealised_usd": acc.aster_residual_markout_guard_unrealised_stats.summary(),
                "cleanup_fee_estimate_usd": acc.aster_residual_markout_guard_cleanup_fee_stats.summary(),
                "refresh_attempts": acc.aster_residual_markout_guard_refresh_attempts,
                "refresh_outcomes": dict(acc.aster_residual_markout_guard_refresh_outcome_counts),
                "refresh_suppressed": dict(acc.aster_residual_markout_guard_refresh_suppressed_counts),
                "refresh_latency_ms": acc.aster_residual_markout_guard_refresh_latency_stats.summary(),
                "fresh_account_age_ms": acc.aster_residual_markout_guard_fresh_account_age_stats.summary(),
                "last": dict(acc.aster_residual_markout_guard_last),
            },
        },
        "canary_breach_response_summary": {
            "total_records": acc.canary_breach_response_total,
            "active_ticks": acc.canary_breach_active_ticks,
            "candidate_target_ticks": acc.canary_breach_candidate_target_ticks,
            "observation_active_ticks": acc.canary_breach_observation_active_ticks,
            "observation_uncovered_target_ticks": acc.canary_breach_observation_uncovered_target_ticks,
            "zero_target_hold_ticks": acc.canary_breach_zero_target_hold_ticks,
            "zero_target_hold_windows": acc.canary_breach_zero_target_hold_windows,
            "max_zero_target_hold_run": acc.canary_breach_zero_target_hold_max_run,
            "first_zero_target_hold_tick": acc.canary_breach_zero_target_hold_first_tick,
            "last_zero_target_hold_tick": acc.canary_breach_zero_target_hold_last_tick,
            "response_modes": dict(acc.canary_breach_response_mode_counts),
        },
        "venue_health": venue_health,
        "orders_per_venue": orders_per_venue,
        "fills": {
            "total_count": int(sum(acc.venue_fill_counts[i] for i in report_indices)),
            "total_base": round(sum(acc.venue_fill_size_base[i] for i in report_indices), 6),
            "by_venue": fills_by_venue,
        },
        "risk": {
            "kill_switch_count": acc.kill_switch_count,
            "kill_reasons": dict(acc.kill_reason_counts),
            "would_send_zero_pct": round(100.0 * acc.would_send_zero_ticks / acc.tick_count, 4)
            if acc.tick_count > 0
            else 0.0,
            "soft_governor_ticks": acc.soft_governor_ticks,
            "soft_governor_reasons": dict(acc.soft_governor_reason_counts),
            "soft_governor_blocked_ticks": {
                acc.venue_name(i): acc.soft_governor_blocked_ticks[i] for i in report_indices
            },
        },
        "hedges": {
            "ticks_with_hedges": acc.hedges_tick_count,
        },
        "pnl_validity": {
            "final_q_global_tao": round(acc.final_q_global_tao, 6) if acc.final_q_global_tao is not None else None,
            "final_pnl_total": round(acc.final_pnl_total, 6) if acc.final_pnl_total is not None else None,
            "final_pnl_realised": round(acc.final_pnl_realised, 6) if acc.final_pnl_realised is not None else None,
            "final_pnl_unrealised": round(acc.final_pnl_unrealised, 6) if acc.final_pnl_unrealised is not None else None,
            "mm_place_total": acc.mm_place_total,
            "mm_keep_total": acc.mm_keep_total,
            "mm_replace_total": acc.mm_replace_total,
        },
        "anomalies": {
            "total": len(acc.anomalies.items),
            "critical": len(acc.anomalies.by_severity("Critical")),
            "warning": len(acc.anomalies.by_severity("Warning")),
            "by_category": dict(acc.anomalies.counts),
        },
        "economics_attribution": economics_attribution_snapshot(acc, report_indices),
    }


def save_metrics_snapshot(acc: TelemetryAccumulator, path: Path) -> None:
    payload = build_metrics_snapshot(acc)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Metrics snapshot saved: {path}", file=sys.stderr)


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
    venue_names: list[str] = []
    for vn in curr_snapshot.get("venue_names", []):
        if isinstance(vn, str) and vn not in venue_names:
            venue_names.append(vn)
    for vn in prev.get("venue_names", []):
        if isinstance(vn, str) and vn not in venue_names:
            venue_names.append(vn)
    for vn in sorted(set(prev.get("venue_health", {})) | set(curr_snapshot.get("venue_health", {}))):
        if vn not in venue_names:
            venue_names.append(vn)

    for vn in venue_names:
        pvh = prev.get("venue_health", {}).get(vn, {})
        cvh = curr_snapshot.get("venue_health", {}).get(vn, {})
        lines.append(compare(f"{vn} healthy%", pvh.get("healthy_pct"), cvh.get("healthy_pct"), lower_is_better=False))
        lines.append(compare(f"{vn} age p95", pvh.get("age_p95"), cvh.get("age_p95")))
        lines.append(compare(f"{vn} age_event p95", pvh.get("age_event_p95"), cvh.get("age_event_p95")))

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
    parser.add_argument(
        "--execution-mode",
        choices=("all", "shadow", "paper", "live"),
        default="all",
        help="Filter rows by execution_mode before analysis (default: all)",
    )
    parser.add_argument(
        "--last-segment",
        action="store_true",
        help="Only analyze the last contiguous segment for the chosen execution mode",
    )
    parser.add_argument("--max-ticks", type=int, default=0, help="Max ticks to process (0=all)")
    parser.add_argument(
        "--tail-bytes",
        type=int,
        default=0,
        help=(
            "Only analyze the last N bytes of the telemetry file. "
            "Use this for production-safe triage on large rolling files."
        ),
    )
    parser.add_argument(
        "--unsafe-allow-large-live-file",
        action="store_true",
        help=(
            "Allow a full scan of the rolling production telemetry file even when it is very large. "
            "This is intentionally unsafe on the live host."
        ),
    )
    parser.add_argument("--checkpoint-json", type=str, default=None, help="Path to save checkpoint JSON")
    parser.add_argument("--metrics-json", type=str, default=None, help="Path to save machine-readable metrics JSON")
    parser.add_argument("--prev-checkpoint", type=str, default=None, help="Path to previous checkpoint JSON for regression comparison")
    parser.add_argument(
        "--validation-profile",
        choices=("none", "pnl-baseline", "mm-churn-probe"),
        default="none",
        help=(
            "Apply measurement validity checks for a specific artifact type. "
            "Use pnl-baseline for coherent flat paper baselines and mm-churn-probe "
            "for decision-exercising paper/shadow probes."
        ),
    )
    parser.add_argument("--output", type=str, default=None, help="Path to write report (default: stdout)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    telemetry_path = Path(args.telemetry)

    if not telemetry_path.exists():
        print(f"ERROR: telemetry file not found: {telemetry_path}", file=sys.stderr)
        return 1

    try:
        guard_unsafe_live_scan(
            telemetry_path,
            tail_bytes=args.tail_bytes,
            unsafe_allow_large_live_file=args.unsafe_allow_large_live_file,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Analyzing {telemetry_path} ...", file=sys.stderr)
    try:
        acc = stream_records(
            telemetry_path,
            max_ticks=args.max_ticks,
            execution_mode=args.execution_mode,
            last_segment=args.last_segment,
            tail_bytes=args.tail_bytes,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if acc.tick_count == 0:
        print("ERROR: no ticks found in telemetry", file=sys.stderr)
        return 1

    report = generate_report(acc, validation_profile=args.validation_profile)

    # Checkpoint
    if args.checkpoint_json:
        cp_path = Path(args.checkpoint_json)
        save_checkpoint(acc, cp_path)

        # Regression scorecard
        if args.prev_checkpoint:
            prev_path = Path(args.prev_checkpoint)
            scorecard = regression_scorecard(prev_path, json.loads(cp_path.read_text()))
            report += "\n\n" + "=" * 80 + "\n  REGRESSION SCORECARD\n" + "=" * 80 + "\n" + scorecard

    if args.metrics_json:
        metrics_path = Path(args.metrics_json)
        save_metrics_snapshot(acc, metrics_path)

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
