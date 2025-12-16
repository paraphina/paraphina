#!/usr/bin/env python3
"""
world_model_dataset.py

Utilities to turn Paraphina telemetry JSONL + per-run metadata into
decision-level trajectories for world-model / RL experiments.

This does NOT depend on any deep learning framework. It just produces
clean numpy arrays / Python lists that you can later feed into PyTorch/JAX.

Assumptions
===========
- Telemetry JSONL comes from the current StrategyRunner telemetry, i.e.
  per-tick rows with at least:

      t, pnl_total, pnl_realised, pnl_unrealised,
      risk_regime, kill_switch,
      q_global_tao, dollar_delta_usd, basis_usd

- For world-model / RL we aggregate ticks into *decision steps* of length
  `decision_horizon` (in ticks), e.g. 10 or 20 ticks.

- For now the "action" is a run-level config vector (band_base, mm_size_eta,
  vol_ref, etc.). Within a single run, actions are constant across time, but
  different runs use different actions.

Outputs
=======
Baseline state-only trajectories:

- `DecisionTrajectory` with:

      states : np.ndarray [T, D_state]
      rewards: np.ndarray [T]
      dones  : np.ndarray [T] (bool)
      meta   : dict (profile, init_q_tao, scenario, etc.)

State+action trajectories:

- `DecisionTrajectorySA` with:

      states  : np.ndarray [T, D_state]
      actions : np.ndarray [T, D_action]
      rewards : np.ndarray [T]
      dones   : np.ndarray [T]
      meta    : dict

- `build_trajectories_from_runs_csv(...)`:
      existing state-only builder (used by wm01, wm02).

- `build_trajectories_sa_from_runs_csv(...)`:
      extended builder that also constructs actions from exp04-style runs CSV.

This is the core building block for Stage 6 (world-model + RL).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ts_metrics import load_telemetry_jsonl


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DecisionTrajectory:
    """Single-episode decision-level trajectory (state-only)."""

    states: np.ndarray  # [T, D_state]
    rewards: np.ndarray  # [T]
    dones: np.ndarray  # [T] bool
    meta: Dict[str, object]


@dataclass
class DecisionTrajectorySA:
    """
    Single-episode decision-level trajectory with actions.

    - states  : [T, D_state]
    - actions : [T, D_action] (per-step actions; here constant within a run)
    - rewards : [T]
    - dones   : [T]
    - meta    : run-level metadata
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    meta: Dict[str, object]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

RISK_REGIME_ORDER = ["Normal", "Warning", "HardLimit"]
RISK_REGIME_TO_ID = {name: idx for idx, name in enumerate(RISK_REGIME_ORDER)}


def _risk_regime_one_hot(series: pd.Series) -> np.ndarray:
    """Map risk_regime string series to one-hot rows [N, 3]."""
    ids = series.map(RISK_REGIME_TO_ID).fillna(0).astype(int).values
    one_hot = np.zeros((len(ids), len(RISK_REGIME_ORDER)), dtype=np.float32)
    one_hot[np.arange(len(ids)), ids] = 1.0
    return one_hot


def _build_decision_steps(
    df_ts: pd.DataFrame,
    decision_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate per-tick telemetry into decision-level states, rewards, dones.

    df_ts: per-tick telemetry with columns:
        t, pnl_total, risk_regime, kill_switch,
        q_global_tao, dollar_delta_usd, basis_usd, ...

    decision_horizon: number of ticks per decision step.

    Returns:
        states  [T, D_state]
        rewards [T]
        dones   [T] bool
    """
    df_ts = df_ts.sort_values("t").reset_index(drop=True)

    # Number of complete decision steps
    num_ticks = len(df_ts)
    if num_ticks < decision_horizon:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )

    num_steps = num_ticks // decision_horizon

    states: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[bool] = []

    # One-hot risk regimes per tick (for aggregation)
    risk_oh = _risk_regime_one_hot(df_ts["risk_regime"])

    for step_idx in range(num_steps):
        start = step_idx * decision_horizon
        end = (step_idx + 1) * decision_horizon
        slice_df = df_ts.iloc[start:end]
        slice_risk_oh = risk_oh[start:end]

        last = slice_df.iloc[-1]

        # State features at end of slice + slice statistics
        pnl_start = float(slice_df["pnl_total"].iloc[0])
        pnl_end = float(slice_df["pnl_total"].iloc[-1])
        pnl_delta = pnl_end - pnl_start

        # Max drawdown within slice (using pnl_total)
        pnl_series = slice_df["pnl_total"].values.astype(float)
        cummax = np.maximum.accumulate(pnl_series)
        drawdowns = cummax - pnl_series
        dd_in_slice = float(drawdowns.max())

        # Regime occupancy in slice
        frac_risk = slice_risk_oh.mean(axis=0)  # [3]

        # Basic exposure at end of slice
        q_global = float(last.get("q_global_tao", np.nan))
        delta_usd = float(last.get("dollar_delta_usd", np.nan))
        basis_usd = float(last.get("basis_usd", np.nan))

        # Optional vol features if present (may be NaN if not logged yet)
        sigma_eff = float(last.get("sigma_eff", np.nan))
        vol_ratio = float(last.get("vol_ratio", np.nan))

        # Build state vector
        state_vec = np.array(
            [
                pnl_end,
                pnl_delta,
                dd_in_slice,
                q_global,
                delta_usd,
                basis_usd,
                sigma_eff,
                vol_ratio,
                *frac_risk.tolist(),  # Normal / Warning / HardLimit fractions
            ],
            dtype=np.float32,
        )
        states.append(state_vec)

        # Reward: pnl_delta minus small penalty for drawdown in slice
        reward = pnl_delta - 0.1 * dd_in_slice
        rewards.append(float(reward))

        # Done if kill_switch happened anywhere in the slice
        done = bool(slice_df["kill_switch"].any())
        dones.append(done)

        if done:
            # Truncate here; subsequent slices are meaningless for RL episodes
            break

    if not states:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )

    states_arr = np.stack(states, axis=0)  # [T, D_state]

    # IMPORTANT: sanitise NaN / inf values so downstream ML gets finite inputs.
    states_arr = np.nan_to_num(states_arr, nan=0.0, posinf=1e6, neginf=-1e6)

    rewards_arr = np.asarray(rewards, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=bool)
    return states_arr, rewards_arr, dones_arr


# ---------------------------------------------------------------------------
# Public API: state-only trajectories
# ---------------------------------------------------------------------------

def build_trajectory_from_telemetry(
    telem_path: Path,
    decision_horizon: int = 10,
) -> Optional[DecisionTrajectory]:
    """
    Load a single telemetry JSONL file and build a DecisionTrajectory.

    Returns None if there is not enough data to form at least one decision step.
    """
    df_ts = load_telemetry_jsonl(telem_path)
    if df_ts.empty:
        return None

    states, rewards, dones = _build_decision_steps(df_ts, decision_horizon)
    if states.size == 0:
        return None

    meta = {
        "telemetry_path": str(telem_path),
    }
    return DecisionTrajectory(states=states, rewards=rewards, dones=dones, meta=meta)


def build_trajectories_from_runs_csv(
    runs_csv: Path,
    decision_horizon: int = 10,
    telemetry_path_column: str = "telemetry_path",
    include_actions: bool = False,
):
    """
    Build trajectories for each run in a per-run CSV.

    Typical usage (wm01, wm02):

        trajs = build_trajectories_from_runs_csv(runs_csv, decision_horizon=10)

    Each CSV row is expected to have:

        - telemetry_path: path to the JSONL telemetry for that run
        - profile, init_q_tao, scenario, etc. (all copied into meta)

    If `include_actions=True`, this is a thin wrapper around
    `build_trajectories_sa_from_runs_csv(...)` and returns
    `List[DecisionTrajectorySA]` (state + action trajectories), which is
    what wm03 / wm04 expect.
    """
    runs_csv = Path(runs_csv)

    # ---------------------------------------------------------------
    # SA MODE: delegate to the state+action builder
    # ---------------------------------------------------------------
    if include_actions:
        return build_trajectories_sa_from_runs_csv(
            runs_csv,
            decision_horizon=decision_horizon,
            telemetry_path_column=telemetry_path_column,
        )

    # ---------------------------------------------------------------
    # State-only mode (original behaviour, used by wm01 / wm02)
    # ---------------------------------------------------------------
    df_runs = pd.read_csv(runs_csv)

    if telemetry_path_column not in df_runs.columns:
        raise ValueError(
            f"Column '{telemetry_path_column}' not found in {runs_csv}. "
            "Make sure exp04/exp05 were run with telemetry enabled and "
            "that telemetry paths are recorded in the runs CSV."
        )

    trajectories: List[DecisionTrajectory] = []

    for _, row in df_runs.iterrows():
        telem_path = Path(row[telemetry_path_column])
        traj = build_trajectory_from_telemetry(telem_path, decision_horizon)
        if traj is None:
            continue

        # Attach run-level metadata
        meta = dict(row)  # copy all columns from runs CSV
        meta["telemetry_path"] = str(telem_path)
        traj.meta.update(meta)

        trajectories.append(traj)

    return trajectories

# ---------------------------------------------------------------------------
# Public API: state+action trajectories (for RL)
# ---------------------------------------------------------------------------

def _build_action_vector_from_row(row: pd.Series) -> np.ndarray:
    """
    Build a *normalised* action vector from a per-run CSV row.

    We treat the following fields (if present) as "knobs" the policy could tune:

        band_base        (TAO)
        mm_size_eta      (dimensionless)
        vol_ref_override or vol_ref_centre (vol level)
        daily_loss_limit (USD)
        init_q_tao       (TAO)
        vol_scale        (scenario vol multiplier)

    We normalise them into a rough O(1) range.
    """
    band_base = float(row.get("band_base", np.nan))
    mm_size_eta = float(row.get("mm_size_eta", np.nan))
    vol_ref = float(
        row.get("vol_ref_override", row.get("vol_ref_centre", np.nan))
    )
    daily_loss_limit = float(row.get("daily_loss_limit", np.nan))
    init_q_tao = float(row.get("init_q_tao", np.nan))
    vol_scale = float(row.get("vol_scale", np.nan))

    a = np.array(
        [
            band_base / 5.0,           # ~ [0.3, 1.2]
            mm_size_eta * 40.0,        # 0.025→1.0, 0.1→4.0
            vol_ref * 100.0,           # 0.015→1.5, etc.
            daily_loss_limit / 5000.0, # 750→0.15, 5000→1.0
            init_q_tao / 40.0,         # -40..40 → -1..1
            vol_scale / 2.0,           # 0.5..1.5 → 0.25..0.75
        ],
        dtype=np.float32,
    )

    a = np.nan_to_num(a, nan=0.0, posinf=1e6, neginf=-1e6)
    return a


def build_trajectories_sa_from_runs_csv(
    runs_csv: Path,
    decision_horizon: int = 10,
    telemetry_path_column: str = "telemetry_path",
) -> List[DecisionTrajectorySA]:
    """
    Build DecisionTrajectorySA objects (state + action) for each run.

    This reuses the state-only builder `build_trajectory_from_telemetry` to
    construct states / rewards / dones, and augments each trajectory with a
    per-step action vector derived from the run-level config.

    Actions are currently **constant within a run**, but vary across runs
    (profile, scenario, band_base, mm_size_eta, vol_ref, etc.).
    """
    runs_csv = Path(runs_csv)
    df_runs = pd.read_csv(runs_csv)

    if telemetry_path_column not in df_runs.columns:
        raise ValueError(
            f"Column '{telemetry_path_column}' not found in {runs_csv}. "
            "Make sure exp04/exp05 were run with telemetry enabled and "
            "that telemetry paths are recorded in the runs CSV."
        )

    trajectories: List[DecisionTrajectorySA] = []

    for _, row in df_runs.iterrows():
        telem_path = Path(row[telemetry_path_column])
        base_traj = build_trajectory_from_telemetry(telem_path, decision_horizon)
        if base_traj is None:
            continue

        action_vec = _build_action_vector_from_row(row)  # [D_action]
        T = base_traj.states.shape[0]
        actions = np.repeat(action_vec[None, :], T, axis=0)  # [T, D_action]

        meta = dict(row)
        meta["telemetry_path"] = str(telem_path)
        base_traj.meta.update(meta)

        trajectories.append(
            DecisionTrajectorySA(
                states=base_traj.states,
                actions=actions,
                rewards=base_traj.rewards,
                dones=base_traj.dones,
                meta=base_traj.meta,
            )
        )

    return trajectories
