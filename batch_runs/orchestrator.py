#!/usr/bin/env python3
"""
orchestrator.py

Generic experiment orchestrator for Paraphina.

Goals
=====
- Provide a single, reusable way to run the `paraphina` binary many times with
  different env vars / labels / seeds.
- Keep ALL assumptions about stdout parsing in pluggable functions.
- Make it trivial for experiments (exp01/02/03/...) to:
    - define a grid of configs,
    - run many sims,
    - get back a tidy DataFrame of per-run metrics,
    - compute summaries grouped by any keys.

This module is intentionally:
- agnostic to CLI shape (you pass the exact `cmd` you want),
- agnostic to metrics (you pass a `parse_metrics(stdout)` function).
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EngineRunConfig:
    """
    Full specification of a single engine run.

    Fields
    ------
    cmd:
        List of command-line arguments to execute, e.g.:
        ["../target/release/paraphina", "--ticks", "2000"]

        The orchestrator makes **no assumptions** about CLI flags. If your
        binary has no flags, you can just pass ["../target/release/paraphina"].

    env:
        Environment variable overrides for this run. Will be overlaid on top
        of the current process os.environ.

        Typical keys include:
            PARAPHINA_RISK_PROFILE
            PARAPHINA_INIT_Q_TAO
            PARAPHINA_HEDGE_BAND_BASE
            PARAPHINA_HEDGE_MAX_STEP
            PARAPHINA_MM_SIZE_ETA
            PARAPHINA_VOL_REF
            PARAPHINA_DAILY_LOSS_LIMIT
            (and any future telemetry/env knobs)

    label:
        Arbitrary metadata describing this run, e.g.:
            {
                "experiment": "exp03_stress_search",
                "profile": "aggressive",
                "init_q_tao": -20.0,
                "repeat": 0,
            }

        This will be merged into the output DataFrame columns, so keep it
        JSON-serialisable (str / int / float / bool).

    run_id:
        Optional unique identifier. If None, the orchestrator will generate a
        simple integer index when converting results to a DataFrame.

    workdir:
        Optional working directory for the subprocess. Defaults to the current
        directory where the script is run.
    """
    cmd: List[str]
    env: Dict[str, str]
    label: Dict[str, Any]
    run_id: Optional[str] = None
    workdir: Optional[Path] = None


@dataclass
class RunResult:
    """
    Result of a single engine run, including raw logs and parsed metrics.

    Fields
    ------
    run_id:
        Identifier of the run (from EngineRunConfig or an index).

    label:
        Copied from EngineRunConfig.label.

    metrics:
        Parsed metrics dictionary returned by parse_metrics(stdout).

    duration_sec:
        Wall-clock runtime in seconds.

    returncode:
        Subprocess return code. Non-zero should be treated as an error.

    stdout:
        Raw stdout from the engine as a single string.

    stderr:
        Raw stderr from the engine as a single string.
    """
    run_id: str
    label: Dict[str, Any]
    metrics: Dict[str, Any]
    duration_sec: float
    returncode: int
    stdout: str
    stderr: str


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_engine(
    cfg: EngineRunConfig,
    parse_metrics: Callable[[str], Dict[str, Any]],
    timeout_sec: Optional[int] = None,
) -> RunResult:
    """
    Run a single Paraphina simulation and parse its metrics.

    Parameters
    ----------
    cfg:
        EngineRunConfig specifying the command, env, labels.

    parse_metrics:
        Function taking stdout (str) and returning a metrics dict.
        This keeps exp01/02/03-specific parsing logic **out** of the core
        orchestrator.

    timeout_sec:
        Optional timeout for the subprocess. If exceeded, subprocess.run will
        raise a TimeoutExpired.

    Returns
    -------
    RunResult
        Contains parsed metrics, stdout/stderr, runtime, and returncode.

    Notes
    -----
    - Any exception (e.g. TimeoutExpired) should be handled by the caller.
    - If returncode != 0, metrics may be empty; caller should decide how to
      handle that.
    """
    env = os.environ.copy()
    env.update({k: str(v) for k, v in cfg.env.items()})

    workdir = str(cfg.workdir) if cfg.workdir is not None else None

    t0 = time.time()
    proc = subprocess.run(
        cfg.cmd,
        env=env,
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    t1 = time.time()

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    # Always try to parse metrics, even if returncode != 0; the caller can
    # decide to drop or inspect them.
    try:
        metrics = parse_metrics(stdout)
    except Exception as e:  # noqa: BLE001
        # Fall back to an empty metrics dict on parse error.
        metrics = {"metrics_parse_error": str(e)}

    # Normalise run_id to a string; if cfg.run_id is None, use placeholder.
    run_id = cfg.run_id if cfg.run_id is not None else ""

    return RunResult(
        run_id=str(run_id),
        label=dict(cfg.label),
        metrics=metrics,
        duration_sec=t1 - t0,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def run_many(
    configs: List[EngineRunConfig],
    parse_metrics: Callable[[str], Dict[str, Any]],
    timeout_sec: Optional[int] = None,
    verbose: bool = True,
) -> List[RunResult]:
    """
    Run many Paraphina simulations sequentially.

    Parameters
    ----------
    configs:
        List of EngineRunConfig objects.

    parse_metrics:
        Function to parse stdout into metrics dicts.

    timeout_sec:
        Optional timeout per run.

    verbose:
        If True, print progress to stdout.

    Returns
    -------
    List[RunResult]
    """
    results: List[RunResult] = []

    for idx, cfg in enumerate(configs):
        if cfg.run_id is None:
            cfg.run_id = f"run_{idx}"

        if verbose:
            print(
                f"[orchestrator] Starting run {idx+1}/{len(configs)} "
                f"(run_id={cfg.run_id}, label={cfg.label})"
            )

        try:
            res = run_engine(cfg, parse_metrics=parse_metrics, timeout_sec=timeout_sec)
        except subprocess.TimeoutExpired as e:  # noqa: BLE001
            if verbose:
                print(f"[orchestrator] TimeoutExpired for run_id={cfg.run_id}: {e}")
            res = RunResult(
                run_id=str(cfg.run_id),
                label=dict(cfg.label),
                metrics={"timeout": True},
                duration_sec=float(timeout_sec) if timeout_sec is not None else float("nan"),
                returncode=-1,
                stdout="",
                stderr=str(e),
            )

        results.append(res)

        if verbose:
            print(
                f"[orchestrator] Finished run_id={cfg.run_id} "
                f"rc={res.returncode} duration={res.duration_sec:.3f}s"
            )

    return results


# ---------------------------------------------------------------------------
# Conversion to DataFrame + summarisation
# ---------------------------------------------------------------------------

def results_to_dataframe(results: List[RunResult]) -> pd.DataFrame:
    """
    Flatten a list of RunResult objects into a single tidy DataFrame.

    Columns
    -------
    - All label keys.
    - All metric keys.
    - run_id
    - duration_sec
    - returncode

    Notes
    -----
    - stderr/stdout are **not** included in the DataFrame by default to keep
      it small. If you need them, you can extend this function or store them
      separately.
    """
    rows: List[Dict[str, Any]] = []

    for res in results:
        row: Dict[str, Any] = {}
        row["run_id"] = res.run_id
        row["duration_sec"] = res.duration_sec
        row["returncode"] = res.returncode

        # Merge label and metrics; metrics take precedence on key collision.
        row.update(res.label)
        row.update(res.metrics)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def summarise_runs(
    df: pd.DataFrame,
    group_keys: List[str],
    numeric_metrics: Optional[List[str]] = None,
    kill_switch_col: str = "kill_switch",
) -> pd.DataFrame:
    """
    Compute grouped summary statistics from a per-run DataFrame.

    Parameters
    ----------
    df:
        Per-run DataFrame (e.g. from results_to_dataframe).

    group_keys:
        Columns to group by (e.g. ["experiment", "profile", "init_q_tao"]).

    numeric_metrics:
        List of numeric metric column names to aggregate. If None, this will
        default to all float/int columns except group_keys.

    kill_switch_col:
        Name of the boolean column indicating kill switch activation. If
        present, we will compute a `kill_switch_frac` column in the summary.

    Returns
    -------
    pd.DataFrame
        Summary with columns:
        - group_keys...
        - <metric>_mean
        - <metric>_std
        - kill_switch_frac (if kill_switch_col present)
        - num_runs
    """
    if df.empty:
        return df

    if numeric_metrics is None:
        numeric_metrics = [
            c
            for c in df.columns
            if c not in group_keys
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    agg_dict: Dict[str, List[str]] = {}
    for m in numeric_metrics:
        agg_dict[m] = ["mean", "std"]

    grouped = df.groupby(group_keys, dropna=False).agg(agg_dict)

    # Flatten MultiIndex columns: metric_mean, metric_std
    grouped.columns = [f"{m}_{stat}" for (m, stat) in grouped.columns]
    grouped = grouped.reset_index()

    # Add kill_switch_frac if requested and present
    if kill_switch_col in df.columns:
        # Treat anything truthy as kill switch "on"
        kfrac = (
            df.assign(_ks=df[kill_switch_col].astype(bool))
            .groupby(group_keys, dropna=False)["_ks"]
            .mean()
            .rename("kill_switch_frac")
        )
        grouped = grouped.merge(kfrac.reset_index(), on=group_keys, how="left")

    # Add num_runs
    counts = df.groupby(group_keys, dropna=False).size().rename("num_runs")
    grouped = grouped.merge(counts.reset_index(), on=group_keys, how="left")

    return grouped
