// src/bin/monte_carlo.rs
//
// Milestone B: Monte Carlo / research harness runner.
//
// Goals:
// - Deterministic multi-run evaluation using seed offsets.
// - Uses existing engine + strategy components (mm/exit/hedge) in the spec order:
//     MM fills -> recompute -> exits -> recompute -> hedge -> recompute
// - Adds controlled stochasticity via jittered tick times (so dummy mids differ per seed)
//   without changing core engine logic.
// - Generates Evidence Pack v1 for audit/verification (Step 7.1).
//
// Run examples:
//   cargo run --bin monte_carlo -- --runs 50 --ticks 600 --seed 1 --jitter-ms 500 --profile Balanced
//   PARAPHINA_RISK_PROFILE=Conservative cargo run --bin monte_carlo -- --runs 100 --ticks 1200 --seed 42 --quiet
//
// Optional CSV export:
//   cargo run --bin monte_carlo -- --runs 200 --ticks 1200 --seed 7 --jitter-ms 250 --csv runs.csv
//
// Default output directory:
//   cargo run --bin monte_carlo -- --output-dir runs/demo_step_7_1
//
// Sharded Monte Carlo (for scale):
//   # Run shard 0 of a 10-shard run with 1000 total runs:
//   cargo run --bin monte_carlo -- --runs 1000 --run-start-index 0 --run-count 100 --seed 42 --output-dir runs/shard_0
//
//   # Summarize aggregated JSONL:
//   cargo run --bin monte_carlo -- summarize --input runs/aggregated/mc_runs.jsonl --out-dir runs/aggregated
//
// Deterministic seed mapping contract:
//   For global run index i, seed_i = base_seed + i (u64 wrapping add).
//   The Monte Carlo loop iterates global indices [run_start_index, run_start_index + run_count).
//
// Notes:
// - This harness applies intents as immediate fills (no gateway), using venue taker fees.
//   This keeps the harness dependency-free and deterministic.

use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use paraphina::config::{resolve_effective_profile, Config, RiskProfile};
use paraphina::engine::Engine;
use paraphina::exit;
use paraphina::hedge;
use paraphina::metrics::{DrawdownTracker, OnlineStats};
use paraphina::mm;
use paraphina::sim_eval::write_evidence_pack;
use paraphina::state::{GlobalState, KillReason};
use paraphina::tail_risk::{TailRiskMetrics, DEFAULT_VAR_ALPHA};
use paraphina::types::{OrderIntent, TimestampMs};
use serde::Serialize;

const DEFAULT_RUNS: usize = 50;
const DEFAULT_TICKS: usize = 600;
const DEFAULT_SEED: u64 = 1;
const DEFAULT_TICK_MS: i64 = 1000;
const DEFAULT_JITTER_MS: i64 = 0;
const DEFAULT_PRINT_EVERY: usize = 1;
const DEFAULT_OUTPUT_DIR: &str = "runs/demo_step_7_1";

/// Command mode for the monte_carlo binary.
#[derive(Debug, Clone)]
enum Command {
    /// Default: run Monte Carlo simulation.
    Run(RunArgs),
    /// Summarize mode: aggregate JSONL into mc_summary.json.
    Summarize(SummarizeArgs),
}

/// Arguments for the `run` command (default Monte Carlo simulation).
#[derive(Debug, Clone)]
struct RunArgs {
    runs: usize,
    ticks: usize,
    seed: u64,
    tick_ms: i64,
    jitter_ms: i64,
    profile: Option<RiskProfile>,
    quiet: bool,
    print_every: usize,
    csv_out: Option<PathBuf>,
    output_dir: PathBuf,
    /// Start index for sharded runs (default: 0).
    /// For global run index i, seed_i = seed + i.
    run_start_index: usize,
    /// Number of runs to execute in this shard (default: same as --runs).
    /// The Monte Carlo loop iterates [run_start_index, run_start_index + run_count).
    run_count: Option<usize>,
    /// Number of threads for parallel execution (default: 1).
    /// When threads > 1, runs are executed in parallel using a rayon thread pool.
    threads: usize,
}

/// Arguments for the `summarize` command.
#[derive(Debug, Clone)]
struct SummarizeArgs {
    /// Path to the input mc_runs.jsonl file.
    input: PathBuf,
    /// Output directory for mc_summary.json and evidence pack.
    out_dir: PathBuf,
    /// Optional base seed for validation (if not provided, seed contract is not validated).
    base_seed: Option<u64>,
}

impl RunArgs {
    fn usage() -> &'static str {
        "\
paraphina Monte Carlo harness (Milestone B)

USAGE:
  cargo run --bin monte_carlo -- [FLAGS]
  cargo run --bin monte_carlo -- summarize --input <PATH> --out-dir <DIR>

COMMANDS:
  (default)            Run Monte Carlo simulation
  summarize            Aggregate JSONL runs into mc_summary.json

PROFILE PRECEDENCE:
  1) --profile overrides environment
  2) else PARAPHINA_RISK_PROFILE
  3) else Balanced

RUN FLAGS:
  --profile NAME       Balanced | Conservative | Aggressive (alias: Loose)
  --runs N             Total runs in the full Monte Carlo (default: 50)
  --ticks N            Ticks per run (default: 600)
  --seed U64           Base seed (default: 1). Run i uses seed + i.
  --run-start-index N  Start index for sharded runs (default: 0)
  --run-count N        Number of runs in this shard (default: --runs)
  --tick-ms MS         Base tick duration in ms (default: 1000)
  --jitter-ms MS       Per-tick jitter added to tick-ms in [-MS, +MS] (default: 0)
  --print-every N      Print every N runs (default: 1). Ignored with --quiet.
  --csv PATH           Write per-run CSV rows to PATH (relative to output-dir)
  --output-dir DIR     Output directory (default: runs/demo_step_7_1)
  --threads N          Number of threads for parallel execution (default: 1)
  --quiet              Suppress per-run lines; only print final summary
  --help               Show this help

SUMMARIZE FLAGS:
  --input PATH         Path to mc_runs.jsonl (required)
  --out-dir DIR        Output directory (required)
  --base-seed U64      Base seed for validation (optional)

OUTPUT:
  The harness writes to <output-dir>/:
    - mc_summary.json     Per-run statistics and aggregate summary
    - mc_runs.jsonl       JSONL of per-run metrics (always written)
    - mc_runs.csv         CSV of per-run metrics (if --csv specified)
    - monte_carlo.yaml    Configuration used for this run
    - evidence_pack/      Evidence Pack v1 for verification

DETERMINISTIC SEED CONTRACT:
  For global run index i, the scenario seed is: seed_i = base_seed + i (u64 wrap).
  The Monte Carlo loop iterates global indices [run_start_index, run_start_index + run_count).

EXAMPLES:
  # Standard run
  cargo run --bin monte_carlo -- --runs 100 --ticks 1200 --seed 7 --jitter-ms 500 --profile Balanced

  # Sharded run (shard 0 of 10 shards, 1000 total runs)
  cargo run --bin monte_carlo -- --runs 1000 --run-start-index 0 --run-count 100 --seed 42 --output-dir runs/shard_0

  # Summarize aggregated JSONL
  cargo run --bin monte_carlo -- summarize --input runs/mc_runs.jsonl --out-dir runs/summary

  PARAPHINA_RISK_PROFILE=Conservative cargo run --bin monte_carlo -- --runs 200 --ticks 800 --seed 42 --csv mc_runs.csv
"
    }

    fn parse_or_exit() -> Self {
        match Self::parse() {
            Ok(a) => a,
            Err(e) => {
                eprintln!("{e}\n\n{}", Self::usage());
                std::process::exit(2);
            }
        }
    }

    fn parse() -> Result<Self, String> {
        let mut out = RunArgs {
            runs: DEFAULT_RUNS,
            ticks: DEFAULT_TICKS,
            seed: DEFAULT_SEED,
            tick_ms: DEFAULT_TICK_MS,
            jitter_ms: DEFAULT_JITTER_MS,
            profile: None,
            quiet: false,
            print_every: DEFAULT_PRINT_EVERY,
            csv_out: None,
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
            run_start_index: 0,
            run_count: None,
            threads: 1,
        };

        let mut it = env::args().skip(1).peekable();

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    println!("{}", Self::usage());
                    std::process::exit(0);
                }
                "--quiet" => out.quiet = true,

                "--profile" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --profile".to_string())?;
                    out.profile = Some(
                        RiskProfile::parse(&v).ok_or_else(|| {
                            "Invalid --profile. Expected: Balanced | Conservative | Aggressive (or Loose)".to_string()
                        })?,
                    );
                }
                "--runs" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --runs".to_string())?;
                    out.runs = v
                        .parse::<usize>()
                        .map_err(|_| "Invalid --runs (expected integer)".to_string())?;
                    if out.runs == 0 {
                        return Err("--runs must be >= 1".to_string());
                    }
                }
                "--ticks" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --ticks".to_string())?;
                    out.ticks = v
                        .parse::<usize>()
                        .map_err(|_| "Invalid --ticks (expected integer)".to_string())?;
                    if out.ticks == 0 {
                        return Err("--ticks must be >= 1".to_string());
                    }
                }
                "--seed" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --seed".to_string())?;
                    out.seed = v
                        .parse::<u64>()
                        .map_err(|_| "Invalid --seed (expected u64)".to_string())?;
                }
                "--run-start-index" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --run-start-index".to_string())?;
                    out.run_start_index = v
                        .parse::<usize>()
                        .map_err(|_| "Invalid --run-start-index (expected integer)".to_string())?;
                }
                "--run-count" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --run-count".to_string())?;
                    let count = v
                        .parse::<usize>()
                        .map_err(|_| "Invalid --run-count (expected integer)".to_string())?;
                    if count == 0 {
                        return Err("--run-count must be >= 1".to_string());
                    }
                    out.run_count = Some(count);
                }
                "--tick-ms" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --tick-ms".to_string())?;
                    out.tick_ms = v
                        .parse::<i64>()
                        .map_err(|_| "Invalid --tick-ms (expected integer ms)".to_string())?;
                    if out.tick_ms <= 0 {
                        return Err("--tick-ms must be > 0".to_string());
                    }
                }
                "--jitter-ms" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --jitter-ms".to_string())?;
                    out.jitter_ms = v
                        .parse::<i64>()
                        .map_err(|_| "Invalid --jitter-ms (expected integer ms)".to_string())?;
                    if out.jitter_ms < 0 {
                        return Err("--jitter-ms must be >= 0".to_string());
                    }
                }
                "--print-every" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --print-every".to_string())?;
                    out.print_every = v
                        .parse::<usize>()
                        .map_err(|_| "Invalid --print-every (expected integer)".to_string())?;
                    if out.print_every == 0 {
                        return Err("--print-every must be >= 1".to_string());
                    }
                }
                "--csv" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --csv".to_string())?;
                    out.csv_out = Some(PathBuf::from(v));
                }
                "--output-dir" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --output-dir".to_string())?;
                    out.output_dir = PathBuf::from(v);
                }
                "--threads" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --threads".to_string())?;
                    out.threads = v
                        .parse::<usize>()
                        .map_err(|_| "Invalid --threads (expected integer)".to_string())?;
                    if out.threads == 0 {
                        return Err("--threads must be >= 1".to_string());
                    }
                }

                // Support --flag=value style for convenience.
                _ if arg.starts_with("--profile=") => {
                    let v = arg["--profile=".len()..].to_string();
                    out.profile = Some(
                        RiskProfile::parse(&v).ok_or_else(|| {
                            "Invalid --profile. Expected: Balanced | Conservative | Aggressive (or Loose)".to_string()
                        })?,
                    );
                }
                _ if arg.starts_with("--runs=") => {
                    let v = &arg["--runs=".len()..];
                    out.runs = v
                        .parse::<usize>()
                        .map_err(|_| "Invalid --runs (expected integer)".to_string())?;
                    if out.runs == 0 {
                        return Err("--runs must be >= 1".to_string());
                    }
                }
                _ if arg.starts_with("--ticks=") => {
                    let v = &arg["--ticks=".len()..];
                    out.ticks = v
                        .parse::<usize>()
                        .map_err(|_| "Invalid --ticks (expected integer)".to_string())?;
                    if out.ticks == 0 {
                        return Err("--ticks must be >= 1".to_string());
                    }
                }
                _ if arg.starts_with("--seed=") => {
                    let v = &arg["--seed=".len()..];
                    out.seed = v
                        .parse::<u64>()
                        .map_err(|_| "Invalid --seed (expected u64)".to_string())?;
                }
                _ if arg.starts_with("--run-start-index=") => {
                    let v = &arg["--run-start-index=".len()..];
                    out.run_start_index = v
                        .parse::<usize>()
                        .map_err(|_| "Invalid --run-start-index (expected integer)".to_string())?;
                }
                _ if arg.starts_with("--run-count=") => {
                    let v = &arg["--run-count=".len()..];
                    let count = v
                        .parse::<usize>()
                        .map_err(|_| "Invalid --run-count (expected integer)".to_string())?;
                    if count == 0 {
                        return Err("--run-count must be >= 1".to_string());
                    }
                    out.run_count = Some(count);
                }
                _ if arg.starts_with("--tick-ms=") => {
                    let v = &arg["--tick-ms=".len()..];
                    out.tick_ms = v
                        .parse::<i64>()
                        .map_err(|_| "Invalid --tick-ms (expected integer ms)".to_string())?;
                    if out.tick_ms <= 0 {
                        return Err("--tick-ms must be > 0".to_string());
                    }
                }
                _ if arg.starts_with("--jitter-ms=") => {
                    let v = &arg["--jitter-ms=".len()..];
                    out.jitter_ms = v
                        .parse::<i64>()
                        .map_err(|_| "Invalid --jitter-ms (expected integer ms)".to_string())?;
                    if out.jitter_ms < 0 {
                        return Err("--jitter-ms must be >= 0".to_string());
                    }
                }
                _ if arg.starts_with("--print-every=") => {
                    let v = &arg["--print-every=".len()..];
                    out.print_every = v
                        .parse::<usize>()
                        .map_err(|_| "Invalid --print-every (expected integer)".to_string())?;
                    if out.print_every == 0 {
                        return Err("--print-every must be >= 1".to_string());
                    }
                }
                _ if arg.starts_with("--csv=") => {
                    let v = &arg["--csv=".len()..];
                    out.csv_out = Some(PathBuf::from(v));
                }
                _ if arg.starts_with("--output-dir=") => {
                    let v = &arg["--output-dir=".len()..];
                    out.output_dir = PathBuf::from(v);
                }
                _ if arg.starts_with("--threads=") => {
                    let v = &arg["--threads=".len()..];
                    out.threads = v
                        .parse::<usize>()
                        .map_err(|_| "Invalid --threads (expected integer)".to_string())?;
                    if out.threads == 0 {
                        return Err("--threads must be >= 1".to_string());
                    }
                }

                other => return Err(format!("Unknown argument: {other}")),
            }
        }

        Ok(out)
    }
}

impl SummarizeArgs {
    fn parse() -> Result<Self, String> {
        let mut input: Option<PathBuf> = None;
        let mut out_dir: Option<PathBuf> = None;
        let mut base_seed: Option<u64> = None;

        let mut it = env::args().skip(2); // Skip binary name and "summarize"

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    println!("{}", RunArgs::usage());
                    std::process::exit(0);
                }
                "--input" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --input".to_string())?;
                    input = Some(PathBuf::from(v));
                }
                "--out-dir" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --out-dir".to_string())?;
                    out_dir = Some(PathBuf::from(v));
                }
                "--base-seed" => {
                    let v = it
                        .next()
                        .ok_or_else(|| "Missing value for --base-seed".to_string())?;
                    base_seed = Some(
                        v.parse::<u64>()
                            .map_err(|_| "Invalid --base-seed (expected u64)".to_string())?,
                    );
                }
                _ if arg.starts_with("--input=") => {
                    let v = &arg["--input=".len()..];
                    input = Some(PathBuf::from(v));
                }
                _ if arg.starts_with("--out-dir=") => {
                    let v = &arg["--out-dir=".len()..];
                    out_dir = Some(PathBuf::from(v));
                }
                _ if arg.starts_with("--base-seed=") => {
                    let v = &arg["--base-seed=".len()..];
                    base_seed = Some(
                        v.parse::<u64>()
                            .map_err(|_| "Invalid --base-seed (expected u64)".to_string())?,
                    );
                }
                other => return Err(format!("Unknown argument for summarize: {other}")),
            }
        }

        let input = input.ok_or_else(|| "Missing required --input for summarize".to_string())?;
        let out_dir =
            out_dir.ok_or_else(|| "Missing required --out-dir for summarize".to_string())?;

        Ok(Self {
            input,
            out_dir,
            base_seed,
        })
    }
}

/// Parse command from arguments.
fn parse_command() -> Command {
    let args: Vec<String> = env::args().collect();

    // Check if first non-binary arg is "summarize"
    if args.len() > 1 && args[1] == "summarize" {
        match SummarizeArgs::parse() {
            Ok(s) => Command::Summarize(s),
            Err(e) => {
                eprintln!("{e}\n\n{}", RunArgs::usage());
                std::process::exit(2);
            }
        }
    } else {
        Command::Run(RunArgs::parse_or_exit())
    }
}

/// Minimal deterministic RNG (xorshift64*) for jitter generation.
/// Not crypto; just stable + fast.
#[derive(Debug, Clone, Copy)]
struct XorShift64Star {
    state: u64,
}

impl XorShift64Star {
    fn new(seed: u64) -> Self {
        // Avoid zero state.
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: s }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn uniform_i64_inclusive(&mut self, lo: i64, hi: i64) -> i64 {
        if lo >= hi {
            return lo;
        }
        let span = (hi - lo) as u64;
        let r = self.next_u64();
        lo + (r % (span + 1)) as i64
    }
}

fn profile_name(p: RiskProfile) -> &'static str {
    match p {
        RiskProfile::Balanced => "Balanced",
        RiskProfile::Conservative => "Conservative",
        RiskProfile::Aggressive => "Aggressive",
    }
}

/// Apply order intents as immediate fills and return the number of fills applied.
///
/// Iterates intents in stable order, applies each valid fill via `apply_perp_fill`,
/// and returns the count. Used by Opt18 to conditionally skip `recompute_after_fills`
/// when no fills occurred.
///
/// This keeps the harness dependency-free and does not require gateway internals.
/// Pragmatically treats all fills as taker.
fn apply_intents_as_fills(cfg: &Config, state: &mut GlobalState, intents: &[OrderIntent]) -> usize {
    let mut fill_count = 0;
    for it in intents {
        // Only intents with (side, price, size) can be treated as synthetic fills.
        let (venue_index, side, size, price) = match it {
            OrderIntent::Place(pi) => (pi.venue_index, pi.side, pi.size, pi.price),
            OrderIntent::Replace(ri) => (ri.venue_index, ri.side, ri.size, ri.price),
            OrderIntent::Cancel(_) | OrderIntent::CancelAll(_) => continue,
        };

        if venue_index >= cfg.venues.len() {
            continue;
        }
        if !price.is_finite() || price <= 0.0 {
            continue;
        }
        if !size.is_finite() || size <= 0.0 {
            continue;
        }

        // Pragmatic: treat everything as taker for now.
        // (If you later want maker-vs-taker modeling, switch on purpose here.)
        let fee_bps = cfg.venues[venue_index].taker_fee_bps;
        state.apply_perp_fill(venue_index, side, size, price, fee_bps);
        fill_count += 1;
    }
    fill_count
}

fn update_peaks(
    state: &GlobalState,
    max_abs_delta: &mut f64,
    max_abs_basis: &mut f64,
    max_abs_q: &mut f64,
    max_venue_toxicity: &mut f64,
) {
    *max_abs_delta = max_abs_delta.max(state.dollar_delta_usd.abs());
    *max_abs_basis = max_abs_basis.max(state.basis_gross_usd.abs());
    *max_abs_q = max_abs_q.max(state.q_global_tao.abs());

    let tox = state.venues.iter().fold(0.0_f64, |m, v| m.max(v.toxicity));
    *max_venue_toxicity = max_venue_toxicity.max(tox);
}

// ============================================================================
// JSON output structures for Monte Carlo summary
// ============================================================================

/// Monte Carlo configuration parameters.
#[derive(Debug, Clone, Serialize)]
struct McConfig {
    runs: usize,
    ticks: usize,
    seed: u64,
    tick_ms: i64,
    jitter_ms: i64,
    profile: String,
}

/// Single run record for JSON output (mc_summary.json).
#[derive(Debug, Clone, Serialize)]
struct McRunRecord {
    run: usize,
    seed: u64,
    ticks_executed: usize,
    kill_tick: Option<usize>,
    kill_switch: bool,
    kill_reason: String,
    final_pnl: f64,
    max_drawdown: f64,
    max_abs_delta_usd: f64,
    max_abs_basis_usd: f64,
    max_abs_q_tao: f64,
    max_venue_toxicity: f64,
}

/// Single run record for JSONL output (mc_runs.jsonl).
/// This is the canonical per-run output format for sharded Monte Carlo.
/// Schema: schemas/mc_runs_schema_v1.json
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
struct McRunJsonlRecord {
    /// Schema version for telemetry contract validation. Always 1 for this version.
    schema_version: u32,
    /// Global run index (0-based, used for deterministic seed mapping).
    run_index: usize,
    /// Seed used for this run (should equal base_seed + run_index).
    seed: u64,
    /// Final PnL at end of run.
    pnl_total: f64,
    /// Maximum drawdown observed during run.
    max_drawdown: f64,
    /// Whether kill switch was triggered.
    kill_switch: bool,
    /// Tick at which kill switch was triggered (if any).
    kill_tick: Option<usize>,
    /// Kill reason (if kill switch triggered).
    kill_reason: String,
    /// Number of ticks executed.
    ticks_executed: usize,
    /// Maximum absolute delta in USD.
    max_abs_delta_usd: f64,
    /// Maximum absolute basis in USD.
    max_abs_basis_usd: f64,
    /// Maximum absolute q in TAO.
    max_abs_q_tao: f64,
    /// Maximum venue toxicity.
    max_venue_toxicity: f64,
}

/// Aggregate statistics for a metric.
#[derive(Debug, Clone, Serialize)]
struct AggregateStats {
    mean: f64,
    std_pop: f64,
    min: f64,
    max: f64,
    p05: f64,
    p50: f64,
    p95: f64,
}

/// Monte Carlo summary output (versioned schema).
#[derive(Debug, Clone, Serialize)]
struct McSummary {
    /// Schema version for mc_summary.json. Increment on breaking changes.
    schema_version: u32,
    paraphina_version: String,
    config: McConfig,
    runs: Vec<McRunRecord>,
    aggregate: McAggregateStats,
    /// Tail risk metrics for Appendix B compliance (Phase A).
    tail_risk: TailRiskMetrics,
}

/// Aggregate statistics across all runs.
#[derive(Debug, Clone, Serialize)]
struct McAggregateStats {
    kill_rate: f64,
    kill_count: u64,
    pnl: AggregateStats,
    max_drawdown: AggregateStats,
    max_abs_delta_usd: SimpleStats,
    max_abs_basis_usd: SimpleStats,
    max_abs_q_tao: SimpleStats,
    max_venue_toxicity: SimpleStats,
}

/// Simple statistics without percentiles.
#[derive(Debug, Clone, Serialize)]
struct SimpleStats {
    mean: f64,
    std_pop: f64,
    min: f64,
    max: f64,
}

#[derive(Debug, Clone)]
struct RunResult {
    /// Global run index (used for deterministic ordering).
    global_idx: usize,
    /// Seed used for this run.
    seed: u64,
    final_pnl: f64,
    max_drawdown: f64,
    max_abs_delta: f64,
    max_abs_basis: f64,
    max_abs_q: f64,
    max_venue_toxicity: f64,
    kill_tick: Option<usize>,
    kill_reason: KillReason,
    kill_switch: bool,
    ticks_executed: usize,
}

fn run_once(
    cfg: &Config,
    global_idx: usize,
    seed: u64,
    ticks: usize,
    tick_ms: i64,
    jitter_ms: i64,
) -> RunResult {
    let engine = Engine::new(cfg);
    let mut state = GlobalState::new(cfg);

    let mut rng = XorShift64Star::new(seed);

    let mut now_ms: TimestampMs = 0;
    let mut dd = DrawdownTracker::new();

    let mut max_abs_delta: f64 = 0.0;
    let mut max_abs_basis: f64 = 0.0;
    let mut max_abs_q: f64 = 0.0;
    let mut max_venue_toxicity: f64 = 0.0;

    update_peaks(
        &state,
        &mut max_abs_delta,
        &mut max_abs_basis,
        &mut max_abs_q,
        &mut max_venue_toxicity,
    );

    let mut kill_tick: Option<usize> = None;
    let mut ticks_executed: usize = 0;

    // Pre-allocate scratch buffers for hot path (avoid per-tick allocations).
    // Reasonable capacity estimates based on typical venue counts.
    let num_venues = cfg.venues.len();
    let mut mm_scratch = mm::MmScratch::with_capacity(num_venues);
    let mut mm_quotes_buf: Vec<mm::MmQuote> = Vec::with_capacity(num_venues);
    let mut mm_intents_buf: Vec<OrderIntent> = Vec::with_capacity(num_venues * 2);
    let mut exit_intents_buf: Vec<OrderIntent> = Vec::with_capacity(num_venues);
    let mut hedge_intents_buf: Vec<OrderIntent> = Vec::with_capacity(num_venues);

    for t in 0..ticks {
        let mut dt = tick_ms;
        if jitter_ms > 0 {
            dt += rng.uniform_i64_inclusive(-jitter_ms, jitter_ms);
        }
        if dt < 1 {
            dt = 1;
        }
        now_ms = now_ms.saturating_add(dt as TimestampMs);

        // Seed book data deterministically from (now_ms).
        engine.seed_dummy_mids(&mut state, now_ms);
        engine.main_tick(&mut state, now_ms);

        // If kill switch is active, stop early before placing any new intents.
        if state.kill_switch {
            kill_tick = Some(t);
            ticks_executed = t;
            break;
        }

        // MM -> fills -> conditional recompute (using scratch buffers)
        mm::compute_mm_quotes_into_with_scratch(cfg, &state, &mut mm_quotes_buf, &mut mm_scratch);
        mm::mm_quotes_to_order_intents_into(&mm_quotes_buf, &mut mm_intents_buf);
        let mm_fills = apply_intents_as_fills(cfg, &mut state, &mm_intents_buf);
        // Opt18: recompute_after_fills is deterministic and pure w.r.t. state.
        // If no fills occurred since the last recompute (in main_tick), results would be
        // identical, so skipping is safe and preserves determinism.
        if mm_fills > 0 {
            state.recompute_after_fills(cfg);
        }
        update_peaks(
            &state,
            &mut max_abs_delta,
            &mut max_abs_basis,
            &mut max_abs_q,
            &mut max_venue_toxicity,
        );

        // Exit -> fills -> conditional recompute (using scratch buffer)
        exit::compute_exit_intents_into(cfg, &state, now_ms, &mut exit_intents_buf);
        let exit_fills = apply_intents_as_fills(cfg, &mut state, &exit_intents_buf);
        // Opt18: Skip recompute if no exit fills applied.
        if exit_fills > 0 {
            state.recompute_after_fills(cfg);
        }
        update_peaks(
            &state,
            &mut max_abs_delta,
            &mut max_abs_basis,
            &mut max_abs_q,
            &mut max_venue_toxicity,
        );

        // Hedge -> fills -> conditional recompute (using scratch buffer)
        hedge::compute_hedge_orders_into(cfg, &state, now_ms, &mut hedge_intents_buf);
        let hedge_fills = apply_intents_as_fills(cfg, &mut state, &hedge_intents_buf);
        // Opt18: Skip recompute if no hedge fills applied.
        if hedge_fills > 0 {
            state.recompute_after_fills(cfg);
        }
        update_peaks(
            &state,
            &mut max_abs_delta,
            &mut max_abs_basis,
            &mut max_abs_q,
            &mut max_venue_toxicity,
        );

        dd.update(state.daily_pnl_total);

        ticks_executed = t + 1;
    }

    RunResult {
        global_idx,
        seed,
        final_pnl: state.daily_pnl_total,
        max_drawdown: dd.max_drawdown(),
        max_abs_delta,
        max_abs_basis,
        max_abs_q,
        max_venue_toxicity,
        kill_tick,
        kill_reason: state.kill_reason,
        kill_switch: state.kill_switch,
        ticks_executed,
    }
}

fn percentile(sorted: &[f64], p01: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let p = p01.clamp(0.0, 1.0);
    let n = sorted.len();
    let idx = p * (n.saturating_sub(1) as f64);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let w = idx - (lo as f64);
    sorted[lo] * (1.0 - w) + sorted[hi] * w
}

fn p05_p50_p95(mut xs: Vec<f64>) -> (f64, f64, f64) {
    xs.retain(|x| x.is_finite());
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (
        percentile(&xs, 0.05),
        percentile(&xs, 0.50),
        percentile(&xs, 0.95),
    )
}

/// Write a file atomically (temp file + rename).
fn atomic_write(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or(Path::new("."));
    let temp_name = format!(
        ".tmp_{}_{}",
        std::process::id(),
        path.file_name()
            .map(|s| s.to_string_lossy())
            .unwrap_or_default()
    );
    let temp_path = parent.join(&temp_name);

    let mut file = File::create(&temp_path)?;
    file.write_all(data)?;
    file.sync_all()?;
    fs::rename(&temp_path, path)?;
    Ok(())
}

fn main() {
    match parse_command() {
        Command::Run(args) => run_monte_carlo(args),
        Command::Summarize(args) => run_summarize(args),
    }
}

/// Run the Monte Carlo simulation (default command).
fn run_monte_carlo(args: RunArgs) {
    // Resolve profile with proper precedence: CLI > env > default
    // (No scenario profile for monte_carlo, so pass None)
    let effective = resolve_effective_profile(args.profile, None);
    let profile = effective.profile;
    let cfg = Config::for_profile(profile);

    // Explicit startup log line (required by spec)
    effective.log_startup();

    // Create output directory
    if let Err(e) = fs::create_dir_all(&args.output_dir) {
        eprintln!(
            "Failed to create output directory {:?}: {e}",
            args.output_dir
        );
        std::process::exit(2);
    }

    // Determine effective run count: if --run-count is specified, use it; otherwise use --runs
    let run_count = args.run_count.unwrap_or(args.runs);
    let run_start_index = args.run_start_index;
    let run_end_index = run_start_index + run_count;

    // Determine CSV path (in output directory)
    let csv_path = args.csv_out.as_ref().map(|p| {
        if p.is_absolute() {
            p.clone()
        } else {
            args.output_dir.join(p)
        }
    });

    let mut csv: Option<File> = match csv_path.as_ref() {
        Some(path) => {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let mut f = File::create(path).unwrap_or_else(|e| {
                eprintln!("Failed to create CSV file {:?}: {e}", path);
                std::process::exit(2);
            });
            writeln!(
                f,
                "run,seed,ticks_executed,kill_tick,kill_switch,kill_reason,final_pnl,max_drawdown,max_abs_delta_usd,max_abs_basis_usd,max_abs_q_tao,max_venue_toxicity"
            )
            .unwrap();
            Some(f)
        }
        None => None,
    };

    // Create JSONL output file
    let jsonl_path = args.output_dir.join("mc_runs.jsonl");
    let mut jsonl_file = File::create(&jsonl_path).unwrap_or_else(|e| {
        eprintln!("Failed to create JSONL file {:?}: {e}", jsonl_path);
        std::process::exit(2);
    });

    println!(
        "paraphina-mc v{} | profile={} ({}) runs={} run_start={} run_count={} ticks={} seed={} tick_ms={} jitter_ms={} threads={} print_every={} output_dir={} csv={}",
        env!("CARGO_PKG_VERSION"),
        profile_name(profile),
        effective.source.as_str(),
        args.runs,
        run_start_index,
        run_count,
        args.ticks,
        args.seed,
        args.tick_ms,
        args.jitter_ms,
        args.threads,
        args.print_every,
        args.output_dir.display(),
        csv_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "-".to_string())
    );

    // =========================================================================
    // Execute runs (parallel or sequential)
    // =========================================================================

    // Collect global indices for all runs
    let global_indices: Vec<usize> = (run_start_index..run_end_index).collect();

    // Execute runs in parallel if threads > 1, otherwise sequential
    let mut results: Vec<RunResult> = if args.threads > 1 {
        // Build a custom thread pool with the specified number of threads
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build()
            .expect("Failed to create rayon thread pool");

        pool.install(|| {
            global_indices
                .par_iter()
                .map(|&global_idx| {
                    // Deterministic seed mapping: seed_i = base_seed + i (u64 wrap)
                    let run_seed = args.seed.wrapping_add(global_idx as u64);
                    run_once(
                        &cfg,
                        global_idx,
                        run_seed,
                        args.ticks,
                        args.tick_ms,
                        args.jitter_ms,
                    )
                })
                .collect()
        })
    } else {
        // Sequential execution for threads=1 (default)
        global_indices
            .iter()
            .map(|&global_idx| {
                let run_seed = args.seed.wrapping_add(global_idx as u64);
                run_once(
                    &cfg,
                    global_idx,
                    run_seed,
                    args.ticks,
                    args.tick_ms,
                    args.jitter_ms,
                )
            })
            .collect()
    };

    // Sort results by global_idx to ensure deterministic output order
    results.sort_by_key(|r| r.global_idx);

    // =========================================================================
    // Process results and write outputs (sequential for determinism)
    // =========================================================================

    let mut pnl_stats = OnlineStats::default();
    let mut dd_stats = OnlineStats::default();
    let mut max_abs_delta_stats = OnlineStats::default();
    let mut max_abs_basis_stats = OnlineStats::default();
    let mut max_abs_q_stats = OnlineStats::default();
    let mut max_tox_stats = OnlineStats::default();

    let mut pnl_samples: Vec<f64> = Vec::with_capacity(run_count);
    let mut dd_samples: Vec<f64> = Vec::with_capacity(run_count);

    let mut kills: u64 = 0;
    let mut kill_tick_stats = OnlineStats::default();

    // Collect run records for JSON output
    let mut run_records: Vec<McRunRecord> = Vec::with_capacity(run_count);

    // Process results in sorted order
    for r in &results {
        pnl_stats.add(r.final_pnl);
        dd_stats.add(r.max_drawdown);
        max_abs_delta_stats.add(r.max_abs_delta);
        max_abs_basis_stats.add(r.max_abs_basis);
        max_abs_q_stats.add(r.max_abs_q);
        max_tox_stats.add(r.max_venue_toxicity);

        pnl_samples.push(r.final_pnl);
        dd_samples.push(r.max_drawdown);

        if let Some(kt) = r.kill_tick {
            kills += 1;
            kill_tick_stats.add(kt as f64);
        }

        let kill_reason_str = format!("{:?}", r.kill_reason);

        // Store run record for mc_summary.json (uses 1-based run number for display)
        let local_idx = r.global_idx - run_start_index;
        run_records.push(McRunRecord {
            run: local_idx + 1,
            seed: r.seed,
            ticks_executed: r.ticks_executed,
            kill_tick: r.kill_tick,
            kill_switch: r.kill_switch,
            kill_reason: kill_reason_str.clone(),
            final_pnl: r.final_pnl,
            max_drawdown: r.max_drawdown,
            max_abs_delta_usd: r.max_abs_delta,
            max_abs_basis_usd: r.max_abs_basis,
            max_abs_q_tao: r.max_abs_q,
            max_venue_toxicity: r.max_venue_toxicity,
        });

        // Write JSONL record (uses global run_index for aggregation)
        // schema_version=1 per schemas/mc_runs_schema_v1.json telemetry contract
        let jsonl_record = McRunJsonlRecord {
            schema_version: 1,
            run_index: r.global_idx,
            seed: r.seed,
            pnl_total: r.final_pnl,
            max_drawdown: r.max_drawdown,
            kill_switch: r.kill_switch,
            kill_tick: r.kill_tick,
            kill_reason: kill_reason_str.clone(),
            ticks_executed: r.ticks_executed,
            max_abs_delta_usd: r.max_abs_delta,
            max_abs_basis_usd: r.max_abs_basis,
            max_abs_q_tao: r.max_abs_q,
            max_venue_toxicity: r.max_venue_toxicity,
        };
        let jsonl_line =
            serde_json::to_string(&jsonl_record).expect("Failed to serialize JSONL record");
        writeln!(jsonl_file, "{}", jsonl_line).expect("Failed to write JSONL record");

        if let Some(f) = csv.as_mut() {
            let kt = r.kill_tick.map(|x| x.to_string()).unwrap_or_default();
            writeln!(
                f,
                "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                r.global_idx,
                r.seed,
                r.ticks_executed,
                kt,
                r.kill_switch,
                kill_reason_str,
                r.final_pnl,
                r.max_drawdown,
                r.max_abs_delta,
                r.max_abs_basis,
                r.max_abs_q,
                r.max_venue_toxicity
            )
            .unwrap();
        }

        // Print progress (only after parallel execution completes, so no interleaving)
        let should_print = !args.quiet
            && (args.print_every == 1
                || (local_idx + 1).is_multiple_of(args.print_every)
                || (local_idx + 1 == run_count));

        if should_print {
            let kill_tick_str = r
                .kill_tick
                .map(|x| x.to_string())
                .unwrap_or_else(|| "-".to_string());
            let kill_reason_display = if r.kill_switch {
                format!("{:?}", r.kill_reason)
            } else {
                "-".to_string()
            };
            println!(
                "run {:>4}/{:<4} (global={}) seed={:<10} pnl={:>10.4} maxDD={:>10.4} |maxΔ|={:>10.4} |basis|={:>10.4} |q|={:>9.4} tox={:>6.3} kill={} kt={} reason={} ticks={}",
                local_idx + 1,
                run_count,
                r.global_idx,
                r.seed,
                r.final_pnl,
                r.max_drawdown,
                r.max_abs_delta,
                r.max_abs_basis,
                r.max_abs_q,
                r.max_venue_toxicity,
                r.kill_switch,
                kill_tick_str,
                kill_reason_display,
                r.ticks_executed
            );
        }
    }

    // Flush JSONL file
    jsonl_file.sync_all().expect("Failed to sync JSONL file");

    let kill_rate = (kills as f64) / (run_count as f64);
    // Clone samples before passing to p05_p50_p95 (which takes ownership)
    // so we can reuse them for tail_risk computation later
    let (pnl_p05, pnl_p50, pnl_p95) = p05_p50_p95(pnl_samples.clone());
    let (dd_p05, dd_p50, dd_p95) = p05_p50_p95(dd_samples.clone());

    println!();
    println!("SUMMARY");
    println!(
        "  runs:              {} (global indices {}..{})",
        run_count, run_start_index, run_end_index
    );
    println!(
        "  kill_rate:         {:.2}% ({} / {})",
        100.0 * kill_rate,
        kills,
        run_count
    );

    if kills > 0 {
        println!(
            "  kill_tick:         mean={:.2}  min={:.0}  max={:.0}",
            kill_tick_stats.mean(),
            kill_tick_stats.min(),
            kill_tick_stats.max()
        );
    }

    println!(
        "  pnl:               mean={:.4}  std(pop)={:.4}  min={:.4}  max={:.4}  p05={:.4}  p50={:.4}  p95={:.4}",
        pnl_stats.mean(),
        pnl_stats.stddev_population(),
        pnl_stats.min(),
        pnl_stats.max(),
        pnl_p05,
        pnl_p50,
        pnl_p95
    );
    println!(
        "  max_drawdown:      mean={:.4}  std(pop)={:.4}  min={:.4}  max={:.4}  p05={:.4}  p50={:.4}  p95={:.4}",
        dd_stats.mean(),
        dd_stats.stddev_population(),
        dd_stats.min(),
        dd_stats.max(),
        dd_p05,
        dd_p50,
        dd_p95
    );
    println!(
        "  max_abs_delta_usd: mean={:.4}  std(pop)={:.4}  min={:.4}  max={:.4}",
        max_abs_delta_stats.mean(),
        max_abs_delta_stats.stddev_population(),
        max_abs_delta_stats.min(),
        max_abs_delta_stats.max()
    );
    println!(
        "  max_abs_basis_usd: mean={:.4}  std(pop)={:.4}  min={:.4}  max={:.4}",
        max_abs_basis_stats.mean(),
        max_abs_basis_stats.stddev_population(),
        max_abs_basis_stats.min(),
        max_abs_basis_stats.max()
    );
    println!(
        "  max_abs_q_tao:     mean={:.4}  std(pop)={:.4}  min={:.4}  max={:.4}",
        max_abs_q_stats.mean(),
        max_abs_q_stats.stddev_population(),
        max_abs_q_stats.min(),
        max_abs_q_stats.max()
    );
    println!(
        "  max_venue_toxicity: mean={:.4}  std(pop)={:.4}  min={:.4}  max={:.4}",
        max_tox_stats.mean(),
        max_tox_stats.stddev_population(),
        max_tox_stats.min(),
        max_tox_stats.max()
    );

    // =========================================================================
    // Write output files
    // =========================================================================

    // Compute tail risk metrics (Phase A)
    let tail_risk = TailRiskMetrics::compute(
        &pnl_samples,
        &dd_samples,
        kills,
        DEFAULT_VAR_ALPHA,
        0.95, // 95% Wilson CI
    );

    // Build summary structure
    let summary = McSummary {
        schema_version: 2, // Version 2: adds tail_risk field
        paraphina_version: env!("CARGO_PKG_VERSION").to_string(),
        config: McConfig {
            runs: args.runs,
            ticks: args.ticks,
            seed: args.seed,
            tick_ms: args.tick_ms,
            jitter_ms: args.jitter_ms,
            profile: profile_name(profile).to_string(),
        },
        runs: run_records,
        aggregate: McAggregateStats {
            kill_rate,
            kill_count: kills,
            pnl: AggregateStats {
                mean: pnl_stats.mean(),
                std_pop: pnl_stats.stddev_population(),
                min: pnl_stats.min(),
                max: pnl_stats.max(),
                p05: pnl_p05,
                p50: pnl_p50,
                p95: pnl_p95,
            },
            max_drawdown: AggregateStats {
                mean: dd_stats.mean(),
                std_pop: dd_stats.stddev_population(),
                min: dd_stats.min(),
                max: dd_stats.max(),
                p05: dd_p05,
                p50: dd_p50,
                p95: dd_p95,
            },
            max_abs_delta_usd: SimpleStats {
                mean: max_abs_delta_stats.mean(),
                std_pop: max_abs_delta_stats.stddev_population(),
                min: max_abs_delta_stats.min(),
                max: max_abs_delta_stats.max(),
            },
            max_abs_basis_usd: SimpleStats {
                mean: max_abs_basis_stats.mean(),
                std_pop: max_abs_basis_stats.stddev_population(),
                min: max_abs_basis_stats.min(),
                max: max_abs_basis_stats.max(),
            },
            max_abs_q_tao: SimpleStats {
                mean: max_abs_q_stats.mean(),
                std_pop: max_abs_q_stats.stddev_population(),
                min: max_abs_q_stats.min(),
                max: max_abs_q_stats.max(),
            },
            max_venue_toxicity: SimpleStats {
                mean: max_tox_stats.mean(),
                std_pop: max_tox_stats.stddev_population(),
                min: max_tox_stats.min(),
                max: max_tox_stats.max(),
            },
        },
        tail_risk,
    };

    // Write mc_summary.json
    let summary_path = args.output_dir.join("mc_summary.json");
    let summary_json =
        serde_json::to_string_pretty(&summary).expect("Failed to serialize mc_summary.json");
    if let Err(e) = atomic_write(&summary_path, summary_json.as_bytes()) {
        eprintln!("Failed to write mc_summary.json: {e}");
        std::process::exit(1);
    }
    println!();
    println!("Wrote: {}", summary_path.display());
    println!("Wrote: {}", jsonl_path.display());

    // Write monte_carlo.yaml (suite file for evidence pack)
    // This is a synthetic suite file capturing the Monte Carlo parameters.
    let suite_yaml = format!(
        r#"# Monte Carlo run configuration
# Generated by paraphina monte_carlo harness
#
# This file captures the parameters used for this Monte Carlo run.
# It serves as the "suite" file for the Evidence Pack.

suite_id: monte_carlo
suite_version: 1
description: Monte Carlo simulation run

config:
  runs: {}
  ticks: {}
  seed: {}
  run_start_index: {}
  run_count: {}
  tick_ms: {}
  jitter_ms: {}
  threads: {}
  profile: {}

paraphina_version: {}
"#,
        args.runs,
        args.ticks,
        args.seed,
        run_start_index,
        run_count,
        args.tick_ms,
        args.jitter_ms,
        args.threads,
        profile_name(profile),
        env!("CARGO_PKG_VERSION")
    );

    let suite_path = args.output_dir.join("monte_carlo.yaml");
    if let Err(e) = atomic_write(&suite_path, suite_yaml.as_bytes()) {
        eprintln!("Failed to write monte_carlo.yaml: {e}");
        std::process::exit(1);
    }
    println!("Wrote: {}", suite_path.display());

    // =========================================================================
    // Generate Evidence Pack v1
    // =========================================================================

    // Collect artifact paths relative to output_dir
    let mut artifact_paths: Vec<PathBuf> = Vec::new();

    // Add mc_summary.json
    artifact_paths.push(PathBuf::from("mc_summary.json"));

    // Add mc_runs.jsonl
    artifact_paths.push(PathBuf::from("mc_runs.jsonl"));

    // Add CSV file if it was written to the output directory
    if let Some(csv_rel_path) = args.csv_out.as_ref() {
        if !csv_rel_path.is_absolute() {
            artifact_paths.push(csv_rel_path.clone());
        }
    }

    // Write evidence pack
    println!();
    println!("Generating Evidence Pack...");
    match write_evidence_pack(&args.output_dir, &suite_path, &artifact_paths) {
        Ok(()) => {
            println!(
                "✓ Evidence Pack written to: {}/evidence_pack/",
                args.output_dir.display()
            );
        }
        Err(e) => {
            eprintln!("✗ EVIDENCE PACK GENERATION FAILED: {e}");
            std::process::exit(1);
        }
    }

    println!();
    println!("Output written to: {}/", args.output_dir.display());
}

/// Run the summarize command: aggregate JSONL runs into mc_summary.json.
fn run_summarize(args: SummarizeArgs) {
    println!(
        "paraphina-mc summarize v{} | input={} out_dir={}",
        env!("CARGO_PKG_VERSION"),
        args.input.display(),
        args.out_dir.display()
    );

    // Read and parse JSONL
    let file = File::open(&args.input).unwrap_or_else(|e| {
        eprintln!("Failed to open input file {:?}: {e}", args.input);
        std::process::exit(2);
    });
    let reader = BufReader::new(file);

    let mut records: Vec<McRunJsonlRecord> = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line.unwrap_or_else(|e| {
            eprintln!(
                "Failed to read line {} in {:?}: {e}",
                line_num + 1,
                args.input
            );
            std::process::exit(2);
        });

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        let record: McRunJsonlRecord = serde_json::from_str(&line).unwrap_or_else(|e| {
            eprintln!(
                "Failed to parse JSON at line {} in {:?}: {e}\nLine: {}",
                line_num + 1,
                args.input,
                line
            );
            std::process::exit(2);
        });
        records.push(record);
    }

    if records.is_empty() {
        eprintln!("Error: No records found in {:?}", args.input);
        std::process::exit(2);
    }

    // Sort by run_index for determinism
    records.sort_by_key(|r| r.run_index);

    // Validate: check for duplicates and contiguity
    let mut seen_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for record in &records {
        if seen_indices.contains(&record.run_index) {
            eprintln!(
                "Error: Duplicate run_index {} found in {:?}",
                record.run_index, args.input
            );
            std::process::exit(2);
        }
        seen_indices.insert(record.run_index);
    }

    // Check contiguity
    let min_idx = records.first().unwrap().run_index;
    let max_idx = records.last().unwrap().run_index;
    let expected_count = max_idx - min_idx + 1;

    if records.len() != expected_count {
        eprintln!(
            "Error: Non-contiguous run indices. Expected {} runs for indices {}..={}, found {}",
            expected_count,
            min_idx,
            max_idx,
            records.len()
        );
        std::process::exit(2);
    }

    // Validate seed contract if base_seed provided
    if let Some(base_seed) = args.base_seed {
        for record in &records {
            let expected_seed = base_seed.wrapping_add(record.run_index as u64);
            if record.seed != expected_seed {
                eprintln!(
                    "Error: Seed mismatch at run_index {}. Expected {} (base_seed {} + {}), found {}",
                    record.run_index, expected_seed, base_seed, record.run_index, record.seed
                );
                std::process::exit(2);
            }
        }
        println!(
            "✓ Seed contract validated for {} runs (base_seed={})",
            records.len(),
            base_seed
        );
    }

    println!(
        "Loaded {} records (run indices {}..={})",
        records.len(),
        min_idx,
        max_idx
    );

    // Create output directory
    if let Err(e) = fs::create_dir_all(&args.out_dir) {
        eprintln!("Failed to create output directory {:?}: {e}", args.out_dir);
        std::process::exit(2);
    }

    // Compute statistics
    let mut pnl_stats = OnlineStats::default();
    let mut dd_stats = OnlineStats::default();
    let mut max_abs_delta_stats = OnlineStats::default();
    let mut max_abs_basis_stats = OnlineStats::default();
    let mut max_abs_q_stats = OnlineStats::default();
    let mut max_tox_stats = OnlineStats::default();

    let mut pnl_samples: Vec<f64> = Vec::with_capacity(records.len());
    let mut dd_samples: Vec<f64> = Vec::with_capacity(records.len());

    let mut kills: u64 = 0;
    let mut kill_tick_stats = OnlineStats::default();

    let mut run_records: Vec<McRunRecord> = Vec::with_capacity(records.len());

    for (local_idx, record) in records.iter().enumerate() {
        pnl_stats.add(record.pnl_total);
        dd_stats.add(record.max_drawdown);
        max_abs_delta_stats.add(record.max_abs_delta_usd);
        max_abs_basis_stats.add(record.max_abs_basis_usd);
        max_abs_q_stats.add(record.max_abs_q_tao);
        max_tox_stats.add(record.max_venue_toxicity);

        pnl_samples.push(record.pnl_total);
        dd_samples.push(record.max_drawdown);

        if record.kill_switch {
            kills += 1;
            if let Some(kt) = record.kill_tick {
                kill_tick_stats.add(kt as f64);
            }
        }

        run_records.push(McRunRecord {
            run: local_idx + 1,
            seed: record.seed,
            ticks_executed: record.ticks_executed,
            kill_tick: record.kill_tick,
            kill_switch: record.kill_switch,
            kill_reason: record.kill_reason.clone(),
            final_pnl: record.pnl_total,
            max_drawdown: record.max_drawdown,
            max_abs_delta_usd: record.max_abs_delta_usd,
            max_abs_basis_usd: record.max_abs_basis_usd,
            max_abs_q_tao: record.max_abs_q_tao,
            max_venue_toxicity: record.max_venue_toxicity,
        });
    }

    let kill_rate = (kills as f64) / (records.len() as f64);
    let (pnl_p05, pnl_p50, pnl_p95) = p05_p50_p95(pnl_samples.clone());
    let (dd_p05, dd_p50, dd_p95) = p05_p50_p95(dd_samples.clone());

    // Compute tail risk metrics
    let tail_risk =
        TailRiskMetrics::compute(&pnl_samples, &dd_samples, kills, DEFAULT_VAR_ALPHA, 0.95);

    // Build summary (note: config comes from aggregated data, not args)
    // We use defaults for config since we're aggregating shards
    let first_seed = records.first().unwrap().seed;
    let base_seed_inferred = first_seed.wrapping_sub(min_idx as u64);

    let summary = McSummary {
        schema_version: 2,
        paraphina_version: env!("CARGO_PKG_VERSION").to_string(),
        config: McConfig {
            runs: records.len(),
            ticks: records.first().unwrap().ticks_executed, // Use first record's ticks
            seed: args.base_seed.unwrap_or(base_seed_inferred),
            tick_ms: DEFAULT_TICK_MS,
            jitter_ms: DEFAULT_JITTER_MS,
            profile: "Unknown".to_string(), // Not available from JSONL
        },
        runs: run_records,
        aggregate: McAggregateStats {
            kill_rate,
            kill_count: kills,
            pnl: AggregateStats {
                mean: pnl_stats.mean(),
                std_pop: pnl_stats.stddev_population(),
                min: pnl_stats.min(),
                max: pnl_stats.max(),
                p05: pnl_p05,
                p50: pnl_p50,
                p95: pnl_p95,
            },
            max_drawdown: AggregateStats {
                mean: dd_stats.mean(),
                std_pop: dd_stats.stddev_population(),
                min: dd_stats.min(),
                max: dd_stats.max(),
                p05: dd_p05,
                p50: dd_p50,
                p95: dd_p95,
            },
            max_abs_delta_usd: SimpleStats {
                mean: max_abs_delta_stats.mean(),
                std_pop: max_abs_delta_stats.stddev_population(),
                min: max_abs_delta_stats.min(),
                max: max_abs_delta_stats.max(),
            },
            max_abs_basis_usd: SimpleStats {
                mean: max_abs_basis_stats.mean(),
                std_pop: max_abs_basis_stats.stddev_population(),
                min: max_abs_basis_stats.min(),
                max: max_abs_basis_stats.max(),
            },
            max_abs_q_tao: SimpleStats {
                mean: max_abs_q_stats.mean(),
                std_pop: max_abs_q_stats.stddev_population(),
                min: max_abs_q_stats.min(),
                max: max_abs_q_stats.max(),
            },
            max_venue_toxicity: SimpleStats {
                mean: max_tox_stats.mean(),
                std_pop: max_tox_stats.stddev_population(),
                min: max_tox_stats.min(),
                max: max_tox_stats.max(),
            },
        },
        tail_risk,
    };

    // Write mc_summary.json
    let summary_path = args.out_dir.join("mc_summary.json");
    let summary_json =
        serde_json::to_string_pretty(&summary).expect("Failed to serialize mc_summary.json");
    if let Err(e) = atomic_write(&summary_path, summary_json.as_bytes()) {
        eprintln!("Failed to write mc_summary.json: {e}");
        std::process::exit(1);
    }

    println!();
    println!("SUMMARY");
    println!("  runs:              {}", records.len());
    println!(
        "  kill_rate:         {:.2}% ({} / {})",
        100.0 * kill_rate,
        kills,
        records.len()
    );
    println!(
        "  pnl:               mean={:.4}  std(pop)={:.4}  min={:.4}  max={:.4}  p05={:.4}  p50={:.4}  p95={:.4}",
        pnl_stats.mean(),
        pnl_stats.stddev_population(),
        pnl_stats.min(),
        pnl_stats.max(),
        pnl_p05,
        pnl_p50,
        pnl_p95
    );
    println!(
        "  max_drawdown:      mean={:.4}  std(pop)={:.4}  min={:.4}  max={:.4}  p05={:.4}  p50={:.4}  p95={:.4}",
        dd_stats.mean(),
        dd_stats.stddev_population(),
        dd_stats.min(),
        dd_stats.max(),
        dd_p05,
        dd_p50,
        dd_p95
    );

    println!();
    println!("Wrote: {}", summary_path.display());
    println!("✓ Summarize complete");
}
