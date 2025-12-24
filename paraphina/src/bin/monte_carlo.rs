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
// Notes:
// - This harness applies intents as immediate fills (no gateway), using venue taker fees.
//   This keeps the harness dependency-free and deterministic.

use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

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

#[derive(Debug, Clone)]
struct Args {
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
}

impl Args {
    fn usage() -> &'static str {
        "\
paraphina Monte Carlo harness (Milestone B)

USAGE:
  cargo run --bin monte_carlo -- [FLAGS]

PROFILE PRECEDENCE:
  1) --profile overrides environment
  2) else PARAPHINA_RISK_PROFILE
  3) else Balanced

FLAGS:
  --profile NAME       Balanced | Conservative | Aggressive (alias: Loose)
  --runs N             Number of runs (default: 50)
  --ticks N            Ticks per run (default: 600)
  --seed U64           Base seed (default: 1). Run i uses seed + i.
  --tick-ms MS         Base tick duration in ms (default: 1000)
  --jitter-ms MS       Per-tick jitter added to tick-ms in [-MS, +MS] (default: 0)
  --print-every N      Print every N runs (default: 1). Ignored with --quiet.
  --csv PATH           Write per-run CSV rows to PATH (relative to output-dir)
  --output-dir DIR     Output directory (default: runs/demo_step_7_1)
  --quiet              Suppress per-run lines; only print final summary
  --help               Show this help

OUTPUT:
  The harness writes to <output-dir>/:
    - mc_summary.json     Per-run statistics and aggregate summary
    - mc_runs.csv         CSV of per-run metrics (if --csv specified)
    - monte_carlo.yaml    Configuration used for this run
    - evidence_pack/      Evidence Pack v1 for verification

EXAMPLES:
  cargo run --bin monte_carlo -- --runs 100 --ticks 1200 --seed 7 --jitter-ms 500 --profile Balanced
  cargo run --bin monte_carlo -- --output-dir runs/my_experiment --runs 50
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
        let mut out = Args {
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
        };

        let mut it = env::args().skip(1);

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

                other => return Err(format!("Unknown argument: {other}")),
            }
        }

        Ok(out)
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

/// Apply intents as immediate fills using taker fees.
/// This keeps the harness dependency-free and does not require gateway internals.
fn apply_intents_as_fills(cfg: &Config, state: &mut GlobalState, intents: &[OrderIntent]) {
    for it in intents {
        if it.venue_index >= cfg.venues.len() {
            continue;
        }
        if !it.price.is_finite() || it.price <= 0.0 {
            continue;
        }
        if !it.size.is_finite() || it.size <= 0.0 {
            continue;
        }

        // Pragmatic: treat everything as taker for now.
        // (If you later want maker-vs-taker modeling, switch on it.purpose here.)
        let fee_bps = cfg.venues[it.venue_index].taker_fee_bps;
        state.apply_perp_fill(it.venue_index, it.side, it.size, it.price, fee_bps);
    }
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

/// Single run record for JSON output.
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

fn run_once(cfg: &Config, seed: u64, ticks: usize, tick_ms: i64, jitter_ms: i64) -> RunResult {
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

        // MM -> fills -> recompute
        let mm_quotes = mm::compute_mm_quotes(cfg, &state);
        let mm_intents = mm::mm_quotes_to_order_intents(&mm_quotes);
        apply_intents_as_fills(cfg, &mut state, &mm_intents);
        state.recompute_after_fills(cfg);
        update_peaks(
            &state,
            &mut max_abs_delta,
            &mut max_abs_basis,
            &mut max_abs_q,
            &mut max_venue_toxicity,
        );

        // Exit -> fills -> recompute
        let exit_intents = exit::compute_exit_intents(cfg, &state, now_ms);
        apply_intents_as_fills(cfg, &mut state, &exit_intents);
        state.recompute_after_fills(cfg);
        update_peaks(
            &state,
            &mut max_abs_delta,
            &mut max_abs_basis,
            &mut max_abs_q,
            &mut max_venue_toxicity,
        );

        // Hedge -> fills -> recompute
        let hedge_intents = hedge::compute_hedge_orders(cfg, &state, now_ms);
        apply_intents_as_fills(cfg, &mut state, &hedge_intents);
        state.recompute_after_fills(cfg);
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
    let args = Args::parse_or_exit();

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

    println!(
        "paraphina-mc v{} | profile={} ({}) runs={} ticks={} seed={} tick_ms={} jitter_ms={} print_every={} output_dir={} csv={}",
        env!("CARGO_PKG_VERSION"),
        profile_name(profile),
        effective.source.as_str(),
        args.runs,
        args.ticks,
        args.seed,
        args.tick_ms,
        args.jitter_ms,
        args.print_every,
        args.output_dir.display(),
        csv_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "-".to_string())
    );

    let mut pnl_stats = OnlineStats::default();
    let mut dd_stats = OnlineStats::default();
    let mut max_abs_delta_stats = OnlineStats::default();
    let mut max_abs_basis_stats = OnlineStats::default();
    let mut max_abs_q_stats = OnlineStats::default();
    let mut max_tox_stats = OnlineStats::default();

    let mut pnl_samples: Vec<f64> = Vec::with_capacity(args.runs);
    let mut dd_samples: Vec<f64> = Vec::with_capacity(args.runs);

    let mut kills: u64 = 0;
    let mut kill_tick_stats = OnlineStats::default();

    // Collect run records for JSON output
    let mut run_records: Vec<McRunRecord> = Vec::with_capacity(args.runs);

    for i in 0..args.runs {
        let run_seed = args.seed.wrapping_add(i as u64);
        let r = run_once(&cfg, run_seed, args.ticks, args.tick_ms, args.jitter_ms);

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

        // Store run record
        run_records.push(McRunRecord {
            run: i + 1,
            seed: run_seed,
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

        if let Some(f) = csv.as_mut() {
            let kt = r.kill_tick.map(|x| x.to_string()).unwrap_or_default();
            writeln!(
                f,
                "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                i + 1,
                run_seed,
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

        let should_print = !args.quiet
            && (args.print_every == 1 || ((i + 1) % args.print_every == 0) || (i + 1 == args.runs));

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
                "run {:>4}/{:<4} seed={:<10} pnl={:>10.4} maxDD={:>10.4} |maxΔ|={:>10.4} |basis|={:>10.4} |q|={:>9.4} tox={:>6.3} kill={} kt={} reason={} ticks={}",
                i + 1,
                args.runs,
                run_seed,
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

    let kill_rate = (kills as f64) / (args.runs as f64);
    // Clone samples before passing to p05_p50_p95 (which takes ownership)
    // so we can reuse them for tail_risk computation later
    let (pnl_p05, pnl_p50, pnl_p95) = p05_p50_p95(pnl_samples.clone());
    let (dd_p05, dd_p50, dd_p95) = p05_p50_p95(dd_samples.clone());

    println!();
    println!("SUMMARY");
    println!("  runs:              {}", args.runs);
    println!(
        "  kill_rate:         {:.2}% ({} / {})",
        100.0 * kill_rate,
        kills,
        args.runs
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
  tick_ms: {}
  jitter_ms: {}
  profile: {}

paraphina_version: {}
"#,
        args.runs,
        args.ticks,
        args.seed,
        args.tick_ms,
        args.jitter_ms,
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
