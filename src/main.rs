// src/main.rs
//
// Thin harness around the Paraphina library.
// All of the real logic lives in the lib crate (engine, strategy, etc).

use clap::Parser;

use paraphina::{
    Config,
    SimGateway,
    StrategyRunner,
    EventSink,
    FileSink,
    NoopSink,
};

/// Command-line arguments for the Paraphina binary.
#[derive(Parser, Debug)]
#[command(
    name = "paraphina",
    version,
    about = "TAO perp MM + hedge research driver"
)]
struct Args {
    /// Number of synthetic ticks to simulate.
    #[arg(long, default_value_t = 200)]
    ticks: u64,

    /// Initial synthetic global position q0 in TAO.
    ///
    /// This is applied once at t = 0 via StrategyRunner::inject_initial_position.
    #[arg(long, allow_hyphen_values = true)]
    initial_q_tao: Option<f64>,

    /// Base hedge band (TAO) before volatility scaling.
    #[arg(long)]
    hedge_band_base: Option<f64>,

    /// JSONL file to log per-tick snapshots.
    ///
    /// If omitted, no JSONL log is written (NoopSink).
    #[arg(long)]
    log_jsonl: Option<String>,
}

/// Build the telemetry sink as a trait object so we can choose between
/// FileSink and NoopSink at runtime.
fn build_sink(log_jsonl: Option<&str>) -> Box<dyn EventSink> {
    if let Some(path) = log_jsonl {
        match FileSink::create(path) {
            Ok(s) => Box::new(s),
            Err(err) => {
                eprintln!(
                    "Failed to create log file ({path}), \
                     falling back to NoopSink: {err}"
                );
                Box::new(NoopSink)
            }
        }
    } else {
        Box::new(NoopSink)
    }
}

/// Build Config from defaults, then apply CLI + env research overrides.
///
/// This keeps src/config.rs as the single source of truth, while letting
/// Python harnesses sweep parameters via environment variables.
fn build_config_from_env_and_args(args: &Args) -> Config {
    let mut cfg = Config::default();

    // ---------- CLI overrides ----------

    if let Some(q0) = args.initial_q_tao {
        cfg.initial_q_tao = q0;
    }

    if let Some(band) = args.hedge_band_base {
        cfg.hedge.hedge_band_base = band;
    }

    // ---------- Env overrides (research knobs) ----------

    // Volatility reference σ_ref (used in vol_ratio).
    if let Ok(raw) = std::env::var("PARAPHINA_VOL_REF") {
        if let Ok(v) = raw.parse::<f64>() {
            cfg.volatility.vol_ref = v;
        }
    }

    // Size risk parameter η in J(Q) = eQ - 0.5 η Q².
    if let Ok(raw) = std::env::var("PARAPHINA_SIZE_ETA") {
        if let Ok(v) = raw.parse::<f64>() {
            cfg.mm.size_eta = v;
        }
    }

    // Base dollar delta hard limit before vol scaling.
    if let Ok(raw) = std::env::var("PARAPHINA_DELTA_LIMIT_USD_BASE") {
        if let Ok(v) = raw.parse::<f64>() {
            cfg.risk.delta_hard_limit_usd_base = v;
        }
    }

    // Daily loss limit (realised + unrealised), negative by convention.
    if let Ok(raw) = std::env::var("PARAPHINA_DAILY_LOSS_LIMIT_USD") {
        if let Ok(v) = raw.parse::<f64>() {
            cfg.risk.daily_loss_limit = v;
        }
    }

    cfg
}

fn main() {
    // 0) Parse CLI args.
    let args = Args::parse();

    // 1) Load / build config with CLI + env overrides.
    let cfg = build_config_from_env_and_args(&args);

    // 2) Choose execution gateway (here: synthetic sim).
    let gateway = SimGateway::new();

    // 3) Build telemetry sink from CLI.
    //
    //    - NoopSink   -> no on-disk logs, just prints to stdout.
    //    - FileSink   -> JSONL file with 1 record per tick for backtesting / RL.
    let sink = build_sink(args.log_jsonl.as_deref());

    // 4) Run the high-level strategy for N ticks.
    let mut runner = StrategyRunner::new(&cfg, gateway, sink);
    runner.run_simulation(args.ticks);
}
