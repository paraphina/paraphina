// src/main.rs
//
// Thin harness around the Paraphina library.
// All of the real logic lives in the lib crate (engine, strategy, etc).

use clap::Parser;

use paraphina::{
    Config,
    EventSink,
    FileSink,
    NoopSink,
    SimGateway,
    StrategyRunner,
};

/// Command-line arguments for the Paraphina binary.
#[derive(Parser, Debug)]
#[command(name = "paraphina")]
struct Cli {
    /// Number of synthetic ticks to run.
    #[arg(long)]
    ticks: u64,

    /// Initial global position q0 in TAO.
    /// (exposed so research harnesses can sweep q0).
    #[arg(long)]
    initial_q_tao: f64,

    /// Base hedge band (TAO).
    #[arg(long)]
    hedge_band_base: f64,

    /// Optional per-day loss limit in USD (positive number).
    ///
    /// NOTE: The risk engine stores this as a NEGATIVE quantity, so we
    /// convert `+4000.0` on the CLI into `-4000.0` in Config.
    #[arg(long)]
    loss_limit_usd: Option<f64>,

    /// Optional JSONL path for tick log.
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
fn build_config_from_env_and_args(cli: &Cli) -> Config {
    let mut cfg = Config::default();

    // ---------- CLI overrides ----------

    // Initial inventory.
    cfg.initial_q_tao = cli.initial_q_tao;

    // Base hedge band (TAO).
    cfg.hedge.hedge_band_base = cli.hedge_band_base;

    // Daily loss limit (realised + unrealised).
    // CLI provides this as a POSITIVE USD number (e.g. 4000.0),
    // but the risk engine expects a NEGATIVE threshold.
    if let Some(loss) = cli.loss_limit_usd {
        cfg.risk.daily_loss_limit = -loss;
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

    // Daily loss limit override from env (already negative by convention).
    if let Ok(raw) = std::env::var("PARAPHINA_DAILY_LOSS_LIMIT_USD") {
        if let Ok(v) = raw.parse::<f64>() {
            cfg.risk.daily_loss_limit = v;
        }
    }

    cfg
}

fn main() {
    // 0) Parse CLI args.
    let cli = Cli::parse();

    // 1) Load / build config with CLI + env overrides.
    let cfg = build_config_from_env_and_args(&cli);

    // 2) Choose execution gateway (here: synthetic sim).
    let gateway = SimGateway::new();

    // 3) Build telemetry sink from CLI.
    //
    //    - NoopSink   -> no on-disk logs, just prints to stdout.
    //    - FileSink   -> JSONL file with 1 record per tick for backtesting / RL.
    let sink = build_sink(cli.log_jsonl.as_deref());

    // 4) Run the high-level strategy for N ticks.
    let mut runner = StrategyRunner::new(&cfg, gateway, sink);
    runner.run_simulation(cli.ticks);
}
