// src/main.rs
//
// Thin harness around the Paraphina library.
// All of the real logic lives in the lib crate (engine, strategy, etc).

use std::env;

use clap::Parser;

use paraphina::{
    Config,
    EventSink,
    FileSink,
    NoopSink,
    SimGateway,
    StrategyRunner,
};

/// Command-line arguments for the Paraphina simulation binary.
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

    /// JSONL file to log per-tick snapshots.
    ///
    /// If omitted, no JSONL log is written (NoopSink).
    #[arg(long)]
    log_jsonl: Option<String>,

    /// Optional override for the initial global position q0 (TAO).
    ///
    /// If omitted, uses cfg.initial_q_tao from Config::default().
    #[arg(long)]
    initial_q_tao: Option<f64>,

    /// Optional override for the base hedge half-band (TAO).
    #[arg(long)]
    hedge_band_base: Option<f64>,

    /// Optional override for the MM size-risk parameter η.
    ///
    /// Smaller η ⇒ more aggressive sizing; larger η ⇒ more conservative.
    #[arg(long)]
    mm_size_eta: Option<f64>,

    /// Optional override for the reference volatility σ_ref (= vol_ref).
    ///
    /// Smaller vol_ref ⇒ larger effective vol_ratio for a given σ_eff.
    #[arg(long)]
    vol_ref: Option<f64>,
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

/// Apply CLI + environment overrides to the base Config.
///
/// This keeps Config::default() as the single source of truth, while
/// allowing research harnesses to move individual knobs.
fn apply_overrides(cfg: &mut Config, args: &Args) {
    // ---- CLI overrides first ----

    if let Some(q0) = args.initial_q_tao {
        cfg.initial_q_tao = q0;
        println!("[config] CLI --initial-q-tao = {q0} (overrode env/default)");
    }

    if let Some(band) = args.hedge_band_base {
        cfg.hedge.hedge_band_base = band.max(0.0);
        println!(
            "[config] CLI --hedge-band-base = {} (overrode env/default)",
            cfg.hedge.hedge_band_base
        );
    }

    if let Some(eta) = args.mm_size_eta {
        let eta_clamped = eta.max(1e-6);
        cfg.mm.size_eta = eta_clamped;
        println!(
            "[config] CLI --mm-size-eta = {} (overrode env/default)",
            cfg.mm.size_eta
        );
    }

    if let Some(vref) = args.vol_ref {
        let vref_clamped = vref.max(cfg.volatility.sigma_min);
        cfg.volatility.vol_ref = vref_clamped;
        println!(
            "[config] CLI --vol-ref = {} (overrode env/default)",
            cfg.volatility.vol_ref
        );
    }

    // ---- Environment overrides (used heavily by Python harnesses) ----

    if let Ok(q_env) = env::var("PARAPHINA_INITIAL_Q_TAO") {
        if let Ok(q0) = q_env.parse::<f64>() {
            cfg.initial_q_tao = q0;
            println!(
                "[config] ENV PARAPHINA_INITIAL_Q_TAO = {} (overrode default/CLI)",
                q0
            );
        }
    }

    if let Ok(band_env) = env::var("PARAPHINA_HEDGE_BAND_BASE") {
        if let Ok(band) = band_env.parse::<f64>() {
            cfg.hedge.hedge_band_base = band.max(0.0);
            println!(
                "[config] ENV PARAPHINA_HEDGE_BAND_BASE = {} (overrode default/CLI)",
                cfg.hedge.hedge_band_base
            );
        }
    }

    if let Ok(eta_env) = env::var("PARAPHINA_MM_SIZE_ETA") {
        if let Ok(eta) = eta_env.parse::<f64>() {
            cfg.mm.size_eta = eta.max(1e-6);
            println!(
                "[config] ENV PARAPHINA_MM_SIZE_ETA = {} (overrode default/CLI)",
                cfg.mm.size_eta
            );
        }
    }

    if let Ok(vref_env) = env::var("PARAPHINA_VOL_REF") {
        if let Ok(vref) = vref_env.parse::<f64>() {
            cfg.volatility.vol_ref = vref.max(cfg.volatility.sigma_min);
            println!(
                "[config] ENV PARAPHINA_VOL_REF = {} (overrode default/CLI)",
                cfg.volatility.vol_ref
            );
        }
    }
}

fn main() {
    // 0) Parse CLI args.
    let args = Args::parse();

    // 1) Load / build config.
    let mut cfg = Config::default();

    // 1b) Apply CLI + env overrides (research knobs).
    apply_overrides(&mut cfg, &args);

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
