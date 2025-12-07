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

/// High-level risk / behaviour profiles, distilled from Exp07/Exp08/Exp09.
///
/// IMPORTANT:
/// - These are *calibrated starting points*, not hard-coded truths.
/// - CLI flags and env-vars still override them.
/// - When you re-run Exp06/07/08/09 in the future, update these numbers
///   from the new summary tables.
#[derive(Clone, Copy, Debug)]
enum Profile {
    Conservative,
    Balanced,
    Aggressive,
}

impl Profile {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "conservative" | "cons" | "c" => Ok(Profile::Conservative),
            "balanced" | "bal" | "b" => Ok(Profile::Balanced),
            "aggressive" | "agg" | "a" => Ok(Profile::Aggressive),
            other => Err(format!(
                "Unknown profile '{other}'. Expected one of: \
                 conservative, balanced, aggressive."
            )),
        }
    }
}

/// Base configs for each profile, generated from Exp07/Exp09 presets.
///
/// These are *starting points*; CLI + env overrides still apply on top.
/// Tuned from:
///   - Exp07 (risk regime sweep)
///   - Exp09 (hyperparam search: band / size_eta / vol_ref / loss_limit)
fn config_for_profile(profile: Profile) -> Config {
    let mut cfg = Config::default();

    // Shared tuned parameters from Exp09 hyperparam search.
    cfg.hedge.hedge_band_base = 2.5;
    cfg.mm.size_eta = 0.05;
    cfg.volatility.vol_ref = 0.01;

    match profile {
        Profile::Conservative => {
            cfg.initial_q_tao = 0.0;
            cfg.risk.daily_loss_limit = -2_000.0;
            cfg
        }
        Profile::Balanced => {
            cfg.initial_q_tao = 0.0;
            cfg.risk.daily_loss_limit = -2_000.0;
            cfg
        }
        Profile::Aggressive => {
            cfg.initial_q_tao = 0.0;
            cfg.risk.daily_loss_limit = -2_000.0;
            cfg
        }
    }
}

/// Command-line arguments for the Paraphina binary.
#[derive(Parser, Debug)]
#[command(name = "paraphina")]
struct Cli {
    /// Number of synthetic ticks to run.
    #[arg(long, value_name = "TICKS")]
    ticks: u64,

    /// Optional initial global position q0 in TAO.
    ///
    /// - Can be NEGATIVE (short inventory).
    /// - If omitted, we use either the profile's q0 (if --profile is set)
    ///   or the default from Config::default().
    #[arg(
        long,
        value_name = "INITIAL_Q_TAO",
        allow_hyphen_values = true
    )]
    initial_q_tao: Option<f64>,

    /// Base hedge band (TAO).
    /// Exposed so research harnesses can sweep hedge_band_base.
    /// If omitted, we keep the value from the profile or Config::default().
    #[arg(long, value_name = "HEDGE_BAND_BASE")]
    hedge_band_base: Option<f64>,

    /// Optional per-day loss limit in USD (positive number).
    ///
    /// NOTE: The risk engine stores this as a NEGATIVE quantity, so we
    /// convert `+4000.0` on the CLI into `-4000.0` in Config.
    #[arg(long, value_name = "LOSS_LIMIT_USD")]
    loss_limit_usd: Option<f64>,

    /// Optional JSONL path for tick log.
    ///
    /// When provided, we stream one JSON record per tick into this file
    /// for offline analysis / research.
    #[arg(long, value_name = "PATH")]
    log_jsonl: Option<String>,

    /// Optional named risk/profile preset:
    ///   - conservative
    ///   - balanced
    ///   - aggressive
    ///
    /// Example:
    ///   paraphina --profile balanced --ticks 2000 --hedge-band-base 5.0
    ///
    /// CLI and env overrides still apply on top of the profile config.
    #[arg(long, value_name = "PROFILE")]
    profile: Option<String>,
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

/// Build Config from defaults and profile, then apply CLI + env overrides.
///
/// Precedence:
///   1) Start from profile config if --profile is set, else Config::default().
///   2) Apply CLI overrides (initial_q_tao, hedge_band_base, loss_limit_usd).
///   3) Apply env overrides (PARAPHINA_*, PARA_* research knobs).
fn build_config_from_env_and_args(cli: &Cli) -> Config {
    // ---------- Base config: profile or default ----------

    let profile = match &cli.profile {
        None => None,
        Some(name) => match Profile::from_str(name) {
            Ok(p) => Some(p),
            Err(msg) => {
                eprintln!("WARNING: {msg}. Falling back to default config.");
                None
            }
        },
    };

    let mut cfg = match profile {
        Some(p) => config_for_profile(p),
        None => Config::default(),
    };

    // ---------- CLI overrides ----------

    // Initial inventory: only override if CLI provided a value.
    if let Some(q) = cli.initial_q_tao {
        cfg.initial_q_tao = q;
    }

    // Base hedge band (TAO).
    // Only override if CLI provided a value; otherwise use profile/default.
    if let Some(band) = cli.hedge_band_base {
        cfg.hedge.hedge_band_base = band;
    }

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

    // 1) Load / build config with profile + CLI + env overrides.
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
