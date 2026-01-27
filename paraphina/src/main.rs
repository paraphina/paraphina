// src/main.rs
//
// Research-harness friendly CLI entrypoint for Paraphina.
//
// Constraints:
// - CLI profile precedence:
//     --profile overrides env;
//     if missing use PARAPHINA_RISK_PROFILE (default Balanced).
// - Deterministic runs via --seed (offset synthetic timebase).
// - Tick count, optional verbosity.
// - Print concise run header (profile, ticks, cfg version/hash).
// - Exit engine is wired in StrategyRunner BEFORE hedge.

use clap::{ArgAction, Parser, ValueEnum};

use paraphina::config::{resolve_effective_profile, Config, RiskProfile};
use paraphina::io::sim::create_sim_adapters;
use paraphina::io::{Gateway, GatewayPolicy};
use paraphina::logging::NoopSink;
use paraphina::strategy_action::StrategyRunner;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ProfileArg {
    Conservative,
    Balanced,
    Aggressive,
}

#[derive(Debug, Parser)]
#[command(
    name = "paraphina",
    about = "Paraphina TAO perp MM + hedge simulator (research harness)",
    version
)]
struct Args {
    /// Number of synthetic ticks to run.
    #[arg(long, default_value_t = 2000)]
    ticks: u64,

    /// Risk profile preset (optional).
    /// If omitted, uses PARAPHINA_RISK_PROFILE (default Balanced).
    #[arg(long, value_enum)]
    profile: Option<ProfileArg>,

    /// Deterministic seed (used to offset synthetic timebase).
    #[arg(long)]
    seed: Option<u64>,

    /// Verbosity: -v, -vv
    #[arg(short, long, action = ArgAction::Count)]
    verbose: u8,
}

fn fnv1a64(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut h = FNV_OFFSET;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn main() {
    let args = Args::parse();

    // Convert CLI ProfileArg to RiskProfile (if provided)
    let cli_profile = args.profile.map(|p| match p {
        ProfileArg::Conservative => RiskProfile::Conservative,
        ProfileArg::Balanced => RiskProfile::Balanced,
        ProfileArg::Aggressive => RiskProfile::Aggressive,
    });

    // Resolve profile with proper precedence: CLI > env > default
    // (No scenario profile for main binary, so pass None)
    let effective = resolve_effective_profile(cli_profile, None);
    let profile = effective.profile;

    // Explicit startup log line (required by spec for observability)
    effective.log_startup();

    // Profile presets + env overrides already handled in Config.
    let cfg = Config::from_env_or_profile(profile);
    let cfg_hash = fnv1a64(&format!("{cfg:?}"));

    println!(
        "paraphina | cfg={} | cfg_hash=0x{:016x} | profile={:?} | ticks={} | seed={}",
        cfg.version,
        cfg_hash,
        profile,
        args.ticks,
        args.seed
            .map(|s| s.to_string())
            .unwrap_or_else(|| "none".to_string())
    );

    // Gateway + sink
    let gateway = Gateway::new(create_sim_adapters(&cfg), GatewayPolicy::for_simulation());
    let sink = NoopSink;

    // Strategy runner owns engine/state/telemetry and runs the loop.
    let mut runner = StrategyRunner::new(&cfg, gateway, sink);
    runner.set_seed(args.seed);
    runner.set_verbosity(args.verbose);
    runner.run_simulation(args.ticks);
}
