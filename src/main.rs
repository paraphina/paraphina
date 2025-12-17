// src/main.rs
//
// CLI entrypoint for the Paraphina research harness.
//
// IMPORTANT precedence rule:
// - If --profile is provided, it wins.
// - If --profile is NOT provided, we fall back to PARAPHINA_RISK_PROFILE,
//   via Config::from_env_or_default().
//
// This makes batch_runs/* experiments (which set env vars) behave as intended.

use clap::{Parser, ValueEnum};

use paraphina::config::{Config, RiskProfile};
use paraphina::gateway::SimGateway;
use paraphina::logging::NoopSink;
use paraphina::strategy::StrategyRunner;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ProfileArg {
    Conservative,
    Balanced,
    Aggressive,
}

#[derive(Parser, Debug)]
#[command(
    name = "paraphina",
    version,
    about = "Paraphina TAO perp MM + hedge simulator",
    long_about = None
)]
struct Cli {
    /// Number of synthetic ticks to run.
    /// Defaults to 2000 if not provided.
    #[arg(long, default_value_t = 2000)]
    ticks: u64,

    /// Risk profile preset: conservative | balanced | aggressive.
    ///
    /// If omitted, we use PARAPHINA_RISK_PROFILE (default Balanced).
    #[arg(long, value_enum)]
    profile: Option<ProfileArg>,
}

fn main() {
    let cli = Cli::parse();

    // Build config:
    // - If CLI profile is provided, use it (with env overrides).
    // - Otherwise use env-selected profile (with env overrides).
    let cfg = match cli.profile {
        Some(p) => {
            let profile = match p {
                ProfileArg::Conservative => RiskProfile::Conservative,
                ProfileArg::Balanced => RiskProfile::Balanced,
                ProfileArg::Aggressive => RiskProfile::Aggressive,
            };
            Config::from_env_or_profile(profile)
        }
        None => Config::from_env_or_default(),
    };

    let gateway = SimGateway;
    let sink = NoopSink;

    let mut runner = StrategyRunner::new(&cfg, gateway, sink);
    runner.run_simulation(cli.ticks);
}
