// src/main.rs
//
// CLI entrypoint for the Paraphina research harness.
//
// Example usage:
//
//   cargo run --release -- --ticks 500
//   cargo run --release -- --ticks 500 --profile aggressive
//
// The `ticks` argument controls the synthetic simulation length.
// The `profile` argument selects a coarse risk preset.

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
    #[arg(long)]
    ticks: u64,

    /// Risk profile preset: conservative | balanced | aggressive.
    #[arg(long, value_enum, default_value_t = ProfileArg::Balanced)]
    profile: ProfileArg,
}

fn main() {
    let cli = Cli::parse();

    let profile = match cli.profile {
        ProfileArg::Conservative => RiskProfile::Conservative,
        ProfileArg::Balanced => RiskProfile::Balanced,
        ProfileArg::Aggressive => RiskProfile::Aggressive,
    };

    // Build config from profile + env overrides.
    let cfg = Config::from_env_or_profile(profile);

    // Pure synthetic gateway + no-op sink (tests / manual sims).
    let gateway = SimGateway; // unit struct – no ::default()
    let sink = NoopSink;

    let mut runner = StrategyRunner::new(&cfg, gateway, sink);
    runner.run_simulation(cli.ticks);
}
