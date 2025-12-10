// src/main.rs
//
// CLI entrypoint for the Paraphina research harness.
//
// Example usage:
//
//   cargo run --release
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
    /// Defaults to 2000 if not provided.
    #[arg(long, default_value_t = 2000)]
    ticks: u64,

    /// Risk profile preset: conservative | balanced | aggressive.
    #[arg(long, value_enum, default_value_t = ProfileArg::Balanced)]
    profile: ProfileArg,
}

fn main() {
    // Parse CLI args.
    let cli = Cli::parse();

    // Map CLI profile to internal RiskProfile enum.
    let profile = match cli.profile {
        ProfileArg::Conservative => RiskProfile::Conservative,
        ProfileArg::Balanced => RiskProfile::Balanced,
        ProfileArg::Aggressive => RiskProfile::Aggressive,
    };

    // Build config from profile + env overrides.
    let cfg = Config::from_env_or_profile(profile);

    // Pure synthetic gateway + no-op event sink (tests / manual sims).
    let gateway = SimGateway; // unit struct – no ::default()
    let sink = NoopSink;

    // Strategy runner owns engine, state, telemetry, and logging.
    let mut runner = StrategyRunner::new(&cfg, gateway, sink);
    runner.run_simulation(cli.ticks);
}
