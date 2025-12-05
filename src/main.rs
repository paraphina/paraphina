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

/// CLI configuration for the simulation / research harness.
///
/// Examples:
///   cargo run -- --ticks 50
///   cargo run -- --ticks 200 --log-jsonl research_run_002.jsonl
#[derive(Parser, Debug)]
#[command(
    name = "paraphina",
    version,
    about = "TAO perp market-making research / simulation harness"
)]
struct Cli {
    /// Number of ticks to simulate
    #[arg(long, default_value_t = 50)]
    ticks: u64,

    /// Optional JSONL file path to log per-tick state.
    ///
    /// If omitted, no JSONL log is written (stdout only).
    #[arg(long, value_name = "PATH")]
    log_jsonl: Option<String>,
}

/// Build the telemetry sink as a trait object so we can choose between
/// FileSink and NoopSink at runtime, based on the CLI options.
fn build_sink(log_jsonl: &Option<String>) -> Box<dyn EventSink> {
    if let Some(path) = log_jsonl {
        match FileSink::create(path) {
            Ok(sink) => Box::new(sink),
            Err(err) => {
                eprintln!(
                    "Failed to create log file ({path}), falling back to NoopSink: {err}"
                );
                Box::new(NoopSink)
            }
        }
    } else {
        // No log file requested → just print to stdout.
        Box::new(NoopSink)
    }
}

fn main() {
    // 0) Parse CLI flags.
    let cli = Cli::parse();

    // 1) Load / build config.
    let cfg = Config::default();

    // 2) Choose execution gateway (here: synthetic sim).
    let gateway = SimGateway::new();

    // 3) Build telemetry sink from CLI options.
    let sink = build_sink(&cli.log_jsonl);

    // 4) Run the high-level strategy for N ticks.
    let mut runner = StrategyRunner::new(&cfg, gateway, sink);
    runner.run_simulation(cli.ticks);
}
