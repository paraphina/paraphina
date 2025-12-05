// src/main.rs
//
// Thin CLI harness around the Paraphina library.
// All trading logic lives in the lib crate (engine, strategy, risk).

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
///
/// Typical usage:
///   cargo run -- --ticks 200
///   cargo run -- --ticks 200 --log-jsonl paraphina_ticks.jsonl
#[derive(Parser, Debug)]
#[command(
    name = "paraphina",
    version,
    author = "Paraphina research stack",
    about = "TAO perp MM + hedge research driver"
)]
struct Args {
    /// Number of synthetic ticks to simulate.
    ///
    /// In research mode each \"tick\" is one full MM + hedge step.
    #[arg(long, default_value_t = 200)]
    ticks: u64,

    /// JSONL file to log per-tick snapshots.
    ///
    /// When set, a single JSON object is written per tick; this is what
    /// tools/research_ticks.py and other research tooling expects.
    /// If omitted, no JSONL log is written and we only stream to stdout.
    #[arg(long)]
    log_jsonl: Option<String>,
}

/// Build the telemetry sink as a trait object so we can choose between
/// FileSink and NoopSink at runtime.
fn build_sink(log_jsonl: Option<&str>) -> Box<dyn EventSink> {
    if let Some(path) = log_jsonl {
        match FileSink::create(path) {
            Ok(sink) => Box::new(sink),
            Err(err) => {
                eprintln!(
                    "Failed to create log file ({path}); \
                     continuing with NoopSink. Error: {err}"
                );
                Box::new(NoopSink)
            }
        }
    } else {
        Box::new(NoopSink)
    }
}

fn main() {
    // 0) Parse CLI args.
    let args = Args::parse();

    // 1) Load / build config (single source of truth for all parameters).
    let cfg = Config::default();

    // 2) Choose execution gateway (synthetic multi-venue simulator).
    let gateway = SimGateway::new();

    // 3) Build telemetry sink based on CLI arguments.
    let sink = build_sink(args.log_jsonl.as_deref());

    // 4) Run the high-level strategy for N ticks.
    let mut runner = StrategyRunner::new(&cfg, gateway, sink);
    runner.run_simulation(args.ticks);
}
