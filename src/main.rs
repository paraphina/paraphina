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

/// Command-line arguments for the Paraphina simulator.
#[derive(Parser, Debug)]
#[command(
    name = "paraphina",
    version,
    about = "Paraphina perp MM / hedge simulator",
    long_about = None
)]
struct Args {
    /// Number of ticks to simulate.
    #[arg(long, default_value_t = 50)]
    num_ticks: u64,

    /// Whether to log ticks to a JSONL file.
    #[arg(long, default_value_t = false)]
    use_file_sink: bool,

    /// Output JSONL filename (used only if --use-file-sink is true).
    #[arg(long, default_value = "paraphina_ticks.jsonl")]
    out: String,
}

/// Build the telemetry sink as a trait object so we can choose between
/// FileSink and NoopSink at runtime.
fn build_sink(use_file_sink: bool, path: &str) -> Box<dyn EventSink> {
    if use_file_sink {
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

fn main() {
    // 0) Parse CLI args.
    let args = Args::parse();

    // 1) Load / build config.
    let cfg = Config::default();

    // 2) Choose execution gateway (here: synthetic sim).
    let gateway = SimGateway::new();

    // 3) Choose telemetry sink from CLI flags.
    let sink = build_sink(args.use_file_sink, &args.out);

    // 4) Run the high-level strategy for N ticks.
    let mut runner = StrategyRunner::new(&cfg, gateway, sink);
    runner.run_simulation(args.num_ticks);
}
