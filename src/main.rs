// src/main.rs
//
// Thin harness around the Paraphina library.
// All of the real logic lives in the lib crate (engine, strategy, etc).

use paraphina::{
    Config,
    SimGateway,
    StrategyRunner,
    EventSink,
    FileSink,
    NoopSink,
};

/// Build the telemetry sink as a trait object so we can choose between
/// FileSink and NoopSink at runtime.
fn build_sink(use_file_sink: bool) -> Box<dyn EventSink> {
    if use_file_sink {
        match FileSink::create("paraphina_ticks.jsonl") {
            Ok(s) => Box::new(s),
            Err(err) => {
                eprintln!(
                    "Failed to create log file (paraphina_ticks.jsonl), \
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
    // 1) Load / build config.
    let cfg = Config::default();

    // 2) Choose execution gateway (here: synthetic sim).
    let gateway = SimGateway::new();

    // 3) Choose telemetry sink.
    //
    //    - NoopSink  -> no on-disk logs, just prints to stdout.
    //    - FileSink  -> JSONL file with 1 record per tick for backtesting / RL.
    //
    // Flip this flag when you want real logs.
    let use_file_sink = false;
    let sink = build_sink(use_file_sink);

    // 4) Run the high-level strategy for N ticks.
    let num_ticks: u64 = 50;
    let mut runner = StrategyRunner::new(&cfg, gateway, sink);
    runner.run_simulation(num_ticks);
}