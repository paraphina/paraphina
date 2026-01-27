// tests/smoke_profiles.rs
//
// Minimal end-to-end smoke tests for the Paraphina engine.
// Goal: if any core wiring breaks (config, gateway, runner, sinks),
// these tests will fail immediately.

use paraphina::io::sim::create_sim_adapters;
use paraphina::io::{Gateway, GatewayPolicy};
use paraphina::{Config, NoopSink, StrategyRunner};

fn run_smoke(cfg: Config, ticks: u64) {
    // Use the same wiring as src/main.rs, but with a NoopSink so tests are fast.
    let gateway = Gateway::new(create_sim_adapters(&cfg), GatewayPolicy::for_simulation());
    let sink = Box::new(NoopSink);
    let mut runner = StrategyRunner::new(&cfg, gateway, sink);

    // We don’t assert on PnL yet; this is a “does not panic / wire is intact” test.
    runner.run_simulation(ticks);
}

/// Basic regression: a "profile-like" default config must be runnable
/// for a modest number of ticks without panicking.
#[test]
fn smoke_profile_like_default() {
    let mut cfg = Config {
        initial_q_tao: 0.0,
        ..Config::default()
    };

    // Mirror the profile presets you just validated:
    cfg.risk.daily_loss_limit = -2_000.0;

    run_smoke(cfg, 1_000);
}

/// Sanity check: a looser daily loss limit should also run cleanly.
#[test]
fn smoke_loose_loss_limit() {
    let mut cfg = Config {
        initial_q_tao: 0.0,
        ..Config::default()
    };

    cfg.risk.daily_loss_limit = -10_000.0;

    run_smoke(cfg, 1_000);
}
