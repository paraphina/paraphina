// src/bin/sim_eval.rs
//
// Simulation & Evaluation runner binary (Option B per ROADMAP.md).
//
// This binary:
// - Loads a scenario YAML file
// - Expands seeds (base_seed + k)
// - Runs the existing simulator pathway (SimEnv)
// - Writes outputs to runs/<scenario_id>/<git_sha>/<seed>/
// - Prints a one-line summary per seed to stdout
//
// Usage:
//   cargo run --bin sim_eval -- scenarios/v1/synth_baseline.yaml
//   cargo run --bin sim_eval -- scenarios/v1/synth_baseline.yaml --output-dir ./my_runs

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use paraphina::config::{Config, RiskProfile};
use paraphina::metrics::DrawdownTracker;
use paraphina::rl::sim_env::SimEnvConfig;
use paraphina::rl::{PolicyAction, SimEnv};
use paraphina::sim_eval::{
    create_output_dir, write_build_info, write_config_resolved, BuildInfo, Engine,
    ExpectKillSwitch, KillSwitchInfo, MarketModelType, RunSummary, ScenarioSpec, SyntheticProcess,
};

/// Command-line arguments.
struct Args {
    /// Path to scenario YAML file.
    scenario_path: PathBuf,
    /// Output directory (default: runs/).
    output_dir: PathBuf,
    /// Whether to write metrics.jsonl.
    write_metrics: bool,
    /// Whether to print verbose output.
    verbose: bool,
}

impl Args {
    fn usage() -> &'static str {
        "\
sim_eval - Simulation & Evaluation runner (Option B)

USAGE:
  cargo run --bin sim_eval -- <SCENARIO_PATH> [OPTIONS]

ARGUMENTS:
  <SCENARIO_PATH>    Path to scenario YAML file (required)

OPTIONS:
  --output-dir DIR   Output directory (default: runs/)
  --metrics          Write metrics.jsonl for each run
  --verbose          Print verbose output
  --help             Show this help

EXAMPLES:
  cargo run --bin sim_eval -- scenarios/v1/synth_baseline.yaml
  cargo run --bin sim_eval -- scenarios/v1/synth_jump.yaml --output-dir ./my_runs --metrics
"
    }

    fn parse_or_exit() -> Self {
        match Self::parse() {
            Ok(args) => args,
            Err(e) => {
                eprintln!("Error: {}\n\n{}", e, Self::usage());
                std::process::exit(2);
            }
        }
    }

    fn parse() -> Result<Self, String> {
        let mut args = Args {
            scenario_path: PathBuf::new(),
            output_dir: PathBuf::from("runs"),
            write_metrics: false,
            verbose: false,
        };

        let mut it = env::args().skip(1);
        let mut scenario_set = false;

        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    println!("{}", Self::usage());
                    std::process::exit(0);
                }
                "--output-dir" => {
                    let val = it
                        .next()
                        .ok_or_else(|| "Missing value for --output-dir".to_string())?;
                    args.output_dir = PathBuf::from(val);
                }
                "--metrics" => {
                    args.write_metrics = true;
                }
                "--verbose" | "-v" => {
                    args.verbose = true;
                }
                _ if arg.starts_with("--output-dir=") => {
                    args.output_dir = PathBuf::from(&arg["--output-dir=".len()..]);
                }
                _ if arg.starts_with('-') => {
                    return Err(format!("Unknown option: {}", arg));
                }
                _ => {
                    if scenario_set {
                        return Err("Multiple scenario paths provided".to_string());
                    }
                    args.scenario_path = PathBuf::from(arg);
                    scenario_set = true;
                }
            }
        }

        if !scenario_set {
            return Err("Missing required argument: <SCENARIO_PATH>".to_string());
        }

        Ok(args)
    }
}

/// Map risk profile string to RiskProfile enum.
fn parse_risk_profile(s: &str) -> RiskProfile {
    match s.to_lowercase().as_str() {
        "conservative" | "cons" | "c" => RiskProfile::Conservative,
        "aggressive" | "agg" | "a" => RiskProfile::Aggressive,
        _ => RiskProfile::Balanced,
    }
}

/// Run result for a single seed.
struct RunResult {
    seed: u64,
    final_pnl: f64,
    max_drawdown: f64,
    kill_switch: KillSwitchInfo,
    steps_executed: u64,
    duration_ms: u128,
}

/// Run simulation for a single seed using SimEnv.
fn run_single_seed(spec: &ScenarioSpec, seed: u64, verbose: bool) -> RunResult {
    let start = Instant::now();

    // Create config from risk profile
    let risk_profile = parse_risk_profile(&spec.initial_state.risk_profile);
    let mut config = Config::for_profile(risk_profile);
    config.initial_q_tao = spec.initial_state.init_q_tao;

    // Apply market model parameters
    if let Some(ref synth) = spec.market_model.synthetic {
        // Map vol to vol_ref for scaling
        if synth.params.vol > 0.0 {
            config.volatility.vol_ref = synth.params.vol;
        }
    }

    // Apply microstructure model
    for venue in &mut config.venues {
        venue.maker_fee_bps = spec.microstructure_model.fees_bps_maker;
        venue.taker_fee_bps = spec.microstructure_model.fees_bps_taker;
    }

    // Create SimEnv config
    let dt_ms = (spec.horizon.dt_seconds * 1000.0) as i64;
    let mut env_config = SimEnvConfig::deterministic();
    env_config.max_ticks = spec.horizon.steps;
    env_config.dt_ms = dt_ms;

    // Apply domain randomization based on market model
    if let Some(ref synth) = spec.market_model.synthetic {
        match synth.process {
            SyntheticProcess::JumpDiffusionStub => {
                // Enable mild domain rand for jump diffusion to add stochasticity
                env_config.apply_domain_rand = true;
                env_config.domain_rand.vol_ref_range = (0.015, 0.025);
            }
            SyntheticProcess::Gbm => {
                // Keep deterministic for GBM baseline
                env_config.apply_domain_rand = false;
            }
        }
    }

    // Create environment
    let mut env = SimEnv::new(config, env_config);

    // Reset with seed
    let _obs = env.reset(Some(seed));

    // Run episode
    let identity_action = PolicyAction::identity(env.num_venues(), "sim_eval");
    let mut dd_tracker = DrawdownTracker::new();
    let mut kill_switch = KillSwitchInfo::default();
    let mut steps_executed = 0u64;

    for step in 0..spec.horizon.steps {
        let result = env.step(&identity_action);
        steps_executed = step + 1;

        // Track drawdown
        dd_tracker.update(result.info.pnl_total);

        // Check for kill switch
        if result.info.kill_switch && !kill_switch.triggered {
            kill_switch.triggered = true;
            kill_switch.step = Some(step);
            kill_switch.reason = result.info.kill_reason.clone();
        }

        if verbose && (step + 1) % 500 == 0 {
            eprintln!(
                "  seed={} step={}/{} pnl={:.2}",
                seed,
                step + 1,
                spec.horizon.steps,
                result.info.pnl_total
            );
        }

        if result.done {
            break;
        }
    }

    let state = env.state();
    let final_pnl = state.daily_pnl_total;
    let max_drawdown = dd_tracker.max_drawdown();
    let duration_ms = start.elapsed().as_millis();

    RunResult {
        seed,
        final_pnl,
        max_drawdown,
        kill_switch,
        steps_executed,
        duration_ms,
    }
}

/// Check invariants for a run result.
fn check_invariants(spec: &ScenarioSpec, result: &RunResult) -> Result<(), String> {
    match spec.invariants.expect_kill_switch {
        ExpectKillSwitch::Always => {
            if !result.kill_switch.triggered {
                return Err(format!(
                    "Invariant violation: kill_switch expected=Always but not triggered (seed={})",
                    result.seed
                ));
            }
        }
        ExpectKillSwitch::Never => {
            if result.kill_switch.triggered {
                return Err(format!(
                    "Invariant violation: kill_switch expected=Never but triggered at step {} (seed={})",
                    result.kill_switch.step.unwrap_or(0),
                    result.seed
                ));
            }
        }
        ExpectKillSwitch::Allowed => {
            // Any outcome is acceptable
        }
    }
    Ok(())
}

fn main() {
    let args = Args::parse_or_exit();

    // Load scenario
    let spec = match ScenarioSpec::from_yaml_file(&args.scenario_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load scenario: {}", e);
            std::process::exit(1);
        }
    };

    // Capture build info
    let build_info = BuildInfo::capture();

    // Print header
    println!(
        "sim_eval | scenario={} version={} engine={:?} steps={} seeds={}",
        spec.scenario_id,
        spec.scenario_version,
        spec.engine,
        spec.horizon.steps,
        spec.rng.num_seeds
    );
    println!(
        "         | risk_profile={} init_q_tao={:.2} dt_seconds={:.3}",
        spec.initial_state.risk_profile, spec.initial_state.init_q_tao, spec.horizon.dt_seconds
    );
    println!(
        "         | git_sha={} dirty={}",
        &build_info.git_sha[..8.min(build_info.git_sha.len())],
        build_info.dirty
    );
    println!();

    // Check engine support
    match spec.engine {
        Engine::RlSimEnv | Engine::CoreSim => {
            // Supported
        }
        Engine::ReplayStub => {
            eprintln!("Warning: replay_stub engine not implemented, using rl_sim_env fallback");
        }
    }

    // Check market model support
    if spec.market_model.model_type == MarketModelType::HistoricalStub {
        eprintln!(
            "Warning: historical_stub market model not implemented, using synthetic fallback"
        );
    }

    // Expand seeds and run
    let seeds = spec.expand_seeds();
    let mut all_results = Vec::with_capacity(seeds.len());
    let mut invariant_failures = Vec::new();

    for (k, seed) in &seeds {
        if args.verbose {
            eprintln!("Running seed {} ({}/{})", seed, k + 1, spec.rng.num_seeds);
        }

        let result = run_single_seed(&spec, *seed, args.verbose);

        // Check invariants
        if let Err(e) = check_invariants(&spec, &result) {
            invariant_failures.push(e.clone());
            eprintln!("  {}", e);
        }

        // Create output directory and write files
        let output_dir = match create_output_dir(
            &args.output_dir,
            &spec.scenario_id,
            &build_info.git_sha,
            *seed,
        ) {
            Ok(dir) => dir,
            Err(e) => {
                eprintln!("Failed to create output directory: {}", e);
                std::process::exit(1);
            }
        };

        // Write run_summary.json
        let summary = RunSummary::new(
            &spec,
            *seed,
            build_info.clone(),
            result.final_pnl,
            result.max_drawdown,
            result.kill_switch.clone(),
        );

        if let Err(e) = summary.write_to_file(output_dir.join("run_summary.json")) {
            eprintln!("Failed to write run_summary.json: {}", e);
        }

        // Write config_resolved.json
        if let Err(e) = write_config_resolved(output_dir.join("config_resolved.json"), &spec) {
            eprintln!("Failed to write config_resolved.json: {}", e);
        }

        // Write build_info.json
        if let Err(e) = write_build_info(output_dir.join("build_info.json"), &build_info) {
            eprintln!("Failed to write build_info.json: {}", e);
        }

        // Print one-line summary
        let kill_str = if result.kill_switch.triggered {
            format!(
                "KILL@{}",
                result
                    .kill_switch
                    .step
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            )
        } else {
            "OK".to_string()
        };

        println!(
            "seed={:<10} pnl={:>10.2} maxDD={:>10.2} steps={:>5} status={:<10} time={:>6}ms checksum={}",
            seed,
            result.final_pnl,
            result.max_drawdown,
            result.steps_executed,
            kill_str,
            result.duration_ms,
            &summary.determinism.checksum[..16]
        );

        all_results.push(result);
    }

    // Print summary
    println!();
    if !all_results.is_empty() {
        let avg_pnl: f64 =
            all_results.iter().map(|r| r.final_pnl).sum::<f64>() / all_results.len() as f64;
        let avg_dd: f64 =
            all_results.iter().map(|r| r.max_drawdown).sum::<f64>() / all_results.len() as f64;
        let kill_count = all_results
            .iter()
            .filter(|r| r.kill_switch.triggered)
            .count();

        println!(
            "SUMMARY | runs={} avg_pnl={:.2} avg_maxDD={:.2} kills={}/{}",
            all_results.len(),
            avg_pnl,
            avg_dd,
            kill_count,
            all_results.len()
        );
    }

    // Report invariant failures
    if !invariant_failures.is_empty() {
        println!();
        println!(
            "INVARIANT FAILURES: {} / {}",
            invariant_failures.len(),
            all_results.len()
        );
        for failure in &invariant_failures {
            println!("  - {}", failure);
        }
        std::process::exit(1);
    }

    println!();
    println!("Output written to: {}/", args.output_dir.display());
}
