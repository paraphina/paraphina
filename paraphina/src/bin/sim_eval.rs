// src/bin/sim_eval.rs
//
// Simulation & Evaluation runner binary (Option B per ROADMAP.md).
//
// This binary supports three subcommands:
// - run <SCENARIO_PATH>: Run a single scenario
// - suite <SUITE_PATH>: Run a CI suite with gates
// - summarize <RUNS_DIR>: Summarize run results from a directory
//
// Usage:
//   cargo run -p paraphina --bin sim_eval -- run scenarios/v1/synth_baseline.yaml
//   cargo run -p paraphina --bin sim_eval -- suite scenarios/suites/ci_smoke_v1.yaml
//   cargo run -p paraphina --bin sim_eval -- summarize runs/

use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use paraphina::config::{Config, RiskProfile};
use paraphina::metrics::DrawdownTracker;
use paraphina::rl::sim_env::SimEnvConfig;
use paraphina::rl::{PolicyAction, SimEnv};
use paraphina::sim_eval::{
    create_output_dir, print_ablations, run_report, summarize_with_format,
    verify_evidence_pack_dir, verify_evidence_pack_tree, write_build_info, write_config_resolved,
    write_config_resolved_with_ablations, write_evidence_pack, AblationSet, BuildInfo, Engine,
    ExpectKillSwitch, KillSwitchInfo, MarketModelType, OutputFormat, ReportArgs, ReportResult,
    RunSummary, ScenarioSpec, SuiteSpec, SummarizeResult, SyntheticProcess,
};

// =============================================================================
// Command-line argument parsing
// =============================================================================

#[derive(Debug)]
enum Command {
    Run(RunArgs),
    Suite(SuiteArgs),
    Summarize(SummarizeArgs),
    Report(ReportArgsLocal),
    Ablations,
    VerifyEvidencePack(VerifyEvidencePackArgs),
    VerifyEvidenceTree(VerifyEvidenceTreeArgs),
}

#[derive(Debug)]
struct ReportArgsLocal {
    baseline_dir: PathBuf,
    variants: Vec<(String, PathBuf)>,
    out_md: PathBuf,
    out_json: PathBuf,
    gate_max_regression_usd: Option<f64>,
    gate_max_regression_pct: Option<f64>,
}

#[derive(Debug)]
struct RunArgs {
    scenario_path: PathBuf,
    output_dir: PathBuf,
    write_metrics: bool,
    verbose: bool,
}

#[derive(Debug)]
struct SuiteArgs {
    suite_path: PathBuf,
    verbose: bool,
    ablations: Vec<String>,
}

#[derive(Debug)]
struct SummarizeArgs {
    runs_dir: PathBuf,
    format: OutputFormat,
}

#[derive(Debug)]
struct VerifyEvidencePackArgs {
    output_root: PathBuf,
}

#[derive(Debug)]
struct VerifyEvidenceTreeArgs {
    root: PathBuf,
}

fn usage() -> &'static str {
    "\
sim_eval - Simulation & Evaluation runner (Option B)

USAGE:
  sim_eval run <SCENARIO_PATH> [OPTIONS]
  sim_eval suite <SUITE_PATH> [OPTIONS]
  sim_eval summarize <RUNS_DIR> [OPTIONS]
  sim_eval report --baseline <DIR> --variant <NAME>=<DIR> ... --out-md <PATH> --out-json <PATH> [OPTIONS]
  sim_eval ablations
  sim_eval verify-evidence-pack <OUTPUT_ROOT>
  sim_eval verify-evidence-tree <ROOT>

SUBCOMMANDS:
  run                  Run a single scenario
  suite                Run a CI suite with determinism and invariant gates
  summarize            Discover and summarize run_summary.json files
  report               Generate baseline-vs-ablations research report in Markdown + JSON
  ablations            List supported ablation IDs and descriptions
  verify-evidence-pack Verify a single evidence pack at <OUTPUT_ROOT>/evidence_pack/
  verify-evidence-tree Verify all evidence packs under <ROOT>

RUN OPTIONS:
  --output-dir DIR   Output directory (default: runs/)
  --metrics          Write metrics.jsonl for each run
  --verbose          Print verbose output

SUITE OPTIONS:
  --verbose              Print verbose output
  --ablation <ID>        Enable an ablation (can be specified multiple times)

SUMMARIZE OPTIONS:
  --format <FORMAT>      Output format: text (default) or md (Markdown)

REPORT OPTIONS:
  --baseline <DIR>              Baseline runs directory (required)
  --variant <NAME>=<DIR>        Variant name and directory (can be specified multiple times, at least one required)
  --out-md <PATH>               Output path for Markdown report (required)
  --out-json <PATH>             Output path for JSON report (required)
  --gate-max-regression-usd <N> Maximum allowed regression in USD (optional gate)
  --gate-max-regression-pct <N> Maximum allowed regression as percentage (optional gate)

COMMON OPTIONS:
  --help             Show this help

EXIT CODES:
  0  Success
  1  Runtime error
  2  Invalid arguments / usage
  3  Verification failure (verify-evidence-pack, verify-evidence-tree)

EXAMPLES:
  sim_eval run scenarios/v1/synth_baseline.yaml
  sim_eval run scenarios/v1/synth_jump.yaml --output-dir ./my_runs --verbose
  sim_eval suite scenarios/suites/ci_smoke_v1.yaml
  sim_eval suite scenarios/suites/research_v1.yaml --ablation disable_vol_floor
  sim_eval suite scenarios/suites/research_v1.yaml --ablation disable_vol_floor --ablation disable_toxicity_gate
  sim_eval summarize runs/
  sim_eval summarize runs/ --format md
  sim_eval report --baseline runs/baseline --variant ablation1=runs/ablation1 --out-md report.md --out-json report.json
  sim_eval report --baseline runs/baseline --variant v1=runs/v1 --variant v2=runs/v2 --out-md report.md --out-json report.json --gate-max-regression-usd 50.0
  sim_eval ablations
  sim_eval verify-evidence-pack runs/my_suite/
  sim_eval verify-evidence-tree runs/
"
}

fn parse_args() -> Result<Command, String> {
    let mut args = env::args().skip(1);

    let subcommand = args
        .next()
        .ok_or_else(|| "Missing subcommand".to_string())?;

    match subcommand.as_str() {
        "--help" | "-h" => {
            println!("{}", usage());
            std::process::exit(0);
        }
        "run" => {
            let mut run_args = RunArgs {
                scenario_path: PathBuf::new(),
                output_dir: PathBuf::from("runs"),
                write_metrics: false,
                verbose: false,
            };
            let mut scenario_set = false;

            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--help" | "-h" => {
                        println!("{}", usage());
                        std::process::exit(0);
                    }
                    "--output-dir" => {
                        let val = args
                            .next()
                            .ok_or_else(|| "Missing value for --output-dir".to_string())?;
                        run_args.output_dir = PathBuf::from(val);
                    }
                    "--metrics" => {
                        run_args.write_metrics = true;
                    }
                    "--verbose" | "-v" => {
                        run_args.verbose = true;
                    }
                    _ if arg.starts_with("--output-dir=") => {
                        run_args.output_dir = PathBuf::from(&arg["--output-dir=".len()..]);
                    }
                    _ if arg.starts_with('-') => {
                        return Err(format!("Unknown option: {}", arg));
                    }
                    _ => {
                        if scenario_set {
                            return Err("Multiple scenario paths provided".to_string());
                        }
                        run_args.scenario_path = PathBuf::from(arg);
                        scenario_set = true;
                    }
                }
            }

            if !scenario_set {
                return Err("Missing required argument: <SCENARIO_PATH>".to_string());
            }

            Ok(Command::Run(run_args))
        }
        "suite" => {
            let mut suite_args = SuiteArgs {
                suite_path: PathBuf::new(),
                verbose: false,
                ablations: Vec::new(),
            };
            let mut suite_set = false;

            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--help" | "-h" => {
                        println!("{}", usage());
                        std::process::exit(0);
                    }
                    "--verbose" | "-v" => {
                        suite_args.verbose = true;
                    }
                    "--ablation" => {
                        let val = args
                            .next()
                            .ok_or_else(|| "Missing value for --ablation".to_string())?;
                        suite_args.ablations.push(val);
                    }
                    _ if arg.starts_with("--ablation=") => {
                        let val = arg["--ablation=".len()..].to_string();
                        suite_args.ablations.push(val);
                    }
                    _ if arg.starts_with('-') => {
                        return Err(format!("Unknown option: {}", arg));
                    }
                    _ => {
                        if suite_set {
                            return Err("Multiple suite paths provided".to_string());
                        }
                        suite_args.suite_path = PathBuf::from(arg);
                        suite_set = true;
                    }
                }
            }

            if !suite_set {
                return Err("Missing required argument: <SUITE_PATH>".to_string());
            }

            Ok(Command::Suite(suite_args))
        }
        "ablations" => Ok(Command::Ablations),
        "report" => {
            let mut report_args = ReportArgsLocal {
                baseline_dir: PathBuf::new(),
                variants: Vec::new(),
                out_md: PathBuf::new(),
                out_json: PathBuf::new(),
                gate_max_regression_usd: None,
                gate_max_regression_pct: None,
            };
            let mut baseline_set = false;
            let mut out_md_set = false;
            let mut out_json_set = false;

            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--help" | "-h" => {
                        println!("{}", usage());
                        std::process::exit(0);
                    }
                    "--baseline" => {
                        let val = args
                            .next()
                            .ok_or_else(|| "Missing value for --baseline".to_string())?;
                        report_args.baseline_dir = PathBuf::from(val);
                        baseline_set = true;
                    }
                    "--variant" => {
                        let val = args
                            .next()
                            .ok_or_else(|| "Missing value for --variant".to_string())?;
                        let parts: Vec<&str> = val.splitn(2, '=').collect();
                        if parts.len() != 2 {
                            return Err(format!(
                                "Invalid --variant format '{}', expected NAME=DIR",
                                val
                            ));
                        }
                        report_args
                            .variants
                            .push((parts[0].to_string(), PathBuf::from(parts[1])));
                    }
                    "--out-md" => {
                        let val = args
                            .next()
                            .ok_or_else(|| "Missing value for --out-md".to_string())?;
                        report_args.out_md = PathBuf::from(val);
                        out_md_set = true;
                    }
                    "--out-json" => {
                        let val = args
                            .next()
                            .ok_or_else(|| "Missing value for --out-json".to_string())?;
                        report_args.out_json = PathBuf::from(val);
                        out_json_set = true;
                    }
                    "--gate-max-regression-usd" => {
                        let val = args.next().ok_or_else(|| {
                            "Missing value for --gate-max-regression-usd".to_string()
                        })?;
                        let parsed: f64 = val.parse().map_err(|_| {
                            format!("Invalid number for --gate-max-regression-usd: {}", val)
                        })?;
                        report_args.gate_max_regression_usd = Some(parsed);
                    }
                    "--gate-max-regression-pct" => {
                        let val = args.next().ok_or_else(|| {
                            "Missing value for --gate-max-regression-pct".to_string()
                        })?;
                        let parsed: f64 = val.parse().map_err(|_| {
                            format!("Invalid number for --gate-max-regression-pct: {}", val)
                        })?;
                        report_args.gate_max_regression_pct = Some(parsed);
                    }
                    _ if arg.starts_with("--baseline=") => {
                        report_args.baseline_dir = PathBuf::from(&arg["--baseline=".len()..]);
                        baseline_set = true;
                    }
                    _ if arg.starts_with("--variant=") => {
                        let val = &arg["--variant=".len()..];
                        let parts: Vec<&str> = val.splitn(2, '=').collect();
                        if parts.len() != 2 {
                            return Err(format!(
                                "Invalid --variant format '{}', expected NAME=DIR",
                                val
                            ));
                        }
                        report_args
                            .variants
                            .push((parts[0].to_string(), PathBuf::from(parts[1])));
                    }
                    _ if arg.starts_with("--out-md=") => {
                        report_args.out_md = PathBuf::from(&arg["--out-md=".len()..]);
                        out_md_set = true;
                    }
                    _ if arg.starts_with("--out-json=") => {
                        report_args.out_json = PathBuf::from(&arg["--out-json=".len()..]);
                        out_json_set = true;
                    }
                    _ if arg.starts_with("--gate-max-regression-usd=") => {
                        let val = &arg["--gate-max-regression-usd=".len()..];
                        let parsed: f64 = val.parse().map_err(|_| {
                            format!("Invalid number for --gate-max-regression-usd: {}", val)
                        })?;
                        report_args.gate_max_regression_usd = Some(parsed);
                    }
                    _ if arg.starts_with("--gate-max-regression-pct=") => {
                        let val = &arg["--gate-max-regression-pct=".len()..];
                        let parsed: f64 = val.parse().map_err(|_| {
                            format!("Invalid number for --gate-max-regression-pct: {}", val)
                        })?;
                        report_args.gate_max_regression_pct = Some(parsed);
                    }
                    _ if arg.starts_with('-') => {
                        return Err(format!("Unknown option: {}", arg));
                    }
                    _ => {
                        return Err(format!("Unexpected argument: {}", arg));
                    }
                }
            }

            if !baseline_set {
                return Err("Missing required argument: --baseline".to_string());
            }
            if report_args.variants.is_empty() {
                return Err("At least one --variant is required".to_string());
            }
            if !out_md_set {
                return Err("Missing required argument: --out-md".to_string());
            }
            if !out_json_set {
                return Err("Missing required argument: --out-json".to_string());
            }

            Ok(Command::Report(report_args))
        }
        "summarize" => {
            let mut summarize_args = SummarizeArgs {
                runs_dir: PathBuf::new(),
                format: OutputFormat::Text,
            };
            let mut runs_dir_set = false;

            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--help" | "-h" => {
                        println!("{}", usage());
                        std::process::exit(0);
                    }
                    "--format" => {
                        let val = args
                            .next()
                            .ok_or_else(|| "Missing value for --format".to_string())?;
                        summarize_args.format = match val.to_lowercase().as_str() {
                            "text" => OutputFormat::Text,
                            "md" | "markdown" => OutputFormat::Markdown,
                            _ => {
                                return Err(format!(
                                    "Unknown format '{}', expected 'text' or 'md'",
                                    val
                                ))
                            }
                        };
                    }
                    _ if arg.starts_with("--format=") => {
                        let val = &arg["--format=".len()..];
                        summarize_args.format = match val.to_lowercase().as_str() {
                            "text" => OutputFormat::Text,
                            "md" | "markdown" => OutputFormat::Markdown,
                            _ => {
                                return Err(format!(
                                    "Unknown format '{}', expected 'text' or 'md'",
                                    val
                                ))
                            }
                        };
                    }
                    _ if arg.starts_with('-') => {
                        return Err(format!("Unknown option: {}", arg));
                    }
                    _ => {
                        if runs_dir_set {
                            return Err("Multiple runs directories provided".to_string());
                        }
                        summarize_args.runs_dir = PathBuf::from(arg);
                        runs_dir_set = true;
                    }
                }
            }

            if !runs_dir_set {
                return Err("Missing required argument: <RUNS_DIR>".to_string());
            }

            Ok(Command::Summarize(summarize_args))
        }
        "verify-evidence-pack" => {
            let mut verify_args = VerifyEvidencePackArgs {
                output_root: PathBuf::new(),
            };
            let mut path_set = false;

            for arg in args.by_ref() {
                match arg.as_str() {
                    "--help" | "-h" => {
                        println!("{}", usage());
                        std::process::exit(0);
                    }
                    _ if arg.starts_with('-') => {
                        return Err(format!("Unknown option: {}", arg));
                    }
                    _ => {
                        if path_set {
                            return Err("Multiple output roots provided".to_string());
                        }
                        verify_args.output_root = PathBuf::from(arg);
                        path_set = true;
                    }
                }
            }

            if !path_set {
                return Err("Missing required argument: <OUTPUT_ROOT>".to_string());
            }

            Ok(Command::VerifyEvidencePack(verify_args))
        }
        "verify-evidence-tree" => {
            let mut verify_args = VerifyEvidenceTreeArgs {
                root: PathBuf::new(),
            };
            let mut path_set = false;

            for arg in args.by_ref() {
                match arg.as_str() {
                    "--help" | "-h" => {
                        println!("{}", usage());
                        std::process::exit(0);
                    }
                    _ if arg.starts_with('-') => {
                        return Err(format!("Unknown option: {}", arg));
                    }
                    _ => {
                        if path_set {
                            return Err("Multiple roots provided".to_string());
                        }
                        verify_args.root = PathBuf::from(arg);
                        path_set = true;
                    }
                }
            }

            if !path_set {
                return Err("Missing required argument: <ROOT>".to_string());
            }

            Ok(Command::VerifyEvidenceTree(verify_args))
        }
        // Legacy support: if first arg looks like a path, treat as `run`
        other if !other.starts_with('-') => {
            let mut run_args = RunArgs {
                scenario_path: PathBuf::from(other),
                output_dir: PathBuf::from("runs"),
                write_metrics: false,
                verbose: false,
            };

            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--output-dir" => {
                        let val = args
                            .next()
                            .ok_or_else(|| "Missing value for --output-dir".to_string())?;
                        run_args.output_dir = PathBuf::from(val);
                    }
                    "--metrics" => {
                        run_args.write_metrics = true;
                    }
                    "--verbose" | "-v" => {
                        run_args.verbose = true;
                    }
                    _ if arg.starts_with("--output-dir=") => {
                        run_args.output_dir = PathBuf::from(&arg["--output-dir=".len()..]);
                    }
                    _ if arg.starts_with('-') => {
                        return Err(format!("Unknown option: {}", arg));
                    }
                    _ => {
                        return Err(format!("Unexpected argument: {}", arg));
                    }
                }
            }

            Ok(Command::Run(run_args))
        }
        other => Err(format!("Unknown subcommand: {}", other)),
    }
}

// =============================================================================
// Simulation logic
// =============================================================================

/// Map risk profile string to RiskProfile enum.
fn parse_risk_profile(s: &str) -> RiskProfile {
    match s.to_lowercase().as_str() {
        "conservative" | "cons" | "c" => RiskProfile::Conservative,
        "aggressive" | "agg" | "a" => RiskProfile::Aggressive,
        _ => RiskProfile::Balanced,
    }
}

/// Run result for a single seed.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub seed: u64,
    pub final_pnl: f64,
    pub max_drawdown: f64,
    pub kill_switch: KillSwitchInfo,
    pub steps_executed: u64,
    pub duration_ms: u128,
    pub checksum: String,
}

/// Run simulation for a single seed using SimEnv.
pub fn run_single_seed(spec: &ScenarioSpec, seed: u64, verbose: bool) -> (RunResult, RunSummary) {
    run_single_seed_with_ablations(spec, seed, verbose, &AblationSet::new())
}

/// Run simulation for a single seed using SimEnv with ablations.
pub fn run_single_seed_with_ablations(
    spec: &ScenarioSpec,
    seed: u64,
    verbose: bool,
    ablations: &AblationSet,
) -> (RunResult, RunSummary) {
    let start = Instant::now();

    // Create config from risk profile
    let risk_profile = parse_risk_profile(&spec.initial_state.risk_profile);
    let mut config = Config::for_profile(risk_profile);
    config.initial_q_tao = spec.initial_state.init_q_tao;

    // Apply market model parameters
    if let Some(ref synth) = spec.market_model.synthetic {
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
    env_config.ablations = ablations.clone();

    // Apply domain randomization based on market model
    if let Some(ref synth) = spec.market_model.synthetic {
        match synth.process {
            SyntheticProcess::JumpDiffusionStub => {
                env_config.apply_domain_rand = true;
                env_config.domain_rand.vol_ref_range = (0.015, 0.025);
            }
            SyntheticProcess::Gbm => {
                env_config.apply_domain_rand = false;
            }
        }
    }

    // Create environment
    let mut env = SimEnv::new(config, env_config);
    let _obs = env.reset(Some(seed));

    // Run episode
    let identity_action = PolicyAction::identity(env.num_venues(), "sim_eval");
    let mut dd_tracker = DrawdownTracker::new();
    let mut kill_switch = KillSwitchInfo::default();
    let mut steps_executed = 0u64;

    for step in 0..spec.horizon.steps {
        let result = env.step(&identity_action);
        steps_executed = step + 1;

        dd_tracker.update(result.info.pnl_total);

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

    // Create summary (includes checksum computation)
    let build_info = BuildInfo::for_test(); // Placeholder for checksum computation
    let summary = RunSummary::with_ablations(
        spec,
        seed,
        build_info,
        final_pnl,
        max_drawdown,
        kill_switch.clone(),
        ablations,
    );

    let run_result = RunResult {
        seed,
        final_pnl,
        max_drawdown,
        kill_switch,
        steps_executed,
        duration_ms,
        checksum: summary.determinism.checksum.clone(),
    };

    (run_result, summary)
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
        ExpectKillSwitch::Allowed => {}
    }
    Ok(())
}

/// Verify run_summary has all required fields (schema completeness gate).
fn verify_schema_completeness(summary: &RunSummary) -> Result<(), String> {
    // Required top-level fields
    if summary.scenario_id.is_empty() {
        return Err("Missing required field: scenario_id".to_string());
    }
    if summary.scenario_version == 0 {
        return Err("Invalid field: scenario_version must be >= 1".to_string());
    }
    // seed can be any value

    // build_info
    if summary.build_info.git_sha.is_empty() {
        return Err("Missing required field: build_info.git_sha".to_string());
    }

    // config
    if summary.config.risk_profile.is_empty() {
        return Err("Missing required field: config.risk_profile".to_string());
    }
    if summary.config.steps == 0 {
        return Err("Invalid field: config.steps must be > 0".to_string());
    }
    if summary.config.dt_seconds <= 0.0 {
        return Err("Invalid field: config.dt_seconds must be > 0".to_string());
    }

    // results - pnl values can be any number (including negative)
    // kill_switch fields are validated by struct defaults

    // determinism
    if summary.determinism.checksum.is_empty() {
        return Err("Missing required field: determinism.checksum".to_string());
    }
    if summary.determinism.checksum.len() != 64 {
        return Err("Invalid field: determinism.checksum must be 64 hex chars".to_string());
    }

    Ok(())
}

// =============================================================================
// Run single scenario subcommand
// =============================================================================

fn cmd_run(args: RunArgs) -> i32 {
    let spec = match ScenarioSpec::from_yaml_file(&args.scenario_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load scenario: {}", e);
            return 1;
        }
    };

    let build_info = BuildInfo::capture();

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

    match spec.engine {
        Engine::RlSimEnv | Engine::CoreSim => {}
        Engine::ReplayStub => {
            eprintln!("Warning: replay_stub engine not implemented, using rl_sim_env fallback");
        }
    }

    if spec.market_model.model_type == MarketModelType::HistoricalStub {
        eprintln!(
            "Warning: historical_stub market model not implemented, using synthetic fallback"
        );
    }

    let seeds = spec.expand_seeds();
    let mut all_results = Vec::with_capacity(seeds.len());
    let mut invariant_failures = Vec::new();

    for (k, seed) in &seeds {
        if args.verbose {
            eprintln!("Running seed {} ({}/{})", seed, k + 1, spec.rng.num_seeds);
        }

        let (result, summary) = run_single_seed(&spec, *seed, args.verbose);

        if let Err(e) = check_invariants(&spec, &result) {
            invariant_failures.push(e.clone());
            eprintln!("  {}", e);
        }

        // Create output directory and write files with real build info
        let output_dir = match create_output_dir(
            &args.output_dir,
            &spec.scenario_id,
            &build_info.git_sha,
            *seed,
        ) {
            Ok(dir) => dir,
            Err(e) => {
                eprintln!("Failed to create output directory: {}", e);
                return 1;
            }
        };

        // Write run_summary.json with real build info
        let summary_with_build = RunSummary::new(
            &spec,
            *seed,
            build_info.clone(),
            result.final_pnl,
            result.max_drawdown,
            result.kill_switch.clone(),
        );

        if let Err(e) = summary_with_build.write_to_file(output_dir.join("run_summary.json")) {
            eprintln!("Failed to write run_summary.json: {}", e);
        }

        if let Err(e) = write_config_resolved(output_dir.join("config_resolved.json"), &spec) {
            eprintln!("Failed to write config_resolved.json: {}", e);
        }

        if let Err(e) = write_build_info(output_dir.join("build_info.json"), &build_info) {
            eprintln!("Failed to write build_info.json: {}", e);
        }

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
        return 1;
    }

    println!();
    println!("Output written to: {}/", args.output_dir.display());
    0
}

// =============================================================================
// Suite subcommand with CI gates
// =============================================================================

/// CI gate failure types
#[derive(Debug, Clone)]
enum GateFailure {
    SchemaCompleteness {
        scenario_id: String,
        seed: u64,
        message: String,
    },
    Determinism {
        scenario_id: String,
        seed: u64,
        checksums: Vec<String>,
    },
    Invariant {
        scenario_id: String,
        seed: u64,
        message: String,
    },
    ScenarioLoad {
        path: String,
        message: String,
    },
}

impl std::fmt::Display for GateFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GateFailure::SchemaCompleteness {
                scenario_id,
                seed,
                message,
            } => {
                write!(f, "[SCHEMA] {}@{}: {}", scenario_id, seed, message)
            }
            GateFailure::Determinism {
                scenario_id,
                seed,
                checksums,
            } => {
                write!(
                    f,
                    "[DETERMINISM] {}@{}: checksums differ across {} runs: {:?}",
                    scenario_id,
                    seed,
                    checksums.len(),
                    checksums.iter().map(|c| &c[..16]).collect::<Vec<_>>()
                )
            }
            GateFailure::Invariant {
                scenario_id,
                seed,
                message,
            } => {
                write!(f, "[INVARIANT] {}@{}: {}", scenario_id, seed, message)
            }
            GateFailure::ScenarioLoad { path, message } => {
                write!(f, "[LOAD] {}: {}", path, message)
            }
        }
    }
}

/// Suite run statistics
struct SuiteStats {
    scenarios_total: usize,
    scenarios_passed: usize,
    seeds_total: usize,
    seeds_passed: usize,
    runs_total: usize,
    gate_failures: Vec<GateFailure>,
}

fn cmd_suite(args: SuiteArgs) -> i32 {
    // Validate and create ablation set
    let ablations = match AblationSet::from_ids(&args.ablations) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {}", e);
            return 2;
        }
    };

    let suite = match SuiteSpec::from_yaml_file(&args.suite_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to load suite: {}", e);
            return 1;
        }
    };

    let build_info = BuildInfo::capture();

    // Compute effective output directory with ablation suffix
    let effective_out_dir = format!("{}{}", suite.out_dir, ablations.dir_suffix());

    println!(
        "sim_eval suite | suite={} version={} repeat_runs={} scenarios={}",
        suite.suite_id,
        suite.suite_version,
        suite.repeat_runs,
        suite.scenarios.len()
    );
    println!(
        "               | git_sha={} dirty={}",
        &build_info.git_sha[..8.min(build_info.git_sha.len())],
        build_info.dirty
    );
    println!("               | output_dir={}", effective_out_dir);
    if !ablations.is_empty() {
        println!("               | ablations={}", ablations);
    }
    println!();

    let out_dir = Path::new(&effective_out_dir);
    let mut stats = SuiteStats {
        scenarios_total: suite.scenarios.len(),
        scenarios_passed: 0,
        seeds_total: 0,
        seeds_passed: 0,
        runs_total: 0,
        gate_failures: Vec::new(),
    };

    // Track artifact paths relative to output_root for evidence pack
    let mut artifact_paths: Vec<PathBuf> = Vec::new();

    for scenario_ref in &suite.scenarios {
        println!("━━━ Scenario: {} ━━━", scenario_ref.path);

        let spec = match ScenarioSpec::from_yaml_file(&scenario_ref.path) {
            Ok(s) => s,
            Err(e) => {
                stats.gate_failures.push(GateFailure::ScenarioLoad {
                    path: scenario_ref.path.clone(),
                    message: e.to_string(),
                });
                println!("  FAILED to load: {}", e);
                println!();
                continue;
            }
        };

        // Warn about unsupported engines/models
        if spec.engine == Engine::ReplayStub {
            println!("  Warning: replay_stub engine not implemented, using rl_sim_env fallback");
        }
        if spec.market_model.model_type == MarketModelType::HistoricalStub {
            println!(
                "  Warning: historical_stub market model not implemented, using synthetic fallback"
            );
        }

        let seeds = spec.expand_seeds();
        stats.seeds_total += seeds.len();

        let mut scenario_passed = true;

        for (_k, seed) in &seeds {
            // Collect checksums across repeat_runs
            let mut checksums: Vec<String> = Vec::with_capacity(suite.repeat_runs as usize);
            let mut last_result: Option<RunResult> = None;
            let mut last_summary: Option<RunSummary> = None;

            for run_idx in 0..suite.repeat_runs {
                if args.verbose {
                    eprintln!(
                        "  Running seed {} run {}/{}",
                        seed,
                        run_idx + 1,
                        suite.repeat_runs
                    );
                }

                let (result, summary) =
                    run_single_seed_with_ablations(&spec, *seed, false, &ablations);
                checksums.push(result.checksum.clone());

                stats.runs_total += 1;

                // Store last result for invariant checking and output writing
                last_result = Some(result);
                last_summary = Some(summary);
            }

            let result = last_result.unwrap();
            let summary = last_summary.unwrap();

            // Gate 1: Schema completeness
            if let Err(e) = verify_schema_completeness(&summary) {
                stats.gate_failures.push(GateFailure::SchemaCompleteness {
                    scenario_id: spec.scenario_id.clone(),
                    seed: *seed,
                    message: e,
                });
                scenario_passed = false;
            }

            // Gate 2: Determinism (if repeat_runs > 1)
            if suite.repeat_runs > 1 {
                let first_checksum = &checksums[0];
                let all_same = checksums.iter().all(|c| c == first_checksum);
                if !all_same {
                    stats.gate_failures.push(GateFailure::Determinism {
                        scenario_id: spec.scenario_id.clone(),
                        seed: *seed,
                        checksums: checksums.clone(),
                    });
                    scenario_passed = false;
                }
            }

            // Gate 3: Invariants
            if let Err(e) = check_invariants(&spec, &result) {
                stats.gate_failures.push(GateFailure::Invariant {
                    scenario_id: spec.scenario_id.clone(),
                    seed: *seed,
                    message: e,
                });
                scenario_passed = false;
            }

            // Write outputs (once per seed, not per repeat)
            // Note: ablation info is already encoded in effective_out_dir suffix
            let output_dir =
                match create_output_dir(out_dir, &spec.scenario_id, &build_info.git_sha, *seed) {
                    Ok(dir) => dir,
                    Err(e) => {
                        eprintln!("  Failed to create output directory: {}", e);
                        continue;
                    }
                };

            let summary_with_build = RunSummary::with_ablations(
                &spec,
                *seed,
                build_info.clone(),
                result.final_pnl,
                result.max_drawdown,
                result.kill_switch.clone(),
                &ablations,
            );

            // Write output files and track relative paths for evidence pack
            let run_summary_path = output_dir.join("run_summary.json");
            let config_resolved_path = output_dir.join("config_resolved.json");
            let build_info_path = output_dir.join("build_info.json");

            let _ = summary_with_build.write_to_file(&run_summary_path);
            let _ = write_config_resolved_with_ablations(&config_resolved_path, &spec, &ablations);
            let _ = write_build_info(&build_info_path, &build_info);

            // Track artifact paths relative to out_dir for evidence pack
            if let Ok(rel_path) = run_summary_path.strip_prefix(out_dir) {
                artifact_paths.push(rel_path.to_path_buf());
            }
            if let Ok(rel_path) = config_resolved_path.strip_prefix(out_dir) {
                artifact_paths.push(rel_path.to_path_buf());
            }
            if let Ok(rel_path) = build_info_path.strip_prefix(out_dir) {
                artifact_paths.push(rel_path.to_path_buf());
            }

            // Print seed status
            let determinism_status = if suite.repeat_runs > 1 {
                let first = &checksums[0];
                if checksums.iter().all(|c| c == first) {
                    "DETERM"
                } else {
                    "DIFFER"
                }
            } else {
                "N/A"
            };

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

            let seed_passed = !stats.gate_failures.iter().any(|f| match f {
                GateFailure::SchemaCompleteness { seed: s, .. }
                | GateFailure::Determinism { seed: s, .. }
                | GateFailure::Invariant { seed: s, .. } => *s == *seed,
                _ => false,
            });

            if seed_passed {
                stats.seeds_passed += 1;
            }

            let status_icon = if seed_passed { "✓" } else { "✗" };

            println!(
                "  {} seed={:<5} pnl={:>10.2} status={:<10} determ={:<6} checksum={}",
                status_icon,
                seed,
                result.final_pnl,
                kill_str,
                determinism_status,
                &result.checksum[..16]
            );
        }

        if scenario_passed {
            stats.scenarios_passed += 1;
            println!("  → PASS");
        } else {
            println!("  → FAIL");
        }
        println!();
    }

    // Print final summary
    println!("═══════════════════════════════════════════════════════════════");
    println!("SUITE SUMMARY: {}", suite.suite_id);
    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "Scenarios: {}/{} passed",
        stats.scenarios_passed, stats.scenarios_total
    );
    println!(
        "Seeds:     {}/{} passed",
        stats.seeds_passed, stats.seeds_total
    );
    println!(
        "Runs:      {} total ({}x repeat)",
        stats.runs_total, suite.repeat_runs
    );
    println!();

    // Generate Evidence Pack v1 (per docs/EVIDENCE_PACK.md)
    // Evidence pack generation failure is fatal (integrity boundary)
    println!();
    println!("Generating Evidence Pack...");
    if let Err(e) = write_evidence_pack(out_dir, &args.suite_path, &artifact_paths) {
        eprintln!("✗ EVIDENCE PACK GENERATION FAILED: {}", e);
        eprintln!();
        eprintln!("Output written to: {}/", effective_out_dir);
        return 1;
    }
    println!(
        "✓ Evidence Pack written to: {}/evidence_pack/",
        effective_out_dir
    );

    if stats.gate_failures.is_empty() {
        println!();
        println!("✓ ALL GATES PASSED");
        println!();
        println!("Output written to: {}/", effective_out_dir);
        0
    } else {
        println!();
        println!("✗ {} GATE FAILURES:", stats.gate_failures.len());
        for failure in &stats.gate_failures {
            println!("  - {}", failure);
        }
        println!();
        println!("Output written to: {}/", effective_out_dir);
        1
    }
}

// =============================================================================
// Summarize subcommand
// =============================================================================

fn cmd_summarize(args: SummarizeArgs) -> i32 {
    if !args.runs_dir.exists() {
        eprintln!("Error: Directory not found: {}", args.runs_dir.display());
        return 1;
    }

    if !args.runs_dir.is_dir() {
        eprintln!("Error: Not a directory: {}", args.runs_dir.display());
        return 1;
    }

    let stdout = std::io::stdout();
    let handle = stdout.lock();

    match summarize_with_format(&args.runs_dir, handle, args.format) {
        Ok(SummarizeResult::Success(count)) => {
            eprintln!("\nFound {} run(s)", count);
            0
        }
        Ok(SummarizeResult::NoFilesFound) => {
            eprintln!(
                "Error: No run_summary.json files found in {}",
                args.runs_dir.display()
            );
            1
        }
        Ok(SummarizeResult::NoParseable) => {
            eprintln!("Error: Found run_summary.json files but none could be parsed");
            1
        }
        Err(e) => {
            eprintln!("Error reading directory: {}", e);
            1
        }
    }
}

// =============================================================================
// Report subcommand
// =============================================================================

fn cmd_report(args: ReportArgsLocal) -> i32 {
    if !args.baseline_dir.exists() {
        eprintln!(
            "Error: Baseline directory not found: {}",
            args.baseline_dir.display()
        );
        return 1;
    }

    for (name, path) in &args.variants {
        if !path.exists() {
            eprintln!(
                "Error: Variant '{}' directory not found: {}",
                name,
                path.display()
            );
            return 1;
        }
    }

    let report_args = ReportArgs {
        baseline_dir: args.baseline_dir,
        variants: args.variants,
        out_md: args.out_md,
        out_json: args.out_json,
        gate_max_regression_usd: args.gate_max_regression_usd,
        gate_max_regression_pct: args.gate_max_regression_pct,
    };

    match run_report(report_args) {
        ReportResult::Success {
            variants,
            gates_passed,
        } => {
            eprintln!("\nAnalyzed {} variant(s)", variants);
            if gates_passed {
                0
            } else {
                1
            }
        }
        ReportResult::NoBaselineRuns => {
            eprintln!("Error: No run_summary.json files found in baseline directory");
            1
        }
        ReportResult::Error(e) => {
            eprintln!("Error: {}", e);
            1
        }
    }
}

// =============================================================================
// Verify subcommands
// =============================================================================

fn cmd_verify_evidence_pack(args: VerifyEvidencePackArgs) -> i32 {
    match verify_evidence_pack_dir(&args.output_root) {
        Ok(report) => {
            println!(
                "OK: verified {} file(s) in 1 evidence pack",
                report.files_verified
            );
            0
        }
        Err(e) => {
            // Get the root cause for concise error message
            let root_cause = e.root_cause();
            eprintln!("ERROR: verification failed: {}", root_cause);
            3
        }
    }
}

fn cmd_verify_evidence_tree(args: VerifyEvidenceTreeArgs) -> i32 {
    match verify_evidence_pack_tree(&args.root) {
        Ok(report) => {
            println!(
                "OK: verified {} file(s) in {} evidence pack(s)",
                report.files_verified, report.packs_verified
            );
            0
        }
        Err(e) => {
            // Get the root cause for concise error message
            let root_cause = e.root_cause();
            eprintln!("ERROR: verification failed: {}", root_cause);
            3
        }
    }
}

// =============================================================================
// Main
// =============================================================================

fn cmd_ablations() -> i32 {
    print_ablations();
    0
}

fn main() {
    let cmd = match parse_args() {
        Ok(cmd) => cmd,
        Err(e) => {
            eprintln!("Error: {}\n\n{}", e, usage());
            std::process::exit(2);
        }
    };

    let exit_code = match cmd {
        Command::Run(args) => cmd_run(args),
        Command::Suite(args) => cmd_suite(args),
        Command::Summarize(args) => cmd_summarize(args),
        Command::Report(args) => cmd_report(args),
        Command::Ablations => cmd_ablations(),
        Command::VerifyEvidencePack(args) => cmd_verify_evidence_pack(args),
        Command::VerifyEvidenceTree(args) => cmd_verify_evidence_tree(args),
    };

    std::process::exit(exit_code);
}
