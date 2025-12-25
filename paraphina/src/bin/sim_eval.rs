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

use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use paraphina::config::{resolve_effective_profile, Config, RiskProfile};
use paraphina::metrics::DrawdownTracker;
use paraphina::rl::sim_env::SimEnvConfig;
use paraphina::rl::{PolicyAction, SimEnv};
use paraphina::sim_eval::{
    create_output_dir, print_ablations, run_report, summarize_with_format,
    verify_evidence_pack_dir, verify_evidence_pack_tree, with_env_overrides, write_build_info,
    write_config_resolved, write_config_resolved_with_ablations, write_evidence_pack, AblationSet,
    BuildInfo, Engine, ExpectKillSwitch, InlineScenario, KillSwitchInfo, MarketModelType,
    OutputFormat, ReportArgs, ReportResult, RunSummary, ScenarioRef, ScenarioSpec, SuiteSpec,
    SummarizeResult, SyntheticProcess,
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
    /// CLI-provided profile override (highest precedence).
    profile: Option<RiskProfile>,
}

#[derive(Debug)]
struct SuiteArgs {
    suite_path: PathBuf,
    /// Base output directory for suite runs (overrides suite's out_dir).
    output_dir: PathBuf,
    verbose: bool,
    ablations: Vec<String>,
    /// CLI-provided profile override (highest precedence).
    profile: Option<RiskProfile>,
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
  --profile NAME     Override risk profile (balanced|conservative|aggressive)

SUITE OPTIONS:
  --output-dir DIR       Base output directory (default: runs/)
                         Output goes exactly to DIR (no suffix unless ablations active)
  --verbose              Print verbose output
  --ablation <ID>        Enable an ablation (can be specified multiple times)
  --profile NAME         Override risk profile (balanced|conservative|aggressive)

  Suites support inline env_overrides per scenario (see SUITE YAML SUPPORT).

SUITE YAML SUPPORT:
  Inline scenarios with env_overrides allow per-scenario environment configuration:
    - id: my_scenario
      seed: 42
      env_overrides: { PARAPHINA_RISK_PROFILE: aggressive }
  env_overrides can be YAML mappings or lists of KEY=VALUE strings.

PROFILE PRECEDENCE:
  1) --profile CLI argument (highest)
  2) PARAPHINA_RISK_PROFILE env var
  3) scenario file risk_profile field
  4) default (Balanced)

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
  sim_eval suite scenarios/suites/ci_smoke_v1.yaml --output-dir ./my_runs
  sim_eval suite scenarios/suites/research_v1.yaml --ablation disable_vol_floor
  sim_eval summarize runs/
  sim_eval summarize runs/ --format md
  sim_eval report --baseline runs/baseline --variant ablation1=runs/ablation1 --out-md report.md --out-json report.json
  sim_eval ablations
  sim_eval verify-evidence-pack runs/demo_step_7_1
  sim_eval verify-evidence-tree runs/demo_step_7_1
"
}

fn verify_evidence_pack_usage() -> &'static str {
    "\
USAGE:
  sim_eval verify-evidence-pack <OUTPUT_ROOT>

DESCRIPTION:
  Verify a single evidence pack at <OUTPUT_ROOT>/evidence_pack/.
  Checks SHA256SUMS integrity and validates all referenced artifacts.

ARGUMENTS:
  <OUTPUT_ROOT>    Directory containing evidence_pack/ subdirectory

OPTIONS:
  --help           Show this help

EXIT CODES:
  0  Verification succeeded
  2  Usage error (missing or extra arguments)
  3  Verification failed

EXAMPLE:
  sim_eval verify-evidence-pack runs/demo_step_7_1
"
}

fn verify_evidence_tree_usage() -> &'static str {
    "\
USAGE:
  sim_eval verify-evidence-tree <ROOT>

DESCRIPTION:
  Recursively find and verify all evidence packs under <ROOT>.
  Searches for evidence_pack/SHA256SUMS files and verifies each pack.

ARGUMENTS:
  <ROOT>           Root directory to search for evidence packs

OPTIONS:
  --help           Show this help

EXIT CODES:
  0  All packs verified successfully
  2  Usage error (missing or extra arguments)
  3  Verification failed (any pack)

EXAMPLE:
  sim_eval verify-evidence-tree runs/demo_step_7_1
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
                profile: None,
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
                    "--profile" => {
                        let val = args
                            .next()
                            .ok_or_else(|| "Missing value for --profile".to_string())?;
                        run_args.profile = Some(
                            RiskProfile::parse(&val).ok_or_else(|| {
                                format!("Invalid --profile '{}'. Expected: balanced|conservative|aggressive", val)
                            })?,
                        );
                    }
                    _ if arg.starts_with("--output-dir=") => {
                        run_args.output_dir = PathBuf::from(&arg["--output-dir=".len()..]);
                    }
                    _ if arg.starts_with("--profile=") => {
                        let val = &arg["--profile=".len()..];
                        run_args.profile = Some(
                            RiskProfile::parse(val).ok_or_else(|| {
                                format!("Invalid --profile '{}'. Expected: balanced|conservative|aggressive", val)
                            })?,
                        );
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
                output_dir: PathBuf::from("runs"),
                verbose: false,
                ablations: Vec::new(),
                profile: None,
            };
            let mut suite_set = false;

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
                        suite_args.output_dir = PathBuf::from(val);
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
                    "--profile" => {
                        let val = args
                            .next()
                            .ok_or_else(|| "Missing value for --profile".to_string())?;
                        suite_args.profile = Some(
                            RiskProfile::parse(&val).ok_or_else(|| {
                                format!("Invalid --profile '{}'. Expected: balanced|conservative|aggressive", val)
                            })?,
                        );
                    }
                    _ if arg.starts_with("--output-dir=") => {
                        suite_args.output_dir = PathBuf::from(&arg["--output-dir=".len()..]);
                    }
                    _ if arg.starts_with("--ablation=") => {
                        let val = arg["--ablation=".len()..].to_string();
                        suite_args.ablations.push(val);
                    }
                    _ if arg.starts_with("--profile=") => {
                        let val = &arg["--profile=".len()..];
                        suite_args.profile = Some(
                            RiskProfile::parse(val).ok_or_else(|| {
                                format!("Invalid --profile '{}'. Expected: balanced|conservative|aggressive", val)
                            })?,
                        );
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
                        println!("{}", verify_evidence_pack_usage());
                        std::process::exit(0);
                    }
                    _ if arg.starts_with('-') => {
                        eprintln!(
                            "Unknown option: {}\n\n{}",
                            arg,
                            verify_evidence_pack_usage()
                        );
                        std::process::exit(2);
                    }
                    _ => {
                        if path_set {
                            eprintln!(
                                "unexpected argument: {}\n\n{}",
                                arg,
                                verify_evidence_pack_usage()
                            );
                            std::process::exit(2);
                        }
                        verify_args.output_root = PathBuf::from(arg);
                        path_set = true;
                    }
                }
            }

            if !path_set {
                eprintln!(
                    "Missing required argument: <OUTPUT_ROOT>\n\n{}",
                    verify_evidence_pack_usage()
                );
                std::process::exit(2);
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
                        println!("{}", verify_evidence_tree_usage());
                        std::process::exit(0);
                    }
                    _ if arg.starts_with('-') => {
                        eprintln!(
                            "Unknown option: {}\n\n{}",
                            arg,
                            verify_evidence_tree_usage()
                        );
                        std::process::exit(2);
                    }
                    _ => {
                        if path_set {
                            eprintln!(
                                "unexpected argument: {}\n\n{}",
                                arg,
                                verify_evidence_tree_usage()
                            );
                            std::process::exit(2);
                        }
                        verify_args.root = PathBuf::from(arg);
                        path_set = true;
                    }
                }
            }

            if !path_set {
                eprintln!(
                    "Missing required argument: <ROOT>\n\n{}",
                    verify_evidence_tree_usage()
                );
                std::process::exit(2);
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
                profile: None,
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
                    "--profile" => {
                        let val = args
                            .next()
                            .ok_or_else(|| "Missing value for --profile".to_string())?;
                        run_args.profile = Some(
                            RiskProfile::parse(&val).ok_or_else(|| {
                                format!("Invalid --profile '{}'. Expected: balanced|conservative|aggressive", val)
                            })?,
                        );
                    }
                    _ if arg.starts_with("--output-dir=") => {
                        run_args.output_dir = PathBuf::from(&arg["--output-dir=".len()..]);
                    }
                    _ if arg.starts_with("--profile=") => {
                        let val = &arg["--profile=".len()..];
                        run_args.profile = Some(
                            RiskProfile::parse(val).ok_or_else(|| {
                                format!("Invalid --profile '{}'. Expected: balanced|conservative|aggressive", val)
                            })?,
                        );
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
    run_single_seed_with_profile(spec, seed, verbose, &AblationSet::new(), None)
}

/// Run simulation for a single seed using SimEnv with ablations.
pub fn run_single_seed_with_ablations(
    spec: &ScenarioSpec,
    seed: u64,
    verbose: bool,
    ablations: &AblationSet,
) -> (RunResult, RunSummary) {
    run_single_seed_with_profile(spec, seed, verbose, ablations, None)
}

/// Run simulation for a single seed using SimEnv with ablations and optional profile override.
///
/// Profile precedence: cli_profile > env var > scenario > default
pub fn run_single_seed_with_profile(
    spec: &ScenarioSpec,
    seed: u64,
    verbose: bool,
    ablations: &AblationSet,
    cli_profile: Option<RiskProfile>,
) -> (RunResult, RunSummary) {
    let start = Instant::now();

    // Resolve profile with proper precedence: CLI > env > scenario > default
    let effective = resolve_effective_profile(cli_profile, Some(&spec.initial_state.risk_profile));
    let mut config = Config::for_profile(effective.profile);
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

    // Resolve profile with proper precedence: CLI > env > scenario > default
    let effective = resolve_effective_profile(args.profile, Some(&spec.initial_state.risk_profile));

    // Explicit startup log line (required by spec)
    effective.log_startup();

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
        "         | risk_profile={} (source={}) init_q_tao={:.2} dt_seconds={:.3}",
        effective.profile.as_str(),
        effective.source.as_str(),
        spec.initial_state.init_q_tao,
        spec.horizon.dt_seconds
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

        let (result, summary) = run_single_seed_with_profile(
            &spec,
            *seed,
            args.verbose,
            &AblationSet::new(),
            args.profile,
        );

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
    SubprocessFailed {
        scenario_id: String,
        exit_code: Option<i32>,
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
            GateFailure::SubprocessFailed {
                scenario_id,
                exit_code,
                message,
            } => {
                let code_str = exit_code
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "signal".to_string());
                write!(
                    f,
                    "[SUBPROCESS] {}: exit={} - {}",
                    scenario_id, code_str, message
                )
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

/// Result of running an inline scenario in-process.
struct InlineScenarioResult {
    output_dir: PathBuf,
    success: bool,
    error_message: Option<String>,
    run_result: Option<RunResult>,
}

/// Create a minimal ScenarioSpec for inline scenarios.
///
/// The inline scenario has limited fields, so we create a default spec
/// and let env_overrides control the actual configuration.
fn create_inline_scenario_spec(inline: &InlineScenario, steps: u64) -> ScenarioSpec {
    use paraphina::sim_eval::{
        Horizon, InitialState, Invariants, MarketModel, MarketModelType, MicrostructureModel,
        PnlLinearityCheck, Rng, SyntheticConfig, SyntheticParams,
    };

    // Parse profile from string (env_overrides may override this)
    let risk_profile = RiskProfile::parse(&inline.profile)
        .unwrap_or(RiskProfile::Balanced)
        .as_str()
        .to_string();

    ScenarioSpec {
        scenario_id: inline.id.clone(),
        scenario_version: 1,
        engine: Engine::RlSimEnv,
        horizon: Horizon {
            steps,
            dt_seconds: 0.1,
        },
        rng: Rng {
            base_seed: inline.seed,
            num_seeds: 1,
        },
        initial_state: InitialState {
            risk_profile,
            init_q_tao: 0.0,
        },
        market_model: MarketModel {
            model_type: MarketModelType::Synthetic,
            synthetic: Some(SyntheticConfig {
                process: SyntheticProcess::Gbm,
                params: SyntheticParams {
                    vol: 0.20,
                    drift: 0.0,
                    jump_intensity: 0.0,
                    jump_sigma: 0.0,
                },
            }),
            historical_stub: None,
        },
        microstructure_model: MicrostructureModel {
            fees_bps_maker: -0.25,
            fees_bps_taker: 0.75,
            latency_ms: 1.0,
        },
        invariants: Invariants {
            expect_kill_switch: ExpectKillSwitch::Allowed,
            pnl_linearity_check: PnlLinearityCheck::Disabled,
        },
    }
}

/// Run an inline scenario in-process with scoped environment overrides.
///
/// Uses with_env_overrides to temporarily set environment variables,
/// runs the simulation, then restores the previous environment.
#[allow(clippy::too_many_arguments)]
fn run_inline_scenario(
    inline: &InlineScenario,
    scenario_index: usize,
    suite_env: &BTreeMap<String, String>,
    out_dir: &Path,
    repeat_runs: u32,
    verbose: bool,
    build_info: &BuildInfo,
    ablations: &AblationSet,
    cli_profile: Option<RiskProfile>,
) -> InlineScenarioResult {
    // Merge environment: suite-level + scenario-level (scenario wins)
    let merged_env = inline.merge_env(suite_env);

    // Create stable output directory: <out_dir>/<index>_<scenario_id>/
    let scenario_out_dir = out_dir.join(format!("{:04}_{}", scenario_index, &inline.id));

    if let Err(e) = std::fs::create_dir_all(&scenario_out_dir) {
        return InlineScenarioResult {
            output_dir: scenario_out_dir,
            success: false,
            error_message: Some(format!("Failed to create output directory: {}", e)),
            run_result: None,
        };
    }

    if verbose {
        eprintln!(
            "    Running inline scenario: {} (seed={}, {} env overrides)",
            inline.id,
            inline.seed,
            merged_env.len()
        );
        for (k, v) in &merged_env {
            eprintln!("      {}={}", k, v);
        }
    }

    // Run simulation with scoped environment overrides
    let steps = 100; // Smoke-level steps for inline scenarios
    let spec = create_inline_scenario_spec(inline, steps);

    // Run repeat_runs times to check determinism
    let run_result: Result<(RunResult, RunSummary, Vec<String>), String> =
        with_env_overrides(&merged_env, || {
            let mut checksums: Vec<String> = Vec::with_capacity(repeat_runs as usize);
            let mut last_result: Option<RunResult> = None;
            let mut last_summary: Option<RunSummary> = None;

            for run_idx in 0..repeat_runs {
                if verbose && run_idx > 0 {
                    eprintln!("      Repeat run {}/{}", run_idx + 1, repeat_runs);
                }

                let (result, summary) =
                    run_single_seed_with_profile(&spec, inline.seed, false, ablations, cli_profile);
                checksums.push(result.checksum.clone());
                last_result = Some(result);
                last_summary = Some(summary);
            }

            Ok((last_result.unwrap(), last_summary.unwrap(), checksums))
        });

    match run_result {
        Ok((result, _summary, checksums)) => {
            // Check determinism if repeat_runs > 1
            if repeat_runs > 1 {
                let first = &checksums[0];
                if !checksums.iter().all(|c| c == first) {
                    return InlineScenarioResult {
                        output_dir: scenario_out_dir,
                        success: false,
                        error_message: Some(format!(
                            "Determinism check failed: checksums differ across {} runs",
                            repeat_runs
                        )),
                        run_result: Some(result),
                    };
                }
            }

            // Write output files
            let summary_with_build = RunSummary::with_ablations(
                &spec,
                inline.seed,
                build_info.clone(),
                result.final_pnl,
                result.max_drawdown,
                result.kill_switch.clone(),
                ablations,
            );

            let run_summary_path = scenario_out_dir.join("run_summary.json");
            let config_resolved_path = scenario_out_dir.join("config_resolved.json");
            let build_info_path = scenario_out_dir.join("build_info.json");

            if let Err(e) = summary_with_build.write_to_file(&run_summary_path) {
                return InlineScenarioResult {
                    output_dir: scenario_out_dir,
                    success: false,
                    error_message: Some(format!("Failed to write run_summary.json: {}", e)),
                    run_result: Some(result),
                };
            }

            if let Err(e) =
                write_config_resolved_with_ablations(&config_resolved_path, &spec, ablations)
            {
                return InlineScenarioResult {
                    output_dir: scenario_out_dir,
                    success: false,
                    error_message: Some(format!("Failed to write config_resolved.json: {}", e)),
                    run_result: Some(result),
                };
            }

            if let Err(e) = write_build_info(&build_info_path, build_info) {
                return InlineScenarioResult {
                    output_dir: scenario_out_dir,
                    success: false,
                    error_message: Some(format!("Failed to write build_info.json: {}", e)),
                    run_result: Some(result),
                };
            }

            // Write evidence pack for this scenario
            let artifact_paths = vec![
                PathBuf::from("run_summary.json"),
                PathBuf::from("config_resolved.json"),
                PathBuf::from("build_info.json"),
            ];

            // Create a synthetic suite path for the evidence pack
            let suite_yaml_content = format!(
                "# Auto-generated for inline scenario\nscenario_id: {}\nseed: {}\nenv_overrides:\n{}",
                inline.id,
                inline.seed,
                merged_env
                    .iter()
                    .map(|(k, v)| format!("  {}: \"{}\"", k, v))
                    .collect::<Vec<_>>()
                    .join("\n")
            );

            let synthetic_suite_path = scenario_out_dir.join("scenario.yaml");
            if let Err(e) = std::fs::write(&synthetic_suite_path, &suite_yaml_content) {
                return InlineScenarioResult {
                    output_dir: scenario_out_dir,
                    success: false,
                    error_message: Some(format!("Failed to write scenario.yaml: {}", e)),
                    run_result: Some(result),
                };
            }

            if let Err(e) =
                write_evidence_pack(&scenario_out_dir, &synthetic_suite_path, &artifact_paths)
            {
                return InlineScenarioResult {
                    output_dir: scenario_out_dir,
                    success: false,
                    error_message: Some(format!("Failed to write evidence pack: {}", e)),
                    run_result: Some(result),
                };
            }

            InlineScenarioResult {
                output_dir: scenario_out_dir,
                success: true,
                error_message: None,
                run_result: Some(result),
            }
        }
        Err(e) => InlineScenarioResult {
            output_dir: scenario_out_dir,
            success: false,
            error_message: Some(e),
            run_result: None,
        },
    }
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

    // Log the CLI/env profile override if present (suite runs multiple scenarios,
    // so we log once at the start if a global override is in effect)
    if args.profile.is_some() {
        let effective = resolve_effective_profile(args.profile, None);
        effective.log_startup();
    } else if let Ok(env_val) = std::env::var("PARAPHINA_RISK_PROFILE") {
        if !env_val.is_empty() {
            eprintln!(
                "effective_risk_profile={} source=env (will override scenario defaults)",
                env_val.to_lowercase()
            );
        }
    }

    let build_info = BuildInfo::capture();

    // Use CLI-provided output_dir as base.
    // Only append ablation suffix if ablations are actually active (not for baseline).
    let base_out_dir = args.output_dir.to_string_lossy();
    let effective_out_dir = format!("{}{}", base_out_dir, ablations.dir_suffix_optional());

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
    if let Some(profile) = args.profile {
        println!(
            "               | profile_override={} (source=cli)",
            profile.as_str()
        );
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

    // Log if suite has inline scenarios
    if suite.has_inline_scenarios() {
        println!("               | mode=subprocess (suite has inline env_overrides)");
        if !suite.env_overrides.is_empty() {
            println!(
                "               | suite_env_overrides={:?}",
                suite.env_overrides.keys().collect::<Vec<_>>()
            );
        }
        println!();
    }

    for (scenario_index, scenario_ref) in suite.scenarios.iter().enumerate() {
        match scenario_ref {
            ScenarioRef::Path { path } => {
                // Path-based scenario: run in-process (original behavior)
                println!("━━━ Scenario: {} ━━━", path);

                let spec = match ScenarioSpec::from_yaml_file(path) {
                    Ok(s) => s,
                    Err(e) => {
                        stats.gate_failures.push(GateFailure::ScenarioLoad {
                            path: path.clone(),
                            message: e.to_string(),
                        });
                        println!("  FAILED to load: {}", e);
                        println!();
                        continue;
                    }
                };

                // Warn about unsupported engines/models
                if spec.engine == Engine::ReplayStub {
                    println!(
                        "  Warning: replay_stub engine not implemented, using rl_sim_env fallback"
                    );
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

                        let (result, summary) = run_single_seed_with_profile(
                            &spec,
                            *seed,
                            false,
                            &ablations,
                            args.profile,
                        );
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
                    let output_dir = match create_output_dir(
                        out_dir,
                        &spec.scenario_id,
                        &build_info.git_sha,
                        *seed,
                    ) {
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
                    let _ = write_config_resolved_with_ablations(
                        &config_resolved_path,
                        &spec,
                        &ablations,
                    );
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

            ScenarioRef::Inline(inline) => {
                // Inline scenario: run in-process with scoped env overrides
                println!(
                    "━━━ Scenario: {} (inline, seed={}) ━━━",
                    inline.id, inline.seed
                );

                stats.seeds_total += 1; // Inline scenarios have a single seed

                let result = run_inline_scenario(
                    inline,
                    scenario_index,
                    &suite.env_overrides,
                    out_dir,
                    suite.repeat_runs,
                    args.verbose,
                    &build_info,
                    &ablations,
                    args.profile,
                );

                stats.runs_total += suite.repeat_runs as usize;

                if result.success {
                    stats.seeds_passed += 1;
                    stats.scenarios_passed += 1;

                    // Track run_summary.json as artifact for evidence pack
                    let run_summary_path = result.output_dir.join("run_summary.json");
                    if let Ok(rel_path) = run_summary_path.strip_prefix(out_dir) {
                        artifact_paths.push(rel_path.to_path_buf());
                    }

                    // Track config_resolved.json
                    let config_resolved_path = result.output_dir.join("config_resolved.json");
                    if let Ok(rel_path) = config_resolved_path.strip_prefix(out_dir) {
                        artifact_paths.push(rel_path.to_path_buf());
                    }

                    // Track build_info.json
                    let build_info_path = result.output_dir.join("build_info.json");
                    if let Ok(rel_path) = build_info_path.strip_prefix(out_dir) {
                        artifact_paths.push(rel_path.to_path_buf());
                    }

                    let pnl_str = result
                        .run_result
                        .as_ref()
                        .map(|r| format!("{:.2}", r.final_pnl))
                        .unwrap_or_else(|| "N/A".to_string());

                    let checksum_str = result
                        .run_result
                        .as_ref()
                        .map(|r| &r.checksum[..16])
                        .unwrap_or("N/A");

                    println!(
                        "  ✓ seed={:<5} pnl={:>10} status=OK checksum={}",
                        inline.seed, pnl_str, checksum_str
                    );
                    println!("  → PASS");
                } else {
                    let error_msg = result
                        .error_message
                        .clone()
                        .unwrap_or_else(|| "unknown error".to_string());
                    stats.gate_failures.push(GateFailure::SubprocessFailed {
                        scenario_id: inline.id.clone(),
                        exit_code: None,
                        message: error_msg.clone(),
                    });
                    println!("  ✗ seed={:<5} status=FAIL", inline.seed);
                    println!("    Error: {}", error_msg);
                    println!("  → FAIL");
                }
                println!();
            }
        }
    }

    // Check for 0 scenarios executed
    if stats.runs_total == 0 {
        eprintln!();
        eprintln!("ERROR: Suite executed 0 scenarios. Check your suite YAML.");
        eprintln!();
        eprintln!("Output written to: {}/", effective_out_dir);
        return 1;
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
