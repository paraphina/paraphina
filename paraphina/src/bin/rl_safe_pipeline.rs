use std::env;
use std::fs;
use std::path::PathBuf;

use paraphina::config::Config;
use paraphina::rl::run_safe_pipeline;

fn parse_u64(args: &[String], flag: &str, default: u64) -> u64 {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_path(args: &[String], flag: &str, default: &str) -> PathBuf {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let seed = parse_u64(&args, "--seed", 42);
    let episodes = parse_u64(&args, "--episodes", 2);
    let ticks = parse_u64(&args, "--ticks", 20);
    let out_path = parse_path(&args, "--out", "runs/rl_safe_pipeline/rl_safe_summary.json");

    let cfg = Config::default();
    let summary = run_safe_pipeline(&cfg, seed, episodes, ticks);

    if let Some(parent) = out_path.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            eprintln!(
                "Failed to create output directory {}: {}",
                parent.display(),
                err
            );
            std::process::exit(2);
        }
    }

    match serde_json::to_string_pretty(&summary) {
        Ok(payload) => {
            if let Err(err) = fs::write(&out_path, payload) {
                eprintln!("Failed to write {}: {}", out_path.display(), err);
                std::process::exit(2);
            }
            println!("rl_safe_pipeline: wrote {}", out_path.display());
        }
        Err(err) => {
            eprintln!("Failed to serialize summary: {}", err);
            std::process::exit(2);
        }
    }
}
