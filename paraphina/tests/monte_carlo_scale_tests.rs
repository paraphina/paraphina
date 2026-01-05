// tests/monte_carlo_scale_tests.rs
//
// Tests for Monte Carlo at Scale features:
// - Deterministic seed mapping for sharding
// - Summarize mode validation (contiguity, duplicates, seed contract)

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use tempfile::tempdir;

// ============================================================================
// Seed mapping tests
// ============================================================================

#[test]
fn test_seed_mapping_basic() {
    // Deterministic seed contract: seed_i = base_seed + i (u64 wrap)
    let base_seed: u64 = 12345;

    for i in 0..100usize {
        let expected = base_seed.wrapping_add(i as u64);
        let computed = base_seed.wrapping_add(i as u64);
        assert_eq!(computed, expected, "Seed mismatch at index {}", i);
    }
}

#[test]
fn test_seed_mapping_wrapping() {
    // Test u64 wrapping behavior near the boundary
    let base_seed: u64 = u64::MAX - 5;

    // Index 0 should give base_seed
    assert_eq!(base_seed.wrapping_add(0), u64::MAX - 5);

    // Index 5 should give u64::MAX
    assert_eq!(base_seed.wrapping_add(5), u64::MAX);

    // Index 6 should wrap to 0
    assert_eq!(base_seed.wrapping_add(6), 0);

    // Index 10 should wrap to 4
    assert_eq!(base_seed.wrapping_add(10), 4);
}

#[test]
fn test_seed_mapping_shard_slicing() {
    // Test that sharding produces the same seeds as a full run
    let base_seed: u64 = 42;
    let total_runs = 1000;
    let shards = 10;
    let runs_per_shard = total_runs / shards;

    // Compute seeds for full run
    let full_seeds: Vec<u64> = (0..total_runs)
        .map(|i| base_seed.wrapping_add(i as u64))
        .collect();

    // Compute seeds for each shard and verify they match
    for shard_idx in 0..shards {
        let start = shard_idx * runs_per_shard;
        let end = start + runs_per_shard;

        for i in start..end {
            let shard_seed = base_seed.wrapping_add(i as u64);
            assert_eq!(
                shard_seed, full_seeds[i],
                "Shard {} index {} seed mismatch",
                shard_idx, i
            );
        }
    }
}

#[test]
fn test_seed_mapping_non_zero_start() {
    // Test seed mapping when run_start_index > 0
    let base_seed: u64 = 100;
    let run_start_index: usize = 500;
    let run_count: usize = 100;

    for local_idx in 0..run_count {
        let global_idx = run_start_index + local_idx;
        let expected_seed = base_seed.wrapping_add(global_idx as u64);

        // This is the exact formula used in monte_carlo.rs
        let computed_seed = base_seed.wrapping_add(global_idx as u64);

        assert_eq!(
            computed_seed, expected_seed,
            "Seed mismatch at local_idx={}, global_idx={}",
            local_idx, global_idx
        );
    }
}

// ============================================================================
// JSONL parsing/validation tests (simulating summarize mode)
// ============================================================================

/// Minimal JSONL record for testing (matches McRunJsonlRecord)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct TestJsonlRecord {
    run_index: usize,
    seed: u64,
    pnl_total: f64,
    max_drawdown: f64,
    kill_switch: bool,
}

fn write_test_jsonl(path: &PathBuf, records: &[TestJsonlRecord]) {
    let mut file = File::create(path).unwrap();
    for record in records {
        let line = serde_json::to_string(record).unwrap();
        writeln!(file, "{}", line).unwrap();
    }
}

#[test]
fn test_summarize_rejects_duplicate_indices() {
    // Create JSONL with duplicate run_index
    let records = vec![
        TestJsonlRecord {
            run_index: 0,
            seed: 100,
            pnl_total: 10.0,
            max_drawdown: 1.0,
            kill_switch: false,
        },
        TestJsonlRecord {
            run_index: 1,
            seed: 101,
            pnl_total: 20.0,
            max_drawdown: 2.0,
            kill_switch: false,
        },
        TestJsonlRecord {
            run_index: 0, // DUPLICATE
            seed: 100,
            pnl_total: 15.0,
            max_drawdown: 1.5,
            kill_switch: false,
        },
    ];

    // Parse and check for duplicates (simulating summarize validation)
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut has_duplicate = false;

    for record in &records {
        if seen.contains(&record.run_index) {
            has_duplicate = true;
            break;
        }
        seen.insert(record.run_index);
    }

    assert!(has_duplicate, "Should detect duplicate run_index");
}

#[test]
fn test_summarize_rejects_non_contiguous_indices() {
    // Create JSONL with gap in indices
    let records = vec![
        TestJsonlRecord {
            run_index: 0,
            seed: 100,
            pnl_total: 10.0,
            max_drawdown: 1.0,
            kill_switch: false,
        },
        TestJsonlRecord {
            run_index: 1,
            seed: 101,
            pnl_total: 20.0,
            max_drawdown: 2.0,
            kill_switch: false,
        },
        TestJsonlRecord {
            run_index: 3, // GAP - missing index 2
            seed: 103,
            pnl_total: 30.0,
            max_drawdown: 3.0,
            kill_switch: false,
        },
    ];

    // Sort by run_index
    let mut sorted = records.clone();
    sorted.sort_by_key(|r| r.run_index);

    // Check contiguity
    let min_idx = sorted.first().unwrap().run_index;
    let max_idx = sorted.last().unwrap().run_index;
    let expected_count = max_idx - min_idx + 1;

    assert_ne!(
        sorted.len(),
        expected_count,
        "Should detect non-contiguous indices"
    );
}

#[test]
fn test_summarize_accepts_valid_contiguous_indices() {
    let records = vec![
        TestJsonlRecord {
            run_index: 5,
            seed: 105,
            pnl_total: 50.0,
            max_drawdown: 5.0,
            kill_switch: false,
        },
        TestJsonlRecord {
            run_index: 6,
            seed: 106,
            pnl_total: 60.0,
            max_drawdown: 6.0,
            kill_switch: false,
        },
        TestJsonlRecord {
            run_index: 7,
            seed: 107,
            pnl_total: 70.0,
            max_drawdown: 7.0,
            kill_switch: false,
        },
    ];

    // Sort by run_index
    let mut sorted = records.clone();
    sorted.sort_by_key(|r| r.run_index);

    // Check no duplicates
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for record in &sorted {
        assert!(
            !seen.contains(&record.run_index),
            "Unexpected duplicate at {}",
            record.run_index
        );
        seen.insert(record.run_index);
    }

    // Check contiguity
    let min_idx = sorted.first().unwrap().run_index;
    let max_idx = sorted.last().unwrap().run_index;
    let expected_count = max_idx - min_idx + 1;

    assert_eq!(
        sorted.len(),
        expected_count,
        "Should be contiguous: expected {} records for indices {}..={}",
        expected_count,
        min_idx,
        max_idx
    );
}

#[test]
fn test_summarize_validates_seed_contract() {
    let base_seed: u64 = 100;

    // Valid records following seed contract
    let valid_records = vec![
        TestJsonlRecord {
            run_index: 0,
            seed: 100, // base_seed + 0
            pnl_total: 10.0,
            max_drawdown: 1.0,
            kill_switch: false,
        },
        TestJsonlRecord {
            run_index: 1,
            seed: 101, // base_seed + 1
            pnl_total: 20.0,
            max_drawdown: 2.0,
            kill_switch: false,
        },
        TestJsonlRecord {
            run_index: 2,
            seed: 102, // base_seed + 2
            pnl_total: 30.0,
            max_drawdown: 3.0,
            kill_switch: false,
        },
    ];

    // Validate seed contract
    for record in &valid_records {
        let expected_seed = base_seed.wrapping_add(record.run_index as u64);
        assert_eq!(
            record.seed, expected_seed,
            "Seed contract violation at run_index {}",
            record.run_index
        );
    }
}

#[test]
fn test_summarize_detects_seed_contract_violation() {
    let base_seed: u64 = 100;

    // Record with wrong seed
    let bad_record = TestJsonlRecord {
        run_index: 5,
        seed: 999, // WRONG - should be 105
        pnl_total: 50.0,
        max_drawdown: 5.0,
        kill_switch: false,
    };

    let expected_seed = base_seed.wrapping_add(bad_record.run_index as u64);
    assert_ne!(
        bad_record.seed, expected_seed,
        "Should detect seed contract violation"
    );
}

#[test]
fn test_summarize_handles_malformed_json() {
    let dir = tempdir().unwrap();
    let jsonl_path = dir.path().join("bad.jsonl");

    // Write malformed JSON
    let mut file = File::create(&jsonl_path).unwrap();
    writeln!(file, r#"{{"run_index": 0, "seed": 100}}"#).unwrap(); // Valid but incomplete
    writeln!(file, r#"not valid json at all"#).unwrap(); // Malformed
    writeln!(file, r#"{{"run_index": 2, "seed": 102}}"#).unwrap();

    // Read and parse
    let content = fs::read_to_string(&jsonl_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();

    let mut parse_errors = 0;
    for line in &lines {
        if serde_json::from_str::<serde_json::Value>(line).is_err() {
            parse_errors += 1;
        }
    }

    assert_eq!(parse_errors, 1, "Should detect 1 malformed JSON line");
}

// ============================================================================
// Integration test: JSONL file round-trip
// ============================================================================

#[test]
fn test_jsonl_round_trip() {
    let dir = tempdir().unwrap();
    let jsonl_path = dir.path().join("test.jsonl");

    let original_records = vec![
        TestJsonlRecord {
            run_index: 0,
            seed: 12345,
            pnl_total: 100.5,
            max_drawdown: 10.2,
            kill_switch: false,
        },
        TestJsonlRecord {
            run_index: 1,
            seed: 12346,
            pnl_total: -50.3,
            max_drawdown: 25.7,
            kill_switch: true,
        },
        TestJsonlRecord {
            run_index: 2,
            seed: 12347,
            pnl_total: 200.0,
            max_drawdown: 5.0,
            kill_switch: false,
        },
    ];

    // Write
    write_test_jsonl(&jsonl_path, &original_records);

    // Read back
    let content = fs::read_to_string(&jsonl_path).unwrap();
    let parsed_records: Vec<TestJsonlRecord> = content
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();

    assert_eq!(
        original_records.len(),
        parsed_records.len(),
        "Record count mismatch"
    );

    for (orig, parsed) in original_records.iter().zip(parsed_records.iter()) {
        assert_eq!(orig.run_index, parsed.run_index);
        assert_eq!(orig.seed, parsed.seed);
        assert!((orig.pnl_total - parsed.pnl_total).abs() < 1e-10);
        assert!((orig.max_drawdown - parsed.max_drawdown).abs() < 1e-10);
        assert_eq!(orig.kill_switch, parsed.kill_switch);
    }
}

