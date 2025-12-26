# Hedge Allocator Verification

This document describes the institutional-grade verification layer for the hedge allocator.

## Overview

The hedge allocator verification consists of two complementary test layers:

1. **Deterministic Invariant Tests** - Property-style tests that explore a wide range of parameter combinations and assert hard safety invariants.

2. **Golden Vector Tests** - Snapshot tests using a curated set of inputs with expected outputs to protect determinism and catch regressions.

## Test Files

| File | Description |
|------|-------------|
| `paraphina/tests/hedge_allocator_invariants.rs` | Deterministic invariant tests |
| `paraphina/tests/hedge_allocator_golden.rs` | Golden vector snapshot tests |
| `paraphina/tests/hedge_testkit.rs` | Shared test utilities |
| `paraphina/tests/golden/hedge_allocator_vectors.json` | Golden test vectors |

## Invariants Enforced

The invariant tests verify that the following properties **always hold** for any valid input:

### 1. No NaNs or Infinities
All outputs (sizes, prices, deltas) must be finite numbers. No NaN or infinity values are allowed.

### 2. Total Hedge Size Bounded
The total hedge size across all venues must never exceed `max_step_tao`.

### 3. Per-Venue Cap Respected
Each venue's allocation must respect its per-venue cap, which is:
```
min(max_venue_tao_per_tick, venue.max_order_size, depth_fraction * depth_usd / fair_value)
```

### 4. Margin Constraint Respected
When a trade would increase absolute exposure, the new position must satisfy:
```
|new_position| <= |old_position| + additional_abs_cap
```
where:
```
additional_abs_cap = (margin_available_usd * max_leverage * safety_buffer) / mark_price_usd
```

### 5. Hedge Direction Correct
- Long position (q > 0) → hedge must **sell** to reduce exposure
- Short position (q < 0) → hedge must **buy** to cover

### 6. Output Determinism
Identical inputs must always produce identical outputs. The allocator must be deterministic with respect to:
- Order of intents
- Sizes and prices
- Venue selection

### 7. Single Order Per Venue
Output contains at most one aggregated order per venue, even with multi-chunk allocation enabled.

### 8. Consistent Ordering
Output intents are always sorted by `venue_index`.

## Running Tests

### Run All Hedge Verification Tests
```bash
cargo test hedge_allocator --test hedge_allocator_invariants --test hedge_allocator_golden
```

### Run Only Invariant Tests
```bash
cargo test --test hedge_allocator_invariants
```

### Run Only Golden Tests
```bash
cargo test --test hedge_allocator_golden
```

### Run a Specific Test
```bash
cargo test --test hedge_allocator_invariants invariant_no_nans_or_infs
cargo test --test hedge_allocator_golden golden_basic_long_hedge
```

## Golden Vectors

### What They Test

The golden vectors cover critical edge cases:

| Vector | Description |
|--------|-------------|
| Basic long/short | Correct hedge direction |
| Deadband | No hedge when inside band |
| Margin cap binding | Limited margin constrains allocation |
| Depth cap binding | Low depth constrains allocation |
| Deterministic tie-break | Equal cost venues sorted by index |
| Convexity spreading | Chunks spread across venues |
| Max step cap | Large position clamped |
| Zero margin | Venue skipped when margin is zero |
| Multi-venue | Fills from cheapest first |

### Updating Golden Vectors

**⚠️ WARNING: Only update golden vectors when behavior changes are intentional!**

To update golden vectors:

1. **Verify the change is intentional** - Understand why the output differs from expected.

2. **Run tests to see differences**:
   ```bash
   cargo test --test hedge_allocator_golden -- --nocapture 2>&1 | head -100
   ```

3. **Compute new expected values** - Run the allocator with each input and capture outputs.

4. **Update the JSON file** - Edit `paraphina/tests/golden/hedge_allocator_vectors.json` with new expected values.

5. **Re-run tests to verify**:
   ```bash
   cargo test --test hedge_allocator_golden
   ```

6. **Document the change** - Add a clear commit message explaining why the expected output changed.

### Golden Vector Format

Each golden vector has:
```json
{
  "description": "Human-readable description",
  "input": {
    "q_global_tao": ...,
    "fair_value": ...,
    "venues": [...]
  },
  "expected": {
    "desired_delta_nano": ...,
    "intents": [
      {
        "venue_index": ...,
        "side": "Buy" or "Sell",
        "size_nano": ...,
        "price_nano": ...
      }
    ]
  }
}
```

**Note**: Sizes and prices are quantized to nano-units (×10⁹) to avoid floating-point precision issues across platforms.

## Test Kit Utilities

The `hedge_testkit.rs` module provides:

### Xorshift64 PRNG
A minimal deterministic pseudo-random number generator:
```rust
let mut rng = Xorshift64::new(SEED);
let value = rng.range_f64(0.0, 100.0);
```

### HedgeTestCase
Generates random test cases with all required parameters:
```rust
let test_case = HedgeTestCase::random(&mut rng, case_id);
let cfg = test_case.build_config();
let state = test_case.build_state(&cfg);
```

### NormalizedPlan
Quantized representation for deterministic comparison:
```rust
let normalized = NormalizedPlan::from_intents(delta, intents);
assert_eq!(actual, expected);
```

## Performance

The verification tests are designed to be fast:

- **Invariant tests**: ~200 random cases, completes in <500ms
- **Golden tests**: ~10 vectors, completes in <50ms
- **Total**: Target is <1s locally

No network I/O or filesystem writes during tests.

## CI Integration

These tests run automatically in CI:
```bash
cargo test -q
```

All hedge verification tests are included in the standard test suite.

## Adding New Tests

### Adding an Invariant
1. Add a new test function in `hedge_allocator_invariants.rs`
2. Use the `HedgeTestCase::random()` generator or systematic enumeration
3. Assert the invariant holds for all cases
4. Include helpful failure messages with case ID and inputs

### Adding a Golden Vector
1. Add a new entry to `golden/hedge_allocator_vectors.json`
2. Include a clear description
3. Compute the expected output by running the allocator
4. Add a corresponding individual test if needed for documentation

## Troubleshooting

### Test Fails with "Non-deterministic result"
The allocator produced different outputs for identical inputs. Check:
- Are there any HashMap/HashSet iterations without sorting?
- Is there any uninitialized memory being read?
- Are floating-point operations order-dependent?

### Golden Vector Mismatch
Expected output differs from actual. This could indicate:
- Intentional behavior change (update the vector)
- Regression bug (investigate and fix)
- Platform-specific floating-point differences (check quantization)

### Invariant Violation
A safety invariant was broken. The failure message includes:
- Case ID (for reproducibility)
- Specific values that violated the invariant
- Full test case configuration

Use this information to identify and fix the underlying bug.

