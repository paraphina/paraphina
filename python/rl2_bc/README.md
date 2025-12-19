# RL-2: Behaviour Cloning Baseline

This directory contains Python scripts for RL-2 behaviour cloning training and evaluation.

## Prerequisites

1. **Build the Rust library** (required):
   ```bash
   cd /path/to/paraphina
   cargo build --release
   ```

2. **Install the Python bindings**:
   ```bash
   cd paraphina_env
   pip install maturin
   maturin develop --release
   ```

3. **Install Python dependencies** (optional, only for training):
   ```bash
   pip install -r python/rl2_bc/requirements.txt
   ```

## Scripts

### 1. Generate Dataset (`generate_dataset.py`)

Collects trajectories from the HeuristicPolicy for behaviour cloning.

```bash
# Generate 100 episodes with deterministic settings
python python/rl2_bc/generate_dataset.py \
    --num-episodes 100 \
    --num-envs 4 \
    --max-ticks 1000 \
    --domain-rand deterministic \
    --seed 42 \
    --output-dir runs/rl2_data

# Generate 1000 episodes with domain randomisation
python python/rl2_bc/generate_dataset.py \
    --num-episodes 1000 \
    --num-envs 8 \
    --domain-rand mild \
    --seed 12345 \
    --output-dir runs/rl2_data_rand
```

Output:
- `{output_dir}/metadata.json` - Dataset metadata with version info
- `{output_dir}/observations.npy` - Observations array [N, obs_dim]
- `{output_dir}/actions.npy` - Actions array [N, action_dim]
- `{output_dir}/rewards.npy` - Rewards array [N]
- `{output_dir}/terminals.npy` - Terminal flags [N]
- `{output_dir}/episode_indices.npy` - Episode indices [N]

### 2. Train BC Model (`train_bc.py`)

Trains an MLP policy to imitate the heuristic.

```bash
# Basic training
python python/rl2_bc/train_bc.py \
    --data-dir runs/rl2_data \
    --output-dir runs/rl2_model \
    --epochs 100 \
    --batch-size 256 \
    --seed 42

# With custom architecture
python python/rl2_bc/train_bc.py \
    --data-dir runs/rl2_data \
    --output-dir runs/rl2_model \
    --hidden-dims 256 256 128 \
    --lr 1e-4 \
    --epochs 200 \
    --seed 42
```

Output:
- `{output_dir}/model.pt` - Trained PyTorch model
- `{output_dir}/training_metadata.json` - Training config and metrics
- `{output_dir}/training_curves.png` - Loss curves (optional)

### 3. Evaluate BC Model (`eval_bc.py`)

Evaluates the BC model on holdout data and/or rollouts.

```bash
# Evaluate action prediction error
python python/rl2_bc/eval_bc.py \
    --model-dir runs/rl2_model \
    --data-dir runs/rl2_data \
    --eval-holdout

# Run rollouts to compare PnL/risk
python python/rl2_bc/eval_bc.py \
    --model-dir runs/rl2_model \
    --num-episodes 100 \
    --eval-rollout

# Full evaluation
python python/rl2_bc/eval_bc.py \
    --model-dir runs/rl2_model \
    --data-dir runs/rl2_data \
    --num-episodes 100 \
    --eval-holdout \
    --eval-rollout \
    --seed 42
```

Output:
- `{model_dir}/eval_results.json` - Evaluation metrics

## Dataset Format

### Metadata (`metadata.json`)

```json
{
  "trajectory_version": 1,
  "obs_version": 1,
  "action_version": 1,
  "policy_version": "heuristic-v1.0.0",
  "base_seed": 42,
  "num_episodes": 100,
  "num_transitions": 50000,
  "num_venues": 5,
  "obs_dim": 87,
  "action_dim": 21,
  "domain_rand_preset": "deterministic",
  "apply_domain_rand": false,
  "max_ticks": 1000,
  "collected_at": "1702000000",
  "kill_rate": 0.05,
  "mean_episode_length": 500.0,
  "mean_pnl": 10.5
}
```

### Observation Features

**Global features (17)**:
- fair_value, sigma_eff, vol_ratio_clipped, spread_mult, size_mult, band_mult
- q_global_tao, dollar_delta_usd, delta_limit_usd
- basis_usd, basis_gross_usd, basis_limit_warn_usd, basis_limit_hard_usd
- daily_realised_pnl, daily_unrealised_pnl, daily_pnl_total
- kill_switch (0/1)

**Per-venue features (14 × num_venues)**:
- mid, spread, depth_near_mid, staleness_ms
- local_vol_short, local_vol_long
- status_healthy, status_warning, status_disabled (one-hot)
- toxicity, position_tao, avg_entry_price, funding_8h, margin_available_usd, dist_liq_sigma

### Action Encoding

**Per-venue (3 × num_venues)**:
- spread_scale: [0.5, 3.0] → normalized to [0, 1]
- size_scale: [0.0, 2.0] → normalized to [0, 1]  
- rprice_offset_usd: [-10, 10] → normalized to [-1, 1]

**Global (1 + num_venues)**:
- hedge_scale: [0.0, 2.0] → normalized to [0, 1]
- hedge_venue_weights: simplex (sum to 1)

## Reproducibility

Same seed + same config = byte-for-byte identical outputs.

**Known floating-point considerations**:
- PyTorch training may have minor non-determinism on GPU due to cuDNN
- Use `torch.use_deterministic_algorithms(True)` for full determinism
- CPU training is fully deterministic

## Exit Criteria (per ROADMAP.md)

BC policy matches baseline within tolerance and does not increase kill rate:
- Action prediction MAE < 0.1 on holdout
- Rollout kill rate ≤ heuristic kill rate × 1.1
- Rollout mean PnL ≥ heuristic mean PnL × 0.9

