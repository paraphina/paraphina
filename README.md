## Risk profiles & starting inventory

Paraphina exposes three coarse risk profiles via `PARAPHINA_RISK_PROFILE`:

- **aggressive**: band≈5.6 TAO, η≈0.025, σ_ref≈0.01875, daily_loss_limit=2000 USD  
- **balanced**:   band≈3.8 TAO, η≈0.050, σ_ref≈0.01500, daily_loss_limit=5000 USD  
- **conservative**: band≈1.9 TAO, η≈0.100, σ_ref≈0.01125, daily_loss_limit=750 USD  

Stress harness `batch_runs/exp03_stress_search.py` sweeps starting inventory
`PARAPHINA_INIT_Q_TAO ∈ {-40, -20, 0, +20, +40}` and confirms:

- PnL is linear in `q0` under the test price path.
- Kill-switch trips exactly when daily PnL crosses the profile’s `daily_loss_limit`.

**Practical guidance**

For live or high-fidelity sim runs we recommend:

- aggressive: `PARAPHINA_INIT_Q_TAO ∈ [-10, +10]`
- balanced:  `PARAPHINA_INIT_Q_TAO ∈ [-20, +20]`
- conservative: `PARAPHINA_INIT_Q_TAO ∈ [-5, +5]`

Larger starting inventories are allowed, but may hit the kill-switch quickly under
adverse moves.
