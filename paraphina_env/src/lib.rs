// paraphina_env/src/lib.rs
//
// Python bindings for the Paraphina RL environment (RL-1).
//
// Provides a Gym-style API for training RL agents:
// - Env: Single environment with reset(seed) and step(action)
// - VecEnv: Vectorised environments for parallel rollouts
//
// All operations are deterministic given seeds.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use paraphina::{
    AblationSet, Config, DomainRandConfig, Observation, PolicyAction, SimEnv as RustSimEnv,
    SimEnvConfig, StepInfo, TrajectoryCollector as RustTrajectoryCollector, TrajectoryMetadata,
    TrajectoryRecord, VecEnv as RustVecEnv, VenueObservation, ACTION_VERSION, OBS_VERSION,
    TRAJECTORY_VERSION,
};

/// Convert a Rust Observation to a Python dictionary.
fn observation_to_dict(py: Python<'_>, obs: &Observation) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);

    // Metadata
    dict.set_item("obs_version", obs.obs_version)?;
    dict.set_item("timestamp_ms", obs.timestamp_ms)?;
    dict.set_item("tick_index", obs.tick_index)?;

    // Global fair value and volatility
    dict.set_item("fair_value", obs.fair_value)?;
    dict.set_item("sigma_eff", obs.sigma_eff)?;
    dict.set_item("vol_ratio_clipped", obs.vol_ratio_clipped)?;

    // Control scalars
    dict.set_item("spread_mult", obs.spread_mult)?;
    dict.set_item("size_mult", obs.size_mult)?;
    dict.set_item("band_mult", obs.band_mult)?;

    // Inventory and exposure
    dict.set_item("q_global_tao", obs.q_global_tao)?;
    dict.set_item("dollar_delta_usd", obs.dollar_delta_usd)?;
    dict.set_item("delta_limit_usd", obs.delta_limit_usd)?;

    // Basis exposure
    dict.set_item("basis_usd", obs.basis_usd)?;
    dict.set_item("basis_gross_usd", obs.basis_gross_usd)?;
    dict.set_item("basis_limit_warn_usd", obs.basis_limit_warn_usd)?;
    dict.set_item("basis_limit_hard_usd", obs.basis_limit_hard_usd)?;

    // PnL
    dict.set_item("daily_realised_pnl", obs.daily_realised_pnl)?;
    dict.set_item("daily_unrealised_pnl", obs.daily_unrealised_pnl)?;
    dict.set_item("daily_pnl_total", obs.daily_pnl_total)?;

    // Risk state
    dict.set_item("risk_regime", format!("{:?}", obs.risk_regime))?;
    dict.set_item("kill_switch", obs.kill_switch)?;
    dict.set_item("kill_reason", format!("{:?}", obs.kill_reason))?;

    // FV gating
    dict.set_item("fv_available", obs.fv_available)?;
    dict.set_item("healthy_venues_used_count", obs.healthy_venues_used_count)?;

    // Per-venue observations
    let venues = PyList::empty_bound(py);
    for v in &obs.venues {
        let vdict = venue_observation_to_dict(py, v)?;
        venues.append(vdict)?;
    }
    dict.set_item("venues", venues)?;

    Ok(dict.into())
}

/// Convert a VenueObservation to a Python dictionary.
fn venue_observation_to_dict(py: Python<'_>, v: &VenueObservation) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);

    dict.set_item("venue_id", &v.venue_id)?;
    dict.set_item("venue_index", v.venue_index)?;
    dict.set_item("mid", v.mid)?;
    dict.set_item("spread", v.spread)?;
    dict.set_item("depth_near_mid", v.depth_near_mid)?;
    dict.set_item("staleness_ms", v.staleness_ms)?;
    dict.set_item("local_vol_short", v.local_vol_short)?;
    dict.set_item("local_vol_long", v.local_vol_long)?;
    dict.set_item("status", format!("{:?}", v.status))?;
    dict.set_item("toxicity", v.toxicity)?;
    dict.set_item("position_tao", v.position_tao)?;
    dict.set_item("avg_entry_price", v.avg_entry_price)?;
    dict.set_item("funding_8h", v.funding_8h)?;
    dict.set_item("margin_available_usd", v.margin_available_usd)?;
    dict.set_item("dist_liq_sigma", v.dist_liq_sigma)?;

    Ok(dict.into())
}

/// Convert a StepInfo to a Python dictionary.
fn step_info_to_dict(py: Python<'_>, info: &StepInfo) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);

    dict.set_item("termination_reason", &info.termination_reason)?;
    dict.set_item("tick", info.tick)?;
    dict.set_item("pnl_total", info.pnl_total)?;
    dict.set_item("pnl_realised", info.pnl_realised)?;
    dict.set_item("pnl_unrealised", info.pnl_unrealised)?;
    dict.set_item("kill_switch", info.kill_switch)?;
    dict.set_item("kill_reason", &info.kill_reason)?;
    dict.set_item("risk_regime", &info.risk_regime)?;
    dict.set_item("q_global_tao", info.q_global_tao)?;
    dict.set_item("dollar_delta_usd", info.dollar_delta_usd)?;
    dict.set_item("basis_usd", info.basis_usd)?;

    Ok(dict.into())
}

/// Parse a Python action dict into a PolicyAction.
fn parse_action(
    _py: Python<'_>,
    action: &Bound<'_, PyDict>,
    num_venues: usize,
) -> PyResult<PolicyAction> {
    // Default identity action
    let mut policy_action = PolicyAction::identity(num_venues, "python-policy");

    // Parse spread_scale if provided
    if let Some(spread_scale) = action.get_item("spread_scale")? {
        let spread_scale: Vec<f64> = spread_scale.extract()?;
        if spread_scale.len() != num_venues {
            return Err(PyValueError::new_err(format!(
                "spread_scale length {} must match num_venues {}",
                spread_scale.len(),
                num_venues
            )));
        }
        policy_action.spread_scale = spread_scale;
    }

    // Parse size_scale if provided
    if let Some(size_scale) = action.get_item("size_scale")? {
        let size_scale: Vec<f64> = size_scale.extract()?;
        if size_scale.len() != num_venues {
            return Err(PyValueError::new_err(format!(
                "size_scale length {} must match num_venues {}",
                size_scale.len(),
                num_venues
            )));
        }
        policy_action.size_scale = size_scale;
    }

    // Parse rprice_offset_usd if provided
    if let Some(rprice_offset) = action.get_item("rprice_offset_usd")? {
        let rprice_offset: Vec<f64> = rprice_offset.extract()?;
        if rprice_offset.len() != num_venues {
            return Err(PyValueError::new_err(format!(
                "rprice_offset_usd length {} must match num_venues {}",
                rprice_offset.len(),
                num_venues
            )));
        }
        policy_action.rprice_offset_usd = rprice_offset;
    }

    // Parse hedge_scale if provided
    if let Some(hedge_scale) = action.get_item("hedge_scale")? {
        policy_action.hedge_scale = hedge_scale.extract()?;
    }

    // Parse hedge_venue_weights if provided
    if let Some(hedge_weights) = action.get_item("hedge_venue_weights")? {
        let hedge_weights: Vec<f64> = hedge_weights.extract()?;
        if hedge_weights.len() != num_venues {
            return Err(PyValueError::new_err(format!(
                "hedge_venue_weights length {} must match num_venues {}",
                hedge_weights.len(),
                num_venues
            )));
        }
        policy_action.hedge_venue_weights = hedge_weights;
    }

    // Clamp to valid ranges
    policy_action.clamp();

    Ok(policy_action)
}

/// Gym-style environment wrapper.
///
/// Provides the standard RL interface:
/// - reset(seed) -> observation
/// - step(action) -> (observation, reward, done, info)
#[pyclass]
pub struct Env {
    inner: RustSimEnv,
    num_venues: usize,
}

#[pymethods]
impl Env {
    /// Create a new environment.
    ///
    /// Args:
    ///     max_ticks: Maximum number of ticks per episode (default: 1000)
    ///     apply_domain_rand: Whether to apply domain randomisation (default: True)
    ///     domain_rand_preset: "default", "mild", or "deterministic" (default: "default")
    #[new]
    #[pyo3(signature = (max_ticks=1000, apply_domain_rand=true, domain_rand_preset="default"))]
    fn new(max_ticks: u64, apply_domain_rand: bool, domain_rand_preset: &str) -> PyResult<Self> {
        let base_config = Config::default();
        let num_venues = base_config.venues.len();

        let domain_rand = match domain_rand_preset {
            "default" => DomainRandConfig::default(),
            "mild" => DomainRandConfig::mild(),
            "deterministic" => DomainRandConfig::deterministic(),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown domain_rand_preset: {}. Use 'default', 'mild', or 'deterministic'",
                    domain_rand_preset
                )))
            }
        };

        let env_config = SimEnvConfig {
            max_ticks,
            dt_ms: 1000,
            domain_rand,
            reward_weights: Default::default(),
            apply_domain_rand,
            ablations: AblationSet::new(),
            ..SimEnvConfig::default()
        };

        let inner = RustSimEnv::new(base_config, env_config);

        Ok(Self { inner, num_venues })
    }

    /// Reset the environment.
    ///
    /// Args:
    ///     seed: Optional seed for deterministic reset
    ///
    /// Returns:
    ///     observation: Dict containing the initial observation
    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, py: Python<'_>, seed: Option<u64>) -> PyResult<Py<PyDict>> {
        let obs = self.inner.reset(seed);
        observation_to_dict(py, &obs)
    }

    /// Take a step in the environment.
    ///
    /// Args:
    ///     action: Dict with optional keys:
    ///         - spread_scale: List[float] per venue (default: 1.0)
    ///         - size_scale: List[float] per venue (default: 1.0)
    ///         - rprice_offset_usd: List[float] per venue (default: 0.0)
    ///         - hedge_scale: float (default: 1.0)
    ///         - hedge_venue_weights: List[float] per venue (default: uniform)
    ///
    /// Returns:
    ///     Tuple of (observation, reward, done, info)
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: &Bound<'py, PyDict>,
    ) -> PyResult<(Py<PyDict>, f64, bool, Py<PyDict>)> {
        let policy_action = parse_action(py, action, self.num_venues)?;
        let result = self.inner.step(&policy_action);

        let obs = observation_to_dict(py, &result.observation)?;
        let info = step_info_to_dict(py, &result.info)?;

        Ok((obs, result.reward, result.done, info))
    }

    /// Get the number of venues.
    #[getter]
    fn num_venues(&self) -> usize {
        self.num_venues
    }

    /// Get the current tick.
    #[getter]
    fn tick(&self) -> u64 {
        self.inner.tick()
    }

    /// Check if the episode is done.
    #[getter]
    fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    /// Get the current seed.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }

    /// Get an identity action (baseline behavior).
    fn identity_action(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let action = PolicyAction::identity(self.num_venues, "identity");
        let dict = PyDict::new_bound(py);
        dict.set_item("spread_scale", action.spread_scale.clone())?;
        dict.set_item("size_scale", action.size_scale.clone())?;
        dict.set_item("rprice_offset_usd", action.rprice_offset_usd.clone())?;
        dict.set_item("hedge_scale", action.hedge_scale)?;
        dict.set_item("hedge_venue_weights", action.hedge_venue_weights.clone())?;
        Ok(dict.into())
    }
}

/// Vectorised environment for parallel rollouts.
///
/// Manages N independent Env instances.
#[pyclass]
pub struct VecEnv {
    inner: RustVecEnv,
    num_venues: usize,
}

#[pymethods]
impl VecEnv {
    /// Create a new vectorised environment.
    ///
    /// Args:
    ///     n: Number of parallel environments
    ///     max_ticks: Maximum ticks per episode (default: 1000)
    ///     apply_domain_rand: Whether to apply domain randomisation (default: True)
    ///     domain_rand_preset: "default", "mild", or "deterministic" (default: "default")
    #[new]
    #[pyo3(signature = (n, max_ticks=1000, apply_domain_rand=true, domain_rand_preset="default"))]
    fn new(
        n: usize,
        max_ticks: u64,
        apply_domain_rand: bool,
        domain_rand_preset: &str,
    ) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err("n must be > 0"));
        }

        let base_config = Config::default();
        let num_venues = base_config.venues.len();

        let domain_rand = match domain_rand_preset {
            "default" => DomainRandConfig::default(),
            "mild" => DomainRandConfig::mild(),
            "deterministic" => DomainRandConfig::deterministic(),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown domain_rand_preset: {}",
                    domain_rand_preset
                )))
            }
        };

        let env_config = SimEnvConfig {
            max_ticks,
            dt_ms: 1000,
            domain_rand,
            reward_weights: Default::default(),
            apply_domain_rand,
            ablations: AblationSet::new(),
            ..SimEnvConfig::default()
        };

        let inner = RustVecEnv::new(n, base_config, env_config);

        Ok(Self { inner, num_venues })
    }

    /// Reset all environments.
    ///
    /// Args:
    ///     seeds: Optional list of seeds (one per environment)
    ///
    /// Returns:
    ///     List of observations
    #[pyo3(signature = (seeds=None))]
    fn reset_all(&mut self, py: Python<'_>, seeds: Option<Vec<u64>>) -> PyResult<Vec<Py<PyDict>>> {
        let observations = self.inner.reset_all(seeds.as_deref());
        observations
            .iter()
            .map(|obs| observation_to_dict(py, obs))
            .collect()
    }

    /// Step all environments.
    ///
    /// Args:
    ///     actions: List of action dicts (one per environment)
    ///
    /// Returns:
    ///     Tuple of (observations, rewards, dones, infos)
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<Bound<'py, PyDict>>,
    ) -> PyResult<(Vec<Py<PyDict>>, Vec<f64>, Vec<bool>, Vec<Py<PyDict>>)> {
        if actions.len() != self.inner.num_envs() {
            return Err(PyValueError::new_err(format!(
                "actions length {} must match num_envs {}",
                actions.len(),
                self.inner.num_envs()
            )));
        }

        let policy_actions: Vec<PolicyAction> = actions
            .iter()
            .map(|a| parse_action(py, a, self.num_venues))
            .collect::<PyResult<Vec<_>>>()?;

        let results = self.inner.step(&policy_actions);

        let observations: PyResult<Vec<_>> = results
            .iter()
            .map(|r| observation_to_dict(py, &r.observation))
            .collect();
        let rewards: Vec<f64> = results.iter().map(|r| r.reward).collect();
        let dones: Vec<bool> = results.iter().map(|r| r.done).collect();
        let infos: PyResult<Vec<_>> = results
            .iter()
            .map(|r| step_info_to_dict(py, &r.info))
            .collect();

        Ok((observations?, rewards, dones, infos?))
    }

    /// Step all environments with identity actions.
    ///
    /// Returns:
    ///     Tuple of (observations, rewards, dones, infos)
    #[allow(clippy::type_complexity)]
    fn step_identity(
        &mut self,
        py: Python<'_>,
    ) -> PyResult<(Vec<Py<PyDict>>, Vec<f64>, Vec<bool>, Vec<Py<PyDict>>)> {
        let results = self.inner.step_identity();

        let observations: PyResult<Vec<_>> = results
            .iter()
            .map(|r| observation_to_dict(py, &r.observation))
            .collect();
        let rewards: Vec<f64> = results.iter().map(|r| r.reward).collect();
        let dones: Vec<bool> = results.iter().map(|r| r.done).collect();
        let infos: PyResult<Vec<_>> = results
            .iter()
            .map(|r| step_info_to_dict(py, &r.info))
            .collect();

        Ok((observations?, rewards, dones, infos?))
    }

    /// Get the number of environments.
    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }

    /// Get the number of venues per environment.
    #[getter]
    fn num_venues(&self) -> usize {
        self.num_venues
    }

    /// Get all current seeds.
    fn seeds(&self) -> Vec<u64> {
        self.inner.seeds()
    }

    /// Get which environments are done.
    fn dones(&self) -> Vec<bool> {
        self.inner.dones()
    }

    /// Get an identity action for all environments.
    fn identity_actions(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let action = PolicyAction::identity(self.num_venues, "identity");
        let mut actions = Vec::with_capacity(self.inner.num_envs());

        for _ in 0..self.inner.num_envs() {
            let dict = PyDict::new_bound(py);
            dict.set_item("spread_scale", action.spread_scale.clone())?;
            dict.set_item("size_scale", action.size_scale.clone())?;
            dict.set_item("rprice_offset_usd", action.rprice_offset_usd.clone())?;
            dict.set_item("hedge_scale", action.hedge_scale)?;
            dict.set_item("hedge_venue_weights", action.hedge_venue_weights.clone())?;
            actions.push(dict.into());
        }

        Ok(actions)
    }
}

/// Get the observation version.
#[pyfunction]
fn obs_version() -> u32 {
    OBS_VERSION
}

/// Get the action encoding version.
#[pyfunction]
fn action_version() -> u32 {
    ACTION_VERSION
}

/// Get the trajectory format version.
#[pyfunction]
fn trajectory_version() -> u32 {
    TRAJECTORY_VERSION
}

/// Get the default number of venues.
#[pyfunction]
fn default_num_venues() -> usize {
    Config::default().venues.len()
}

/// Convert TrajectoryRecord to Python dict.
fn trajectory_record_to_dict(py: Python<'_>, record: &TrajectoryRecord) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);

    dict.set_item("obs_features", record.obs_features.clone())?;
    dict.set_item("action_target", record.action_target.clone())?;
    dict.set_item("reward", record.reward)?;
    dict.set_item("terminal", record.terminal)?;
    dict.set_item("kill_reason", record.kill_reason.clone())?;
    dict.set_item("episode_idx", record.episode_idx)?;
    dict.set_item("step_idx", record.step_idx)?;
    dict.set_item("seed", record.seed)?;

    Ok(dict.into())
}

/// Convert TrajectoryMetadata to Python dict.
fn trajectory_metadata_to_dict(
    py: Python<'_>,
    metadata: &TrajectoryMetadata,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);

    dict.set_item("trajectory_version", metadata.trajectory_version)?;
    dict.set_item("obs_version", metadata.obs_version)?;
    dict.set_item("action_version", metadata.action_version)?;
    dict.set_item("policy_version", &metadata.policy_version)?;
    dict.set_item("base_seed", metadata.base_seed)?;
    dict.set_item("num_episodes", metadata.num_episodes)?;
    dict.set_item("num_transitions", metadata.num_transitions)?;
    dict.set_item("num_venues", metadata.num_venues)?;
    dict.set_item("obs_dim", metadata.obs_dim)?;
    dict.set_item("action_dim", metadata.action_dim)?;
    dict.set_item("domain_rand_preset", &metadata.domain_rand_preset)?;
    dict.set_item("apply_domain_rand", metadata.apply_domain_rand)?;
    dict.set_item("max_ticks", metadata.max_ticks)?;
    dict.set_item("collected_at", &metadata.collected_at)?;
    dict.set_item("kill_rate", metadata.kill_rate)?;
    dict.set_item("mean_episode_length", metadata.mean_episode_length)?;
    dict.set_item("mean_pnl", metadata.mean_pnl)?;

    Ok(dict.into())
}

/// Trajectory collector for generating BC training data.
///
/// Collects trajectories from the HeuristicPolicy running in the simulation
/// environment. Supports vectorised rollouts and domain randomisation.
#[pyclass]
pub struct TrajectoryCollector {
    inner: RustTrajectoryCollector,
}

#[pymethods]
impl TrajectoryCollector {
    /// Create a new trajectory collector.
    ///
    /// Args:
    ///     num_envs: Number of parallel environments (default: 4)
    ///     max_ticks: Maximum ticks per episode (default: 1000)
    ///     apply_domain_rand: Whether to apply domain randomisation (default: True)
    ///     domain_rand_preset: "default", "mild", or "deterministic" (default: "default")
    ///     base_seed: Base seed for reproducibility (default: 42)
    #[new]
    #[pyo3(signature = (num_envs=4, max_ticks=1000, apply_domain_rand=true, domain_rand_preset="default", base_seed=42))]
    fn new(
        num_envs: usize,
        max_ticks: u64,
        apply_domain_rand: bool,
        domain_rand_preset: &str,
        base_seed: u64,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be > 0"));
        }

        let inner = RustTrajectoryCollector::new(
            num_envs,
            max_ticks,
            apply_domain_rand,
            domain_rand_preset,
            base_seed,
        );

        Ok(Self { inner })
    }

    /// Collect trajectories for the specified number of episodes.
    ///
    /// Args:
    ///     num_episodes: Number of episodes to collect
    ///
    /// Returns:
    ///     Tuple of (records, metadata) where:
    ///       - records is a list of dicts with obs_features, action_target, reward, etc.
    ///       - metadata is a dict with version info, stats, etc.
    fn collect(
        &self,
        py: Python<'_>,
        num_episodes: u32,
    ) -> PyResult<(Vec<Py<PyDict>>, Py<PyDict>)> {
        let (records, metadata) = self.inner.collect(num_episodes);

        let py_records: PyResult<Vec<_>> = records
            .iter()
            .map(|r| trajectory_record_to_dict(py, r))
            .collect();

        let py_metadata = trajectory_metadata_to_dict(py, &metadata)?;

        Ok((py_records?, py_metadata))
    }

    /// Collect trajectories and return as numpy-compatible arrays.
    ///
    /// Args:
    ///     num_episodes: Number of episodes to collect
    ///
    /// Returns:
    ///     Dict with arrays:
    ///       - observations: [N, obs_dim] float32
    ///       - actions: [N, action_dim] float32
    ///       - rewards: [N] float32
    ///       - terminals: [N] bool
    ///       - episode_indices: [N] uint32
    ///       - metadata: dict
    fn collect_arrays(&self, py: Python<'_>, num_episodes: u32) -> PyResult<Py<PyDict>> {
        let (records, metadata) = self.inner.collect(num_episodes);

        let n = records.len();
        let obs_dim = metadata.obs_dim;
        let action_dim = metadata.action_dim;

        // Flatten arrays
        let mut observations: Vec<f32> = Vec::with_capacity(n * obs_dim);
        let mut actions: Vec<f32> = Vec::with_capacity(n * action_dim);
        let mut rewards: Vec<f32> = Vec::with_capacity(n);
        let mut terminals: Vec<bool> = Vec::with_capacity(n);
        let mut episode_indices: Vec<u32> = Vec::with_capacity(n);

        for record in &records {
            observations.extend(&record.obs_features);
            actions.extend(&record.action_target);
            rewards.push(record.reward);
            terminals.push(record.terminal);
            episode_indices.push(record.episode_idx);
        }

        let dict = PyDict::new_bound(py);
        dict.set_item("observations", observations)?;
        dict.set_item("actions", actions)?;
        dict.set_item("rewards", rewards)?;
        dict.set_item("terminals", terminals)?;
        dict.set_item("episode_indices", episode_indices)?;
        dict.set_item("obs_dim", obs_dim)?;
        dict.set_item("action_dim", action_dim)?;
        dict.set_item("num_samples", n)?;

        let py_metadata = trajectory_metadata_to_dict(py, &metadata)?;
        dict.set_item("metadata", py_metadata)?;

        Ok(dict.into())
    }
}

/// Python module definition.
#[pymodule]
fn paraphina_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Env>()?;
    m.add_class::<VecEnv>()?;
    m.add_class::<TrajectoryCollector>()?;
    m.add_function(wrap_pyfunction!(obs_version, m)?)?;
    m.add_function(wrap_pyfunction!(action_version, m)?)?;
    m.add_function(wrap_pyfunction!(trajectory_version, m)?)?;
    m.add_function(wrap_pyfunction!(default_num_venues, m)?)?;
    Ok(())
}
