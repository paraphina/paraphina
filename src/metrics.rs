// src/metrics.rs
//
// Small, dependency-free online metrics helpers for research harnesses.
// - OnlineStats: Welford running mean/variance + min/max.
// - DrawdownTracker: running peak and max drawdown on an equity curve.
//
// Intentionally simple + deterministic.

#[derive(Debug, Clone, Copy)]
pub struct OnlineStats {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

impl Default for OnlineStats {
    fn default() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
}

impl OnlineStats {
    /// Adds a sample if finite. Non-finite samples are ignored.
    pub fn add(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }

        self.n += 1;
        self.min = self.min.min(x);
        self.max = self.max.max(x);

        // Welford online variance.
        let delta = x - self.mean;
        self.mean += delta / (self.n as f64);
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn n(&self) -> u64 {
        self.n
    }

    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.mean
        }
    }

    pub fn min(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.min
        }
    }

    pub fn max(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.max
        }
    }

    /// Population variance (divide by n).
    pub fn variance_population(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.m2 / (self.n as f64)
        }
    }

    /// Sample variance (divide by n-1).
    pub fn variance_sample(&self) -> f64 {
        if self.n <= 1 {
            0.0
        } else {
            self.m2 / ((self.n as f64) - 1.0)
        }
    }

    pub fn stddev_population(&self) -> f64 {
        self.variance_population().sqrt()
    }

    pub fn stddev_sample(&self) -> f64 {
        self.variance_sample().sqrt()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DrawdownTracker {
    peak: f64,
    max_drawdown: f64,
    initialised: bool,
}

impl Default for DrawdownTracker {
    fn default() -> Self {
        Self {
            peak: 0.0,
            max_drawdown: 0.0,
            initialised: false,
        }
    }
}

impl DrawdownTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update with the current equity value. Tracks:
    /// - peak = max equity so far
    /// - max_drawdown = max(peak - equity)
    pub fn update(&mut self, equity: f64) {
        if !equity.is_finite() {
            return;
        }

        if !self.initialised {
            self.peak = equity;
            self.max_drawdown = 0.0;
            self.initialised = true;
            return;
        }

        if equity > self.peak {
            self.peak = equity;
        } else {
            let dd = (self.peak - equity).max(0.0);
            if dd > self.max_drawdown {
                self.max_drawdown = dd;
            }
        }
    }

    pub fn peak(&self) -> f64 {
        if !self.initialised {
            0.0
        } else {
            self.peak
        }
    }

    pub fn max_drawdown(&self) -> f64 {
        if !self.initialised {
            0.0
        } else {
            self.max_drawdown
        }
    }
}
