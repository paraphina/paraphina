//! Instrument normalization and TAO-equivalent conversions (feature-gated).

use serde::{Deserialize, Serialize};

use crate::config::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentSpec {
    pub venue_id: String,
    pub tick_size: f64,
    pub lot_size_tao: f64,
    pub size_step_tao: f64,
    pub min_notional_usd: f64,
    pub contract_multiplier: f64,
}

impl InstrumentSpec {
    pub fn from_config(cfg: &Config) -> Vec<Self> {
        cfg.venues
            .iter()
            .map(|v| InstrumentSpec {
                venue_id: v.id.clone(),
                tick_size: v.tick_size,
                lot_size_tao: v.lot_size_tao,
                size_step_tao: v.size_step_tao,
                min_notional_usd: v.min_notional_usd,
                contract_multiplier: 1.0,
            })
            .collect()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.tick_size <= 0.0 || !self.tick_size.is_finite() {
            return Err(format!("invalid tick_size for {}", self.venue_id));
        }
        if self.lot_size_tao <= 0.0 || !self.lot_size_tao.is_finite() {
            return Err(format!("invalid lot_size_tao for {}", self.venue_id));
        }
        if self.size_step_tao <= 0.0 || !self.size_step_tao.is_finite() {
            return Err(format!("invalid size_step_tao for {}", self.venue_id));
        }
        if self.min_notional_usd < 0.0 || !self.min_notional_usd.is_finite() {
            return Err(format!("invalid min_notional_usd for {}", self.venue_id));
        }
        if self.contract_multiplier <= 0.0 || !self.contract_multiplier.is_finite() {
            return Err(format!("invalid contract_multiplier for {}", self.venue_id));
        }
        Ok(())
    }

    pub fn tao_to_contracts(&self, size_tao: f64) -> f64 {
        size_tao / self.contract_multiplier
    }

    pub fn contracts_to_tao(&self, contracts: f64) -> f64 {
        contracts * self.contract_multiplier
    }

    pub fn round_price(&self, price: f64) -> f64 {
        let ticks = (price / self.tick_size).round();
        ticks * self.tick_size
    }

    pub fn round_size(&self, size_tao: f64) -> f64 {
        let steps = (size_tao / self.size_step_tao).floor();
        steps * self.size_step_tao
    }

    pub fn meets_min_notional(&self, size_tao: f64, price: f64) -> bool {
        size_tao * price >= self.min_notional_usd
    }
}

pub fn validate_specs(specs: &[InstrumentSpec]) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    for spec in specs {
        if let Err(err) = spec.validate() {
            errors.push(err);
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn conversion_roundtrip_is_consistent() {
        let cfg = Config::default();
        let spec = InstrumentSpec::from_config(&cfg).remove(0);
        let tao = 3.5;
        let contracts = spec.tao_to_contracts(tao);
        let back = spec.contracts_to_tao(contracts);
        assert!((tao - back).abs() < 1e-9);
    }

    #[test]
    fn rounding_and_min_notional() {
        let cfg = Config::default();
        let spec = InstrumentSpec::from_config(&cfg).remove(0);
        let rounded_price = spec.round_price(100.123);
        assert!((rounded_price / spec.tick_size).fract().abs() < 1e-9);
        let rounded_size = spec.round_size(spec.size_step_tao * 1.9);
        assert_eq!(rounded_size, spec.size_step_tao);
        let ok = spec.meets_min_notional(1.0, spec.min_notional_usd.max(1.0));
        assert!(ok);
    }
}
