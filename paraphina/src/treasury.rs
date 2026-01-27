// src/treasury.rs
//
// Treasury guidance (slow-timescale) for operator actions.
// Guidance is telemetry-only and does not affect trading decisions.

use serde_json::Value as JsonValue;

use crate::state::GlobalState;

const TREASURY_GUIDANCE_CADENCE_TICKS: u64 = 60;
const TREASURY_MARGIN_USAGE_WARN: f64 = 0.80;
const TREASURY_MARGIN_USAGE_TARGET: f64 = 0.50;

#[derive(Debug, Clone, Default)]
struct TreasuryVenueAccumulator {
    sum_margin_balance_usd: f64,
    sum_margin_used_usd: f64,
    sum_abs_position_tao: f64,
    sum_funding_8h: f64,
    sum_funding_cost_usd_8h: f64,
}

#[derive(Debug, Clone)]
pub struct TreasuryGuidanceEngine {
    cadence_ticks: u64,
    samples: u64,
    venues: Vec<TreasuryVenueAccumulator>,
}

impl TreasuryGuidanceEngine {
    pub fn new(venue_count: usize) -> Self {
        Self {
            cadence_ticks: TREASURY_GUIDANCE_CADENCE_TICKS.max(1),
            samples: 0,
            venues: vec![TreasuryVenueAccumulator::default(); venue_count],
        }
    }

    pub fn update(&mut self, state: &GlobalState, fair_value: f64) {
        self.samples = self.samples.saturating_add(1);
        let fair = if fair_value.is_finite() && fair_value > 0.0 {
            fair_value
        } else {
            0.0
        };
        for (idx, venue) in state.venues.iter().enumerate() {
            let acc = &mut self.venues[idx];
            acc.sum_margin_balance_usd += venue.margin_balance_usd;
            acc.sum_margin_used_usd += venue.margin_used_usd;
            let abs_pos = venue.position_tao.abs();
            acc.sum_abs_position_tao += abs_pos;
            acc.sum_funding_8h += venue.funding_8h;
            acc.sum_funding_cost_usd_8h += venue.funding_8h * abs_pos * fair;
        }
    }

    pub fn build_guidance(&self, state: &GlobalState, tick: u64, now_ms: i64) -> JsonValue {
        let samples = self.samples.max(1) as f64;
        let emit = tick % self.cadence_ticks == 0;
        let mut venues = Vec::with_capacity(self.venues.len());
        let mut recommendations = Vec::new();

        for (idx, venue) in state.venues.iter().enumerate() {
            let acc = &self.venues[idx];
            let avg_margin_balance = acc.sum_margin_balance_usd / samples;
            let avg_margin_used = acc.sum_margin_used_usd / samples;
            let avg_margin_usage = if avg_margin_balance > 0.0 {
                avg_margin_used / avg_margin_balance
            } else {
                0.0
            };
            let avg_abs_position = acc.sum_abs_position_tao / samples;
            let avg_funding_8h = acc.sum_funding_8h / samples;
            let avg_funding_cost = acc.sum_funding_cost_usd_8h / samples;

            venues.push(serde_json::json!({
                "venue_index": idx as i64,
                "venue_id": venue.id.as_ref(),
                "avg_margin_usage": avg_margin_usage,
                "avg_abs_position_tao": avg_abs_position,
                "avg_funding_8h": avg_funding_8h,
                "avg_funding_cost_usd_8h": avg_funding_cost,
                "avg_margin_balance_usd": avg_margin_balance,
                "avg_margin_used_usd": avg_margin_used,
            }));

            if emit {
                if avg_margin_usage >= TREASURY_MARGIN_USAGE_WARN {
                    let suggested_margin = if avg_margin_balance > 0.0 {
                        let target_balance = avg_margin_used / TREASURY_MARGIN_USAGE_TARGET;
                        (target_balance - avg_margin_balance).max(0.0)
                    } else {
                        0.0
                    };
                    recommendations.push(serde_json::json!({
                        "kind": "increase_margin",
                        "venue_index": idx as i64,
                        "venue_id": venue.id.as_ref(),
                        "value": avg_margin_usage,
                        "threshold": TREASURY_MARGIN_USAGE_WARN,
                        "suggested_margin_usd": suggested_margin,
                        "suggested_position_tao": JsonValue::Null,
                        "message": format!(
                            "Increase margin on venue {} by {:.2} USD to support target usage {:.2}.",
                            venue.id, suggested_margin, TREASURY_MARGIN_USAGE_TARGET
                        ),
                    }));
                }

                if avg_funding_8h > 0.0 && avg_abs_position > 0.0 {
                    recommendations.push(serde_json::json!({
                        "kind": "reduce_exposure",
                        "venue_index": idx as i64,
                        "venue_id": venue.id.as_ref(),
                        "value": avg_funding_8h,
                        "threshold": 0.0,
                        "suggested_margin_usd": JsonValue::Null,
                        "suggested_position_tao": 0.0,
                        "message": format!(
                            "Reduce overall exposure to venue {} given persistent poor funding (avg 8h {:.6}).",
                            venue.id, avg_funding_8h
                        ),
                    }));
                }
            }
        }

        recommendations.sort_by(|a, b| {
            let kind = a.get("kind").and_then(|v| v.as_str()).unwrap_or("");
            let kind_b = b.get("kind").and_then(|v| v.as_str()).unwrap_or("");
            let venue = a.get("venue_index").and_then(|v| v.as_i64()).unwrap_or(-1);
            let venue_b = b.get("venue_index").and_then(|v| v.as_i64()).unwrap_or(-1);
            (venue, kind).cmp(&(venue_b, kind_b))
        });

        serde_json::json!({
            "cadence_ticks": self.cadence_ticks,
            "samples": self.samples,
            "emit": emit,
            "as_of_tick": tick,
            "as_of_ms": now_ms,
            "venues": venues,
            "recommendations": recommendations,
        })
    }
}
