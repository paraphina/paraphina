use std::env;

/// Canonical Roadmap-B venue registry (stable order).
pub const ROADMAP_B_VENUES: [&str; 5] = ["extended", "hyperliquid", "aster", "lighter", "paradex"];

/// Stable venue ordering used across live telemetry + connector selection + gating.
pub const CANONICAL_VENUE_ORDER: [&str; 5] = ROADMAP_B_VENUES;

pub fn canonical_venue_ids() -> &'static [&'static str] {
    &CANONICAL_VENUE_ORDER
}

pub fn roadmap_b_enabled() -> bool {
    match env::var("PARAPHINA_ROADMAP_B") {
        Ok(val) => matches!(
            val.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "enabled"
        ),
        Err(_) => false,
    }
}

pub fn warn_if_noncanonical_venue_order(venue_ids: &[&str], context: &str) {
    if venue_ids != CANONICAL_VENUE_ORDER {
        eprintln!(
            "paraphina | warn=noncanonical_venue_order context={} expected={:?} actual={:?}",
            context, CANONICAL_VENUE_ORDER, venue_ids
        );
    }
}
