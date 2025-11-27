// src/toxicity.rs
//
// Per-venue toxicity + venue health scaffold.
// This will later be replaced by the full Section 10 whitepaper logic,
// but for now it just keeps toxicity in [0,1] and leaves status alone.

use crate::state::GlobalState;

pub fn update_toxicity(state: &mut GlobalState) {
    // Placeholder: clamp any existing toxicity into [0, 1].
    // Later we will update this based on recent fills, rejects, etc.
    for venue in &mut state.venues {
        if !venue.toxicity.is_finite() {
            venue.toxicity = 0.0;
        }

        venue.toxicity = venue.toxicity.clamp(0.0, 1.0);
        // `venue.status` stays as-is for now (usually Healthy).
    }
}

/// Entry point used by the engine / main loop.
///
/// Later this will also update `venue.status` based on toxicity,
/// outages, error rates, etc., but for now it just calls
/// the basic toxicity updater.
pub fn update_toxicity_and_health(state: &mut GlobalState) {
    update_toxicity(state);
}
