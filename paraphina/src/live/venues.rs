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

fn is_canonical_venue_order_subset(venue_ids: &[&str]) -> bool {
    let mut cursor = 0;
    for venue_id in venue_ids {
        let Some(offset) = CANONICAL_VENUE_ORDER[cursor..]
            .iter()
            .position(|expected| expected == venue_id)
        else {
            return false;
        };
        cursor += offset + 1;
    }
    true
}

pub fn warn_if_noncanonical_venue_order(venue_ids: &[&str], context: &str) {
    if !is_canonical_venue_order_subset(venue_ids) {
        eprintln!(
            "paraphina | warn=noncanonical_venue_order context={} expected={:?} actual={:?}",
            context, CANONICAL_VENUE_ORDER, venue_ids
        );
    }
}

#[cfg(test)]
mod tests {
    use super::is_canonical_venue_order_subset;

    #[test]
    fn canonical_subset_accepts_single_lighter() {
        assert!(is_canonical_venue_order_subset(&["lighter"]));
    }

    #[test]
    fn canonical_subset_accepts_ordered_subsets() {
        assert!(is_canonical_venue_order_subset(&[
            "hyperliquid",
            "lighter",
            "paradex",
        ]));
    }

    #[test]
    fn canonical_subset_rejects_reordered_or_unknown_venues() {
        assert!(!is_canonical_venue_order_subset(&[
            "lighter",
            "hyperliquid",
        ]));
        assert!(!is_canonical_venue_order_subset(&["unknown"]));
    }
}
