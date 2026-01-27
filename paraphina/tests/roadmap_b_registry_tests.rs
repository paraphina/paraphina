#![cfg(feature = "roadmap_b")]

use paraphina::connector_registry::{roadmap_b_selectable_venues, validate_roadmap_b_connector_coverage};
use paraphina::venues::{ROADMAP_B_VENUE_IDS, VenueId};

#[test]
fn roadmap_b_registry_has_five_venues() {
    let expected = ["extended", "hyperliquid", "aster", "lighter", "paradex"];
    assert_eq!(ROADMAP_B_VENUE_IDS, expected);
}

#[test]
fn roadmap_b_connector_selection_covers_all_venues() {
    validate_roadmap_b_connector_coverage().expect("roadmap-b coverage should be complete");
    let selectable = roadmap_b_selectable_venues()
        .into_iter()
        .map(|v| v.as_str())
        .collect::<Vec<_>>();
    assert_eq!(selectable, ROADMAP_B_VENUE_IDS);
}

#[test]
fn roadmap_b_venue_id_names_are_stable() {
    let ids = [
        VenueId::Extended,
        VenueId::Hyperliquid,
        VenueId::Aster,
        VenueId::Lighter,
        VenueId::Paradex,
    ];
    let names = ids.iter().map(|v| v.name()).collect::<Vec<_>>();
    assert_eq!(names, ["Extended", "Hyperliquid", "Aster", "Lighter", "Paradex"]);
}
