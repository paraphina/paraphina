#![cfg(feature = "roadmap_b")]

use paraphina::live::connector_registry::{
    roadmap_b_selectable_venues, validate_roadmap_b_connector_coverage,
};
use paraphina::live::venues::ROADMAP_B_VENUES;

#[test]
fn roadmap_b_registry_has_five_venues() {
    let expected = ["extended", "hyperliquid", "aster", "lighter", "paradex"];
    assert_eq!(ROADMAP_B_VENUES, expected);
}

#[test]
fn roadmap_b_connector_selection_covers_all_venues() {
    validate_roadmap_b_connector_coverage().expect("roadmap-b coverage should be complete");
    let selectable = roadmap_b_selectable_venues()
        .into_iter()
        .collect::<Vec<_>>();
    assert_eq!(selectable, ROADMAP_B_VENUES);
}

#[test]
fn roadmap_b_connector_venue_ids_are_stable() {
    use paraphina::live::connector_registry::ConnectorArg;

    let ids = [
        ConnectorArg::Extended.venue_id(),
        ConnectorArg::Hyperliquid.venue_id(),
        ConnectorArg::Aster.venue_id(),
        ConnectorArg::Lighter.venue_id(),
        ConnectorArg::Paradex.venue_id(),
    ];
    assert_eq!(
        ids,
        [
            Some("extended"),
            Some("hyperliquid"),
            Some("aster"),
            Some("lighter"),
            Some("paradex")
        ]
    );
}
