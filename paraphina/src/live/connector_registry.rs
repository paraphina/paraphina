//! Live connector registry (CLI + audit helpers).

use std::collections::BTreeSet;

use clap::ValueEnum;

use crate::venues::{canonical_order_indices, VenueId, ROADMAP_B_VENUES};

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq, Hash)]
pub enum ConnectorArg {
    Mock,
    Extended,
    Hyperliquid,
    Aster,
    Lighter,
    Paradex,
    HyperliquidFixture,
}

impl ConnectorArg {
    pub fn as_str(&self) -> &'static str {
        match self {
            ConnectorArg::Mock => "mock",
            ConnectorArg::Extended => "extended",
            ConnectorArg::Hyperliquid => "hyperliquid",
            ConnectorArg::Aster => "aster",
            ConnectorArg::Lighter => "lighter",
            ConnectorArg::Paradex => "paradex",
            ConnectorArg::HyperliquidFixture => "hyperliquid_fixture",
        }
    }

    pub fn parse_env(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mock" => Some(ConnectorArg::Mock),
            "extended" => Some(ConnectorArg::Extended),
            "hyperliquid" | "hl" => Some(ConnectorArg::Hyperliquid),
            "aster" => Some(ConnectorArg::Aster),
            "lighter" => Some(ConnectorArg::Lighter),
            "paradex" => Some(ConnectorArg::Paradex),
            "hyperliquid_fixture" | "hl_fixture" | "fixture" => Some(ConnectorArg::HyperliquidFixture),
            _ => None,
        }
    }

    pub fn venue_id(&self) -> Option<VenueId> {
        match self {
            ConnectorArg::Extended => Some(VenueId::Extended),
            ConnectorArg::Hyperliquid | ConnectorArg::HyperliquidFixture => Some(VenueId::Hyperliquid),
            ConnectorArg::Aster => Some(VenueId::Aster),
            ConnectorArg::Lighter => Some(VenueId::Lighter),
            ConnectorArg::Paradex => Some(VenueId::Paradex),
            ConnectorArg::Mock => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConnectorSpec {
    pub connector: ConnectorArg,
    pub venue: Option<VenueId>,
    pub feature_flags: &'static [&'static str],
    pub supports_market_data: bool,
    pub supports_account: bool,
    pub supports_execution: bool,
    pub supports_cancel_all: bool,
    pub supports_shadow: bool,
    pub supports_paper: bool,
    pub supports_testnet: bool,
    pub supports_live: bool,
    pub supports_multi_connector: bool,
    pub notes: &'static str,
}

pub const CONNECTOR_SPECS: [ConnectorSpec; 7] = [
    ConnectorSpec {
        connector: ConnectorArg::Mock,
        venue: None,
        feature_flags: &[],
        supports_market_data: true,
        supports_account: false,
        supports_execution: false,
        supports_cancel_all: false,
        supports_shadow: true,
        supports_paper: true,
        supports_testnet: false,
        supports_live: false,
        supports_multi_connector: false,
        notes: "Synthetic snapshots only.",
    },
    ConnectorSpec {
        connector: ConnectorArg::Hyperliquid,
        venue: Some(VenueId::Hyperliquid),
        feature_flags: &["live_hyperliquid"],
        supports_market_data: true,
        supports_account: true,
        supports_execution: true,
        supports_cancel_all: false,
        supports_shadow: true,
        supports_paper: true,
        supports_testnet: true,
        supports_live: true,
        supports_multi_connector: false,
        notes: "REST exec; cancel_all not implemented.",
    },
    ConnectorSpec {
        connector: ConnectorArg::HyperliquidFixture,
        venue: Some(VenueId::Hyperliquid),
        feature_flags: &["live_hyperliquid"],
        supports_market_data: true,
        supports_account: true,
        supports_execution: false,
        supports_cancel_all: false,
        supports_shadow: true,
        supports_paper: true,
        supports_testnet: false,
        supports_live: false,
        supports_multi_connector: false,
        notes: "Offline fixture replay.",
    },
    ConnectorSpec {
        connector: ConnectorArg::Lighter,
        venue: Some(VenueId::Lighter),
        feature_flags: &["live_lighter"],
        supports_market_data: true,
        supports_account: true,
        supports_execution: true,
        supports_cancel_all: false,
        supports_shadow: true,
        supports_paper: true,
        supports_testnet: true,
        supports_live: true,
        supports_multi_connector: false,
        notes: "Paper mode suppresses exec.",
    },
    ConnectorSpec {
        connector: ConnectorArg::Extended,
        venue: Some(VenueId::Extended),
        feature_flags: &[],
        supports_market_data: false,
        supports_account: false,
        supports_execution: false,
        supports_cancel_all: false,
        supports_shadow: true,
        supports_paper: false,
        supports_testnet: false,
        supports_live: false,
        supports_multi_connector: false,
        notes: "Stub connector (not implemented).",
    },
    ConnectorSpec {
        connector: ConnectorArg::Aster,
        venue: Some(VenueId::Aster),
        feature_flags: &[],
        supports_market_data: false,
        supports_account: false,
        supports_execution: false,
        supports_cancel_all: false,
        supports_shadow: true,
        supports_paper: false,
        supports_testnet: false,
        supports_live: false,
        supports_multi_connector: false,
        notes: "Stub connector (not implemented).",
    },
    ConnectorSpec {
        connector: ConnectorArg::Paradex,
        venue: Some(VenueId::Paradex),
        feature_flags: &[],
        supports_market_data: false,
        supports_account: false,
        supports_execution: false,
        supports_cancel_all: false,
        supports_shadow: true,
        supports_paper: false,
        supports_testnet: false,
        supports_live: false,
        supports_multi_connector: false,
        notes: "Stub connector (not implemented).",
    },
];

pub fn roadmap_b_selectable_venues() -> Vec<VenueId> {
    let mut present = BTreeSet::new();
    for spec in &CONNECTOR_SPECS {
        if let Some(venue) = spec.venue {
            if ROADMAP_B_VENUES.contains(&venue) {
                present.insert(venue);
            }
        }
    }
    let canonical = canonical_order_indices(&ROADMAP_B_VENUES, |v| v.as_str());
    let mut ordered = Vec::with_capacity(present.len());
    for idx in canonical {
        let venue = ROADMAP_B_VENUES[idx];
        if present.contains(&venue) {
            ordered.push(venue);
        }
    }
    ordered
}

pub fn validate_roadmap_b_connector_coverage() -> Result<(), String> {
    let expected = ROADMAP_B_VENUES.len();
    let present = roadmap_b_selectable_venues().len();
    if present < expected {
        Err(format!(
            "roadmap_b_enabled=true but selectable_venues={present} expected={expected}"
        ))
    } else {
        Ok(())
    }
}

pub fn roadmap_b_gate_enabled() -> bool {
    std::env::var("PARAPHINA_ROADMAP_B")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}
