//! Canonical venue registry and ordering helpers.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VenueId {
    Extended,
    Hyperliquid,
    Aster,
    Lighter,
    Paradex,
}

impl VenueId {
    pub const fn as_str(&self) -> &'static str {
        match self {
            VenueId::Extended => "extended",
            VenueId::Hyperliquid => "hyperliquid",
            VenueId::Aster => "aster",
            VenueId::Lighter => "lighter",
            VenueId::Paradex => "paradex",
        }
    }

    pub const fn name(&self) -> &'static str {
        match self {
            VenueId::Extended => "Extended",
            VenueId::Hyperliquid => "Hyperliquid",
            VenueId::Aster => "Aster",
            VenueId::Lighter => "Lighter",
            VenueId::Paradex => "Paradex",
        }
    }
}

pub const CANONICAL_VENUES: [VenueId; 5] = [
    VenueId::Extended,
    VenueId::Hyperliquid,
    VenueId::Aster,
    VenueId::Lighter,
    VenueId::Paradex,
];

pub const CANONICAL_VENUE_IDS: [&str; 5] = [
    "extended",
    "hyperliquid",
    "aster",
    "lighter",
    "paradex",
];

pub const ROADMAP_B_VENUES: [VenueId; 5] = CANONICAL_VENUES;
pub const ROADMAP_B_VENUE_IDS: [&str; 5] = CANONICAL_VENUE_IDS;

pub fn canonical_venue_count(config_len: usize) -> usize {
    if config_len == CANONICAL_VENUE_IDS.len() {
        CANONICAL_VENUE_IDS.len()
    } else {
        config_len
    }
}

pub fn canonical_order_indices<T, F>(items: &[T], id_fn: F) -> Vec<usize>
where
    F: Fn(&T) -> &str,
{
    if items.is_empty() {
        return Vec::new();
    }
    let mut indices = Vec::with_capacity(items.len());
    let mut seen = vec![false; items.len()];
    for id in CANONICAL_VENUE_IDS {
        if let Some((idx, _)) = items
            .iter()
            .enumerate()
            .find(|(_, item)| id_fn(item).eq_ignore_ascii_case(id))
        {
            indices.push(idx);
            seen[idx] = true;
        }
    }
    for idx in 0..items.len() {
        if !seen[idx] {
            indices.push(idx);
        }
    }
    indices
}
