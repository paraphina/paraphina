#!/usr/bin/env python3
"""
profile_centres.py

Canonical per-profile "centre" knobs for Paraphina research harnesses.

These are the single source of truth for the coarse risk presets
(aggressive / balanced / conservative) used by:

  - exp02_profile_risk_grid.py
  - exp03 / exp04 stress-search style experiments
  - exp05 / exp06 telemetry validation + safe region
  - any future RL / hypersearch that wants a sane starting point.

Values are derived from:

  - wm04 world-model constant-policy optimisation (exp07)
  - exp08_profile_validation presets

and are mirrored in src/config.rs::Config::for_profile in Rust.
"""

from dataclasses import dataclass


@dataclass
class ProfileCentre:
    band_base: float        # TAO, hedge_band_base
    mm_size_eta: float      # η in J(Q)=eQ - 0.5 η Q²
    vol_ref: float          # reference volatility σ_ref
    daily_loss_limit: float # USD
    init_q_tao: float       # TAO
    vol_scale: float        # scenario vol multiplier (for sims only)


# ---------------------------------------------------------------------
# World-model tuned structural knobs (same for all three profiles)
# ---------------------------------------------------------------------

_BAND_BASE = 5.625
_MM_SIZE_ETA = 0.10
_VOL_REF = 0.028125
_VOL_SCALE = 1.5
_INIT_Q_TAO = 0.0

PROFILE_CENTRES = {
    "aggressive": ProfileCentre(
        band_base=_BAND_BASE,
        mm_size_eta=_MM_SIZE_ETA,
        vol_ref=_VOL_REF,
        daily_loss_limit=2000.0,
        init_q_tao=_INIT_Q_TAO,
        vol_scale=_VOL_SCALE,
    ),
    "balanced": ProfileCentre(
        band_base=_BAND_BASE,
        mm_size_eta=_MM_SIZE_ETA,
        vol_ref=_VOL_REF,
        daily_loss_limit=5000.0,
        init_q_tao=_INIT_Q_TAO,
        vol_scale=_VOL_SCALE,
    ),
    "conservative": ProfileCentre(
        band_base=_BAND_BASE,
        mm_size_eta=_MM_SIZE_ETA,
        vol_ref=_VOL_REF,
        daily_loss_limit=750.0,
        init_q_tao=_INIT_Q_TAO,
        vol_scale=_VOL_SCALE,
    ),
}


def get_profile_centre(name: str) -> ProfileCentre:
    """Return the canonical centre for a given profile name."""
    key = name.lower()
    if key not in PROFILE_CENTRES:
        raise KeyError(f"Unknown profile centre: {name!r}")
    return PROFILE_CENTRES[key]


# ---------------------------------------------------------------------
# Backwards-compat shims for older experiments
# ---------------------------------------------------------------------

# Older scripts expect a dict called PROFILES – keep it pointing at the
# canonical centres so they continue to work.
PROFILES = PROFILE_CENTRES


def load_profile_centres_from_exp02(*_args, **_kwargs):
    """
    Legacy helper used by older expXX scripts.

    Historically this loaded best-profile centres from exp02 results.
    We now treat PROFILE_CENTRES as the single source of truth, so this
    shim simply returns that dict, ignoring any arguments.
    """
    return PROFILE_CENTRES
