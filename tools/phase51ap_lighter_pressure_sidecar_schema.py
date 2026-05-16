#!/usr/bin/env python3
"""Synthetic-only Phase 5.1 Lighter pressure sidecar packet schema.

This module is a design-gate validator for synthetic fixtures only. It performs
no network access, reads no evidence files, writes no evidence rows, does not
call transaction-submission APIs, and does not observe runtime traffic. Its only
purpose is to prove the packet contract, provenance gates, redaction gates, and
completeness gates that a future source-owner sidecar would need to satisfy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


SCHEMA_VERSION = 1
PRODUCER = "Phase51LighterPressureSource"
TARGET_TYPE = "lighter_native_limit"
VENUE_ID = "lighter"
EVENT_TIME_STATUS = "SYNTHETIC_EVENT_TIME_ALIGNED"
TARGET_KEY_PROVENANCE = "SYNTHETIC_EXPLICIT_TARGET_KEY"
OBSERVED_PROVENANCE = "SYNTHETIC_EVENT_TIME_SOURCE"
PACKET_STATE = "SYNTHETIC_SANITIZED_EVENT_TIME_COMPLETE"
REDACTION_STATUS = "PASS"
FIXTURE_PROVENANCE = "SYNTHETIC_FIXTURE_ONLY"
PRESSURE_SOURCE = "SYNTHETIC_FIXTURE_PRESSURE_SOURCE"
MAX_OBSERVATION_LAG_MS = 60_000

REQUIRED_FALSE_FLAGS = {
    "approved_for_model_training",
    "approved_for_live",
    "approved_for_canary",
    "approved_for_capital_escalation",
    "approved_for_financial_claim",
    "admissible_for_model_training",
    "admissible_for_financial_claim",
    "admissible_for_ev_admission",
    "live_orders_allowed",
    "capital_change_allowed",
    "risk_limit_relaxation_allowed",
}

REQUIRED_STRINGS = {
    "producer",
    "target_type",
    "venue_id",
    "baseline_commit",
    "run_id",
    "canonical_group_id",
    "order_key",
    "public_market_key",
    "native_limit_event_time_status",
    "target_key_provenance_state",
    "active_order_provenance_state",
    "sendtx_provenance_state",
    "request_pressure_provenance_state",
    "pressure_packet_state",
    "raw_identifier_redaction_status",
    "fixture_provenance",
    "native_limit_pressure_source",
    "transform_version",
    "redaction_policy_version",
    "gate_status",
}

REQUIRED_INTS = {
    "source_event_time_ms",
    "observed_at_ms",
    "active_order_headroom_account",
    "active_order_headroom_market",
    "sendtx_per_minute_limit",
    "sendtx_per_minute_remaining",
    "source_count",
}

REQUEST_LIMIT_PAIRS = (
    ("rest_requests_per_minute_limit", "rest_requests_per_minute_remaining"),
    ("weighted_requests_per_minute_limit", "weighted_requests_per_minute_remaining"),
)

OPTIONAL_INTS = {field for pair in REQUEST_LIMIT_PAIRS for field in pair}

OPTIONAL_SHA256_FIELDS = {
    "sanitized_source_record_sha256",
    "sanitized_packet_sha256",
}

REQUIRED_BOOLS = {
    "no_live_flag",
    "completeness_flag",
    "is_synthetic_fixture",
    "derived_from_real_evidence",
    "runtime_observation",
    "capture_enabled",
    "gap_or_staleness_flag",
    *REQUIRED_FALSE_FLAGS,
}

ALLOWED_FIELDS = {
    "schema_version",
    *REQUIRED_STRINGS,
    *REQUIRED_INTS,
    *OPTIONAL_INTS,
    *OPTIONAL_SHA256_FIELDS,
    *REQUIRED_BOOLS,
}

RAW_IDENTIFIER_FIELDS = {
    "account_id",
    "account_index",
    "address",
    "ask_account_id",
    "ask_client_id",
    "ask_id",
    "bid_account_id",
    "bid_client_id",
    "bid_id",
    "client_id",
    "clientId",
    "client_order_id",
    "clientOrderId",
    "cloid",
    "fill_id",
    "fillId",
    "id",
    "oid",
    "order_id",
    "orderId",
    "raw_account_id",
    "raw_client_order_id",
    "raw_order_id",
    "tid",
    "trade_id",
    "tradeId",
    "tx_hash",
    "txHash",
    "venue_order_id",
    "wallet",
    "wallet_address",
}

FORBIDDEN_KEY_FRAGMENTS = (
    ".env",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "bearer",
    "jwt",
    "mnemonic",
    "passphrase",
    "password",
    "private_key",
    "secret",
    "session_token",
    "signature",
    "signed_payload",
    "signing_key",
    "token",
)

REJECTED_PROVENANCE_STATES = {
    "ACCOUNT_TIER",
    "CONFIGURED_CAP",
    "EMPTY_HEADER",
    "CURRENT_SNAPSHOT",
    "NONLIVE_TESTNET_OR_PAPER_CAPTURE",
    "OBSERVED_EVENT_TIME_SOURCE",
    "DERIVED_FROM_CLIENT_ORDER_ID",
    "DERIVED_FROM_DOCS",
    "DERIVED_FROM_LOCAL_ORDER_STATE",
    "DERIVED_FROM_PRICE_SIZE_SIDE",
    "DERIVED_FROM_PROXIMITY",
    "DERIVED_FROM_TIMING",
    "DOCS",
    "INFERRED",
    "LOCAL_COUNTER",
    "MANUAL",
    "MISSING",
    "PARTIAL",
    "PHASE51AA",
    "PHASE51B",
    "PHASE51C",
    "PHASE51Z",
    "READONLY_CAPTURE",
    "REAL_EVIDENCE",
    "RUNTIME_CONFIG",
    "RUNTIME_OBSERVATION",
    "STALE_SNAPSHOT",
}

SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)^bearer\s+\S+"),
    re.compile(r"(?i)^sk-[a-z0-9]"),
    re.compile(r"(?i)^pk_[a-z0-9]"),
    re.compile(r"^eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+$"),
)

HEX_0X_RE = re.compile(r"^0x[0-9a-fA-F]{32,}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reject_reasons: tuple[str, ...]
    sanitized_packet: dict[str, Any] | None = None


def packet_digest(packet: Mapping[str, Any]) -> str:
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_packet(packet: Mapping[str, Any]) -> ValidationResult:
    reasons: list[str] = []
    if not isinstance(packet, Mapping):
        return ValidationResult(False, ("packet must be a JSON object",), None)

    for key, value in packet.items():
        if not isinstance(key, str):
            reasons.append("field name must be string")
            continue
        _check_field_name(key, reasons)
        if key not in ALLOWED_FIELDS:
            reasons.append(f"unsupported field {key}")
        if isinstance(value, (dict, list, tuple)):
            reasons.append(f"nested payload field {key} is prohibited")
        if isinstance(value, str) and _value_looks_secret_or_raw_identifier(value):
            reasons.append(f"field {key} contains secret-shaped or raw-identifier-shaped value")
        _check_nested_safety(value, reasons)

    _require_exact(packet, "schema_version", SCHEMA_VERSION, reasons)
    _require_exact(packet, "producer", PRODUCER, reasons)
    _require_exact(packet, "target_type", TARGET_TYPE, reasons)
    _require_exact(packet, "venue_id", VENUE_ID, reasons)
    _require_exact(packet, "gate_status", "HOLD", reasons)
    _require_exact(packet, "native_limit_event_time_status", EVENT_TIME_STATUS, reasons)
    _require_exact(packet, "target_key_provenance_state", TARGET_KEY_PROVENANCE, reasons)
    _require_exact(packet, "active_order_provenance_state", OBSERVED_PROVENANCE, reasons)
    _require_exact(packet, "sendtx_provenance_state", OBSERVED_PROVENANCE, reasons)
    _require_exact(packet, "request_pressure_provenance_state", OBSERVED_PROVENANCE, reasons)
    _require_exact(packet, "pressure_packet_state", PACKET_STATE, reasons)
    _require_exact(packet, "raw_identifier_redaction_status", REDACTION_STATUS, reasons)
    _require_exact(packet, "fixture_provenance", FIXTURE_PROVENANCE, reasons)
    _require_exact(packet, "native_limit_pressure_source", PRESSURE_SOURCE, reasons)
    _require_exact(packet, "no_live_flag", True, reasons)
    _require_exact(packet, "completeness_flag", True, reasons)
    _require_exact(packet, "is_synthetic_fixture", True, reasons)
    _require_exact(packet, "derived_from_real_evidence", False, reasons)
    _require_exact(packet, "runtime_observation", False, reasons)
    _require_exact(packet, "capture_enabled", False, reasons)
    _require_exact(packet, "gap_or_staleness_flag", False, reasons)

    for flag in sorted(REQUIRED_FALSE_FLAGS):
        _require_exact(packet, flag, False, reasons)

    for field in sorted(REQUIRED_STRINGS):
        _require_nonempty_string(packet, field, reasons)

    for field in sorted(REQUIRED_BOOLS):
        _require_bool(packet, field, reasons)

    for field in sorted(REQUIRED_INTS):
        _require_nonnegative_int(packet, field, reasons)

    for field in sorted(OPTIONAL_INTS):
        if field in packet:
            _require_nonnegative_int(packet, field, reasons)

    for field in sorted(OPTIONAL_SHA256_FIELDS):
        if field in packet and not _valid_sha256(packet[field]):
            reasons.append(f"{field} must be a lowercase sanitized sha256")

    _check_provenance_values(packet, reasons)
    _check_event_time(packet, reasons)
    _check_pressure_pairs(packet, reasons)
    _check_source_count(packet, reasons)

    if reasons:
        return ValidationResult(False, tuple(dict.fromkeys(reasons)), None)

    return ValidationResult(True, (), dict(packet))


def _check_field_name(key: str, reasons: list[str]) -> None:
    normalized = key.replace("-", "_").lower()
    if key in RAW_IDENTIFIER_FIELDS:
        reasons.append(f"raw identifier field {key} is prohibited")
    if any(fragment in normalized for fragment in FORBIDDEN_KEY_FRAGMENTS):
        reasons.append(f"secret-shaped field {key} is prohibited")


def _check_nested_safety(value: Any, reasons: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str):
                _check_field_name(key, reasons)
            _check_nested_safety(child, reasons)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_nested_safety(child, reasons)
    elif isinstance(value, str) and _value_looks_secret_or_raw_identifier(value):
        reasons.append("nested value contains secret-shaped or raw-identifier-shaped value")


def _value_looks_secret_or_raw_identifier(value: str) -> bool:
    stripped = value.strip()
    if ".env" in stripped.lower():
        return True
    if HEX_0X_RE.match(stripped):
        return True
    return any(pattern.match(stripped) for pattern in SECRET_VALUE_PATTERNS)


def _require_exact(packet: Mapping[str, Any], field: str, expected: Any, reasons: list[str]) -> None:
    if field not in packet:
        reasons.append(f"missing required field {field}")
        return
    if isinstance(expected, bool):
        if packet[field] is not expected:
            reasons.append(f"{field} must be {expected!r}")
        return
    if packet[field] != expected:
        reasons.append(f"{field} must be {expected!r}")


def _require_nonempty_string(packet: Mapping[str, Any], field: str, reasons: list[str]) -> None:
    value = packet.get(field)
    if not isinstance(value, str) or not value.strip():
        reasons.append(f"{field} must be a non-empty string")


def _require_bool(packet: Mapping[str, Any], field: str, reasons: list[str]) -> None:
    if field in packet and not isinstance(packet[field], bool):
        reasons.append(f"{field} must be a JSON boolean")


def _require_nonnegative_int(packet: Mapping[str, Any], field: str, reasons: list[str]) -> None:
    value = packet.get(field)
    if not _is_nonnegative_int(value):
        reasons.append(f"{field} must be a non-negative integer")


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _check_provenance_values(packet: Mapping[str, Any], reasons: list[str]) -> None:
    for field in (
        "target_key_provenance_state",
        "active_order_provenance_state",
        "sendtx_provenance_state",
        "request_pressure_provenance_state",
    ):
        value = packet.get(field)
        if isinstance(value, str) and value in REJECTED_PROVENANCE_STATES:
            reasons.append(f"{field} has rejected provenance {value}")


def _check_event_time(packet: Mapping[str, Any], reasons: list[str]) -> None:
    source_event_time_ms = packet.get("source_event_time_ms")
    observed_at_ms = packet.get("observed_at_ms")
    if not (_is_nonnegative_int(source_event_time_ms) and _is_nonnegative_int(observed_at_ms)):
        return
    if observed_at_ms < source_event_time_ms:
        reasons.append("observed_at_ms must not precede source_event_time_ms")
        return
    if observed_at_ms - source_event_time_ms > MAX_OBSERVATION_LAG_MS:
        reasons.append("event-time observation lag exceeds allowed synthetic sidecar bound")


def _check_pressure_pairs(packet: Mapping[str, Any], reasons: list[str]) -> None:
    _check_limit_pair(packet, "sendtx_per_minute_limit", "sendtx_per_minute_remaining", reasons)
    complete_request_pairs = 0
    for limit_field, remaining_field in REQUEST_LIMIT_PAIRS:
        limit_present = limit_field in packet
        remaining_present = remaining_field in packet
        if limit_present != remaining_present:
            reasons.append(f"{limit_field}/{remaining_field} must be present as a complete pair")
            continue
        if limit_present and remaining_present:
            complete_request_pairs += 1
            _check_limit_pair(packet, limit_field, remaining_field, reasons)
    if complete_request_pairs == 0:
        reasons.append("REST-or-weighted request pressure pair is required")


def _check_limit_pair(packet: Mapping[str, Any], limit_field: str, remaining_field: str, reasons: list[str]) -> None:
    limit_value = packet.get(limit_field)
    remaining_value = packet.get(remaining_field)
    if not (_is_nonnegative_int(limit_value) and _is_nonnegative_int(remaining_value)):
        return
    if remaining_value > limit_value:
        reasons.append(f"{remaining_field} must be <= {limit_field}")


def _check_source_count(packet: Mapping[str, Any], reasons: list[str]) -> None:
    source_count = packet.get("source_count")
    if _is_nonnegative_int(source_count) and source_count == 0:
        reasons.append("source_count must be positive")
    if isinstance(source_count, float) and not math.isfinite(source_count):
        reasons.append("source_count must be finite")


__all__ = [
    "EVENT_TIME_STATUS",
    "FIXTURE_PROVENANCE",
    "OBSERVED_PROVENANCE",
    "PACKET_STATE",
    "PRESSURE_SOURCE",
    "PRODUCER",
    "REDACTION_STATUS",
    "SCHEMA_VERSION",
    "TARGET_KEY_PROVENANCE",
    "TARGET_TYPE",
    "VENUE_ID",
    "ValidationResult",
    "packet_digest",
    "validate_packet",
]
