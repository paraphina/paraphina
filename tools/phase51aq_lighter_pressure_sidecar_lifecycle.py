#!/usr/bin/env python3
"""Synthetic-only in-memory lifecycle for Phase 5.1 Lighter pressure facts.

This module proves the future sidecar state machine with synthetic fixtures
only. It has no CLI, performs no file or network I/O, writes no rows, observes
no runtime traffic, and cannot affect live/order/capital state. Partial facts
remain in memory and produce a packet only when every required synthetic fact
family is present for the exact same synthetic sample id.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

import phase51ap_lighter_pressure_sidecar_schema as packet_schema


FACT_TYPES = (
    "target_key_fact",
    "active_order_pressure_fact",
    "sendtx_pressure_fact",
    "request_pressure_fact",
    "event_time_fact",
)

COMMON_REQUIRED_FIELDS = {
    "schema_version",
    "fact_type",
    "sample_id",
    "venue_id",
    "gate_status",
    "fixture_provenance",
    "raw_identifier_redaction_status",
    "is_synthetic_fixture",
    "derived_from_real_evidence",
    "runtime_observation",
    "capture_enabled",
    "no_live_flag",
    *packet_schema.REQUIRED_FALSE_FLAGS,
}

TARGET_KEY_FIELDS = {
    "baseline_commit",
    "run_id",
    "canonical_group_id",
    "order_key",
    "public_market_key",
    "transform_version",
    "redaction_policy_version",
}

FACT_SPECIFIC_FIELDS = {
    "target_key_fact": TARGET_KEY_FIELDS,
    "active_order_pressure_fact": {
        "active_order_headroom_account",
        "active_order_headroom_market",
    },
    "sendtx_pressure_fact": {
        "sendtx_per_minute_limit",
        "sendtx_per_minute_remaining",
    },
    "request_pressure_fact": {
        "rest_requests_per_minute_limit",
        "rest_requests_per_minute_remaining",
        "weighted_requests_per_minute_limit",
        "weighted_requests_per_minute_remaining",
    },
    "event_time_fact": {
        "source_event_time_ms",
        "observed_at_ms",
    },
}

STRING_FIELDS = {
    "fact_type",
    "sample_id",
    "venue_id",
    "gate_status",
    "fixture_provenance",
    "raw_identifier_redaction_status",
    *TARGET_KEY_FIELDS,
}

BOOL_FIELDS = {
    "is_synthetic_fixture",
    "derived_from_real_evidence",
    "runtime_observation",
    "capture_enabled",
    "no_live_flag",
    *packet_schema.REQUIRED_FALSE_FLAGS,
}

INT_FIELDS = {
    "schema_version",
    "active_order_headroom_account",
    "active_order_headroom_market",
    "sendtx_per_minute_limit",
    "sendtx_per_minute_remaining",
    "rest_requests_per_minute_limit",
    "rest_requests_per_minute_remaining",
    "weighted_requests_per_minute_limit",
    "weighted_requests_per_minute_remaining",
    "source_event_time_ms",
    "observed_at_ms",
}


@dataclass(frozen=True)
class LifecycleResult:
    accepted: bool
    state: str
    reject_reasons: tuple[str, ...]
    packet: dict[str, Any] | None = None


class SyntheticLighterPressureSidecarLifecycle:
    def __init__(self, *, max_samples: int = 128):
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._max_samples = max_samples
        self._samples: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def ingest_fact(self, fact: Mapping[str, Any]) -> LifecycleResult:
        validation_reasons = validate_fact(fact)
        if validation_reasons:
            return LifecycleResult(False, "REJECTED", validation_reasons)

        sample_id = str(fact["sample_id"])
        fact_type = str(fact["fact_type"])
        fact_digest = _stable_hash(fact)
        sample = self._samples.get(sample_id)
        if sample is None:
            if len(self._samples) >= self._max_samples:
                return LifecycleResult(False, "REJECTED", ("sample capacity exceeded",))
            sample = {"facts": OrderedDict(), "digests": {}, "emitted": False}
            self._samples[sample_id] = sample
        self._samples.move_to_end(sample_id)

        existing_digest = sample["digests"].get(fact_type)
        if existing_digest is not None:
            if existing_digest == fact_digest:
                return LifecycleResult(True, "DUPLICATE_FACT_IGNORED", ())
            return LifecycleResult(False, "REJECTED", ("conflicting duplicate fact",))

        sample["facts"][fact_type] = dict(fact)
        sample["digests"][fact_type] = fact_digest

        missing = tuple(fact_type for fact_type in FACT_TYPES if fact_type not in sample["facts"])
        if missing:
            return LifecycleResult(True, "OBSERVED_PARTIAL_INTERNAL_ONLY", ())

        if sample["emitted"]:
            return LifecycleResult(True, "EMITTED_ALREADY", ())

        packet = _assemble_packet(sample["facts"])
        packet_result = packet_schema.validate_packet(packet)
        if not packet_result.accepted:
            return LifecycleResult(False, "REJECTED", packet_result.reject_reasons)
        sample["emitted"] = True
        return LifecycleResult(True, "EMITTABLE_SYNTHETIC_PACKET", (), packet_result.sanitized_packet)

    def pending_sample_count(self) -> int:
        return len(self._samples)


def validate_fact(fact: Mapping[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    if not isinstance(fact, Mapping):
        return ("fact must be a JSON object",)
    for key, value in fact.items():
        if not isinstance(key, str):
            reasons.append("field name must be string")
            continue
        _check_field_name(key, reasons)
        if isinstance(value, (dict, list, tuple)):
            reasons.append(f"nested payload field {key} is prohibited")
        if isinstance(value, str) and _value_looks_secret_or_raw_identifier(value):
            reasons.append(f"field {key} contains secret-shaped or raw-identifier-shaped value")
        _check_nested_safety(value, reasons)

    fact_type = fact.get("fact_type")
    if fact_type not in FACT_TYPES:
        reasons.append("fact_type is unsupported")
        allowed = set(COMMON_REQUIRED_FIELDS)
    else:
        allowed = COMMON_REQUIRED_FIELDS | FACT_SPECIFIC_FIELDS[str(fact_type)]
    for field in sorted(set(fact) - allowed):
        reasons.append(f"unsupported field {field}")
    for field in sorted(COMMON_REQUIRED_FIELDS - set(fact)):
        reasons.append(f"missing required field {field}")
    if fact_type in FACT_TYPES:
        for field in sorted(FACT_SPECIFIC_FIELDS[str(fact_type)] - set(fact)):
            if str(fact_type) == "request_pressure_fact" and field in {
                "rest_requests_per_minute_limit",
                "rest_requests_per_minute_remaining",
                "weighted_requests_per_minute_limit",
                "weighted_requests_per_minute_remaining",
            }:
                continue
            reasons.append(f"missing required field {field}")

    _require_exact(fact, "schema_version", packet_schema.SCHEMA_VERSION, reasons)
    _require_exact(fact, "venue_id", packet_schema.VENUE_ID, reasons)
    _require_exact(fact, "gate_status", "HOLD", reasons)
    _require_exact(fact, "fixture_provenance", packet_schema.FIXTURE_PROVENANCE, reasons)
    _require_exact(fact, "raw_identifier_redaction_status", packet_schema.REDACTION_STATUS, reasons)
    _require_exact(fact, "is_synthetic_fixture", True, reasons)
    _require_exact(fact, "derived_from_real_evidence", False, reasons)
    _require_exact(fact, "runtime_observation", False, reasons)
    _require_exact(fact, "capture_enabled", False, reasons)
    _require_exact(fact, "no_live_flag", True, reasons)
    for flag in sorted(packet_schema.REQUIRED_FALSE_FLAGS):
        _require_exact(fact, flag, False, reasons)
    for field in sorted(STRING_FIELDS & set(fact)):
        _require_nonempty_string(fact, field, reasons)
    for field in sorted(BOOL_FIELDS & set(fact)):
        _require_bool(fact, field, reasons)
    for field in sorted(INT_FIELDS & set(fact)):
        _require_nonnegative_int(fact, field, reasons)

    sample_id = fact.get("sample_id")
    if isinstance(sample_id, str) and not sample_id.startswith("synthetic-"):
        reasons.append("sample_id must be synthetic-only")
    if fact_type == "request_pressure_fact":
        _check_request_pressure_fact(fact, reasons)
    if fact_type == "sendtx_pressure_fact":
        _check_limit_pair(fact, "sendtx_per_minute_limit", "sendtx_per_minute_remaining", reasons)

    return tuple(dict.fromkeys(reasons))


def _assemble_packet(facts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    target = facts["target_key_fact"]
    active = facts["active_order_pressure_fact"]
    sendtx = facts["sendtx_pressure_fact"]
    request = facts["request_pressure_fact"]
    event_time = facts["event_time_fact"]
    source_hash = _stable_hash([facts[fact_type] for fact_type in FACT_TYPES])
    packet: dict[str, Any] = {
        "schema_version": packet_schema.SCHEMA_VERSION,
        "producer": packet_schema.PRODUCER,
        "target_type": packet_schema.TARGET_TYPE,
        "venue_id": packet_schema.VENUE_ID,
        "baseline_commit": target["baseline_commit"],
        "run_id": target["run_id"],
        "gate_status": "HOLD",
        "canonical_group_id": target["canonical_group_id"],
        "order_key": target["order_key"],
        "public_market_key": target["public_market_key"],
        "source_event_time_ms": event_time["source_event_time_ms"],
        "observed_at_ms": event_time["observed_at_ms"],
        "native_limit_event_time_status": packet_schema.EVENT_TIME_STATUS,
        "active_order_headroom_account": active["active_order_headroom_account"],
        "active_order_headroom_market": active["active_order_headroom_market"],
        "sendtx_per_minute_limit": sendtx["sendtx_per_minute_limit"],
        "sendtx_per_minute_remaining": sendtx["sendtx_per_minute_remaining"],
        "target_key_provenance_state": packet_schema.TARGET_KEY_PROVENANCE,
        "active_order_provenance_state": packet_schema.OBSERVED_PROVENANCE,
        "sendtx_provenance_state": packet_schema.OBSERVED_PROVENANCE,
        "request_pressure_provenance_state": packet_schema.OBSERVED_PROVENANCE,
        "pressure_packet_state": packet_schema.PACKET_STATE,
        "pressure_state": packet_schema.PRESSURE_COMPLETE,
        "raw_identifier_redaction_status": packet_schema.REDACTION_STATUS,
        "fixture_provenance": packet_schema.FIXTURE_PROVENANCE,
        "native_limit_pressure_source": packet_schema.PRESSURE_SOURCE,
        "transform_version": target["transform_version"],
        "redaction_policy_version": target["redaction_policy_version"],
        "source_count": len(FACT_TYPES),
        "completeness_flag": True,
        "is_synthetic_fixture": True,
        "derived_from_real_evidence": False,
        "runtime_observation": False,
        "capture_enabled": False,
        "gap_or_staleness_flag": False,
        "no_live_flag": True,
        "sanitized_source_record_sha256": source_hash,
    }
    for flag in packet_schema.REQUIRED_FALSE_FLAGS:
        packet[flag] = False
    for field in (
        "rest_requests_per_minute_limit",
        "rest_requests_per_minute_remaining",
        "weighted_requests_per_minute_limit",
        "weighted_requests_per_minute_remaining",
    ):
        if field in request:
            packet[field] = request[field]
    return packet


def _check_request_pressure_fact(fact: Mapping[str, Any], reasons: list[str]) -> None:
    complete_pairs = 0
    for limit_field, remaining_field in packet_schema.REQUEST_LIMIT_PAIRS:
        limit_present = limit_field in fact
        remaining_present = remaining_field in fact
        if limit_present != remaining_present:
            reasons.append(f"{limit_field}/{remaining_field} must be present as a complete pair")
            continue
        if limit_present and remaining_present:
            complete_pairs += 1
            _check_limit_pair(fact, limit_field, remaining_field, reasons)
    if complete_pairs == 0:
        reasons.append("REST-or-weighted request pressure pair is required")


def _check_limit_pair(fact: Mapping[str, Any], limit_field: str, remaining_field: str, reasons: list[str]) -> None:
    limit_value = fact.get(limit_field)
    remaining_value = fact.get(remaining_field)
    if not (_is_nonnegative_int(limit_value) and _is_nonnegative_int(remaining_value)):
        return
    if remaining_value > limit_value:
        reasons.append(f"{remaining_field} must be <= {limit_field}")


def _check_field_name(key: str, reasons: list[str]) -> None:
    normalized = key.replace("-", "_").lower()
    if key in packet_schema.RAW_IDENTIFIER_FIELDS:
        reasons.append(f"raw identifier field {key} is prohibited")
    if any(fragment in normalized for fragment in packet_schema.FORBIDDEN_KEY_FRAGMENTS):
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
    return packet_schema._value_looks_secret_or_raw_identifier(value)


def _require_exact(fact: Mapping[str, Any], field: str, expected: Any, reasons: list[str]) -> None:
    if field not in fact:
        return
    if isinstance(expected, bool):
        if fact[field] is not expected:
            reasons.append(f"{field} must be {expected!r}")
        return
    if fact[field] != expected:
        reasons.append(f"{field} must be {expected!r}")


def _require_nonempty_string(fact: Mapping[str, Any], field: str, reasons: list[str]) -> None:
    value = fact.get(field)
    if not isinstance(value, str) or not value.strip():
        reasons.append(f"{field} must be a non-empty string")
    elif _value_looks_secret_or_raw_identifier(value):
        reasons.append(f"{field} contains secret-shaped or raw-identifier-shaped value")


def _require_bool(fact: Mapping[str, Any], field: str, reasons: list[str]) -> None:
    if not isinstance(fact.get(field), bool):
        reasons.append(f"{field} must be a JSON boolean")


def _require_nonnegative_int(fact: Mapping[str, Any], field: str, reasons: list[str]) -> None:
    if not _is_nonnegative_int(fact.get(field)):
        reasons.append(f"{field} must be a non-negative integer")


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "FACT_TYPES",
    "LifecycleResult",
    "SyntheticLighterPressureSidecarLifecycle",
    "validate_fact",
]
