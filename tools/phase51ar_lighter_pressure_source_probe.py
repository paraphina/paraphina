#!/usr/bin/env python3
"""Phase 5.1AR sanitized Lighter pressure source-availability probe.

This probe answers one narrow question: does the authenticated read-only
Lighter accountLimits surface expose the pressure dimensions Phase 5.1 needs?

It never submits, cancels, modifies, replaces, or signs orders. It never calls
sendTx or sendTxBatch. It does not load .env files or private keys. If HTTP mode
is used, the only secret-shaped input it consumes is an auth token already
present in the process environment, and the value is never logged or persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51ar_lighter_pressure_source_probe"
DEFAULT_BASE_URL = "https://mainnet.zklighter.elliot.ai"
ACCOUNT_LIMITS_ENDPOINT = "/api/v1/accountLimits"

SENSITIVE_HEADER_FRAGMENTS = (
    "auth",
    "cookie",
    "credential",
    "jwt",
    "password",
    "secret",
    "session",
    "set-cookie",
    "signature",
    "token",
)
RESPONSE_HEADER_FRAGMENTS = (
    "cooldown",
    "quota",
    "rate",
    "remaining",
    "retry-after",
    "limit",
)

FORBIDDEN_RAW_VALUE_KEYS = {
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "client_order_id",
    "clientorderid",
    "fill_id",
    "fillid",
    "jwt",
    "order_id",
    "orderid",
    "private_key",
    "privatekey",
    "secret",
    "signature",
    "token",
    "trade_id",
    "tradeid",
    "tx_hash",
    "txhash",
}

PRESSURE_FIELD_KEYS: dict[str, tuple[str, ...]] = {
    "active_order_headroom_account": (
        "activeorderheadroomaccount",
        "activeordersheadroomaccount",
        "activeorderheadroom",
        "activeordersheadroom",
    ),
    "active_orders_per_account_limit": (
        "activeordersperaccountlimit",
        "activeordersperaccount",
    ),
    "active_orders_per_market_limit": (
        "activeorderspermarketlimit",
        "activeorderspermarket",
    ),
    "pending_orders_per_account_limit": (
        "pendingordersperaccountlimit",
        "pendingordersperaccount",
    ),
    "pending_orders_per_market_limit": (
        "pendingorderspermarketlimit",
        "pendingorderspermarket",
    ),
    "sendtx_per_minute_limit": (
        "sendtxperminutelimit",
        "sendtxperminute",
        "sendtxlimit",
        "sendtxbatchlimit",
        "sendtxsendtxbatchperminutelimit",
    ),
    "sendtx_per_minute_remaining": (
        "sendtxperminuteremaining",
        "sendtxremaining",
        "sendtxbatchremaining",
        "sendtxsendtxbatchperminuteremaining",
    ),
    "rest_requests_per_minute_limit": (
        "restrequestsperminutelimit",
        "restrequestslimit",
        "standardrequestsperminute",
        "standardrequestsperminutelimit",
        "restlimit",
    ),
    "rest_requests_per_minute_remaining": (
        "restrequestsperminuteremaining",
        "restrequestsremaining",
        "standardrequestsperminuteremaining",
        "restremaining",
    ),
    "weighted_requests_per_minute_limit": (
        "weightedrequestsperminutelimit",
        "premiumweightedrequests",
        "premiumweightedrequestsperminute",
        "premiumweightedrequestsperminutelimit",
        "weightedlimit",
    ),
    "weighted_requests_per_minute_remaining": (
        "weightedrequestsperminuteremaining",
        "premiumweightedrequestsremaining",
        "premiumweightedrequestsperminuteremaining",
        "weightedremaining",
    ),
    "volume_quota_remaining": (
        "volumequotaremaining",
        "quotaremaining",
    ),
}

OUTPUT_PRESSURE_FIELDS = tuple(PRESSURE_FIELD_KEYS.keys())


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"expected object or array JSON in {path}")
    return payload


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _sanitize_response_headers(headers: dict[str, Any]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for raw_key, raw_value in headers.items():
        key = str(raw_key)
        normalized = key.lower()
        if any(fragment in normalized for fragment in SENSITIVE_HEADER_FRAGMENTS):
            continue
        if not any(fragment in normalized for fragment in RESPONSE_HEADER_FRAGMENTS):
            continue
        sanitized[key] = str(raw_value)
    return dict(sorted(sanitized.items()))


def _http_get_account_limits(
    *,
    base_url: str,
    account_index: int,
    auth_token: str,
    timeout_s: float,
) -> tuple[dict[str, Any] | list[Any], dict[str, str]]:
    query = urllib.parse.urlencode({"account_index": account_index})
    url = f"{base_url.rstrip('/')}{ACCOUNT_LIMITS_ENDPOINT}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": auth_token,
            "User-Agent": "paraphina-phase51ar-readonly/1",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
        headers = _sanitize_response_headers(dict(response.headers.items()))
    if not isinstance(payload, (dict, list)):
        raise ValueError("accountLimits returned non-object/non-array JSON")
    return payload, headers


def _find_first_value(payload: Any, candidate_keys: tuple[str, ...]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if _normalize_key(key) in candidate_keys:
                return value
        for value in payload.values():
            found = _find_first_value(value, candidate_keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_first_value(item, candidate_keys)
            if found is not None:
                return found
    return None


def _field_paths(payload: Any, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_path = f"{prefix}.{key}" if prefix else str(key)
            paths.append(next_path)
            paths.extend(_field_paths(value, prefix=next_path))
    elif isinstance(payload, list):
        for item in payload[:3]:
            paths.extend(_field_paths(item, prefix=f"{prefix}[]"))
    return paths


def _header_numeric_candidates(headers: dict[str, str]) -> dict[str, int | None]:
    candidates: dict[str, int | None] = {}
    for key, value in headers.items():
        normalized = _normalize_key(key)
        numeric = _as_int(value)
        if numeric is None:
            continue
        has_remaining = "remaining" in normalized
        has_limit = "limit" in normalized and "remaining" not in normalized
        if "sendtx" in normalized or "sendtxbatch" in normalized:
            if has_remaining:
                candidates.setdefault("sendtx_per_minute_remaining", numeric)
            elif has_limit:
                candidates.setdefault("sendtx_per_minute_limit", numeric)
        elif "weighted" in normalized or "weight" in normalized:
            if has_remaining:
                candidates.setdefault("weighted_requests_per_minute_remaining", numeric)
            elif has_limit:
                candidates.setdefault("weighted_requests_per_minute_limit", numeric)
        elif "rest" in normalized or "request" in normalized or "rate" in normalized:
            if has_remaining:
                candidates.setdefault("rest_requests_per_minute_remaining", numeric)
            elif has_limit:
                candidates.setdefault("rest_requests_per_minute_limit", numeric)
    return candidates


def _extract_pressure(payload: Any, headers: dict[str, str]) -> dict[str, Any]:
    body_values: dict[str, int | float | None] = {}
    for output_field, candidate_keys in PRESSURE_FIELD_KEYS.items():
        raw_value = _find_first_value(payload, candidate_keys)
        if output_field == "volume_quota_remaining":
            body_values[output_field] = _as_float(raw_value)
        else:
            body_values[output_field] = _as_int(raw_value)

    header_values = _header_numeric_candidates(headers)
    observed_values: dict[str, int | float | None] = {}
    value_sources: dict[str, str | None] = {}
    for field in OUTPUT_PRESSURE_FIELDS:
        if body_values.get(field) is not None:
            observed_values[field] = body_values[field]
            value_sources[field] = "accountLimits_body"
        elif header_values.get(field) is not None:
            observed_values[field] = header_values[field]
            value_sources[field] = "accountLimits_response_header"
        else:
            observed_values[field] = None
            value_sources[field] = None

    has_sendtx_pair = (
        observed_values["sendtx_per_minute_limit"] is not None
        and observed_values["sendtx_per_minute_remaining"] is not None
    )
    has_rest_pair = (
        observed_values["rest_requests_per_minute_limit"] is not None
        and observed_values["rest_requests_per_minute_remaining"] is not None
    )
    has_weighted_pair = (
        observed_values["weighted_requests_per_minute_limit"] is not None
        and observed_values["weighted_requests_per_minute_remaining"] is not None
    )
    has_active_headroom = observed_values["active_order_headroom_account"] is not None
    pressure_dimensions_complete = has_active_headroom and has_sendtx_pair and (has_rest_pair or has_weighted_pair)
    missing = [field for field, value in observed_values.items() if value is None]
    required_missing: list[str] = []
    if not has_active_headroom:
        required_missing.append("active_order_headroom_account")
    if not has_sendtx_pair:
        required_missing.append("sendtx_per_minute_limit/sendtx_per_minute_remaining")
    if not (has_rest_pair or has_weighted_pair):
        required_missing.append(
            "rest_requests_per_minute_limit/rest_requests_per_minute_remaining "
            "or weighted_requests_per_minute_limit/weighted_requests_per_minute_remaining"
        )

    return {
        "observed_values": observed_values,
        "value_sources": value_sources,
        "field_presence": {field: value is not None for field, value in observed_values.items()},
        "missing_fields": missing,
        "required_missing_dimensions": required_missing,
        "has_sendtx_pair": has_sendtx_pair,
        "has_rest_pair": has_rest_pair,
        "has_weighted_pair": has_weighted_pair,
        "has_active_order_headroom_account": has_active_headroom,
        "pressure_dimensions_complete_from_account_limits_surface": pressure_dimensions_complete,
        "event_time_aligned": False,
        "native_limit_event_time_status": "READONLY_ACCOUNT_LIMITS_SOURCE_AVAILABILITY_ONLY",
    }


def _safe_payload_fingerprint(payload: Any) -> dict[str, Any]:
    paths = sorted(set(_field_paths(payload)))
    normalized_paths = [_normalize_key(path.rsplit(".", 1)[-1]) for path in paths]
    raw_or_secret_key_count = sum(1 for key in normalized_paths if key in FORBIDDEN_RAW_VALUE_KEYS)
    top_level_keys = sorted(str(key) for key in payload.keys()) if isinstance(payload, dict) else []
    return {
        "payload_sha256": _stable_hash(payload),
        "top_level_keys": top_level_keys,
        "field_paths": paths[:500],
        "field_path_count": len(paths),
        "raw_or_secret_shaped_key_count": raw_or_secret_key_count,
        "raw_payload_persisted": False,
    }


def _base_summary(
    *,
    run_id: str,
    created_utc: str,
    base_url: str,
    account_index_present: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tool": "phase51ar_lighter_pressure_source_probe",
        "run_id": run_id,
        "created_utc": created_utc,
        "venue_id": "lighter",
        "source_endpoint": f"{base_url.rstrip('/')}{ACCOUNT_LIMITS_ENDPOINT}",
        "account_index_present": account_index_present,
        "no_live_flag": True,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "sendtx_allowed": False,
        "sendtxbatch_allowed": False,
        "orders_submitted": False,
        "orders_cancelled": False,
        "orders_modified": False,
        "secrets_persisted": False,
        "raw_identifiers_persisted": False,
        "phase51_validators_run": False,
        "blocker_cleared": False,
    }


def _write_probe_artifacts(out_dir: Path, summary: dict[str, Any], pressure: dict[str, Any] | None) -> None:
    _write_json(out_dir / "lighter_pressure_source_probe_summary.json", summary)
    if pressure is not None:
        _write_json(out_dir / "account_limits_pressure.sanitized.json", pressure)
    manifest = {
        "schema_version": 1,
        "run_id": summary["run_id"],
        "created_utc": summary["created_utc"],
        "artifacts": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
        "no_live_flag": True,
        "blocker_cleared": False,
        "phase51_validators_run": False,
    }
    _write_json(out_dir / "manifest.json", manifest)


def run(
    *,
    output_root: Path,
    run_id: str | None,
    account_limits_json: Path | None,
    fetch_readonly: bool,
    base_url: str,
    account_index: int | None,
    auth_token_env: str,
    timeout_s: float,
) -> Path:
    if output_root.is_symlink():
        raise ValueError(f"output root must not be a symlink: {output_root}")
    if not fetch_readonly and account_limits_json is None:
        raise ValueError("provide --account-limits-json or --fetch-readonly")
    if fetch_readonly and account_limits_json is not None:
        raise ValueError("--fetch-readonly and --account-limits-json are mutually exclusive")
    if timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")

    resolved_run_id = run_id or f"PHASE51AR-LIGHTER-PRESSURE-SOURCE-PROBE-{_utc_stamp()}"
    out_dir = output_root / resolved_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    created_utc = _utc_now()

    resolved_account_index = account_index
    if resolved_account_index is None:
        raw_account_index = os.environ.get("LIGHTER_ACCOUNT_INDEX", "").strip()
        resolved_account_index = int(raw_account_index) if raw_account_index else None

    summary = _base_summary(
        run_id=resolved_run_id,
        created_utc=created_utc,
        base_url=base_url,
        account_index_present=resolved_account_index is not None,
    )

    payload: dict[str, Any] | list[Any] | None = None
    headers: dict[str, str] = {}
    if account_limits_json is not None:
        payload = _load_json(account_limits_json)
        summary["fetch_status"] = "OBSERVED_FROM_CAPTURED_JSON"
        summary["source_file_sha256"] = _stable_hash(payload)
    elif fetch_readonly:
        if resolved_account_index is None:
            summary["fetch_status"] = "NOT_RUN_MISSING_ACCOUNT_INDEX"
            summary["next_required_action"] = "set LIGHTER_ACCOUNT_INDEX or pass --account-index"
            _write_probe_artifacts(out_dir, summary, None)
            return out_dir
        auth_token = os.environ.get(auth_token_env, "").strip()
        if not auth_token:
            summary["fetch_status"] = "NOT_RUN_MISSING_AUTH_TOKEN"
            summary["auth_token_env_present"] = False
            summary["next_required_action"] = f"provide {auth_token_env} in process env without printing it"
            _write_probe_artifacts(out_dir, summary, None)
            return out_dir
        summary["auth_token_env_present"] = True
        try:
            payload, headers = _http_get_account_limits(
                base_url=base_url,
                account_index=resolved_account_index,
                auth_token=auth_token,
                timeout_s=timeout_s,
            )
            summary["fetch_status"] = "OBSERVED_FROM_READONLY_HTTP"
        except Exception as exc:  # noqa: BLE001 - sanitized source-availability result
            summary["fetch_status"] = "ERROR"
            summary["error_type"] = type(exc).__name__
            summary["error_status_code"] = getattr(exc, "code", None)
            summary["error_message_sha256"] = _stable_hash(str(exc))
            summary["next_required_action"] = "resolve authenticated read-only accountLimits fetch"
            _write_probe_artifacts(out_dir, summary, None)
            return out_dir

    if payload is None:
        raise ValueError("internal error: missing payload")

    pressure = _extract_pressure(payload, headers)
    payload_fingerprint = _safe_payload_fingerprint(payload)
    header_artifact = {
        "schema_version": 1,
        "response_header_names": sorted(headers.keys()),
        "response_headers": headers,
        "response_headers_sha256": _stable_hash(headers) if headers else None,
    }
    _write_json(out_dir / "response_headers.sanitized.json", header_artifact)

    summary.update({
        "response_header_names": header_artifact["response_header_names"],
        "payload_fingerprint": payload_fingerprint,
        "pressure_field_presence": pressure["field_presence"],
        "required_missing_dimensions": pressure["required_missing_dimensions"],
        "pressure_dimensions_complete_from_account_limits_surface": pressure[
            "pressure_dimensions_complete_from_account_limits_surface"
        ],
        "native_limit_event_time_status": pressure["native_limit_event_time_status"],
        "blocker_cleared": False,
        "safe_to_run_phase51_validators": False,
        "recommended_next_path": (
            "event_time_alignment_observation_window"
            if pressure["pressure_dimensions_complete_from_account_limits_surface"]
            else "passive_sendtx_sendtxbatch_observation_tap"
        ),
    })
    _write_probe_artifacts(out_dir, summary, pressure)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--account-limits-json", type=Path, default=None)
    parser.add_argument("--fetch-readonly", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--account-index", type=int, default=None)
    parser.add_argument("--auth-token-env", default="LIGHTER_AUTH_TOKEN")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()
    try:
        out_dir = run(
            output_root=args.output_root,
            run_id=args.run_id,
            account_limits_json=args.account_limits_json,
            fetch_readonly=args.fetch_readonly,
            base_url=args.base_url,
            account_index=args.account_index,
            auth_token_env=args.auth_token_env,
            timeout_s=args.timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"phase51ar_lighter_pressure_source_probe: ERROR: {exc}", file=sys.stderr)
        return 2
    summary_path = out_dir / "lighter_pressure_source_probe_summary.json"
    try:
        summary = _load_json(summary_path)
        fetch_status = summary.get("fetch_status") if isinstance(summary, dict) else "UNKNOWN"
    except Exception:  # noqa: BLE001
        fetch_status = "UNKNOWN"
    print(f"phase51ar_lighter_pressure_source_probe: wrote {out_dir}")
    print(f"phase51ar_lighter_pressure_source_probe: fetch_status {fetch_status}")
    print("phase51ar_lighter_pressure_source_probe: blocker_cleared false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
