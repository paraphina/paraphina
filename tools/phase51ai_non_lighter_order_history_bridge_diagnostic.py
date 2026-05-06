#!/usr/bin/env python3
"""Phase 5.1ai Extended/Paradex order-history bridge diagnostic.

This HOLD-only utility tests whether read-only Extended and Paradex order
history can bridge already-requested native role source rows to current Phase
5.1 targets. It never submits, signs, edits, cancels, or replaces orders.

A proposed source link is emitted only when all conditions hold:

1. the native trade/fill row hashes exactly to a source_record_sha256 already
   present in the current source-link request pack;
2. order-history raw identifiers uniquely hash to a current target; and
3. the native row and order-history row share a deterministic raw order key.

No source links are inferred from time, price, size, or account role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import phase51z_readonly_native_role_capture as phase51z


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51ai_non_lighter_order_history_bridge_diagnostic"
VENUES = ("extended", "paradex")

OFFICIAL_DOCS = {
    "extended": [
        "https://api.docs.extended.exchange/",
        "GET /api/v1/user/orders/history",
        "GET /api/v1/user/trades",
    ],
    "paradex": [
        "https://docs.paradex.trade/api/prod/orders/get-orders",
        "https://docs.paradex.trade/api/prod/account/list-fills/",
        "GET /v1/orders-history",
        "GET /v1/fills",
    ],
}

UNSAFE_TRUE_FLAGS = {
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
SECRET_FIELD_FRAGMENTS = {
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
    "signature",
    "signing_key",
    "token",
}
RAW_IDENTIFIER_VALUE_KEYS = {
    "account",
    "accountid",
    "accountindex",
    "address",
    "clientid",
    "clientorderid",
    "cursor",
    "externalid",
    "externalorderid",
    "fillid",
    "hash",
    "id",
    "l1address",
    "next",
    "orderid",
    "prev",
    "requestid",
    "seqno",
    "tradeid",
    "txhash",
    "wallet",
}
OUTPUT_RAW_IDENTIFIER_FIELDS = {
    "account",
    "account_id",
    "accountId",
    "address",
    "client_id",
    "clientId",
    "client_order_id",
    "clientOrderId",
    "cursor",
    "external_id",
    "externalId",
    "external_order_id",
    "externalOrderId",
    "fill_id",
    "hash",
    "id",
    "l1_address",
    "next",
    "order_id",
    "orderId",
    "prev",
    "request_id",
    "seq_no",
    "trade_id",
    "tx_hash",
    "wallet",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _stable_hash(value: Any) -> str:
    return phase51z._stable_hash(value)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    _check_output_safe(data, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            _check_output_safe(record, path)
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            f.write("\n")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            yield line_no, row


def _iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _field_looks_secret(key: str) -> bool:
    normalized = key.replace("-", "_").lower()
    if "nonsecret" in normalized:
        return False
    if "authorization" in normalized and normalized.startswith(("no_", "not_")):
        return False
    return any(fragment in normalized for fragment in SECRET_FIELD_FRAGMENTS)


def _check_no_secret_fields(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for key in obj:
            if _field_looks_secret(str(key)):
                raise ValueError(f"{path} has secret-shaped {label} field {key!r}")


def _check_unsafe_flags(record: Any, path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for flag in UNSAFE_TRUE_FLAGS:
            if obj.get(flag) is True:
                raise ValueError(f"{path} has unsafe {label} flag {flag}=true")


def _check_no_raw_identifier_fields(record: Any, path: Path, *, label: str) -> None:
    raw_fields = {field.replace("-", "_").lower() for field in OUTPUT_RAW_IDENTIFIER_FIELDS}
    for obj in _iter_dicts(record):
        for key in obj:
            normalized = str(key).replace("-", "_").lower()
            if normalized in raw_fields and not normalized.endswith("_sha256"):
                raise ValueError(f"{path} has raw identifier-shaped {label} field {key!r}")


def _check_output_safe(record: Any, path: Path) -> None:
    _check_unsafe_flags(record, path, label="output")
    _check_no_secret_fields(record, path, label="output")
    _check_no_raw_identifier_fields(record, path, label="output")


def _check_local_file(path: Path, *, label: str) -> Path:
    if "://" in str(path):
        raise ValueError(f"network {label} path is prohibited: {path}")
    resolved = _resolve_path(path)
    if any(part == ".env" or part.endswith(".env") for part in resolved.parts):
        raise ValueError(f"env files are prohibited as {label} inputs")
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{label} path does not exist or is not a file: {resolved}")
    if resolved.suffix not in {".json", ".jsonl"}:
        raise ValueError(f"{label} path must be .json or .jsonl: {resolved}")
    return resolved


def _artifact_infos(root_dir: Path, paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(paths)
    ]


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _normalize_key(key)
            if _field_looks_secret(str(key)):
                prefix = f"redacted_field_{_stable_hash(str(key))[:12]}"
                redacted[f"{prefix}_present"] = item not in (None, "")
            elif normalized in RAW_IDENTIFIER_VALUE_KEYS:
                redacted[f"{key}_sha256"] = _stable_hash(item) if item not in (None, "") else None
            elif isinstance(item, str) and item.strip()[:1] in {"{", "["}:
                redacted[f"{key}_present"] = bool(item)
                redacted[f"{key}_sha256"] = _stable_hash(item)
            else:
                redacted[str(key)] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _payload_records(payloads: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for payload in payloads:
        out.extend(phase51z._payload_records(payload))
    return out


def _load_payload_files(paths: list[Path], *, label: str) -> list[Any]:
    payloads: list[Any] = []
    for path in paths:
        resolved = _check_local_file(path, label=label)
        payload = _load_json(resolved)
        _check_unsafe_flags(payload, resolved, label=label)
        payloads.append(payload)
    return payloads


def _candidate_hashes(*values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in (None, ""):
            continue
        candidates = [value]
        if not isinstance(value, str):
            candidates.append(str(value))
        for candidate in candidates:
            hashed = _stable_hash(candidate).lower()
            if hashed not in seen:
                seen.add(hashed)
                out.append(hashed)
    return out


def _raw_key_values(*values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in (None, ""):
            continue
        for candidate in (value, str(value)):
            key = str(candidate)
            if key and key not in seen:
                seen.add(key)
                out.append(key)
    return out


def _extended_identity_hashes(row: dict[str, Any]) -> list[str]:
    return _candidate_hashes(
        row.get("externalId"),
        row.get("externalOrderId"),
        row.get("client_order_id"),
        row.get("clientOrderId"),
        row.get("orderId"),
        row.get("order_id"),
        row.get("id"),
    )


def _paradex_identity_hashes(row: dict[str, Any]) -> list[str]:
    return _candidate_hashes(
        row.get("client_id"),
        row.get("clientId"),
        row.get("client_order_id"),
        row.get("clientOrderId"),
        row.get("order_id"),
        row.get("orderId"),
        row.get("id"),
    )


def _identity_hashes(venue: str, row: dict[str, Any]) -> list[str]:
    if venue == "extended":
        return _extended_identity_hashes(row)
    if venue == "paradex":
        return _paradex_identity_hashes(row)
    return []


def _order_join_keys(venue: str, row: dict[str, Any]) -> list[str]:
    if venue == "extended":
        return _raw_key_values(row.get("id"), row.get("orderId"), row.get("order_id"))
    if venue == "paradex":
        return _raw_key_values(row.get("id"), row.get("order_id"), row.get("orderId"))
    return []


def _native_join_keys(venue: str, row: dict[str, Any]) -> list[str]:
    if venue == "extended":
        return _raw_key_values(row.get("orderId"), row.get("order_id"))
    if venue == "paradex":
        return _raw_key_values(row.get("order_id"), row.get("orderId"))
    return []


def _match_target(
    venue: str,
    hashes: list[str],
    targets_by_hash: dict[str, dict[str, set[str]]],
) -> tuple[str | None, str]:
    matched: set[str] = set()
    venue_targets = targets_by_hash.get(venue, {})
    for hashed in hashes:
        matched.update(venue_targets.get(hashed.lower(), set()))
    if len(matched) == 1:
        return next(iter(matched)), "TARGET_MATCHED_BY_REDACTED_ID_HASH"
    if len(matched) > 1:
        return None, "AMBIGUOUS_REDACTED_ID_HASH"
    return None, "NO_TARGET_MATCH"


def _load_request_pack(request_pack: Path) -> tuple[dict[str, dict[str, set[str]]], dict[str, set[str]], dict[str, Any], list[phase51z.CaptureTarget]]:
    target_path = request_pack / "source_link_request_targets.jsonl"
    source_path = request_pack / "source_link_request_sources.jsonl"
    targets_by_hash: dict[str, dict[str, set[str]]] = {venue: {} for venue in VENUES}
    request_source_hashes: dict[str, set[str]] = {venue: set() for venue in VENUES}
    target_counts = {venue: 0 for venue in VENUES}
    source_counts = {venue: 0 for venue in VENUES}
    target_rows: list[phase51z.CaptureTarget] = []

    for _, target in _iter_jsonl(target_path):
        _check_output_safe(target, target_path)
        if target.get("baseline_commit") != BASELINE_COMMIT:
            raise ValueError("request target baseline_commit mismatch")
        venue = str(target.get("venue_id") or "").lower()
        if venue not in VENUES:
            continue
        target_id = str(target.get("canonical_group_id") or target.get("order_key") or "")
        if not target_id:
            continue
        target_counts[venue] += 1
        for hashed in (target.get("order_id_hash"), target.get("client_order_id_hash")):
            if isinstance(hashed, str) and len(hashed) == 64:
                targets_by_hash[venue].setdefault(hashed.lower(), set()).add(target_id)
        target_rows.append(
            phase51z.CaptureTarget(
                venue_id=venue,
                canonical_group_id=str(target.get("canonical_group_id") or ""),
                order_key=str(target.get("order_key") or ""),
                order_id_hash=str(target.get("order_id_hash") or "") or None,
                client_order_id_hash=str(target.get("client_order_id_hash") or "") or None,
                first_fill_time_ms=phase51z._safe_int(target.get("first_fill_time_ms")),
                last_fill_time_ms=phase51z._safe_int(target.get("last_fill_time_ms")),
            )
        )

    for _, source in _iter_jsonl(source_path):
        _check_output_safe(source, source_path)
        if source.get("baseline_commit") != BASELINE_COMMIT:
            raise ValueError("request source baseline_commit mismatch")
        venue = str(source.get("venue_id") or "").lower()
        source_hash = source.get("source_record_sha256")
        if venue in VENUES and isinstance(source_hash, str) and len(source_hash) == 64:
            request_source_hashes[venue].add(source_hash.lower())
            source_counts[venue] += 1

    metadata = {
        "request_target_counts_by_venue": target_counts,
        "request_source_counts_by_venue": source_counts,
        "target_hash_counts_by_venue": {
            venue: len(targets_by_hash[venue])
            for venue in VENUES
        },
    }
    return targets_by_hash, request_source_hashes, metadata, target_rows


def _target_join_map(
    venue: str,
    order_rows: list[dict[str, Any]],
    targets_by_hash: dict[str, dict[str, set[str]]],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    join_map: dict[str, set[str]] = {}
    stats = {
        "order_history_row_count": len(order_rows),
        "order_history_target_match_count": 0,
        "order_history_join_key_count": 0,
    }
    for row in order_rows:
        target_id, _status = _match_target(venue, _identity_hashes(venue, row), targets_by_hash)
        if target_id is None:
            continue
        stats["order_history_target_match_count"] += 1
        for key in _order_join_keys(venue, row):
            join_map.setdefault(key, set()).add(target_id)
    stats["order_history_join_key_count"] = len(join_map)
    return join_map, stats


def _bridge_native_rows(
    venue: str,
    native_rows: list[dict[str, Any]],
    *,
    request_source_hashes: dict[str, set[str]],
    targets_by_hash: dict[str, dict[str, set[str]]],
    join_map: dict[str, set[str]],
    run_id: str,
    timestamp_ns: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    links_by_source: dict[str, dict[str, Any]] = {}
    conflicted_sources: set[str] = set()
    status_counts: dict[str, int] = {}
    stats = {
        "native_source_row_count": len(native_rows),
        "request_source_hash_overlap_count": 0,
        "native_row_direct_target_match_count": 0,
        "native_row_order_history_join_match_count": 0,
        "native_row_order_history_join_ambiguous_count": 0,
    }

    for seq, row in enumerate(native_rows, start=1):
        source_hash = _stable_hash(row).lower()
        source_in_request = source_hash in request_source_hashes.get(venue, set())
        if source_in_request:
            stats["request_source_hash_overlap_count"] += 1
        direct_target, direct_status = _match_target(venue, _identity_hashes(venue, row), targets_by_hash)
        if direct_target is not None:
            stats["native_row_direct_target_match_count"] += 1

        joined_targets: set[str] = set()
        ambiguous_join = False
        for key in _native_join_keys(venue, row):
            targets = join_map.get(key, set())
            if len(targets) > 1:
                ambiguous_join = True
            elif len(targets) == 1:
                joined_targets.update(targets)
        if len(joined_targets) == 1:
            stats["native_row_order_history_join_match_count"] += 1
        if len(joined_targets) > 1 or ambiguous_join:
            stats["native_row_order_history_join_ambiguous_count"] += 1

        target_id: str | None = None
        if direct_target is not None and len(joined_targets) == 1:
            joined = next(iter(joined_targets))
            if joined == direct_target:
                target_id = joined
                status = "SOURCE_LINK_PROPOSED_DIRECT_AND_ORDER_HISTORY_AGREE"
            else:
                status = "CONFLICTING_NATIVE_AND_ORDER_HISTORY_TARGET_REJECTED"
        elif direct_target is not None:
            status = "DIRECT_NATIVE_TARGET_MATCH_WITHOUT_ORDER_HISTORY_BRIDGE_REJECTED"
        elif len(joined_targets) == 1:
            target_id = next(iter(joined_targets))
            status = "SOURCE_LINK_PROPOSED_ORDER_HISTORY_BRIDGE"
        elif len(joined_targets) > 1 or ambiguous_join:
            status = "AMBIGUOUS_ORDER_HISTORY_JOIN_REJECTED"
        else:
            status = direct_status if direct_status != "NO_TARGET_MATCH" else "NO_ORDER_HISTORY_TARGET_MATCH"

        if not source_in_request:
            status = "NATIVE_SOURCE_HASH_NOT_IN_REQUEST"
            target_id = None

        if source_in_request and target_id is not None:
            existing = links_by_source.get(source_hash)
            if existing is None:
                links_by_source[source_hash] = {
                    "source_record_sha256": source_hash,
                    "canonical_group_id": target_id,
                }
            elif existing.get("canonical_group_id") != target_id:
                conflicted_sources.add(source_hash)
                links_by_source.pop(source_hash, None)
                status = "CONFLICTING_SOURCE_LINK_REJECTED"

        label = {
            "schema_version": 1,
            "label_type": "PHASE51AI_NON_LIGHTER_ORDER_HISTORY_BRIDGE_LABEL",
            "event_seq": seq,
            "timestamp_local_ns": timestamp_ns + seq,
            "run_id": run_id,
            "baseline_commit": BASELINE_COMMIT,
            "gate_status": "HOLD",
            "venue_id": venue,
            "source_record_sha256": source_hash,
            "source_hash_in_request": source_in_request,
            "direct_target_match_status": direct_status,
            "bridge_status": status,
            "canonical_group_id": target_id,
            "native_join_key_count": len(_native_join_keys(venue, row)),
            "no_live_flag": True,
            "approved_for_live": False,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "risk_limit_relaxation_allowed": False,
        }
        labels.append(label)
        status_counts[status] = status_counts.get(status, 0) + 1

    links = [
        row
        for source_hash, row in sorted(links_by_source.items())
        if source_hash not in conflicted_sources
    ]
    return links, labels, dict(sorted(status_counts.items())), stats


def _fetch_extended_orders_history(env: dict[str, str], timeout_s: float, limit: int, max_pages: int) -> list[dict[str, Any]]:
    base_url = env.get("EXTENDED_REST_URL", "https://api.starknet.extended.exchange").rstrip("/")
    market = env.get("EXTENDED_MARKET", "ETH-USD").strip() or "ETH-USD"
    headers = {"X-Api-Key": env["EXTENDED_API_KEY"]}
    rows: list[dict[str, Any]] = []
    cursor: int | None = None
    seen: set[int] = set()
    for _ in range(max_pages):
        payload = phase51z._http_get_json(
            base_url,
            "/api/v1/user/orders/history",
            headers=headers,
            params={"market": market, "limit": limit, "cursor": cursor},
            timeout_s=timeout_s,
        )
        batch = [item for item in (payload.get("data") if isinstance(payload, dict) else []) or [] if isinstance(item, dict)]
        for row in batch:
            enriched = dict(row)
            enriched["venue_id"] = "extended"
            rows.append(enriched)
        next_cursor = ((payload.get("pagination") or {}).get("cursor") if isinstance(payload, dict) else None)
        parsed = phase51z._safe_int(next_cursor)
        if not batch or parsed is None or parsed in seen:
            break
        seen.add(parsed)
        cursor = parsed
    return rows


def _fetch_paradex_orders_history(
    env: dict[str, str],
    targets: list[phase51z.CaptureTarget],
    timeout_s: float,
    limit: int,
    max_pages: int,
    pad_ms: int,
    *,
    allow_jwt_cmd: bool,
) -> list[dict[str, Any]]:
    start_ms, end_ms = phase51z._time_window_for_targets(targets, "paradex", pad_ms)
    if start_ms is None or end_ms is None:
        return []
    base_url = env.get("PARADEX_REST_URL", "https://api.prod.paradex.trade/v1").rstrip("/")
    token = phase51z._paradex_token(env, allow_jwt_cmd=allow_jwt_cmd)
    headers = {"Authorization": f"Bearer {token}"}
    market = env.get("PARADEX_MARKET", "ETH-USD-PERP").strip() or "ETH-USD-PERP"
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(max_pages):
        payload = phase51z._http_get_json(
            base_url,
            "/orders-history",
            headers=headers,
            params={
                "market": market,
                "start_at": start_ms,
                "end_at": end_ms,
                "page_size": min(limit, 5000),
                "cursor": cursor,
            },
            timeout_s=timeout_s,
        )
        batch = [item for item in (payload.get("results") if isinstance(payload, dict) else []) or [] if isinstance(item, dict)]
        for row in batch:
            enriched = dict(row)
            enriched["venue_id"] = "paradex"
            rows.append(enriched)
        cursor = payload.get("next") if isinstance(payload, dict) else None
        if not cursor or not batch:
            break
    return rows


def _credential_presence_summary(env: dict[str, str]) -> dict[str, dict[str, bool]]:
    return {
        "extended": {
            "read_access_present": bool(env.get("EXTENDED_API_KEY", "").strip()),
            "market_config_present": bool(env.get("EXTENDED_MARKET", "").strip()),
            "rest_url_config_present": bool(env.get("EXTENDED_REST_URL", "").strip()),
        },
        "paradex": {
            "session_material_present": bool(
                env.get("PARADEX_JWT", "").strip()
                or env.get("PARADEX_READONLY_TOKEN", "").strip()
            ),
            "command_material_present": bool(env.get("PARADEX_JWT_CMD", "").strip()),
            "market_config_present": bool(env.get("PARADEX_MARKET", "").strip()),
            "rest_url_config_present": bool(env.get("PARADEX_REST_URL", "").strip()),
        },
    }


def _candidate_manifest(request_pack: Path, sidecar_path: Path) -> dict[str, Any]:
    template = request_pack / "candidate_manifest_with_empty_sidecar.json"
    if template.exists():
        manifest = _load_json(template)
        if not isinstance(manifest, dict):
            raise ValueError(f"candidate manifest is not an object: {template}")
    else:
        manifest = {"schema_version": 1, "sources": [], "source_links": []}
    manifest["source_links"] = [
        {
            "source_link_id": "phase51ai_non_lighter_order_history_proposed_source_links",
            "path": str(sidecar_path),
        }
    ]
    return manifest


def build_non_lighter_order_history_bridge_diagnostic(
    *,
    request_pack: Path,
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
    extended_trades_json: list[Path],
    extended_orders_history_json: list[Path],
    paradex_fills_json: list[Path],
    paradex_orders_history_json: list[Path],
    fetch_readonly: bool,
    env_file: Path | None,
    venues: list[str],
    timeout_s: float,
    limit: int,
    max_pages: int,
    window_pad_ms: int,
    allow_paradex_jwt_cmd: bool,
) -> Path:
    request_pack = _check_local_file(request_pack / "manifest.json", label="request pack manifest").parent
    selected_venues = sorted(set(venue.lower() for venue in venues))
    unsupported = sorted(set(selected_venues) - set(VENUES))
    if unsupported:
        raise ValueError(f"unsupported venues: {unsupported}")

    targets_by_hash, request_source_hashes, request_metadata, target_rows = _load_request_pack(request_pack)
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    source_dir = out_dir / "source_snapshots"
    evidence_dir = out_dir / "evidence_pack"
    source_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    native_rows_by_venue: dict[str, list[dict[str, Any]]] = {venue: [] for venue in VENUES}
    order_rows_by_venue: dict[str, list[dict[str, Any]]] = {venue: [] for venue in VENUES}
    fetch_status: dict[str, dict[str, Any]] = {}
    credential_presence: dict[str, dict[str, bool]] = {}

    native_payloads = {
        "extended": _load_payload_files(extended_trades_json, label="extended trades"),
        "paradex": _load_payload_files(paradex_fills_json, label="paradex fills"),
    }
    order_payloads = {
        "extended": _load_payload_files(extended_orders_history_json, label="extended orders history"),
        "paradex": _load_payload_files(paradex_orders_history_json, label="paradex orders history"),
    }
    for venue in VENUES:
        native_rows_by_venue[venue].extend(_payload_records(native_payloads[venue]))
        order_rows_by_venue[venue].extend(_payload_records(order_payloads[venue]))

    if fetch_readonly:
        if env_file is None:
            raise ValueError("--env-file is required with --fetch-readonly")
        env_path = _resolve_path(env_file)
        env = phase51z._load_env_file(env_path)
        credential_presence = _credential_presence_summary(env)
        for venue in selected_venues:
            try:
                if venue == "extended":
                    native = phase51z._fetch_extended(env, timeout_s, limit, max_pages)
                    orders = _fetch_extended_orders_history(env, timeout_s, limit, max_pages)
                elif venue == "paradex":
                    native = phase51z._fetch_paradex(
                        env,
                        target_rows,
                        timeout_s,
                        limit,
                        window_pad_ms,
                        max_pages,
                        allow_jwt_cmd=allow_paradex_jwt_cmd,
                    )
                    orders = _fetch_paradex_orders_history(
                        env,
                        target_rows,
                        timeout_s,
                        limit,
                        max_pages,
                        window_pad_ms,
                        allow_jwt_cmd=allow_paradex_jwt_cmd,
                    )
                else:
                    native, orders = [], []
                native_rows_by_venue[venue].extend(native)
                order_rows_by_venue[venue].extend(orders)
                fetch_status[venue] = {
                    "status": "FETCHED",
                    "native_row_count": len(native),
                    "order_history_row_count": len(orders),
                }
            except Exception as exc:  # noqa: BLE001 - read-only diagnostic remains auditable on failure
                fetch_status[venue] = {
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "message_sha256": _stable_hash(str(exc)),
                }

    snapshot_paths: list[Path] = []
    for venue in VENUES:
        native_path = source_dir / f"{venue}_native_source_rows.sanitized.json"
        order_path = source_dir / f"{venue}_orders_history.sanitized.json"
        _write_json(native_path, {"venue_id": venue, "rows": _redact(native_rows_by_venue[venue])})
        _write_json(order_path, {"venue_id": venue, "rows": _redact(order_rows_by_venue[venue])})
        snapshot_paths.extend([native_path, order_path])

    all_links: list[dict[str, Any]] = []
    all_labels: list[dict[str, Any]] = []
    bridge_status_counts: dict[str, int] = {}
    diagnostics_by_venue: dict[str, Any] = {}
    label_offset = 0
    for venue in VENUES:
        join_map, order_stats = _target_join_map(venue, order_rows_by_venue[venue], targets_by_hash)
        links, labels, status_counts, native_stats = _bridge_native_rows(
            venue,
            native_rows_by_venue[venue],
            request_source_hashes=request_source_hashes,
            targets_by_hash=targets_by_hash,
            join_map=join_map,
            run_id=run_id,
            timestamp_ns=timestamp_ns + label_offset,
        )
        label_offset += len(labels)
        all_links.extend(links)
        all_labels.extend(labels)
        for status, count in status_counts.items():
            bridge_status_counts[status] = bridge_status_counts.get(status, 0) + count
        diagnostics_by_venue[venue] = {
            **order_stats,
            **native_stats,
            "proposed_source_link_count": len(links),
            "proposed_target_count": len({row["canonical_group_id"] for row in links}),
            "proposed_source_count": len({row["source_record_sha256"] for row in links}),
            "bridge_status_counts": status_counts,
        }

    unique_links: dict[tuple[str, str], dict[str, Any]] = {}
    for row in all_links:
        unique_links[(row["source_record_sha256"], row["canonical_group_id"])] = row
    source_links = sorted(unique_links.values(), key=lambda row: (row["canonical_group_id"], row["source_record_sha256"]))

    sidecar_path = out_dir / "source_links.proposed.sanitized.jsonl"
    labels_path = out_dir / "phase51ai_non_lighter_order_history_bridge_labels.jsonl"
    candidate_manifest_path = out_dir / "candidate_manifest_with_order_history_sidecar.json"
    summary_path = out_dir / "phase51ai_non_lighter_order_history_bridge_summary.json"
    command_log_path = out_dir / "command_log.json"
    artifact_index_path = evidence_dir / "artifact_index.json"
    manifest_path = out_dir / "manifest.json"

    _write_jsonl(sidecar_path, source_links)
    _write_jsonl(labels_path, all_labels)
    _write_json(candidate_manifest_path, _candidate_manifest(request_pack, sidecar_path))

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51ai_non_lighter_order_history_bridge_diagnostic_nonlive_hold",
        "request_pack": str(request_pack),
        "request_metadata": request_metadata,
        "selected_venues": selected_venues,
        "fetch_readonly_requested": fetch_readonly,
        "fetch_status": fetch_status,
        "credential_presence": credential_presence,
        "official_docs": OFFICIAL_DOCS,
        "diagnostics_by_venue": diagnostics_by_venue,
        "bridge_status_counts": dict(sorted(bridge_status_counts.items())),
        "materializable_source_link_count": len(source_links),
        "materializable_target_count": len({row["canonical_group_id"] for row in source_links}),
        "materializable_source_count": len({row["source_record_sha256"] for row in source_links}),
        "candidate_manifest_path": str(candidate_manifest_path),
        "source_links_path": str(sidecar_path),
        "source_links_sha256": _sha256_file(sidecar_path),
        "no_live_flag": True,
        "approved_for_model_training": False,
        "approved_for_live": False,
        "approved_for_canary": False,
        "approved_for_capital_escalation": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "raw_identifier_redaction_status": "PASS",
        "clears_phase51_blockers": False,
    }
    _write_json(summary_path, summary)
    _write_json(
        command_log_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "timestamp_local_ns": timestamp_ns,
            "fetch_readonly_requested": fetch_readonly,
            "selected_venues": selected_venues,
            "no_live_flag": True,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "risk_limit_relaxation_allowed": False,
        },
    )
    artifacts = [
        *snapshot_paths,
        sidecar_path,
        labels_path,
        candidate_manifest_path,
        summary_path,
        command_log_path,
    ]
    _write_json(
        artifact_index_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "created_utc": _timestamp_ns_to_utc(timestamp_ns),
            "baseline_commit": BASELINE_COMMIT,
            "gate_status": "HOLD",
            "artifacts": _artifact_infos(out_dir, artifacts),
            "no_live_flag": True,
            "approved_for_live": False,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "risk_limit_relaxation_allowed": False,
        },
    )
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "created_utc": _timestamp_ns_to_utc(timestamp_ns),
            "baseline_commit": BASELINE_COMMIT,
            "gate_status": "HOLD",
            "artifacts": _artifact_infos(out_dir, [artifact_index_path, *artifacts]),
            "no_live_flag": True,
            "approved_for_live": False,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "risk_limit_relaxation_allowed": False,
        },
    )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-pack", type=Path, required=True)
    parser.add_argument("--extended-trades-json", action="append", type=Path, default=[])
    parser.add_argument("--extended-orders-history-json", action="append", type=Path, default=[])
    parser.add_argument("--paradex-fills-json", action="append", type=Path, default=[])
    parser.add_argument("--paradex-orders-history-json", action="append", type=Path, default=[])
    parser.add_argument("--fetch-readonly", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--venue", action="append", choices=VENUES, default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"PHASE51AI-NON-LIGHTER-ORDER-HISTORY-BRIDGE-DIAGNOSTIC-HOLD-{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--window-pad-ms", type=int, default=30 * 60 * 1000)
    parser.add_argument("--allow-paradex-jwt-cmd", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp_ns = args.timestamp_ns if args.timestamp_ns is not None else time.time_ns()
    try:
        out_dir = build_non_lighter_order_history_bridge_diagnostic(
            request_pack=args.request_pack,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
            extended_trades_json=args.extended_trades_json,
            extended_orders_history_json=args.extended_orders_history_json,
            paradex_fills_json=args.paradex_fills_json,
            paradex_orders_history_json=args.paradex_orders_history_json,
            fetch_readonly=args.fetch_readonly,
            env_file=args.env_file,
            venues=args.venue or list(VENUES),
            timeout_s=args.timeout_s,
            limit=args.limit,
            max_pages=args.max_pages,
            window_pad_ms=args.window_pad_ms,
            allow_paradex_jwt_cmd=args.allow_paradex_jwt_cmd,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary fails closed
        print(f"phase51ai_non_lighter_order_history_bridge_diagnostic: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51ai_non_lighter_order_history_bridge_diagnostic: wrote {out_dir}")
    print("phase51ai_non_lighter_order_history_bridge_diagnostic: status HOLD (read-only bridge diagnostic only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
