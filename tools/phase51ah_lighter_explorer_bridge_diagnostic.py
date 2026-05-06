#!/usr/bin/env python3
"""Phase 5.1ah Lighter Explorer log/tx bridge diagnostic.

This HOLD-only utility tests the last non-duplicative historical Lighter source
surface identified after Phase 5.1ag: read-only Explorer account logs and
transaction GET endpoints. It never submits, signs, cancels, or modifies
orders. It emits proposed source links only when both conditions hold:

1. a fetched raw row hashes exactly to a Lighter source_record_sha256 already
   present in the current Phase 5.1 source-link request pack; and
2. raw identifier values on that same row uniquely hash to a current Lighter
   target canonical_group_id/order_key.

No links are inferred from time, price, size, account role, or proximity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import phase51z_readonly_native_role_capture as phase51z


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51ah_lighter_explorer_bridge_diagnostic"
DEFAULT_EXPLORER_BASE_URL = "https://explorer.elliot.ai"
DEFAULT_MAINNET_BASE_URL = "https://mainnet.zklighter.elliot.ai"
DEFAULT_USER_AGENT = "paraphina-phase51ah-readonly/1"

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
    "authorization",
    "auth_token",
    "bearer",
    "jwt",
    "password",
    "private_key",
    "secret",
    "signature",
    "token",
}
SECRET_FIELD_EXACT = {"sig"}
RAW_IDENTIFIER_VALUE_KEYS = {
    "askclientid",
    "askclientidstr",
    "askid",
    "askidstr",
    "bidclientid",
    "bidclientidstr",
    "bidid",
    "bididstr",
    "clientid",
    "clientidstr",
    "clientorderid",
    "clientorderidstr",
    "hash",
    "id",
    "idstr",
    "index",
    "l1address",
    "nonce",
    "orderid",
    "orderidstr",
    "parenthash",
    "parentorderid",
    "parentorderindex",
    "sequenceindex",
    "tradeid",
    "tradeidstr",
    "transactionhash",
    "transactionid",
    "txhash",
}
OUTPUT_RAW_IDENTIFIER_FIELDS = {
    "ask_client_id",
    "ask_client_id_str",
    "ask_id",
    "ask_id_str",
    "bid_client_id",
    "bid_client_id_str",
    "bid_id",
    "bid_id_str",
    "client_id",
    "client_order_id",
    "hash",
    "id",
    "index",
    "l1_address",
    "nonce",
    "order_id",
    "parent_hash",
    "parent_order_id",
    "sequence_index",
    "trade_id",
    "transaction_hash",
    "transaction_id",
    "tx_hash",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _stable_hash(value: Any) -> str:
    return phase51z._stable_hash(value)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, data: Any) -> None:
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


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
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
    compact = _normalize_key(key)
    return compact in SECRET_FIELD_EXACT or any(fragment in normalized for fragment in SECRET_FIELD_FRAGMENTS)


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
    for obj in _iter_dicts(record):
        for key in obj:
            normalized = str(key).replace("-", "_").lower()
            if normalized in OUTPUT_RAW_IDENTIFIER_FIELDS and not normalized.endswith("_sha256"):
                raise ValueError(f"{path} has raw identifier-shaped {label} field {key!r}")


def _check_output_safe(record: Any, path: Path) -> None:
    _check_unsafe_flags(record, path, label="output")
    _check_no_secret_fields(record, path, label="output")
    _check_no_raw_identifier_fields(record, path, label="output")


def _check_local_file(path: Path, *, label: str) -> Path:
    if "://" in str(path):
        raise ValueError(f"network {label} path is prohibited: {path}")
    resolved = path if path.is_absolute() else ROOT / path
    if any(part == ".env" or part.endswith(".env") for part in resolved.parts):
        raise ValueError(f"env files are prohibited as {label} inputs")
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"{label} path does not exist or is not a file: {resolved}")
    if resolved.suffix not in {".json", ".jsonl"}:
        raise ValueError(f"{label} path must be .json or .jsonl: {resolved}")
    return resolved


def _normalize_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _normalize_key(key)
            if _field_looks_secret(str(key)):
                redacted[str(key)] = "<redacted>"
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


def _artifact_infos(root_dir: Path, paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(paths)
    ]


def _load_request_pack(request_pack: Path) -> tuple[dict[str, set[str]], set[str], dict[str, Any]]:
    target_path = request_pack / "source_link_request_targets.jsonl"
    source_path = request_pack / "source_link_request_sources.jsonl"
    targets_by_hash: dict[str, set[str]] = {}
    request_source_hashes: set[str] = set()
    target_count = 0
    source_count = 0
    for _, target in _iter_jsonl(target_path):
        _check_output_safe(target, target_path)
        if target.get("baseline_commit") != BASELINE_COMMIT:
            raise ValueError("request target baseline_commit mismatch")
        if target.get("venue_id") != "lighter":
            continue
        target_id = str(target.get("canonical_group_id") or target.get("order_key") or "")
        if not target_id:
            continue
        target_count += 1
        for hashed in (target.get("order_id_hash"), target.get("client_order_id_hash")):
            if isinstance(hashed, str) and len(hashed) == 64:
                targets_by_hash.setdefault(hashed.lower(), set()).add(target_id)
    for _, source in _iter_jsonl(source_path):
        _check_output_safe(source, source_path)
        if source.get("baseline_commit") != BASELINE_COMMIT:
            raise ValueError("request source baseline_commit mismatch")
        if source.get("venue_id") == "lighter" and isinstance(source.get("source_record_sha256"), str):
            request_source_hashes.add(str(source["source_record_sha256"]).lower())
            source_count += 1
    return targets_by_hash, request_source_hashes, {
        "lighter_request_target_count": target_count,
        "lighter_request_source_count": source_count,
        "lighter_target_hash_count": len(targets_by_hash),
    }


def _http_get_json(
    base_url: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    timeout_s: float = 20.0,
) -> Any:
    encoded = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None}, doseq=True)
    url = f"{base_url.rstrip('/')}{endpoint}"
    if encoded:
        url = f"{url}?{encoded}"
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"unexpected JSON payload from {endpoint}")
    return payload


def _payload_records(payload: Any) -> list[dict[str, Any]]:
    records = [*phase51z._payload_records(payload), *_nested_dict_records(payload)]
    expanded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        for candidate in _expand_record(record):
            key = _stable_hash(candidate)
            if key not in seen:
                seen.add(key)
                expanded.append(candidate)
    return expanded


def _nested_dict_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        out = [payload]
        for value in payload.values():
            out.extend(_nested_dict_records(value))
        return out
    if isinstance(payload, list):
        out: list[dict[str, Any]] = []
        for item in payload:
            out.extend(_nested_dict_records(item))
        return out
    return []


def _expand_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    out = [record]
    for value in record.values():
        parsed = _parse_embedded_json(value)
        if isinstance(parsed, dict):
            out.append(parsed)
            out.extend(_payload_records(parsed))
        elif isinstance(parsed, list):
            out.extend(_payload_records(parsed))
    return out


def _parse_embedded_json(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _collect_hash_values(record: Any) -> list[str]:
    hashes: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if value in (None, ""):
            return
        for candidate in (value, str(value)):
            hashed = _stable_hash(candidate)
            if hashed not in seen:
                seen.add(hashed)
                hashes.append(hashed)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = _normalize_key(key)
                if normalized in RAW_IDENTIFIER_VALUE_KEYS:
                    add(item)
                elif normalized.endswith("sha256") and isinstance(item, str) and len(item) == 64:
                    lowered = item.lower()
                    if lowered not in seen:
                        seen.add(lowered)
                        hashes.append(lowered)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(record)
    return hashes


def _match_target(hashes: list[str], targets_by_hash: dict[str, set[str]]) -> tuple[str | None, str]:
    matched: set[str] = set()
    for hashed in hashes:
        matched.update(targets_by_hash.get(hashed.lower(), set()))
    if len(matched) == 1:
        return next(iter(matched)), "TARGET_MATCHED_BY_REDACTED_ID_HASH"
    if len(matched) > 1:
        return None, "AMBIGUOUS_REDACTED_ID_HASH"
    return None, "NO_TARGET_MATCH"


def _extract_tx_hashes(payloads: list[Any], limit: int) -> list[str]:
    hashes: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        stripped = value.strip()
        if len(stripped) < 32:
            return
        hexish = stripped[2:] if stripped.startswith("0x") else stripped
        if not all(ch in "0123456789abcdefABCDEF" for ch in hexish):
            return
        if stripped not in seen:
            seen.add(stripped)
            hashes.append(stripped)

    def walk(value: Any) -> None:
        if len(hashes) >= limit:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = _normalize_key(key)
                if normalized in {"hash", "txhash", "transactionhash", "parenthash"}:
                    add(item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for payload in payloads:
        walk(payload)
        if len(hashes) >= limit:
            break
    return hashes[:limit]


def _load_payload_files(paths: list[Path], *, label: str) -> list[Any]:
    payloads: list[Any] = []
    for path in paths:
        resolved = _check_local_file(path, label=label)
        payload = _load_json(resolved)
        _check_unsafe_flags(payload, resolved, label=label)
        _check_no_secret_fields(payload, resolved, label=label)
        payloads.append(payload)
    return payloads


def _fetch_readonly_payloads(
    *,
    account_param: str,
    explorer_base_url: str,
    mainnet_base_url: str,
    pages: int,
    limit: int,
    offset: int,
    sleep_s: float,
    tx_detail_limit: int,
    timeout_s: float,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    if not account_param:
        raise ValueError("--account-param is required for --fetch-readonly")
    log_payloads: list[Any] = []
    for page in range(max(pages, 0)):
        payload = _http_get_json(
            explorer_base_url,
            f"/api/accounts/{urllib.parse.quote(account_param, safe='')}/logs",
            params={"limit": min(max(limit, 1), 100), "offset": str(offset + page * limit)},
            timeout_s=timeout_s,
        )
        log_payloads.append(payload)
        if sleep_s > 0 and page + 1 < pages:
            time.sleep(sleep_s)
    tx_hashes = _extract_tx_hashes(log_payloads, tx_detail_limit)
    tx_payloads: list[Any] = []
    tx_errors: list[dict[str, Any]] = []
    for tx_hash in tx_hashes:
        for source_name, base_url, endpoint, params in (
            ("explorer_log_by_hash", explorer_base_url, f"/api/logs/{urllib.parse.quote(tx_hash, safe='')}", None),
            ("mainnet_tx_by_hash", mainnet_base_url, "/api/v1/tx", {"by": "hash", "value": tx_hash}),
        ):
            try:
                tx_payloads.append(_http_get_json(base_url, endpoint, params=params, timeout_s=timeout_s))
            except Exception as exc:  # noqa: BLE001 - optional read-only enrichment stays auditable
                tx_errors.append({
                    "source": source_name,
                    "tx_hash_sha256": _stable_hash(tx_hash),
                    "error_type": type(exc).__name__,
                    "error_message_sha256": _stable_hash(str(exc)),
                })
        if sleep_s > 0:
            time.sleep(sleep_s)
    return log_payloads, tx_payloads, {
        "account_param_sha256": _stable_hash(account_param),
        "log_pages_requested": pages,
        "log_limit": min(max(limit, 1), 100),
        "tx_hash_count": len(tx_hashes),
        "tx_detail_payload_count": len(tx_payloads),
        "tx_detail_error_count": len(tx_errors),
        "tx_detail_errors": tx_errors,
    }


def _build_links(
    payloads: list[Any],
    *,
    request_source_hashes: set[str],
    targets_by_hash: dict[str, set[str]],
    run_id: str,
    timestamp_ns: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    links_by_source: dict[str, dict[str, Any]] = {}
    label_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    seq = 0
    for payload in payloads:
        for record in _payload_records(payload):
            seq += 1
            source_hash = _stable_hash(record)
            source_in_request = source_hash in request_source_hashes
            target_id, match_status = _match_target(_collect_hash_values(record), targets_by_hash)
            if source_in_request and target_id is not None:
                existing = links_by_source.get(source_hash)
                if existing is None:
                    links_by_source[source_hash] = {
                        "source_record_sha256": source_hash,
                        "canonical_group_id": target_id,
                    }
                    status = "SOURCE_LINK_PROPOSED"
                elif existing.get("canonical_group_id") == target_id:
                    status = "DUPLICATE_SOURCE_LINK_CONFIRMATION"
                else:
                    links_by_source.pop(source_hash, None)
                    status = "CONFLICTING_SOURCE_LINK_REJECTED"
            elif source_in_request:
                status = match_status
            elif target_id is not None:
                status = "TARGET_MATCHED_BUT_SOURCE_HASH_NOT_IN_REQUEST"
            else:
                status = "NO_REQUEST_SOURCE_OR_TARGET_MATCH"
            label = {
                "schema_version": 1,
                "label_type": "PHASE51AH_LIGHTER_EXPLORER_BRIDGE_LABEL",
                "event_seq": seq,
                "timestamp_local_ns": timestamp_ns + seq,
                "run_id": run_id,
                "baseline_commit": BASELINE_COMMIT,
                "gate_status": "HOLD",
                "venue_id": "lighter",
                "source_record_sha256": source_hash,
                "source_hash_in_request": source_in_request,
                "target_match_status": match_status,
                "bridge_status": status,
                "canonical_group_id": target_id,
                "no_live_flag": True,
                "approved_for_live": False,
                "live_orders_allowed": False,
                "capital_change_allowed": False,
                "risk_limit_relaxation_allowed": False,
            }
            label_rows.append(label)
            status_counts[status] = status_counts.get(status, 0) + 1
    return sorted(links_by_source.values(), key=lambda row: row["source_record_sha256"]), label_rows, dict(sorted(status_counts.items()))


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
            "source_link_id": "phase51ah_lighter_explorer_proposed_source_links",
            "venue_id": "lighter",
            "path": str(sidecar_path),
        }
    ]
    return manifest


def build_lighter_explorer_bridge_diagnostic(
    *,
    request_pack: Path,
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
    explorer_logs_json: list[Path],
    tx_json: list[Path],
    fetch_readonly: bool,
    account_param: str | None,
    explorer_base_url: str,
    mainnet_base_url: str,
    pages: int,
    limit: int,
    offset: int,
    sleep_s: float,
    tx_detail_limit: int,
    timeout_s: float,
) -> Path:
    request_pack = request_pack if request_pack.is_absolute() else ROOT / request_pack
    if not request_pack.exists() or not request_pack.is_dir():
        raise ValueError(f"request pack directory not found: {request_pack}")
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    output_root = output_root if output_root.is_absolute() else ROOT / output_root
    run_id = run_id or f"PHASE51AH-LIGHTER-EXPLORER-BRIDGE-DIAGNOSTIC-HOLD-{_utc_stamp()}"
    timestamp_ns = timestamp_ns or time.time_ns()
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    targets_by_hash, request_source_hashes, request_summary = _load_request_pack(request_pack)
    log_payloads = _load_payload_files(explorer_logs_json, label="explorer logs input")
    tx_payloads = _load_payload_files(tx_json, label="tx input")
    fetch_summary: dict[str, Any] = {
        "fetch_readonly": fetch_readonly,
        "account_param_present": bool(account_param),
        "account_param_sha256": _stable_hash(account_param) if account_param else None,
        "tx_detail_errors": [],
    }
    if fetch_readonly:
        fetched_logs, fetched_txs, fetched_summary = _fetch_readonly_payloads(
            account_param=account_param or "",
            explorer_base_url=explorer_base_url,
            mainnet_base_url=mainnet_base_url,
            pages=pages,
            limit=limit,
            offset=offset,
            sleep_s=sleep_s,
            tx_detail_limit=tx_detail_limit,
            timeout_s=timeout_s,
        )
        log_payloads.extend(fetched_logs)
        tx_payloads.extend(fetched_txs)
        fetch_summary.update(fetched_summary)

    all_payloads = [*log_payloads, *tx_payloads]
    sidecars, labels, status_counts = _build_links(
        all_payloads,
        request_source_hashes=request_source_hashes,
        targets_by_hash=targets_by_hash,
        run_id=run_id,
        timestamp_ns=timestamp_ns,
    )

    source_dir = out_dir / "source_snapshots"
    logs_path = source_dir / "explorer_logs.sanitized.json"
    tx_path = source_dir / "tx_details.sanitized.json"
    sidecar_path = out_dir / "source_links.proposed.sanitized.jsonl"
    labels_path = out_dir / "phase51ah_lighter_explorer_bridge_labels.jsonl"
    summary_path = out_dir / "phase51ah_lighter_explorer_bridge_summary.json"
    candidate_manifest_path = out_dir / "candidate_manifest_with_explorer_sidecar.json"
    command_log_path = out_dir / "command_log.json"

    _write_json(logs_path, _redact(log_payloads))
    _write_json(tx_path, _redact(tx_payloads))
    _write_jsonl(sidecar_path, sidecars)
    _write_jsonl(labels_path, labels)
    _write_json(candidate_manifest_path, _candidate_manifest(request_pack, sidecar_path))

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51ah_lighter_explorer_bridge_diagnostic_nonlive_hold",
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
        "request_pack": str(request_pack),
        **request_summary,
        **fetch_summary,
        "explorer_log_payload_count": len(log_payloads),
        "tx_detail_payload_count": len(tx_payloads),
        "explorer_bridge_label_count": len(labels),
        "bridge_status_counts": status_counts,
        "materializable_source_link_count": len(sidecars),
        "candidate_manifest_with_explorer_sidecar": str(candidate_manifest_path),
        "source_links_sanitized_path": str(sidecar_path),
        "next_required_action": (
            "run_phase51ad_or_phase51ae_then_phase51v_if_materializable_source_link_count_positive"
            if sidecars
            else "historical_lighter_explorer_bridge_exhausted_for_current_request_pack"
        ),
    }
    _write_json(summary_path, summary)
    _write_json(command_log_path, {
        "argv": [
            arg if "TOKEN" not in arg.upper() and "PRIVATE" not in arg.upper() and "SECRET" not in arg.upper() else "<redacted>"
            for arg in sys.argv
        ],
        "fetch_readonly": fetch_readonly,
        "created_utc": summary["created_utc"],
    })

    artifacts = [logs_path, tx_path, sidecar_path, labels_path, candidate_manifest_path, summary_path, command_log_path]
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, artifacts),
    })
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": summary["created_utc"],
        "metadata": summary,
        "files": _artifact_infos(out_dir, [*artifacts, artifact_index_path]),
    })
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-pack", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--explorer-logs-json", type=Path, action="append", default=[])
    parser.add_argument("--tx-json", type=Path, action="append", default=[])
    parser.add_argument("--fetch-readonly", action="store_true")
    parser.add_argument("--account-param", default=None)
    parser.add_argument("--explorer-base-url", default=DEFAULT_EXPLORER_BASE_URL)
    parser.add_argument("--mainnet-base-url", default=DEFAULT_MAINNET_BASE_URL)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep-s", type=float, default=1.0)
    parser.add_argument("--tx-detail-limit", type=int, default=80)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()
    try:
        out_dir = build_lighter_explorer_bridge_diagnostic(
            request_pack=args.request_pack,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
            explorer_logs_json=args.explorer_logs_json,
            tx_json=args.tx_json,
            fetch_readonly=args.fetch_readonly,
            account_param=args.account_param,
            explorer_base_url=args.explorer_base_url,
            mainnet_base_url=args.mainnet_base_url,
            pages=args.pages,
            limit=args.limit,
            offset=args.offset,
            sleep_s=args.sleep_s,
            tx_detail_limit=args.tx_detail_limit,
            timeout_s=args.timeout_s,
        )
    except Exception as exc:
        print(f"phase51ah_lighter_explorer_bridge_diagnostic: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51ah_lighter_explorer_bridge_diagnostic: wrote {out_dir}")
    print("phase51ah_lighter_explorer_bridge_diagnostic: status HOLD (read-only nonlive diagnostic)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
