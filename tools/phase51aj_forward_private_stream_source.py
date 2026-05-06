#!/usr/bin/env python3
"""Phase 5.1aj forward private-stream source normalizer.

This HOLD-only utility materializes the third accepted blocker-clearance path:
a directly target-linkable read-only private/native source surface. It ingests
already-local JSON/JSONL private stream or native fill rows, extracts native
maker/taker role fields plus deterministic order/client identifiers, and emits
Phase 5.1v-compatible sanitized source rows.

It never connects to a venue, signs, submits, cancels, modifies, or replaces
orders. Links are produced only from exact raw identifier hashes; there is no
time/price/size/account-role/proximity inference.
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


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_REQUEST_PACK = (
    ROOT
    / "runs/phase51z_source_link_request_pack"
    / "PHASE51Z-CURRENT-TARGET-WIDE-SOURCE-LINK-REQUEST-PACK-HOLD-20260505T000000Z"
)
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51aj_forward_private_stream_source"
VENUES = {"aster", "extended", "lighter", "paradex"}
SOURCE_LIST_KEYS = {
    "data",
    "events",
    "fills",
    "orders",
    "results",
    "rows",
    "trade_history",
    "tradeHistory",
    "trades",
}
OFFICIAL_DOCS = {
    "extended": [
        "https://api.docs.extended.exchange/",
        "GET /stream.extended.exchange/v1/account",
        "Account update TRADE rows expose orderId, externalOrderId, and isTaker.",
    ],
    "paradex": [
        "https://docs.paradex.trade/ws/web-socket-channels/fills/fills",
        "https://docs.paradex.trade/websocket-reference/web-socket-channels/orders/orders",
        "Fills expose client_id, order_id, and liquidity; orders expose client_id and id.",
    ],
    "lighter": [
        "https://apidocs.lighter.xyz/docs/websocket-reference",
        "Read-only account streams are admissible only when trade rows expose account side and maker flags.",
    ],
    "aster": [
        "official native order-trade update stream",
        "ORDER_TRADE_UPDATE-equivalent rows are admissible only when maker flag and positive fill quantity are present.",
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
OUTPUT_RAW_IDENTIFIER_FIELDS = {
    "account",
    "account_id",
    "accountId",
    "address",
    "ask_client_id",
    "ask_client_id_str",
    "ask_id",
    "ask_id_str",
    "bid_client_id",
    "bid_client_id_str",
    "bid_id",
    "bid_id_str",
    "client_id",
    "clientId",
    "client_order_id",
    "clientOrderId",
    "cloid",
    "external_id",
    "externalId",
    "external_order_id",
    "externalOrderId",
    "fill_id",
    "fillId",
    "hash",
    "id",
    "order_id",
    "orderId",
    "raw_client_order_id",
    "raw_order_id",
    "tid",
    "trade_id",
    "tradeId",
    "tx_hash",
    "txHash",
    "venue_order_id",
}
OUTPUT_HASH_FIELD_EXCEPTIONS = {
    "client_order_id_hash",
    "order_id_hash",
    "source_record_sha256",
    "source_row_sha256",
}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _is_uri_like(value: str) -> bool:
    return "://" in value or value.startswith(("http:", "https:", "s3:", "gs:", "wss:"))


def _is_env_path(path: Path) -> bool:
    return any(part == ".env" or part.endswith(".env") for part in path.parts)


def _check_no_symlink(path: Path) -> None:
    current = path if path.is_absolute() else _resolve_path(path)
    chain = [current]
    chain.extend(current.parents)
    for candidate in chain:
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"symlink path is prohibited: {candidate}")


def _check_local_file(path: Path, *, label: str) -> Path:
    if _is_uri_like(str(path)):
        raise ValueError(f"network {label} path is prohibited: {path}")
    resolved = _resolve_path(path)
    if _is_env_path(resolved):
        raise ValueError(f"env files are prohibited as Phase 5.1aj {label} inputs")
    _check_no_symlink(resolved)
    if not resolved.exists():
        raise ValueError(f"{label} path does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} path is not a file: {resolved}")
    if resolved.suffix not in {".json", ".jsonl", ".ndjson"}:
        raise ValueError(f"{label} path must be .json, .jsonl, or .ndjson: {resolved}")
    return resolved


def _check_run_id(run_id: str) -> str:
    path = Path(run_id)
    if path.name != run_id or ".." in path.parts:
        raise ValueError("run_id must be a single local path segment")
    return run_id


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    for obj in _iter_dicts(record):
        for key in obj:
            key_str = str(key)
            normalized = key_str.replace("-", "_")
            if normalized.endswith("_sha256") or normalized in OUTPUT_HASH_FIELD_EXCEPTIONS:
                continue
            if key_str in OUTPUT_RAW_IDENTIFIER_FIELDS or normalized in OUTPUT_RAW_IDENTIFIER_FIELDS:
                raise ValueError(f"{path} has raw identifier-shaped {label} field {key!r}")


def _check_output_safe(record: Any, path: Path) -> None:
    _check_unsafe_flags(record, path, label="output")
    _check_no_secret_fields(record, path, label="output")
    _check_no_raw_identifier_fields(record, path, label="output")


def _base_record(run_id: str, seq: int, timestamp_ns: int, label_type: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "label_type": label_type,
        "label_seq": seq,
        "timestamp_local_ns": timestamp_ns + seq,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
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
    }


def _positive_float(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _candidate_hashes(*values: Any, prehashed_values: tuple[Any, ...] = ()) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value)
        for candidate in (text, _stable_hash(text)):
            if candidate and candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
    for value in prehashed_values:
        if not isinstance(value, str):
            continue
        candidate = value.strip().lower()
        if len(candidate) == 64 and all(ch in "0123456789abcdef" for ch in candidate) and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _nested_get(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    payload = row.get("o")
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                return payload.get(key)
    return None


def _extract_native_payload(venue: str, row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if venue == "aster":
        maker = _nested_get(row, "maker", "m")
        qty = _nested_get(row, "qty", "l", "lastFilledQty", "last_filled_qty")
        event_type = _nested_get(row, "e", "event_type", "eventType") or "ORDER_TRADE_UPDATE"
        if not isinstance(maker, bool):
            return None, "ASTER_MAKER_FIELD_MISSING"
        if not _positive_float(qty):
            return None, "ASTER_POSITIVE_FILL_QTY_MISSING"
        return {"e": str(event_type), "o": {"m": maker, "l": str(qty)}}, "NATIVE_ROLE_FIELDS_READY"

    if venue == "extended":
        is_taker = _nested_get(row, "isTaker", "is_taker")
        if not isinstance(is_taker, bool):
            return None, "EXTENDED_ISTAKER_MISSING"
        return {"isTaker": is_taker}, "NATIVE_ROLE_FIELDS_READY"

    if venue == "lighter":
        account_index = _safe_int(_nested_get(row, "account_index", "accountIndex"))
        ask_account = _safe_int(_nested_get(row, "ask_account_id", "askAccountId"))
        bid_account = _safe_int(_nested_get(row, "bid_account_id", "bidAccountId"))
        is_maker_ask = _nested_get(row, "is_maker_ask", "isMakerAsk")
        if account_index is None:
            return None, "LIGHTER_ACCOUNT_INDEX_MISSING"
        if ask_account is None or bid_account is None:
            return None, "LIGHTER_SIDE_ACCOUNT_ID_MISSING"
        if not isinstance(is_maker_ask, bool):
            return None, "LIGHTER_IS_MAKER_ASK_MISSING"
        return {
            "account_index": account_index,
            "is_maker_ask": is_maker_ask,
            "ask_account_id": ask_account,
            "bid_account_id": bid_account,
        }, "NATIVE_ROLE_FIELDS_READY"

    if venue == "paradex":
        liquidity = str(_nested_get(row, "liquidity") or "").upper()
        if liquidity not in {"MAKER", "TAKER"}:
            return None, "PARADEX_LIQUIDITY_MISSING"
        return {"liquidity": liquidity}, "NATIVE_ROLE_FIELDS_READY"

    return None, "UNSUPPORTED_VENUE"


def _identity_hashes(venue: str, row: dict[str, Any]) -> tuple[list[str], list[str]]:
    if venue == "aster":
        client_hashes = _candidate_hashes(
            _nested_get(row, "clientOrderId", "client_order_id", "origClientOrderId"),
            prehashed_values=(
                _nested_get(row, "clientOrderId_sha256", "client_order_id_sha256", "origClientOrderId_sha256"),
            ),
        )
        order_hashes = _candidate_hashes(
            _nested_get(row, "orderId", "order_id", "i"),
            prehashed_values=(_nested_get(row, "orderId_sha256", "order_id_sha256", "i_sha256"),),
        )
        return client_hashes, order_hashes

    if venue == "extended":
        client_hashes = _candidate_hashes(
            _nested_get(row, "externalOrderId", "externalId", "client_order_id", "clientOrderId"),
            prehashed_values=(
                _nested_get(
                    row,
                    "externalOrderId_sha256",
                    "externalId_sha256",
                    "client_order_id_sha256",
                    "clientOrderId_sha256",
                ),
            ),
        )
        order_hashes = _candidate_hashes(
            _nested_get(row, "orderId", "order_id", "id"),
            prehashed_values=(_nested_get(row, "orderId_sha256", "order_id_sha256", "id_sha256"),),
        )
        return client_hashes, order_hashes

    if venue == "lighter":
        client_hashes = _candidate_hashes(
            _nested_get(row, "ask_client_id", "ask_client_id_str", "askClientId"),
            _nested_get(row, "bid_client_id", "bid_client_id_str", "bidClientId"),
            prehashed_values=(
                _nested_get(row, "ask_client_id_sha256", "ask_client_id_str_sha256", "askClientId_sha256"),
                _nested_get(row, "bid_client_id_sha256", "bid_client_id_str_sha256", "bidClientId_sha256"),
            ),
        )
        order_hashes = _candidate_hashes(
            _nested_get(row, "ask_id", "ask_id_str", "askId"),
            _nested_get(row, "bid_id", "bid_id_str", "bidId"),
            prehashed_values=(
                _nested_get(row, "ask_id_sha256", "ask_id_str_sha256", "askId_sha256"),
                _nested_get(row, "bid_id_sha256", "bid_id_str_sha256", "bidId_sha256"),
            ),
        )
        return client_hashes, order_hashes

    if venue == "paradex":
        client_hashes = _candidate_hashes(
            _nested_get(row, "client_id", "clientId"),
            prehashed_values=(_nested_get(row, "client_id_sha256", "clientId_sha256"),),
        )
        order_hashes = _candidate_hashes(
            _nested_get(row, "order_id", "orderId", "id"),
            prehashed_values=(_nested_get(row, "order_id_sha256", "orderId_sha256", "id_sha256"),),
        )
        return client_hashes, order_hashes

    return [], []


def _flatten_records(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            rows.extend(_flatten_records(item))
        return rows
    if not isinstance(value, dict):
        return rows

    emitted_child = False
    for key, child in value.items():
        if key in SOURCE_LIST_KEYS and isinstance(child, list):
            emitted_child = True
            for item in child:
                rows.extend(_flatten_records(item))
        elif key in SOURCE_LIST_KEYS and isinstance(child, dict):
            emitted_child = True
            rows.extend(_flatten_records(child))

    params = value.get("params")
    if isinstance(params, dict) and isinstance(params.get("data"), dict):
        emitted_child = True
        rows.extend(_flatten_records(params["data"]))
    data = value.get("data")
    if isinstance(data, dict) and any(key in data for key in SOURCE_LIST_KEYS):
        emitted_child = True
        rows.extend(_flatten_records(data))

    if not emitted_child:
        rows.append(value)
    return rows


def _load_source_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.suffix == ".json":
        rows.extend(_flatten_records(_load_json(path)))
    else:
        for _, row in _iter_jsonl(path):
            rows.extend(_flatten_records(row))
    return rows


def _load_request_targets(request_pack: Path) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], set[tuple[str, str]]]:
    target_path = request_pack / "source_link_request_targets.jsonl"
    if not target_path.exists():
        raise ValueError(f"missing request-pack targets: {target_path}")
    targets_by_hash: dict[tuple[str, str], list[dict[str, Any]]] = {}
    source_hashes_by_venue: set[tuple[str, str]] = set()
    for line_no, row in _iter_jsonl(target_path):
        _check_unsafe_flags(row, target_path, label=f"target:{line_no}")
        venue = str(row.get("venue_id") or "").lower()
        if venue not in VENUES:
            continue
        for key in ("client_order_id_hash", "order_id_hash"):
            value = str(row.get(key) or "").lower()
            if value:
                targets_by_hash.setdefault((venue, value), []).append(row)
    source_path = request_pack / "source_link_request_sources.jsonl"
    if source_path.exists():
        for line_no, row in _iter_jsonl(source_path):
            _check_unsafe_flags(row, source_path, label=f"request-source:{line_no}")
            venue = str(row.get("venue_id") or "").lower()
            source_hash = str(row.get("source_record_sha256") or "").lower()
            if venue in VENUES and source_hash:
                source_hashes_by_venue.add((venue, source_hash))
    return targets_by_hash, source_hashes_by_venue


def _match_target(
    venue: str,
    client_hashes: list[str],
    order_hashes: list[str],
    targets_by_hash: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    matched: dict[str, dict[str, Any]] = {}
    ambiguous = False
    for value in client_hashes + order_hashes:
        targets = targets_by_hash.get((venue, value.lower()), [])
        if len(targets) > 1:
            ambiguous = True
        elif len(targets) == 1:
            target = targets[0]
            target_id = str(target.get("canonical_group_id") or target.get("order_key") or "")
            if target_id:
                matched[target_id] = target
    if ambiguous or len(matched) > 1:
        return None, "AMBIGUOUS_TARGET_HASH_MATCH"
    if matched:
        return next(iter(matched.values())), "DIRECT_TARGET_HASH_MATCH"
    return None, "NO_TARGET_HASH_MATCH"


def _source_row(
    *,
    run_id: str,
    seq: int,
    timestamp_ns: int,
    venue: str,
    source_hash: str,
    target: dict[str, Any],
    native_payload: dict[str, Any],
) -> dict[str, Any]:
    row = _base_record(run_id, seq, timestamp_ns, "PHASE51AJ_FORWARD_PRIVATE_STREAM_SOURCE")
    row.update(
        {
            "venue_id": venue,
            "canonical_group_id": str(target.get("canonical_group_id") or ""),
            "order_key": str(target.get("order_key") or ""),
            "source_record_sha256": source_hash,
            **native_payload,
        }
    )
    return row


def _source_link_row(source_hash: str, target: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_record_sha256": source_hash,
        "canonical_group_id": str(target.get("canonical_group_id") or ""),
        "order_key": str(target.get("order_key") or ""),
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
    }


def _status_counts(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def _parse_source_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--source-json entries must be in venue=/path/to/file form")
    venue, path_text = spec.split("=", 1)
    venue = venue.strip().lower()
    if venue not in VENUES:
        raise ValueError(f"unsupported venue in --source-json: {venue}")
    return venue, _check_local_file(Path(path_text.strip()), label=f"{venue} source")


def _candidate_manifest(source_path: Path, source_link_path: Path) -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "baseline_commit": BASELINE_COMMIT,
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
        "sources": [
            {
                "source_id": "phase51aj_forward_private_stream_source_rows",
                "venue_id": "all5",
                "path": str(source_path),
            }
        ],
        "source_links": [
            {
                "source_link_id": "phase51aj_forward_private_stream_proposed_source_links",
                "path": str(source_link_path),
            }
        ],
    }


def build_forward_private_stream_source(
    *,
    request_pack: Path,
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
    source_specs: list[str],
) -> Path:
    request_pack = _check_local_file(request_pack / "manifest.json", label="request pack manifest").parent
    output_root = _resolve_path(output_root)
    run_id = _check_run_id(run_id)
    out_dir = output_root / run_id
    snapshot_dir = out_dir / "source_snapshots"
    evidence_dir = out_dir / "evidence_pack"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets_by_hash, request_source_hashes = _load_request_targets(request_pack)
    source_files = [_parse_source_spec(spec) for spec in source_specs]

    labels: list[dict[str, Any]] = []
    source_rows_by_target: dict[str, dict[str, Any]] = {}
    source_links_by_source: dict[str, dict[str, Any]] = {}
    seq = 0
    raw_row_count_by_venue: dict[str, int] = {}
    direct_target_count_by_venue: dict[str, int] = {}
    request_source_overlap_count_by_venue: dict[str, int] = {}

    for venue, path in source_files:
        for row in _load_source_rows(path):
            _check_unsafe_flags(row, path, label=f"{venue} source input")
            _check_no_secret_fields(row, path, label=f"{venue} source input")
            seq += 1
            raw_row_count_by_venue[venue] = raw_row_count_by_venue.get(venue, 0) + 1
            source_hash = _stable_hash(row).lower()
            native_payload, native_status = _extract_native_payload(venue, row)
            client_hashes, order_hashes = _identity_hashes(venue, row)
            target, target_status = _match_target(venue, client_hashes, order_hashes, targets_by_hash)
            source_hash_in_request = (venue, source_hash) in request_source_hashes
            if source_hash_in_request:
                request_source_overlap_count_by_venue[venue] = request_source_overlap_count_by_venue.get(venue, 0) + 1

            materialization_status = "NOT_MATERIALIZED"
            if native_payload is not None and target is not None:
                target_id = str(target.get("canonical_group_id") or target.get("order_key") or "")
                if target_id and target_id not in source_rows_by_target:
                    source_rows_by_target[target_id] = _source_row(
                        run_id=run_id,
                        seq=len(source_rows_by_target),
                        timestamp_ns=timestamp_ns,
                        venue=venue,
                        source_hash=source_hash,
                        target=target,
                        native_payload=native_payload,
                    )
                    direct_target_count_by_venue[venue] = direct_target_count_by_venue.get(venue, 0) + 1
                    materialization_status = "TARGET_LINKED_SOURCE_ROW_EMITTED"
                else:
                    materialization_status = "DUPLICATE_TARGET_SUPPRESSED"
                if source_hash_in_request and source_hash not in source_links_by_source:
                    source_links_by_source[source_hash] = _source_link_row(source_hash, target)
            elif native_payload is None:
                materialization_status = native_status
            else:
                materialization_status = target_status

            label = _base_record(run_id, seq, timestamp_ns, "PHASE51AJ_FORWARD_PRIVATE_STREAM_SOURCE_LABEL")
            label.update(
                {
                    "venue_id": venue,
                    "source_record_sha256": source_hash,
                    "source_hash_in_request": source_hash_in_request,
                    "native_field_status": native_status,
                    "target_match_status": target_status,
                    "materialization_status": materialization_status,
                    "canonical_group_id": str(target.get("canonical_group_id") or "") if target else None,
                    "order_key": str(target.get("order_key") or "") if target else None,
                    "client_order_id_hash_candidate_count": len(client_hashes),
                    "order_id_hash_candidate_count": len(order_hashes),
                    "raw_identifier_redaction_status": "PASS",
                }
            )
            labels.append(label)

    source_rows = sorted(
        source_rows_by_target.values(),
        key=lambda row: (str(row.get("venue_id") or ""), str(row.get("canonical_group_id") or "")),
    )
    source_links = sorted(
        source_links_by_source.values(),
        key=lambda row: (str(row.get("canonical_group_id") or ""), str(row.get("source_record_sha256") or "")),
    )

    source_rows_path = snapshot_dir / "phase51aj_forward_private_stream_source_rows.jsonl"
    source_links_path = out_dir / "source_links.proposed.sanitized.jsonl"
    labels_path = out_dir / "phase51aj_forward_private_stream_source_labels.jsonl"
    candidate_manifest_path = out_dir / "phase51aj_candidate_manifest.json"
    summary_path = out_dir / "phase51aj_forward_private_stream_source_summary.json"
    manifest_path = out_dir / "manifest.json"
    artifact_index_path = evidence_dir / "artifact_index.json"

    _write_jsonl(source_rows_path, source_rows)
    _write_jsonl(source_links_path, source_links)
    _write_jsonl(labels_path, labels)
    _write_json(candidate_manifest_path, _candidate_manifest(source_rows_path, source_links_path))

    status_counts = _status_counts(labels, "materialization_status")
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51aj_forward_private_stream_source_nonlive_hold",
        "timestamp_local_ns": timestamp_ns,
        "timestamp_utc": _timestamp_ns_to_utc(timestamp_ns),
        "request_pack": str(request_pack),
        "source_file_count": len(source_files),
        "raw_row_count": len(labels),
        "raw_row_count_by_venue": dict(sorted(raw_row_count_by_venue.items())),
        "direct_target_linked_source_row_count": len(source_rows),
        "direct_target_linked_source_row_count_by_venue": dict(sorted(direct_target_count_by_venue.items())),
        "request_source_hash_overlap_count": sum(request_source_overlap_count_by_venue.values()),
        "request_source_hash_overlap_count_by_venue": dict(sorted(request_source_overlap_count_by_venue.items())),
        "source_link_count": len(source_links),
        "materialization_status_counts": status_counts,
        "candidate_manifest_path": str(candidate_manifest_path),
        "source_rows_path": str(source_rows_path),
        "source_links_path": str(source_links_path),
        "source_rows_sha256": _sha256_file(source_rows_path),
        "source_links_sha256": _sha256_file(source_links_path),
        "raw_identifier_redaction_status": "PASS",
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
        "official_docs": OFFICIAL_DOCS,
        "next_required_action": "compose_phase51aj_candidate_manifest_into_phase51v_bundle",
    }
    _write_json(summary_path, summary)

    artifacts = [
        source_rows_path,
        source_links_path,
        labels_path,
        candidate_manifest_path,
        summary_path,
    ]
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "created_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "artifacts": _artifact_infos(out_dir, artifacts),
        "no_live_flag": True,
        "approved_for_live": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
    }
    _write_json(manifest_path, manifest)
    _write_json(
        artifact_index_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "baseline_commit": BASELINE_COMMIT,
            "artifacts": _artifact_infos(out_dir, artifacts + [manifest_path]),
            "no_live_flag": True,
            "approved_for_live": False,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "risk_limit_relaxation_allowed": False,
        },
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-pack", type=Path, default=DEFAULT_REQUEST_PACK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"phase51aj_forward_private_stream_source_{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int, default=time.time_ns())
    parser.add_argument(
        "--source-json",
        action="append",
        default=[],
        help="Local source artifact in venue=/path/to/file form. May be .json, .jsonl, or .ndjson.",
    )
    args = parser.parse_args(argv)

    if not args.source_json:
        print("phase51aj_forward_private_stream_source: ERROR: provide at least one --source-json venue=path", file=sys.stderr)
        return 2

    try:
        out_dir = build_forward_private_stream_source(
            request_pack=args.request_pack,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
            source_specs=args.source_json,
        )
    except Exception as exc:
        print(f"phase51aj_forward_private_stream_source: ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"phase51aj_forward_private_stream_source: wrote {out_dir}")
    print("phase51aj_forward_private_stream_source: status HOLD (forward private source only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
