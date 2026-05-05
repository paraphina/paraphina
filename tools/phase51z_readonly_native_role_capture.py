#!/usr/bin/env python3
"""Phase 5.1z read-only native-role source capture.

This HOLD-only utility attempts to acquire venue-native maker/taker fields
from read-only private source surfaces, maps them to the existing redacted
Phase 5.1u targets, and emits sanitized local rows for the established
5.1y/5.1v gates.

It never submits, edits, cancels, or replaces orders. It never writes secrets
or raw venue identifiers to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51z_readonly_native_role_capture"
DEFAULT_USER_AGENT = "paraphina-phase51z-readonly-native-role-capture/1.0"
VENUES = ("aster", "extended", "lighter", "paradex")
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
    "private_key",
    "privatekey",
    "secret_key",
    "secretkey",
    "access_key",
    "accesskey",
    "auth_token",
    "authtoken",
    "authorization",
    "bearer",
    "jwt",
    "mnemonic",
    "passphrase",
    "password",
    "session_token",
    "signature",
    "signing_key",
    "token",
}
SOURCE_LIST_KEYS = {
    "data",
    "events",
    "fills",
    "results",
    "rows",
    "trade_history",
    "tradeHistory",
    "trades",
}
OFFICIAL_DOCS = {
    "aster": [
        "https://docs.asterdex.com/product/aster-perpetuals/api/api-documentation",
    ],
    "extended": [
        "https://api.docs.extended.exchange/",
    ],
    "lighter": [
        "https://apidocs.lighter.xyz/reference/trades",
    ],
    "paradex": [
        "https://docs.paradex.trade/api/prod/account/list-fills/",
    ],
}


@dataclass(frozen=True)
class CaptureTarget:
    venue_id: str
    canonical_group_id: str
    order_key: str
    order_id_hash: str | None
    client_order_id_hash: str | None
    first_fill_time_ms: int | None
    last_fill_time_ms: int | None


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            yield line_no, record


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            if not key or not key.replace("_", "").isalnum():
                continue
            try:
                parsed = shlex.split(raw_value, comments=False, posix=True)
            except ValueError:
                parsed = [raw_value.strip().strip("'\"")]
            values[key] = parsed[0] if parsed else ""
    return values


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


def _check_output_safe(record: dict[str, Any], path: Path) -> None:
    _check_unsafe_flags(record, path, label="output")
    _check_no_secret_fields(record, path, label="output")


def _path_text_is_unsafe(path_text: str) -> bool:
    lower = path_text.lower()
    return lower.startswith(("http://", "https://")) or "://" in lower


def _is_env_path(path: Path) -> bool:
    return any(part == ".env" or part.endswith(".env") for part in path.parts)


def _check_no_symlink(path: Path) -> None:
    current = path if path.is_absolute() else _resolve_path(path)
    chain = [current]
    chain.extend(current.parents)
    for candidate in chain:
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"symlink source path is prohibited: {candidate}")


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive_float(value: Any) -> bool:
    parsed = _safe_float(value)
    return parsed is not None and parsed > 0.0


def _payload_records(payload: Any, inherited: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    inherited = dict(inherited or {})
    if isinstance(payload, list):
        out: list[dict[str, Any]] = []
        for item in payload:
            out.extend(_payload_records(item, inherited))
        return out
    if not isinstance(payload, dict):
        return []
    merged = {**inherited}
    for key in ("account_index", "market", "market_id", "market_type", "venue", "venue_id"):
        if key in payload:
            merged[key] = payload[key]
    for key in SOURCE_LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            out: list[dict[str, Any]] = []
            for item in value:
                out.extend(_payload_records(item, merged))
            return out
    return [{**merged, **payload}]


def _iter_source_records(source_specs: list[str]):
    for spec in source_specs:
        if "=" in spec:
            venue, raw_path = spec.split("=", 1)
            venue = venue.strip().lower()
        else:
            venue, raw_path = "", spec
        if venue and venue not in VENUES:
            raise ValueError(f"unsupported source venue {venue!r}")
        if _path_text_is_unsafe(raw_path):
            raise ValueError("network source paths are prohibited")
        path = _resolve_path(Path(raw_path))
        if _is_env_path(path):
            raise ValueError("env files are prohibited as native source input")
        _check_no_symlink(path)
        if not path.exists() or not path.is_file():
            raise ValueError(f"source path does not exist or is not a file: {path}")
        if path.suffix not in {".json", ".jsonl"}:
            raise ValueError(f"unsupported source suffix for {path}")
        if path.suffix == ".jsonl":
            for line_no, row in _iter_jsonl(path):
                _check_unsafe_flags(row, path, label="source row")
                _check_no_secret_fields(row, path, label="source row")
                for item in _payload_records(row):
                    yield path, line_no, venue, item
        else:
            payload = _load_json(path)
            _check_unsafe_flags(payload, path, label="source payload")
            _check_no_secret_fields(payload, path, label="source payload")
            for line_no, row in enumerate(_payload_records(payload), start=1):
                _check_unsafe_flags(row, path, label="source row")
                _check_no_secret_fields(row, path, label="source row")
                yield path, line_no, venue, row


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
        "admissible_for_model_training": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "raw_identifier_redaction_status": "PASS",
    }


def _target_id(target: CaptureTarget) -> str:
    return target.canonical_group_id or target.order_key


def _load_targets(target_run: Path) -> tuple[dict[str, Any], list[CaptureTarget]]:
    target_run = _resolve_path(target_run)
    summary = _load_json(target_run / "phase51u_forward_capture_target_manifest_summary.json")
    if not isinstance(summary, dict):
        raise ValueError("Phase 5.1u target summary must be a JSON object")
    _check_unsafe_flags(summary, target_run / "phase51u_forward_capture_target_manifest_summary.json", label="summary")
    if summary.get("baseline_commit") != BASELINE_COMMIT:
        raise ValueError("Phase 5.1u baseline_commit mismatch")
    observed_run = _resolve_path(Path(str(summary["observed_pfill_run"])))
    pfill_by_key: dict[str, dict[str, Any]] = {}
    pfill_by_group: dict[str, dict[str, Any]] = {}
    for _, label in _iter_jsonl(observed_run / "pfill_order_labels.jsonl"):
        if label.get("label_type") != "ORDER_PFILL_OUTCOME_LABEL":
            continue
        pfill_by_key[str(label.get("order_key") or "")] = label
        pfill_by_group[str(label.get("canonical_group_id") or "")] = label

    targets: list[CaptureTarget] = []
    for _, row in _iter_jsonl(target_run / "native_role_capture_targets.jsonl"):
        _check_unsafe_flags(row, target_run / "native_role_capture_targets.jsonl", label="native role target")
        group = str(row.get("canonical_group_id") or "")
        order_key = str(row.get("order_key") or "")
        pfill = pfill_by_key.get(order_key) or pfill_by_group.get(group) or {}
        targets.append(
            CaptureTarget(
                venue_id=str(row.get("venue_id") or "").lower(),
                canonical_group_id=group,
                order_key=order_key,
                order_id_hash=str(pfill.get("order_id_hash") or "") or None,
                client_order_id_hash=str(pfill.get("client_order_id_hash") or "") or None,
                first_fill_time_ms=_safe_int(pfill.get("first_fill_time_ms")),
                last_fill_time_ms=_safe_int(pfill.get("last_fill_time_ms")),
            )
        )
    return summary, targets


def _build_hash_index(targets: list[CaptureTarget]) -> dict[tuple[str, str], CaptureTarget | None]:
    index: dict[tuple[str, str], CaptureTarget | None] = {}
    for target in targets:
        for hashed in (target.order_id_hash, target.client_order_id_hash):
            if not hashed:
                continue
            key = (target.venue_id, hashed)
            if key in index and index[key] != target:
                index[key] = None
            else:
                index[key] = target
    return index


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
            hashed = _stable_hash(candidate)
            if hashed not in seen:
                seen.add(hashed)
                out.append(hashed)
    return out


def _match_target(
    venue: str,
    hashes: list[str],
    target_index: dict[tuple[str, str], CaptureTarget | None],
) -> tuple[CaptureTarget | None, str]:
    matched: dict[str, CaptureTarget] = {}
    ambiguous = False
    for hashed in hashes:
        target = target_index.get((venue, hashed))
        if target is None and (venue, hashed) in target_index:
            ambiguous = True
            continue
        if target is not None:
            matched[_target_id(target)] = target
    if len(matched) == 1:
        return next(iter(matched.values())), "TARGET_MATCHED_BY_REDACTED_ID_HASH"
    if len(matched) > 1 or ambiguous:
        return None, "AMBIGUOUS_REDACTED_ID_HASH"
    return None, "NO_TARGET_MATCH"


def _http_get_json(
    base_url: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout_s: float = 20.0,
) -> Any:
    encoded = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None}, doseq=True)
    url = f"{base_url.rstrip('/')}{path}"
    if encoded:
        url = f"{url}?{encoded}"
    req_headers = {"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT}
    req_headers.update(headers or {})
    request = urllib.request.Request(url, headers=req_headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:  # pragma: no cover - exercised only against live APIs
        raise RuntimeError(f"GET {path} failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - exercised only against live APIs
        raise RuntimeError(f"GET {path} failed: {exc.reason}") from exc


def _aster_signed_query(env: dict[str, str], params: dict[str, Any]) -> str:
    signed = {k: v for k, v in params.items() if v is not None}
    signed["timestamp"] = str(int(time.time() * 1000))
    recv_window = env.get("ASTER_RECV_WINDOW", "").strip()
    if recv_window:
        signed["recvWindow"] = recv_window
    canonical = urllib.parse.urlencode(sorted((str(k), str(v)) for k, v in signed.items()))
    signature = hmac.new(
        env["ASTER_API_SECRET"].encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{canonical}&signature={signature}"


def _aster_get(env: dict[str, str], path: str, params: dict[str, Any], timeout_s: float) -> Any:
    base_url = env.get("ASTER_REST_URL", "https://fapi.asterdex.com").rstrip("/")
    query = _aster_signed_query(env, params)
    headers = {"X-MBX-APIKEY": env["ASTER_API_KEY"]}
    request = urllib.request.Request(f"{base_url}{path}?{query}", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:  # pragma: no cover
        raise RuntimeError(f"GET {path} failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover
        raise RuntimeError(f"GET {path} failed: {exc.reason}") from exc


def _time_window_for_targets(targets: list[CaptureTarget], venue: str, pad_ms: int) -> tuple[int | None, int | None]:
    times: list[int] = []
    for target in targets:
        if target.venue_id != venue:
            continue
        if target.first_fill_time_ms is not None:
            times.append(target.first_fill_time_ms)
        if target.last_fill_time_ms is not None:
            times.append(target.last_fill_time_ms)
    if not times:
        return None, None
    return max(0, min(times) - pad_ms), max(times) + pad_ms


def _paginate_aster(
    env: dict[str, str],
    path: str,
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    timeout_s: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor = start_ms
    chunk_ms = 6 * 24 * 60 * 60 * 1000
    while cursor <= end_ms:
        chunk_end = min(end_ms, cursor + chunk_ms)
        while cursor <= chunk_end:
            payload = _aster_get(
                env,
                path,
                {"symbol": symbol, "startTime": cursor, "endTime": chunk_end, "limit": limit},
                timeout_s,
            )
            batch = [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []
            out.extend(batch)
            batch_times = [_safe_int(item.get("time") or item.get("updateTime")) or cursor for item in batch]
            batch_max = max(batch_times or [cursor])
            if len(batch) < limit or batch_max < cursor:
                break
            next_cursor = batch_max + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
        cursor = chunk_end + 1
    return out


def _fetch_aster(env: dict[str, str], targets: list[CaptureTarget], timeout_s: float, limit: int, pad_ms: int) -> list[dict[str, Any]]:
    start_ms, end_ms = _time_window_for_targets(targets, "aster", pad_ms)
    if start_ms is None or end_ms is None:
        return []
    symbol = env.get("ASTER_MARKET", "ETHUSDT").strip() or "ETHUSDT"
    trades = _paginate_aster(
        env,
        "/fapi/v1/userTrades",
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=min(limit, 1000),
        timeout_s=timeout_s,
    )
    orders = _paginate_aster(
        env,
        "/fapi/v1/allOrders",
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=min(limit, 1000),
        timeout_s=timeout_s,
    )
    client_by_order: dict[str, Any] = {}
    for order in orders:
        order_id = order.get("orderId") or order.get("order_id")
        client_id = order.get("clientOrderId") or order.get("client_order_id") or order.get("origClientOrderId")
        if order_id not in (None, "") and client_id not in (None, ""):
            client_by_order[str(order_id)] = client_id
    rows: list[dict[str, Any]] = []
    for trade in trades:
        row = dict(trade)
        row["venue_id"] = "aster"
        order_id = row.get("orderId") or row.get("order_id")
        if order_id not in (None, "") and (row.get("clientOrderId") in (None, "")):
            client_id = client_by_order.get(str(order_id))
            if client_id not in (None, ""):
                row["clientOrderId"] = client_id
        rows.append(row)
    return rows


def _paradex_token(env: dict[str, str], *, allow_jwt_cmd: bool) -> str:
    for key in ("PARADEX_READONLY_TOKEN", "PARADEX_JWT"):
        token = env.get(key, "").strip()
        if token:
            return token
    if not allow_jwt_cmd:
        raise RuntimeError("PARADEX_JWT_CMD present but --allow-paradex-jwt-cmd not set")
    command = env.get("PARADEX_JWT_CMD", "").strip()
    if not command:
        raise RuntimeError("PARADEX_JWT or PARADEX_JWT_CMD is required")
    argv = shlex.split(command)
    if not argv:
        raise RuntimeError("PARADEX_JWT_CMD is empty")
    merged_env = os.environ.copy()
    merged_env.update(env)
    completed = subprocess.run(argv, text=True, capture_output=True, check=True, env=merged_env)
    token = completed.stdout.strip()
    if not token:
        raise RuntimeError("PARADEX_JWT_CMD returned an empty token")
    return token


def _fetch_paradex(
    env: dict[str, str],
    targets: list[CaptureTarget],
    timeout_s: float,
    limit: int,
    pad_ms: int,
    max_pages: int,
    *,
    allow_jwt_cmd: bool,
) -> list[dict[str, Any]]:
    start_ms, end_ms = _time_window_for_targets(targets, "paradex", pad_ms)
    if start_ms is None or end_ms is None:
        return []
    base_url = env.get("PARADEX_REST_URL", "https://api.prod.paradex.trade/v1").rstrip("/")
    token = _paradex_token(env, allow_jwt_cmd=allow_jwt_cmd)
    headers = {"Authorization": f"Bearer {token}"}
    market = env.get("PARADEX_MARKET", "ETH-USD-PERP").strip() or "ETH-USD-PERP"
    cursor: str | None = None
    rows: list[dict[str, Any]] = []
    for _ in range(max_pages):
        payload = _http_get_json(
            base_url,
            "/fills",
            headers=headers,
            params={"market": market, "start_at": start_ms, "end_at": end_ms, "page_size": min(limit, 5000), "cursor": cursor},
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


def _fetch_extended(env: dict[str, str], timeout_s: float, limit: int, max_pages: int) -> list[dict[str, Any]]:
    base_url = env.get("EXTENDED_REST_URL", "https://api.starknet.extended.exchange").rstrip("/")
    market = env.get("EXTENDED_MARKET", "ETH-USD").strip() or "ETH-USD"
    headers = {"X-Api-Key": env["EXTENDED_API_KEY"]}
    rows: list[dict[str, Any]] = []
    cursor: int | None = None
    seen: set[int] = set()
    for _ in range(max_pages):
        payload = _http_get_json(
            base_url,
            "/api/v1/user/trades",
            headers=headers,
            params={"market": market, "type": "trade", "limit": limit, "cursor": cursor},
            timeout_s=timeout_s,
        )
        batch = [item for item in (payload.get("data") if isinstance(payload, dict) else []) or [] if isinstance(item, dict)]
        for row in batch:
            enriched = dict(row)
            enriched["venue_id"] = "extended"
            rows.append(enriched)
        next_cursor = ((payload.get("pagination") or {}).get("cursor") if isinstance(payload, dict) else None)
        parsed = _safe_int(next_cursor)
        if not batch or parsed is None or parsed in seen:
            break
        seen.add(parsed)
        cursor = parsed
    return rows


def _extract_aster_source(row: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None, str]:
    client_id = row.get("clientOrderId") or row.get("client_order_id") or row.get("origClientOrderId")
    hashes = _candidate_hashes(client_id)
    maker = row.get("maker")
    if maker is None:
        maker = row.get("m")
    qty = row.get("qty") or row.get("l") or row.get("lastFilledQty")
    if not isinstance(maker, bool):
        return hashes, None, "ASTER_MAKER_FIELD_MISSING"
    if not _positive_float(qty):
        return hashes, None, "ASTER_POSITIVE_FILL_QTY_MISSING"
    return hashes, {"e": "ORDER_TRADE_UPDATE", "o": {"m": maker, "l": str(qty)}}, "NATIVE_ROLE_FIELDS_READY"


def _extract_extended_source(row: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None, str]:
    client_id = row.get("externalOrderId") or row.get("externalId") or row.get("client_order_id")
    hashes = _candidate_hashes(client_id)
    is_taker = row.get("isTaker")
    if is_taker is None:
        is_taker = row.get("is_taker")
    if not isinstance(is_taker, bool):
        return hashes, None, "EXTENDED_ISTAKER_MISSING"
    return hashes, {"isTaker": is_taker}, "NATIVE_ROLE_FIELDS_READY"


def _extract_lighter_source(row: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None, str]:
    account_index = _safe_int(row.get("account_index"))
    ask_account = _safe_int(row.get("ask_account_id") or row.get("askAccountId"))
    bid_account = _safe_int(row.get("bid_account_id") or row.get("bidAccountId"))
    is_maker_ask = row.get("is_maker_ask")
    if is_maker_ask is None:
        is_maker_ask = row.get("isMakerAsk")
    hashes = _candidate_hashes(
        row.get("ask_id"),
        row.get("ask_id_str"),
        row.get("askId"),
        row.get("ask_client_id"),
        row.get("ask_client_id_str"),
        row.get("askClientId"),
        row.get("bid_id"),
        row.get("bid_id_str"),
        row.get("bidId"),
        row.get("bid_client_id"),
        row.get("bid_client_id_str"),
        row.get("bidClientId"),
    )
    if account_index is None:
        return hashes, None, "LIGHTER_ACCOUNT_INDEX_MISSING"
    if ask_account is None or bid_account is None:
        return hashes, None, "LIGHTER_SIDE_ACCOUNT_ID_MISSING"
    if not isinstance(is_maker_ask, bool):
        return hashes, None, "LIGHTER_IS_MAKER_ASK_MISSING"
    return hashes, {
        "account_index": account_index,
        "is_maker_ask": is_maker_ask,
        "ask_account_id": ask_account,
        "bid_account_id": bid_account,
    }, "NATIVE_ROLE_FIELDS_READY"


def _extract_paradex_source(row: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None, str]:
    hashes = _candidate_hashes(row.get("client_id") or row.get("clientId"))
    liquidity = str(row.get("liquidity") or "").upper()
    if liquidity not in {"MAKER", "TAKER"}:
        return hashes, None, "PARADEX_LIQUIDITY_MISSING"
    return hashes, {"liquidity": liquidity}, "NATIVE_ROLE_FIELDS_READY"


def _extract_source(venue: str, row: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None, str]:
    if venue == "aster":
        return _extract_aster_source(row)
    if venue == "extended":
        return _extract_extended_source(row)
    if venue == "lighter":
        return _extract_lighter_source(row)
    if venue == "paradex":
        return _extract_paradex_source(row)
    return [], None, "UNSUPPORTED_VENUE"


def _venue_id(row: dict[str, Any], fallback: str) -> str:
    return str(row.get("venue_id") or row.get("venue") or fallback or "").lower()


def _sanitize_rows(
    *,
    run_id: str,
    timestamp_ns: int,
    rows_by_venue: dict[str, list[dict[str, Any]]],
    target_index: dict[tuple[str, str], CaptureTarget | None],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    output_rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    emitted_targets: set[str] = set()
    seq = 0
    for venue in VENUES:
        for row in rows_by_venue.get(venue, []):
            seq += 1
            source_hash = _stable_hash(row)
            hashes, native_payload, field_status = _extract_source(venue, row)
            target, match_status = _match_target(venue, hashes, target_index)
            status = field_status if native_payload is None else match_status
            if native_payload is not None and target is not None:
                target_id = _target_id(target)
                if target_id not in emitted_targets:
                    out = _base_record(run_id, len(output_rows), timestamp_ns, "PHASE51Z_READONLY_NATIVE_ROLE_SOURCE")
                    out.update(
                        {
                            "venue_id": venue,
                            "canonical_group_id": target.canonical_group_id,
                            "order_key": target.order_key,
                            "source_record_sha256": source_hash,
                            **native_payload,
                        }
                    )
                    output_rows.append(out)
                    emitted_targets.add(target_id)
                    status = "SANITIZED_SOURCE_ROW_EMITTED"
            label = _base_record(run_id, seq, timestamp_ns, "PHASE51Z_READONLY_NATIVE_ROLE_CAPTURE_LABEL")
            label.update(
                {
                    "venue_id": venue,
                    "source_record_sha256": source_hash,
                    "capture_status": status,
                    "native_field_status": field_status,
                    "target_match_status": match_status,
                    "canonical_group_id": target.canonical_group_id if target else None,
                    "order_key": target.order_key if target else None,
                    "redacted_id_hash_candidate_count": len(hashes),
                }
            )
            labels.append(label)
            status_counts[status] = status_counts.get(status, 0) + 1
    return output_rows, labels, dict(sorted(status_counts.items()))


def _artifact_infos(root_dir: Path, artifact_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(artifact_paths)
    ]


def _count_by_venue(targets: list[CaptureTarget]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for target in targets:
        counts[target.venue_id] = counts.get(target.venue_id, 0) + 1
    return dict(sorted(counts.items()))


def _target_time_windows_by_venue(targets: list[CaptureTarget], pad_ms: int) -> dict[str, dict[str, Any]]:
    windows: dict[str, dict[str, Any]] = {}
    for venue in VENUES:
        start_ms, end_ms = _time_window_for_targets(targets, venue, pad_ms)
        windows[venue] = {
            "available": start_ms is not None and end_ms is not None,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_utc": (
                datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()
                if start_ms is not None
                else None
            ),
            "end_utc": (
                datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat()
                if end_ms is not None
                else None
            ),
            "pad_ms": pad_ms,
        }
    return windows


def _increment_nested(
    counts: dict[str, dict[str, int]],
    venue: str,
    key: str | None,
) -> None:
    normalized_key = str(key or "UNKNOWN")
    venue_counts = counts.setdefault(venue, {})
    venue_counts[normalized_key] = venue_counts.get(normalized_key, 0) + 1


def _build_capture_diagnostics(
    *,
    targets: list[CaptureTarget],
    rows_by_venue: dict[str, list[dict[str, Any]]],
    output_rows: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    target_counts = _count_by_venue(targets)
    emitted_counts: dict[str, int] = {}
    for row in output_rows:
        venue = str(row.get("venue_id") or "unknown")
        emitted_counts[venue] = emitted_counts.get(venue, 0) + 1

    field_status_counts: dict[str, dict[str, int]] = {}
    target_match_status_counts: dict[str, dict[str, int]] = {}
    capture_status_counts: dict[str, dict[str, int]] = {}
    native_field_ready_counts: dict[str, int] = {}
    target_matched_row_counts: dict[str, int] = {}
    duplicate_matched_row_counts: dict[str, int] = {}
    no_target_match_counts: dict[str, int] = {}
    rows_with_hash_candidates: dict[str, int] = {}
    hash_candidate_count_sum: dict[str, int] = {}

    for label in labels:
        venue = str(label.get("venue_id") or "unknown")
        field_status = str(label.get("native_field_status") or "UNKNOWN")
        match_status = str(label.get("target_match_status") or "UNKNOWN")
        capture_status = str(label.get("capture_status") or "UNKNOWN")
        hash_count = _safe_int(label.get("redacted_id_hash_candidate_count")) or 0

        _increment_nested(field_status_counts, venue, field_status)
        _increment_nested(target_match_status_counts, venue, match_status)
        _increment_nested(capture_status_counts, venue, capture_status)

        if field_status == "NATIVE_ROLE_FIELDS_READY":
            native_field_ready_counts[venue] = native_field_ready_counts.get(venue, 0) + 1
        if match_status == "TARGET_MATCHED_BY_REDACTED_ID_HASH":
            target_matched_row_counts[venue] = target_matched_row_counts.get(venue, 0) + 1
            if capture_status != "SANITIZED_SOURCE_ROW_EMITTED":
                duplicate_matched_row_counts[venue] = duplicate_matched_row_counts.get(venue, 0) + 1
        if match_status == "NO_TARGET_MATCH":
            no_target_match_counts[venue] = no_target_match_counts.get(venue, 0) + 1
        if hash_count > 0:
            rows_with_hash_candidates[venue] = rows_with_hash_candidates.get(venue, 0) + 1
            hash_candidate_count_sum[venue] = hash_candidate_count_sum.get(venue, 0) + hash_count

    diagnostics: dict[str, dict[str, Any]] = {}
    for venue in VENUES:
        source_row_count = len(rows_by_venue.get(venue, []))
        emitted_count = emitted_counts.get(venue, 0)
        target_count = target_counts.get(venue, 0)
        hash_rows = rows_with_hash_candidates.get(venue, 0)
        diagnostics[venue] = {
            "target_count": target_count,
            "target_ready_count": emitted_count,
            "target_missing_count": max(target_count - emitted_count, 0),
            "source_row_count": source_row_count,
            "native_field_ready_count": native_field_ready_counts.get(venue, 0),
            "target_matched_row_count": target_matched_row_counts.get(venue, 0),
            "duplicate_matched_row_count": duplicate_matched_row_counts.get(venue, 0),
            "no_target_match_count": no_target_match_counts.get(venue, 0),
            "rows_with_redacted_hash_candidates": hash_rows,
            "average_redacted_hash_candidate_count": (
                round(hash_candidate_count_sum.get(venue, 0) / hash_rows, 6)
                if hash_rows
                else 0.0
            ),
            "field_status_counts": dict(sorted(field_status_counts.get(venue, {}).items())),
            "target_match_status_counts": dict(sorted(target_match_status_counts.get(venue, {}).items())),
            "capture_status_counts": dict(sorted(capture_status_counts.get(venue, {}).items())),
        }
    return diagnostics


def _presence(env: dict[str, str], keys: list[str]) -> dict[str, bool]:
    return {key: bool(env.get(key, "").strip()) for key in keys}


def build_readonly_native_role_capture(
    *,
    target_run: Path,
    source_json: list[str],
    env_file: Path | None,
    fetch_readonly: bool,
    venues: list[str],
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
    timeout_s: float,
    limit: int,
    max_pages: int,
    window_pad_ms: int,
    allow_paradex_jwt_cmd: bool,
) -> Path:
    target_summary, targets = _load_targets(target_run)
    target_index = _build_hash_index(targets)
    output_root = _resolve_path(output_root)
    out_dir = output_root / run_id
    source_dir = out_dir / "source_snapshots"
    source_dir.mkdir(parents=True, exist_ok=True)

    selected_venues = sorted(set(venue.lower() for venue in venues))
    unsupported = sorted(set(selected_venues) - set(VENUES))
    if unsupported:
        raise ValueError(f"unsupported venues: {unsupported}")

    rows_by_venue: dict[str, list[dict[str, Any]]] = {venue: [] for venue in VENUES}
    source_file_counts: dict[str, int] = {}
    for path, _line_no, fallback_venue, row in _iter_source_records(source_json):
        venue = _venue_id(row, fallback_venue)
        if venue not in VENUES:
            continue
        rows_by_venue.setdefault(venue, []).append(row)
        source_file_counts[path.name] = source_file_counts.get(path.name, 0) + 1

    env: dict[str, str] = {}
    fetch_status: dict[str, dict[str, Any]] = {}
    credential_presence: dict[str, dict[str, bool]] = {}
    if fetch_readonly:
        if env_file is None:
            raise ValueError("--env-file is required with --fetch-readonly")
        env_path = _resolve_path(env_file)
        env = _load_env_file(env_path)
        credential_presence = {
            "aster": _presence(env, ["ASTER_API_KEY", "ASTER_API_SECRET", "ASTER_MARKET"]),
            "extended": _presence(env, ["EXTENDED_API_KEY", "EXTENDED_MARKET"]),
            "paradex": _presence(env, ["PARADEX_JWT", "PARADEX_READONLY_TOKEN", "PARADEX_JWT_CMD", "PARADEX_MARKET"]),
        }
        for venue in selected_venues:
            if venue == "lighter":
                fetch_status[venue] = {
                    "status": "SKIPPED",
                    "reason": "lighter_uses_existing_phase51b_phase51c_readonly_collectors_or_local_source_json",
                }
                continue
            try:
                if venue == "aster":
                    fetched = _fetch_aster(env, targets, timeout_s, limit, window_pad_ms)
                elif venue == "extended":
                    fetched = _fetch_extended(env, timeout_s, limit, max_pages)
                elif venue == "paradex":
                    fetched = _fetch_paradex(
                        env,
                        targets,
                        timeout_s,
                        limit,
                        window_pad_ms,
                        max_pages,
                        allow_jwt_cmd=allow_paradex_jwt_cmd,
                    )
                else:
                    fetched = []
                rows_by_venue.setdefault(venue, []).extend(fetched)
                fetch_status[venue] = {"status": "FETCHED", "row_count": len(fetched)}
            except Exception as exc:  # noqa: BLE001 - live API boundary must continue with evidence
                fetch_status[venue] = {"status": "ERROR", "error_type": type(exc).__name__, "message": str(exc)}

    output_rows, labels, status_counts = _sanitize_rows(
        run_id=run_id,
        timestamp_ns=timestamp_ns,
        rows_by_venue=rows_by_venue,
        target_index=target_index,
    )

    source_path = source_dir / "phase51z_forward_native_role_rows.jsonl"
    labels_path = out_dir / "phase51z_readonly_native_role_capture_labels.jsonl"
    candidate_manifest_path = out_dir / "phase51z_candidate_manifest.json"
    summary_path = out_dir / "phase51z_readonly_native_role_capture_summary.json"
    manifest_path = out_dir / "manifest.json"
    _write_jsonl(source_path, output_rows)
    _write_jsonl(labels_path, labels)
    _write_json(
        candidate_manifest_path,
        {
            "manifest_version": 1,
            "baseline_commit": BASELINE_COMMIT,
            "no_live_flag": True,
            "approved_for_live": False,
            "approved_for_canary": False,
            "approved_for_model_training": False,
            "approved_for_capital_escalation": False,
            "admissible_for_financial_claim": False,
            "admissible_for_ev_admission": False,
            "live_orders_allowed": False,
            "capital_change_allowed": False,
            "risk_limit_relaxation_allowed": False,
            "sources": [
                {
                    "source_id": "phase51z_forward_native_role_rows",
                    "venue_id": "all5",
                    "path": str(source_path),
                }
            ],
            "source_links": [],
        },
    )

    recovered_by_venue: dict[str, int] = {}
    for row in output_rows:
        venue = str(row.get("venue_id") or "unknown")
        recovered_by_venue[venue] = recovered_by_venue.get(venue, 0) + 1
    capture_diagnostics = _build_capture_diagnostics(
        targets=targets,
        rows_by_venue=rows_by_venue,
        output_rows=output_rows,
        labels=labels,
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51z_readonly_native_role_capture_nonlive_hold",
        "target_run": str(_resolve_path(target_run)),
        "target_summary_gate_status": target_summary.get("gate_status"),
        "native_role_target_count": len(targets),
        "native_role_target_counts_by_venue": _count_by_venue(targets),
        "target_time_windows_by_venue": _target_time_windows_by_venue(targets, window_pad_ms),
        "sanitized_source_row_count": len(output_rows),
        "sanitized_source_row_counts_by_venue": dict(sorted(recovered_by_venue.items())),
        "capture_label_count": len(labels),
        "capture_status_counts": status_counts,
        "capture_diagnostics_by_venue": capture_diagnostics,
        "local_source_file_counts": dict(sorted(source_file_counts.items())),
        "fetch_readonly_requested": fetch_readonly,
        "fetch_status": fetch_status,
        "credential_presence": credential_presence,
        "env_file_path_hash": _stable_hash(str(_resolve_path(env_file))) if env_file else None,
        "official_docs": OFFICIAL_DOCS,
        "candidate_manifest_path": str(candidate_manifest_path),
        "source_path": str(source_path),
        "source_sha256": _sha256_file(source_path),
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
    artifacts = [source_path, labels_path, candidate_manifest_path, summary_path]
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
            "baseline_commit": BASELINE_COMMIT,
            "gate_status": "HOLD",
            "artifacts": _artifact_infos(out_dir, artifacts),
        },
    )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--source-json", action="append", default=[], help="Local .json/.jsonl source, optionally venue=/path")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--fetch-readonly", action="store_true")
    parser.add_argument("--venue", action="append", choices=VENUES, default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"phase51z_readonly_native_role_capture_{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--window-pad-ms", type=int, default=30 * 60 * 1000)
    parser.add_argument("--allow-paradex-jwt-cmd", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source_json and not args.fetch_readonly:
        print("phase51z_readonly_native_role_capture: ERROR: provide --source-json or --fetch-readonly", file=sys.stderr)
        return 2
    timestamp_ns = args.timestamp_ns if args.timestamp_ns is not None else time.time_ns()
    try:
        out_dir = build_readonly_native_role_capture(
            target_run=args.target_run,
            source_json=args.source_json,
            env_file=args.env_file,
            fetch_readonly=args.fetch_readonly,
            venues=args.venue or list(VENUES),
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
            timeout_s=args.timeout_s,
            limit=args.limit,
            max_pages=args.max_pages,
            window_pad_ms=args.window_pad_ms,
            allow_paradex_jwt_cmd=args.allow_paradex_jwt_cmd,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"phase51z_readonly_native_role_capture: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51z_readonly_native_role_capture: wrote {out_dir}")
    print("phase51z_readonly_native_role_capture: status HOLD (read-only native role source only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
