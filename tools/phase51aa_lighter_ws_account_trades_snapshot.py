#!/usr/bin/env python3
"""Capture a read-only Lighter account-trades WebSocket snapshot.

This Phase 5.1aa utility is non-live and HOLD-only. It subscribes only to
read-only account snapshot channels, sanitizes raw trade/order identifiers
before writing to disk, and emits source rows that can be replayed through the
existing Phase 5.1z/5.1v gates.

It never sends jsonapi/sendtx, jsonapi/sendtxbatch, create, cancel, modify, or
any other order-affecting message.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import ssl
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase51b_lighter_account_limits import (
    BASELINE_COMMIT,
    DEFAULT_BASE_URL,
    DEFAULT_TESTNET_BASE_URL,
    RAW_IDENTIFIER_VALUE_KEYS,
    _artifact_infos,
    _as_int,
    _get_auth_token,
    _load_env_file,
    _redact,
    _resolve_base_url,
    _sha256_file,
    _stable_hash,
    _write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51aa_lighter_ws_account_trades_snapshot"
OFFICIAL_WS_DOC_URL = "https://apidocs.lighter.xyz/docs/websocket-reference"
READONLY_WS_CHANNELS = {"account_all", "account_all_trades"}
DEFAULT_WS_CHANNELS = ["account_all_trades"]
UNSAFE_TRUE_FLAGS = {
    "approved_for_model_training",
    "approved_for_live",
    "approved_for_canary",
    "approved_for_capital_escalation",
    "approved_for_financial_claim",
    "admissible_for_financial_claim",
    "admissible_for_ev_admission",
    "live_orders_allowed",
    "capital_change_allowed",
    "risk_limit_relaxation_allowed",
}
ALLOWED_RAW_ID_KEYS = {
    "accountid",
    "accountindex",
    "askaccountid",
    "bidaccountid",
    "marketid",
    "runid",
    "venueid",
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


class WebSocketProtocolError(RuntimeError):
    """Raised for a WebSocket protocol failure."""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


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
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object at {path}:{line_no}")
            yield line_no, record


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


def _normalized_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _raw_identifier_key_violations(value: Any, path: str = "$") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_str = str(key)
            normalized = _normalized_key(key)
            if normalized.endswith("sha256") or normalized in ALLOWED_RAW_ID_KEYS:
                pass
            elif (
                normalized in RAW_IDENTIFIER_VALUE_KEYS
                or normalized.endswith("id")
                or normalized.endswith("hash")
                or "cursor" in normalized
            ):
                violations.append(f"{path}.{key_str}")
            violations.extend(_raw_identifier_key_violations(item, f"{path}.{key_str}"))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            violations.extend(_raw_identifier_key_violations(item, f"{path}[{idx}]"))
    return violations


def _check_output_safe(record: dict[str, Any], path: Path) -> None:
    _check_unsafe_flags(record, path, label="output")
    _check_no_secret_fields(record, path, label="output")
    violations = _raw_identifier_key_violations(record)
    if violations:
        raise ValueError(f"{path} has raw identifier-like output fields: {violations[:10]}")


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
        "raw_identifier_redaction_status": "PASS",
    }


def _ws_url_from_base_url(base_url: str, readonly: bool) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported Lighter base URL scheme: {parsed.scheme}")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = "readonly=true" if readonly else ""
    return urllib.parse.urlunparse((scheme, parsed.netloc, "/stream", "", query, ""))


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise WebSocketProtocolError("socket closed while reading WebSocket frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_ws_frame(sock: socket.socket, payload: bytes, opcode: int = 0x1) -> None:
    first = 0x80 | opcode
    length = len(payload)
    header = bytearray([first])
    if length < 126:
        header.append(0x80 | length)
    elif length < (1 << 16):
        header.append(0x80 | 126)
        header.extend(length.to_bytes(2, "big"))
    else:
        header.append(0x80 | 127)
        header.extend(length.to_bytes(8, "big"))
    mask = os.urandom(4)
    masked = bytes(payload[idx] ^ mask[idx % 4] for idx in range(length))
    sock.sendall(bytes(header) + mask + masked)


def _recv_ws_message(sock: socket.socket) -> str | None:
    parts: list[bytes] = []
    opcode_initial: int | None = None
    while True:
        header = _recv_exact(sock, 2)
        first, second = header
        fin = bool(first & 0x80)
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = int.from_bytes(_recv_exact(sock, 2), "big")
        elif length == 127:
            length = int.from_bytes(_recv_exact(sock, 8), "big")
        mask = _recv_exact(sock, 4) if masked else b""
        payload = _recv_exact(sock, length) if length else b""
        if masked:
            payload = bytes(payload[idx] ^ mask[idx % 4] for idx in range(len(payload)))
        if opcode == 0x8:
            return None
        if opcode == 0x9:
            _send_ws_frame(sock, payload, opcode=0xA)
            continue
        if opcode == 0xA:
            continue
        if opcode in {0x1, 0x2}:
            opcode_initial = opcode
            parts = [payload]
        elif opcode == 0x0 and opcode_initial is not None:
            parts.append(payload)
        else:
            raise WebSocketProtocolError(f"unsupported WebSocket opcode {opcode}")
        if fin:
            data = b"".join(parts)
            if opcode_initial == 0x2:
                return data.decode("utf-8")
            return data.decode("utf-8")


def _connect_websocket(ws_url: str, timeout_s: float) -> socket.socket:
    parsed = urllib.parse.urlparse(ws_url)
    if parsed.scheme not in {"ws", "wss"}:
        raise ValueError(f"unsupported WebSocket URL scheme: {parsed.scheme}")
    host = parsed.hostname
    if not host:
        raise ValueError("WebSocket URL missing host")
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    raw_sock = socket.create_connection((host, port), timeout=timeout_s)
    raw_sock.settimeout(timeout_s)
    sock: socket.socket
    if parsed.scheme == "wss":
        context = ssl.create_default_context()
        sock = context.wrap_socket(raw_sock, server_hostname=host)
    else:
        sock = raw_sock
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {parsed.netloc}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "User-Agent: paraphina-phase51aa-readonly-ws/1\r\n"
        "\r\n"
    )
    sock.sendall(request.encode("ascii"))
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            raise WebSocketProtocolError("socket closed during WebSocket handshake")
        response += chunk
        if len(response) > 65536:
            raise WebSocketProtocolError("oversized WebSocket handshake response")
    header_text = response.split(b"\r\n\r\n", 1)[0].decode("iso-8859-1")
    lines = header_text.split("\r\n")
    if not lines or " 101 " not in lines[0]:
        raise WebSocketProtocolError(f"WebSocket handshake failed: {lines[0] if lines else '<empty>'}")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
    accept_expected = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
    ).decode("ascii")
    if headers.get("sec-websocket-accept") != accept_expected:
        raise WebSocketProtocolError("WebSocket accept header mismatch")
    return sock


def _handle_lighter_ws_payload(
    *,
    payload: dict[str, Any],
    send_json,
    subscribed: bool,
    account_index: int,
    auth_token: str,
    channels: list[str],
) -> tuple[bool, dict[str, Any] | None]:
    message_type = str(payload.get("type") or "")
    if message_type == "ping":
        send_json({"type": "pong"})
        return subscribed, None
    if not subscribed:
        if message_type != "connected":
            return subscribed, payload
        for channel in channels:
            subscription = {
                "type": "subscribe",
                "channel": f"{channel}/{account_index}",
            }
            if channel == "account_all_trades":
                subscription["auth"] = auth_token
            send_json(subscription)
        return True, None
    return subscribed, payload


def _fetch_ws_messages_with_optional_websockets(
    *,
    ws_url: str,
    account_index: int,
    auth_token: str,
    channels: list[str],
    timeout_s: float,
    max_messages: int,
) -> list[dict[str, Any]]:
    from websockets.sync.client import connect  # type: ignore

    messages: list[dict[str, Any]] = []
    for channel in channels:
        if channel not in READONLY_WS_CHANNELS:
            raise ValueError(f"unsupported read-only Lighter WS channel: {channel}")
    deadline = time.monotonic() + timeout_s
    subscribed = False
    with connect(
        ws_url,
        open_timeout=timeout_s,
        close_timeout=1,
        ping_interval=None,
        user_agent_header="paraphina-phase51aa-readonly-ws/1",
    ) as ws:
        while len(messages) < max_messages and time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                raw_message = ws.recv(timeout=remaining)
            except TimeoutError:
                break
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")
            payload = json.loads(raw_message)
            if not isinstance(payload, dict):
                continue
            subscribed, accepted = _handle_lighter_ws_payload(
                payload=payload,
                send_json=lambda data: ws.send(json.dumps(data, separators=(",", ":"))),
                subscribed=subscribed,
                account_index=account_index,
                auth_token=auth_token,
                channels=channels,
            )
            if accepted is not None:
                messages.append(accepted)
    return messages


def _fetch_ws_messages_raw(
    *,
    ws_url: str,
    account_index: int,
    auth_token: str,
    channels: list[str],
    timeout_s: float,
    max_messages: int,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    sock = _connect_websocket(ws_url, timeout_s)
    try:
        for channel in channels:
            if channel not in READONLY_WS_CHANNELS:
                raise ValueError(f"unsupported read-only Lighter WS channel: {channel}")
        deadline = time.monotonic() + timeout_s
        subscribed = False
        while len(messages) < max_messages and time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            sock.settimeout(remaining)
            try:
                text = _recv_ws_message(sock)
            except socket.timeout:
                break
            if text is None:
                break
            payload = json.loads(text)
            if isinstance(payload, dict):
                subscribed, accepted = _handle_lighter_ws_payload(
                    payload=payload,
                    send_json=lambda data: _send_ws_frame(
                        sock,
                        json.dumps(data, separators=(",", ":")).encode("utf-8"),
                    ),
                    subscribed=subscribed,
                    account_index=account_index,
                    auth_token=auth_token,
                    channels=channels,
                )
                if accepted is not None:
                    messages.append(accepted)
        try:
            _send_ws_frame(sock, b"", opcode=0x8)
        except OSError:
            pass
    finally:
        sock.close()
    return messages


def _fetch_ws_messages(
    *,
    ws_url: str,
    account_index: int,
    auth_token: str,
    channels: list[str],
    timeout_s: float,
    max_messages: int,
) -> list[dict[str, Any]]:
    try:
        return _fetch_ws_messages_with_optional_websockets(
            ws_url=ws_url,
            account_index=account_index,
            auth_token=auth_token,
            channels=channels,
            timeout_s=timeout_s,
            max_messages=max_messages,
        )
    except ImportError:
        return _fetch_ws_messages_raw(
            ws_url=ws_url,
            account_index=account_index,
            auth_token=auth_token,
            channels=channels,
            timeout_s=timeout_s,
            max_messages=max_messages,
        )


def _load_message_files(paths: list[Path]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for raw_path in paths:
        path = _resolve_path(raw_path)
        if path.suffix == ".jsonl":
            for _, row in _iter_jsonl(path):
                messages.append(row)
        elif path.suffix == ".json":
            payload = _load_json(path)
            if isinstance(payload, list):
                messages.extend(row for row in payload if isinstance(row, dict))
            elif isinstance(payload, dict):
                messages.append(payload)
            else:
                raise ValueError(f"unsupported JSON payload in {path}")
        else:
            raise ValueError(f"unsupported message file suffix: {path}")
    return messages


def _coerce_trade_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        if any(str(key).lower() in {"trade_id", "tradeid", "ask_id", "askid", "bid_id", "bidid"} for key in value):
            return [value]
        out: list[dict[str, Any]] = []
        for nested in value.values():
            out.extend(_coerce_trade_list(nested))
        return out
    return []


def _readonly_message_channel(message: dict[str, Any]) -> str | None:
    channel = str(message.get("channel") or "")
    message_type = str(message.get("type") or "")
    for readonly_channel in sorted(READONLY_WS_CHANNELS, key=len, reverse=True):
        if readonly_channel in channel or message_type in {
            f"update/{readonly_channel}",
            f"subscribed/{readonly_channel}",
        }:
            return readonly_channel
    return None


def _flatten_trade_rows(messages: list[dict[str, Any]], account_index: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in messages:
        source_channel = _readonly_message_channel(message)
        if source_channel is None:
            continue
        trades_payload = message.get("trades")
        if not isinstance(trades_payload, (dict, list)):
            continue
        if isinstance(trades_payload, dict):
            for market_key, trade_value in trades_payload.items():
                for trade in _coerce_trade_list(trade_value):
                    row = dict(trade)
                    row.setdefault("market_id", _as_int(market_key))
                    row["venue_id"] = "lighter"
                    row["source_channel"] = source_channel
                    row["account_index"] = account_index
                    rows.append(row)
        else:
            for trade in _coerce_trade_list(trades_payload):
                row = dict(trade)
                row["venue_id"] = "lighter"
                row["source_channel"] = source_channel
                row["account_index"] = account_index
                rows.append(row)
    return rows


def _message_metadata_records(messages: list[dict[str, Any]], *, run_id: str, timestamp_ns: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seq, message in enumerate(messages, start=1):
        trades_payload = message.get("trades")
        trade_market_count = len(trades_payload) if isinstance(trades_payload, dict) else None
        trade_row_count = len(_coerce_trade_list(trades_payload)) if isinstance(trades_payload, (dict, list)) else 0
        record = _base_record(run_id, seq, timestamp_ns, "PHASE51AA_LIGHTER_WS_MESSAGE_METADATA")
        record.update({
            "message_record_sha256": _stable_hash(message),
            "message_type": str(message.get("type") or ""),
            "source_channel": _readonly_message_channel(message),
            "top_level_keys": sorted(str(key) for key in message.keys()),
            "has_trades_field": "trades" in message,
            "trades_container_type": type(trades_payload).__name__ if trades_payload is not None else None,
            "trade_market_count": trade_market_count,
            "trade_row_count": trade_row_count,
            "capture_status": "SANITIZED_WS_MESSAGE_METADATA_EMITTED",
        })
        _check_output_safe(record, Path("phase51aa_sanitized_message_metadata"))
        records.append(record)
    return records


def _trade_timestamp_ms(row: dict[str, Any]) -> int | None:
    ts = _as_int(row.get("timestamp") or row.get("transaction_time") or row.get("created_at"))
    if ts is None:
        return None
    return ts * 1000 if ts < 10_000_000_000 else ts


def _sanitize_trade_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for row in rows:
        redacted = _redact(row)
        if not isinstance(redacted, dict):
            raise ValueError("sanitized trade row is not an object")
        redacted["venue_id"] = "lighter"
        redacted["source_record_sha256"] = _stable_hash(row)
        redacted.setdefault("approved_for_model_training", False)
        redacted.setdefault("approved_for_live", False)
        redacted.setdefault("approved_for_canary", False)
        redacted.setdefault("approved_for_capital_escalation", False)
        redacted.setdefault("admissible_for_financial_claim", False)
        redacted.setdefault("admissible_for_ev_admission", False)
        redacted.setdefault("live_orders_allowed", False)
        redacted.setdefault("capital_change_allowed", False)
        redacted.setdefault("risk_limit_relaxation_allowed", False)
        redacted.setdefault("raw_identifier_redaction_status", "PASS")
        _check_output_safe(redacted, Path("phase51aa_sanitized_trade_row"))
        sanitized.append(redacted)
    return sanitized


def _summarize_trades(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps = [ts for ts in (_trade_timestamp_ms(row) for row in rows) if ts is not None]
    return {
        "trade_count": len(rows),
        "timestamp_min_ms": min(timestamps) if timestamps else None,
        "timestamp_max_ms": max(timestamps) if timestamps else None,
    }


def build_ws_account_trades_snapshot(
    *,
    target_run: Path,
    env_file: Path | None,
    message_json: list[Path],
    output_root: Path | None,
    run_id: str | None,
    timestamp_ns: int | None,
    base_url: str | None,
    account_index: int | None,
    allow_sdk_auth: bool,
    lighter_sdk_path: Path | None,
    timeout_s: float,
    max_messages: int,
    readonly_url: bool,
    channels: list[str] | None,
) -> Path:
    run_id = run_id or f"PHASE51AA-LIGHTER-WS-ACCOUNT-TRADES-SNAPSHOT-{_utc_stamp()}"
    output_root = _resolve_path(output_root or DEFAULT_OUTPUT_ROOT)
    out_dir = output_root / run_id
    source_dir = out_dir / "source_snapshots"
    source_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ns = timestamp_ns or time.time_ns()
    created_utc = _timestamp_ns_to_utc(timestamp_ns)

    env = dict(os.environ)
    if env_file is not None:
        env.update(_load_env_file(_resolve_path(env_file)))
    resolved_account_index = account_index
    if resolved_account_index is None:
        configured = env.get("LIGHTER_ACCOUNT_INDEX")
        resolved_account_index = int(configured) if configured not in (None, "") else None
    if resolved_account_index is None:
        raise ValueError("LIGHTER_ACCOUNT_INDEX or --account-index is required")
    resolved_channels = list(dict.fromkeys(channels or DEFAULT_WS_CHANNELS))
    if not resolved_channels:
        raise ValueError("at least one read-only Lighter WS channel is required")
    unsupported_channels = sorted(set(resolved_channels) - READONLY_WS_CHANNELS)
    if unsupported_channels:
        raise ValueError(f"unsupported read-only Lighter WS channel(s): {unsupported_channels}")

    resolved_base_url = _resolve_base_url(env, base_url)
    ws_url = _ws_url_from_base_url(resolved_base_url, readonly_url)
    source_mode = "offline_message_json"
    fetch_status: dict[str, Any] = {"status": "SKIPPED", "reason": "offline_message_json_supplied"}
    if message_json:
        messages = _load_message_files(message_json)
    else:
        source_mode = "readonly_lighter_websocket"
        auth_token = _get_auth_token(env, allow_sdk_auth=allow_sdk_auth, sdk_path=lighter_sdk_path)
        try:
            messages = _fetch_ws_messages(
                ws_url=ws_url,
                account_index=resolved_account_index,
                auth_token=auth_token,
                channels=resolved_channels,
                timeout_s=timeout_s,
                max_messages=max_messages,
            )
            fetch_status = {"status": "FETCHED", "message_count": len(messages)}
        except Exception as exc:  # noqa: BLE001 - evidence boundary must record HOLD failures
            messages = []
            fetch_status = {"status": "ERROR", "error_type": type(exc).__name__, "message": str(exc)}

    _check_unsafe_flags(messages, Path("phase51aa_input_messages"), label="input message")
    _check_no_secret_fields(messages, Path("phase51aa_input_messages"), label="input message")
    raw_trade_rows = _flatten_trade_rows(messages, resolved_account_index)
    sanitized_trade_rows = _sanitize_trade_rows(raw_trade_rows)
    raw_identifier_violations = _raw_identifier_key_violations(sanitized_trade_rows)

    source_path = source_dir / "lighter_ws_account_trades.sanitized.jsonl"
    messages_path = source_dir / "lighter_ws_messages.sanitized.jsonl"
    labels_path = out_dir / "phase51aa_lighter_ws_account_trades_snapshot_labels.jsonl"
    candidate_manifest_path = out_dir / "phase51aa_candidate_manifest.json"
    summary_path = out_dir / "phase51aa_lighter_ws_account_trades_snapshot_summary.json"
    manifest_path = out_dir / "manifest.json"

    _write_jsonl(messages_path, _message_metadata_records(messages, run_id=run_id, timestamp_ns=timestamp_ns))
    _write_jsonl(source_path, sanitized_trade_rows)

    labels: list[dict[str, Any]] = []
    for seq, row in enumerate(sanitized_trade_rows, start=1):
        label = _base_record(run_id, seq, timestamp_ns, "PHASE51AA_LIGHTER_WS_ACCOUNT_TRADES_LABEL")
        label.update({
            "venue_id": "lighter",
            "source_record_sha256": row.get("source_record_sha256"),
            "capture_status": "SANITIZED_WS_TRADE_ROW_EMITTED",
            "source_channel": row.get("source_channel"),
            "market_id": row.get("market_id"),
            "raw_identifier_redaction_status": "PASS",
        })
        labels.append(label)
    _write_jsonl(labels_path, labels)

    _write_json(candidate_manifest_path, {
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
                "source_id": "phase51aa_lighter_ws_account_trades",
                "venue_id": "lighter",
                "path": str(source_path),
            }
        ],
        "source_links": [],
    })

    trade_summary = _summarize_trades(raw_trade_rows)
    ws_url_public = urllib.parse.urlparse(ws_url)
    ws_url_fingerprint = urllib.parse.urlunparse((
        ws_url_public.scheme,
        ws_url_public.netloc,
        ws_url_public.path,
        "",
        "readonly=true" if readonly_url else "",
        "",
    ))
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": created_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51aa_lighter_ws_account_trades_snapshot_nonlive_hold",
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
        "raw_identifier_redaction_status": "PASS" if not raw_identifier_violations else "FAIL",
        "raw_identifier_key_violation_count": len(raw_identifier_violations),
        "clears_phase51_blockers": False,
        "target_run": str(_resolve_path(target_run)),
        "source_mode": source_mode,
        "fetch_status": fetch_status,
        "requested_channels": resolved_channels,
        "account_index_present": True,
        "env_file_path_hash": _stable_hash(str(_resolve_path(env_file))) if env_file else None,
        "lighter_auth_token_present": bool(env.get("LIGHTER_AUTH_TOKEN", "").strip()),
        "lighter_api_private_key_present": bool(env.get("LIGHTER_API_PRIVATE_KEY_HEX", "").strip()),
        "ws_url": ws_url_fingerprint,
        "official_docs": [OFFICIAL_WS_DOC_URL],
        "message_count": len(messages),
        "trade_count": trade_summary["trade_count"],
        "timestamp_min_ms": trade_summary["timestamp_min_ms"],
        "timestamp_max_ms": trade_summary["timestamp_max_ms"],
        "source_path": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "candidate_manifest_path": str(candidate_manifest_path),
        "promotion_boundary": "Phase 5.1z/5.1v target-ready counts only",
    }
    if raw_identifier_violations:
        raise ValueError(f"sanitized output still contains raw identifier-like fields: {raw_identifier_violations[:10]}")
    _write_json(summary_path, summary)

    artifacts = [source_path, messages_path, labels_path, candidate_manifest_path, summary_path]
    _write_json(manifest_path, {
        "schema_version": 1,
        "created_utc": created_utc,
        "metadata": summary,
        "files": _artifact_infos(out_dir, artifacts),
    })
    artifact_index_path = out_dir / "evidence_pack" / "artifact_index.json"
    _write_json(artifact_index_path, {
        "schema_version": 1,
        "metadata": summary,
        "artifacts": _artifact_infos(out_dir, artifacts),
    })
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--message-json", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--account-index", type=int, default=None)
    parser.add_argument("--allow-sdk-auth", action="store_true")
    parser.add_argument("--lighter-sdk-path", type=Path, default=None)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--max-messages", type=int, default=4)
    parser.add_argument("--no-readonly-url", action="store_true")
    parser.add_argument(
        "--channel",
        choices=sorted(READONLY_WS_CHANNELS),
        action="append",
        default=None,
        help="Read-only Lighter account WebSocket channel to subscribe; repeat for multiple channels.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        out_dir = build_ws_account_trades_snapshot(
            target_run=args.target_run,
            env_file=args.env_file,
            message_json=args.message_json,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=args.timestamp_ns,
            base_url=args.base_url,
            account_index=args.account_index,
            allow_sdk_auth=args.allow_sdk_auth,
            lighter_sdk_path=args.lighter_sdk_path,
            timeout_s=args.timeout_s,
            max_messages=args.max_messages,
            readonly_url=not args.no_readonly_url,
            channels=args.channel,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"phase51aa_lighter_ws_account_trades_snapshot: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51aa_lighter_ws_account_trades_snapshot: wrote {out_dir}")
    print("phase51aa_lighter_ws_account_trades_snapshot: status HOLD (read-only WS source only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
