#!/usr/bin/env python3
"""Phase 5.1an source-owner forward-refresh capture scaffold.

This HOLD-only wrapper owns the source-owner inbox handoff for the Phase 5.1
forward-refresh route. It does not connect to venues, read env files, place or
cancel orders, call sendTx/sendTxBatch, infer joins, infer roles, infer Lighter
pressure, authorize live/canary/capital/risk changes, or make financial claims.

The only operator-authored evidence file is a sanitized local JSONL:

    <inbox>/forward_refresh.jsonl

When that file is empty, the tool emits a capture contract and an empty waiting
intake manifest. When it contains rows, the tool first safety-scans them, then
materializes a Phase 5.1al forward-refresh pack and writes an intake manifest
that points Phase 5.1am at the Phase 5.1al summary.
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

try:
    from phase51al_forward_refresh_capture_gate import build_forward_refresh_capture_gate
except ImportError:  # pragma: no cover - supports module execution from repo root
    from tools.phase51al_forward_refresh_capture_gate import build_forward_refresh_capture_gate


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_INBOX = Path("/home/ubuntu/source_owner_inbox/phase51")
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51an_source_owner_forward_refresh_capture"
DEFAULT_PHASE51AL_OUTPUT_ROOT = ROOT / "runs/phase51al_forward_refresh_capture_gate"

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
    "session_token",
    "signing_key",
    "token",
}

RAW_IDENTIFIER_FIELDS = {
    "ask_client_id",
    "ask_id",
    "bid_client_id",
    "bid_id",
    "client_id",
    "clientId",
    "client_order_id",
    "clientOrderId",
    "cloid",
    "decision_id",
    "fill_id",
    "fillId",
    "i",
    "id",
    "oid",
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


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _check_run_id(run_id: str) -> str:
    path = Path(run_id)
    if path.name != run_id or ".." in path.parts:
        raise ValueError("run_id must be a single local path segment")
    return run_id


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _path_text_is_unsafe(path_text: str) -> bool:
    return "://" in path_text or path_text.lower().startswith(("http:", "https:", "s3:", "gs:"))


def _path_contains_env_part(path: Path) -> bool:
    return any(part == ".env" or part.endswith(".env") for part in path.parts)


def _path_has_symlink(path: Path) -> bool:
    candidates = [path, *path.parents]
    for candidate in candidates:
        if candidate.exists() and candidate.is_symlink():
            return True
        if candidate == candidate.parent:
            break
    return False


def _check_local_path(path: Path, *, label: str, must_exist: bool = True) -> Path:
    if _path_text_is_unsafe(str(path)):
        raise ValueError(f"{label} must be a local filesystem path: {path}")
    resolved = path if path.is_absolute() else ROOT / path
    if _path_contains_env_part(resolved):
        raise ValueError(f"{label} must not reference .env files: {resolved}")
    if must_exist and not resolved.exists():
        raise ValueError(f"{label} does not exist: {resolved}")
    if resolved.exists() and _path_has_symlink(resolved):
        raise ValueError(f"{label} must not traverse symlinks: {resolved}")
    return resolved


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


def _check_safety(record: dict[str, Any], path: Path, *, label: str) -> None:
    for obj in _iter_dicts(record):
        for flag in UNSAFE_TRUE_FLAGS:
            if obj.get(flag) is True:
                raise ValueError(f"{path} has unsafe {label} flag {flag}=true")
        for key in obj:
            if _field_looks_secret(str(key)):
                raise ValueError(f"{path} has secret-shaped {label} field {key!r}")
        raw_fields = sorted(str(key) for key in obj if str(key) in RAW_IDENTIFIER_FIELDS)
        if raw_fields:
            raise ValueError(f"{path} leaked raw identifier {label} fields: {raw_fields}")


def _load_forward_refresh_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            _check_safety(row, path, label=f"input row {line_no}")
            rows.append(row)
    return rows


def _row_counts(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    by_target_type: dict[str, int] = {}
    by_target_type_venue: dict[str, int] = {}
    for row in rows:
        target_type = str(row.get("target_type") or "missing_target_type").lower()
        venue = str(row.get("venue_id") or row.get("venue") or "missing_venue").lower()
        by_target_type[target_type] = by_target_type.get(target_type, 0) + 1
        key = f"{target_type}:{venue}"
        by_target_type_venue[key] = by_target_type_venue.get(key, 0) + 1
    return dict(sorted(by_target_type.items())), dict(sorted(by_target_type_venue.items()))


def _intake_manifest(*, material_change_reason: str, phase51al_summary: Path | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "material_change_reason": material_change_reason,
        "phase51al_summaries": [str(phase51al_summary)] if phase51al_summary else [],
        "validated_mappings": [],
        "phase51aj_source_json": [],
        "phase51ab_pressure_jsonls": [],
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
    }


def _capture_contract(inbox: Path, forward_refresh_path: Path) -> str:
    return f"""# Phase 5.1an Source-Owner Forward-Refresh Capture Contract

Purpose: populate one sanitized source-owner JSONL file that Phase 5.1al can
materialize into a forward-refresh target pack.

Actual evidence file:

```text
{forward_refresh_path}
```

Required row families:

- `target_type=native_role`: include `venue_id`, `canonical_group_id` or
  `order_key`, and the venue-native maker/taker field.
- `target_type=lighter_native_limit`: include `venue_id=lighter`,
  `canonical_group_id` or `order_key`, active-order headroom, sendTx
  limit/remaining, REST-or-weighted limit/remaining, and
  `native_limit_event_time_status=EVENT_TIME_ALIGNED`.

Accepted native-role fields:

- Aster: `m` or `maker_side`, plus positive `l` or `lastFilledQty`.
- Extended: `isTaker` or `is_taker`.
- Hyperliquid: `crossed`.
- Lighter: `account_index`, `is_maker_ask`, `ask_account_id`,
  `bid_account_id`.
- Paradex: `liquidity` as `MAKER` or `TAKER`.

Forbidden in source-owner files:

- raw order, client, trade, fill, transaction, or venue identifiers;
- API keys, private keys, JWTs, tokens, signed payloads, or `.env` content;
- `approved_for_live=true`, `approved_for_canary=true`,
  `live_orders_allowed=true`, `capital_change_allowed=true`, or
  `risk_limit_relaxation_allowed=true`;
- inferred joins from time, price, size, account role, docs, or snapshots.

After rows are added, run:

```bash
python3 tools/phase51an_source_owner_forward_refresh_capture.py --inbox {inbox} --update-intake-manifest
```

Then run Phase 5.1am against:

```text
{inbox / "intake.json"}
```
"""


def _artifact_infos(root_dir: Path, paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        try:
            display = path.relative_to(root_dir).as_posix()
        except ValueError:
            display = str(path)
        records.append(
            {
                "path": display,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def build_source_owner_capture(
    *,
    inbox: Path,
    output_root: Path,
    phase51al_output_root: Path,
    run_id: str,
    phase51al_run_id: str | None,
    timestamp_ns: int,
    update_intake_manifest: bool,
) -> Path:
    run_id = _check_run_id(run_id)
    if phase51al_run_id is not None:
        phase51al_run_id = _check_run_id(phase51al_run_id)
    inbox = _check_local_path(inbox, label="source-owner inbox", must_exist=False)
    output_root = _check_local_path(output_root, label="output-root", must_exist=False)
    phase51al_output_root = _check_local_path(phase51al_output_root, label="Phase 5.1al output-root", must_exist=False)
    inbox.mkdir(parents=True, exist_ok=True)
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    forward_refresh_path = inbox / "forward_refresh.jsonl"
    if forward_refresh_path.exists() and _path_has_symlink(forward_refresh_path):
        raise ValueError(f"forward-refresh JSONL must not be a symlink: {forward_refresh_path}")
    if not forward_refresh_path.exists():
        forward_refresh_path.write_text("", encoding="utf-8")
    if forward_refresh_path.suffix != ".jsonl":
        raise ValueError(f"forward-refresh evidence must be a .jsonl file: {forward_refresh_path}")

    rows = _load_forward_refresh_rows(forward_refresh_path)
    by_target_type, by_target_type_venue = _row_counts(rows)
    row_count = len(rows)

    phase51al_summary_path: Path | None = None
    phase51al_run_dir: Path | None = None
    control_status = "AWAITING_SOURCE_OWNER_ROWS"
    next_required_action = "populate_forward_refresh_jsonl_with_sanitized_event_time_rows"
    material_change_reason = "awaiting sanitized source-owner forward-refresh rows"

    if row_count > 0:
        phase51al_run_id = phase51al_run_id or f"{run_id}-PHASE51AL-HOLD"
        phase51al_run_dir = build_forward_refresh_capture_gate(
            input_jsonl=forward_refresh_path,
            output_root=phase51al_output_root,
            run_id=phase51al_run_id,
            timestamp_ns=timestamp_ns,
        )
        phase51al_summary_path = phase51al_run_dir / "phase51al_forward_refresh_capture_summary.json"
        control_status = "PHASE51AL_FORWARD_REFRESH_PACK_MATERIALIZED"
        next_required_action = "run_phase51am_with_generated_intake_manifest"
        material_change_reason = "source owner supplied sanitized forward-refresh rows materialized by Phase 5.1al"

    intake_payload = _intake_manifest(
        material_change_reason=material_change_reason,
        phase51al_summary=phase51al_summary_path,
    )
    generated_intake_path = inbox / "intake.generated.json"
    _write_json(generated_intake_path, intake_payload)
    if update_intake_manifest:
        _write_json(inbox / "intake.json", intake_payload)

    capture_contract_path = inbox / "SOURCE_OWNER_CAPTURE_CONTRACT.md"
    _write_text(capture_contract_path, _capture_contract(inbox, forward_refresh_path))

    summary_path = out_dir / "phase51an_source_owner_forward_refresh_capture_summary.json"
    generated_at_utc = _timestamp_ns_to_utc(timestamp_ns)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": generated_at_utc,
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51an_source_owner_forward_refresh_capture_nonlive_hold",
        "control_status": control_status,
        "source_owner_inbox": str(inbox),
        "forward_refresh_jsonl": str(forward_refresh_path),
        "forward_refresh_jsonl_exists": forward_refresh_path.exists(),
        "forward_refresh_jsonl_sha256": _sha256_file(forward_refresh_path),
        "forward_refresh_row_count": row_count,
        "forward_refresh_row_counts_by_target_type": by_target_type,
        "forward_refresh_row_counts_by_target_type_venue": by_target_type_venue,
        "capture_contract_path": str(capture_contract_path),
        "generated_intake_manifest_path": str(generated_intake_path),
        "updated_intake_manifest_path": str(inbox / "intake.json") if update_intake_manifest else None,
        "phase51al_run_dir": str(phase51al_run_dir) if phase51al_run_dir else None,
        "phase51al_summary_path": str(phase51al_summary_path) if phase51al_summary_path else None,
        "next_required_action": next_required_action,
        "source_link_inference_allowed": False,
        "time_price_size_inference_allowed": False,
        "role_inference_allowed": False,
        "lighter_pressure_inference_allowed": False,
        "clears_phase51_blockers": False,
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
        "raw_identifier_redaction_status": "PASS",
    }
    _write_json(summary_path, summary)

    manifest_path = out_dir / "manifest.json"
    artifacts = [summary_path, generated_intake_path, capture_contract_path]
    if update_intake_manifest:
        artifacts.append(inbox / "intake.json")
    if phase51al_summary_path:
        artifacts.append(phase51al_summary_path)
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "generated_at_utc": generated_at_utc,
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
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--phase51al-output-root", type=Path, default=DEFAULT_PHASE51AL_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"PHASE51AN-SOURCE-OWNER-FORWARD-REFRESH-CAPTURE-HOLD-{_utc_stamp()}")
    parser.add_argument("--phase51al-run-id", default=None)
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--update-intake-manifest", action="store_true")
    args = parser.parse_args()

    try:
        out_dir = build_source_owner_capture(
            inbox=args.inbox,
            output_root=args.output_root,
            phase51al_output_root=args.phase51al_output_root,
            run_id=args.run_id,
            phase51al_run_id=args.phase51al_run_id,
            timestamp_ns=args.timestamp_ns or time.time_ns(),
            update_intake_manifest=args.update_intake_manifest,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should fail closed with concise stderr
        print(f"phase51an_source_owner_forward_refresh_capture: ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"phase51an_source_owner_forward_refresh_capture: wrote {out_dir}")
    print("phase51an_source_owner_forward_refresh_capture: status HOLD (source-owner capture scaffold only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
