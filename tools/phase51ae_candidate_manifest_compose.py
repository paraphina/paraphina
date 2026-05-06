#!/usr/bin/env python3
"""Compose HOLD-only Phase 5.1v candidate manifests.

This utility turns manual candidate-manifest stitching into a repo-owned,
deterministic gate. It only combines already-local source and source-link
artifacts. It does not infer source links, read secrets, place orders, enable
live/canary behavior, escalate capital, or relax risk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = "18dd09512288a85e440d3977e32432c3aabc1190"
DEFAULT_OUTPUT_ROOT = ROOT / "runs/phase51ae_candidate_manifest_compose"

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
    "signing_key",
    "token",
}

RAW_IDENTIFIER_FIELDS = {
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
    "decision_id",
    "fill_id",
    "fillId",
    "i",
    "id",
    "oid",
    "order_id",
    "order_id_str",
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

MANIFEST_ALLOWED_FIELDS = {
    "admissible_for_ev_admission",
    "admissible_for_financial_claim",
    "admissible_for_model_training",
    "approved_for_canary",
    "approved_for_capital_escalation",
    "approved_for_financial_claim",
    "approved_for_live",
    "approved_for_model_training",
    "baseline_commit",
    "capital_change_allowed",
    "live_orders_allowed",
    "manifest_version",
    "no_live_flag",
    "risk_limit_relaxation_allowed",
    "source_links",
    "sources",
}
SOURCE_ALLOWED_FIELDS = {"path", "source_id", "venue", "venue_id"}
SOURCE_LINK_ALLOWED_FIELDS = {"path", "source_link_id"}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _timestamp_ns_to_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _is_uri_like(value: str) -> bool:
    return "://" in value or value.startswith(("http:", "https:", "s3:", "gs:"))


def _is_env_path(path: Path) -> bool:
    return any(part == ".env" or part.endswith(".env") for part in path.parts)


def _check_no_symlink(path: Path) -> None:
    current = path if path.is_absolute() else _resolve_path(path)
    chain = [current]
    chain.extend(current.parents)
    for candidate in chain:
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"symlink path is prohibited: {candidate}")


def _check_local_path(path: Path, *, label: str) -> Path:
    raw = str(path)
    if _is_uri_like(raw):
        raise ValueError(f"network {label} path is prohibited: {path}")
    resolved = _resolve_path(path)
    if _is_env_path(resolved):
        raise ValueError(f"env files are prohibited as Phase 5.1ae {label} inputs")
    _check_no_symlink(resolved)
    return resolved


def _check_existing_artifact(path: Path, *, label: str) -> Path:
    resolved = _check_local_path(path, label=label)
    if not resolved.exists():
        raise ValueError(f"{label} path does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} path is not a file: {resolved}")
    if resolved.suffix not in {".json", ".jsonl"}:
        raise ValueError(f"{label} path must be .json or .jsonl: {resolved}")
    return resolved


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


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _check_no_raw_identifier_fields(record: dict[str, Any], path: Path, *, label: str) -> None:
    raw_fields = RAW_IDENTIFIER_FIELDS & set(record)
    if raw_fields:
        raise ValueError(f"{path} {label} leaked raw identifier fields: {sorted(raw_fields)}")


def _check_output_safe(record: dict[str, Any], path: Path) -> None:
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
        "admissible_for_model_training": False,
        "admissible_for_financial_claim": False,
        "admissible_for_ev_admission": False,
        "live_orders_allowed": False,
        "capital_change_allowed": False,
        "risk_limit_relaxation_allowed": False,
        "raw_identifier_redaction_status": "PASS",
    }


def _count_by_venue(sources: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in sources:
        venue = str(source.get("venue_id") or source.get("venue") or "unknown").lower()
        counts[venue] = counts.get(venue, 0) + 1
    return dict(sorted(counts.items()))


def _parse_source_spec(spec: str) -> dict[str, str]:
    parts = spec.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("--source must use SOURCE_ID=VENUE_ID=PATH")
    return {"source_id": parts[0], "venue_id": parts[1].lower(), "path": parts[2]}


def _parse_source_link_spec(spec: str) -> dict[str, str]:
    parts = spec.split("=", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError("--source-link must use SOURCE_LINK_ID=PATH")
    return {"source_link_id": parts[0], "path": parts[1]}


def _validate_manifest(manifest: dict[str, Any], path: Path) -> None:
    _check_unsafe_flags(manifest, path, label="candidate manifest")
    _check_no_secret_fields(manifest, path, label="candidate manifest")
    _check_no_raw_identifier_fields(manifest, path, label="candidate manifest")
    unexpected = sorted(set(manifest) - MANIFEST_ALLOWED_FIELDS)
    if unexpected:
        raise ValueError(f"{path} candidate manifest has unsupported fields: {unexpected}")
    baseline = manifest.get("baseline_commit")
    if baseline not in (None, BASELINE_COMMIT):
        raise ValueError(f"{path} baseline_commit mismatch")
    if manifest.get("no_live_flag") is False:
        raise ValueError(f"{path} has no_live_flag=false")
    if not isinstance(manifest.get("sources", []), list):
        raise ValueError(f"{path} sources must be a list")
    if not isinstance(manifest.get("source_links", []), list):
        raise ValueError(f"{path} source_links must be a list")


def _validate_source_entry(entry: dict[str, Any], origin: Path, index: int) -> dict[str, str]:
    _check_unsafe_flags(entry, origin, label=f"source[{index}]")
    _check_no_secret_fields(entry, origin, label=f"source[{index}]")
    _check_no_raw_identifier_fields(entry, origin, label=f"source[{index}]")
    unexpected = sorted(set(entry) - SOURCE_ALLOWED_FIELDS)
    if unexpected:
        raise ValueError(f"{origin} source[{index}] has unsupported fields: {unexpected}")
    source_id = str(entry.get("source_id") or "").strip()
    path_text = str(entry.get("path") or "").strip()
    venue_id = str(entry.get("venue_id") or entry.get("venue") or "unknown").lower()
    if not source_id:
        raise ValueError(f"{origin} source[{index}] missing source_id")
    if not path_text:
        raise ValueError(f"{origin} source[{index}] missing path")
    path = _check_existing_artifact(Path(path_text), label=f"source[{index}]")
    return {"source_id": source_id, "venue_id": venue_id, "path": str(path)}


def _validate_source_link_entry(entry: dict[str, Any], origin: Path, index: int) -> dict[str, str]:
    _check_unsafe_flags(entry, origin, label=f"source_links[{index}]")
    _check_no_secret_fields(entry, origin, label=f"source_links[{index}]")
    _check_no_raw_identifier_fields(entry, origin, label=f"source_links[{index}]")
    unexpected = sorted(set(entry) - SOURCE_LINK_ALLOWED_FIELDS)
    if unexpected:
        raise ValueError(f"{origin} source_links[{index}] has unsupported fields: {unexpected}")
    source_link_id = str(entry.get("source_link_id") or "").strip()
    path_text = str(entry.get("path") or "").strip()
    if not source_link_id:
        raise ValueError(f"{origin} source_links[{index}] missing source_link_id")
    if not path_text:
        raise ValueError(f"{origin} source_links[{index}] missing path")
    path = _check_existing_artifact(Path(path_text), label=f"source_links[{index}]")
    return {"source_link_id": source_link_id, "path": str(path)}


def _load_manifest_entries(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    resolved = _check_existing_artifact(path, label="candidate-manifest")
    payload = _load_json(resolved)
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved} candidate manifest must be a JSON object")
    _validate_manifest(payload, resolved)
    sources = [
        _validate_source_entry(entry, resolved, index)
        for index, entry in enumerate(payload.get("sources") or [])
        if isinstance(entry, dict)
    ]
    source_links = [
        _validate_source_link_entry(entry, resolved, index)
        for index, entry in enumerate(payload.get("source_links") or [])
        if isinstance(entry, dict)
    ]
    if len(sources) != len(payload.get("sources") or []):
        raise ValueError(f"{resolved} sources entries must be objects")
    if len(source_links) != len(payload.get("source_links") or []):
        raise ValueError(f"{resolved} source_links entries must be objects")
    return sources, source_links


def _dedupe_sources(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    by_path: dict[str, dict[str, str]] = {}
    for source in sources:
        key = str(Path(source["path"]).resolve())
        existing = by_path.get(key)
        if existing is not None and (
            existing["source_id"] != source["source_id"] or existing["venue_id"] != source["venue_id"]
        ):
            raise ValueError(f"source path {key} has conflicting source metadata")
        by_path[key] = source
    return sorted(by_path.values(), key=lambda item: (item["venue_id"], item["source_id"], item["path"]))


def _dedupe_source_links(source_links: list[dict[str, str]]) -> list[dict[str, str]]:
    by_path: dict[str, dict[str, str]] = {}
    for source_link in source_links:
        key = str(Path(source_link["path"]).resolve())
        existing = by_path.get(key)
        if existing is not None and existing["source_link_id"] != source_link["source_link_id"]:
            raise ValueError(f"source-link path {key} has conflicting source_link_id")
        by_path[key] = source_link
    return sorted(by_path.values(), key=lambda item: (item["source_link_id"], item["path"]))


def build_candidate_manifest_composition(
    *,
    candidate_manifests: list[Path],
    source_specs: list[str],
    source_link_specs: list[str],
    output_root: Path,
    run_id: str,
    timestamp_ns: int,
    target_run: Path | None,
) -> Path:
    if not candidate_manifests and not source_specs and not source_link_specs:
        raise ValueError("at least one candidate manifest, source, or source-link is required")

    output_root = _resolve_path(output_root)
    out_dir = output_root / _check_run_id(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, str]] = []
    source_links: list[dict[str, str]] = []
    input_manifest_infos: list[dict[str, Any]] = []
    for manifest_path in candidate_manifests:
        resolved = _check_existing_artifact(manifest_path, label="candidate-manifest")
        manifest_sources, manifest_source_links = _load_manifest_entries(resolved)
        sources.extend(manifest_sources)
        source_links.extend(manifest_source_links)
        input_manifest_infos.append(
            {
                "path_hash": _stable_hash(str(resolved)),
                "sha256": _sha256_file(resolved),
                "source_count": len(manifest_sources),
                "source_link_count": len(manifest_source_links),
            }
        )

    for seq, spec in enumerate(source_specs):
        sources.append(_validate_source_entry(_parse_source_spec(spec), Path(f"--source[{seq}]"), seq))
    for seq, spec in enumerate(source_link_specs):
        source_links.append(_validate_source_link_entry(_parse_source_link_spec(spec), Path(f"--source-link[{seq}]"), seq))

    sources = _dedupe_sources(sources)
    source_links = _dedupe_source_links(source_links)
    if not sources and source_links:
        raise ValueError("source-link-only composition is prohibited; at least one source is required")

    candidate_manifest_path = out_dir / "candidate_manifest.composed.json"
    labels_path = out_dir / "phase51ae_candidate_manifest_compose_labels.jsonl"
    summary_path = out_dir / "phase51ae_candidate_manifest_compose_summary.json"
    manifest_path = out_dir / "manifest.json"

    candidate_manifest = {
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
        "sources": sources,
        "source_links": source_links,
    }
    _write_json(candidate_manifest_path, candidate_manifest)

    labels: list[dict[str, Any]] = []
    for seq, source in enumerate(sources):
        label = _base_record(run_id, seq, timestamp_ns, "PHASE51AE_COMPOSED_SOURCE_LABEL")
        label.update(
            {
                "source_id": source["source_id"],
                "venue_id": source["venue_id"],
                "path_hash": _stable_hash(source["path"]),
                "sha256": _sha256_file(Path(source["path"])),
                "compose_status": "SOURCE_INCLUDED",
            }
        )
        labels.append(label)
    offset = len(labels)
    for seq, source_link in enumerate(source_links):
        label = _base_record(run_id, offset + seq, timestamp_ns, "PHASE51AE_COMPOSED_SOURCE_LINK_LABEL")
        label.update(
            {
                "source_link_id": source_link["source_link_id"],
                "path_hash": _stable_hash(source_link["path"]),
                "sha256": _sha256_file(Path(source_link["path"])),
                "compose_status": "SOURCE_LINK_INCLUDED",
            }
        )
        labels.append(label)
    _write_jsonl(labels_path, labels)

    phase51v_validation_command = None
    if target_run is not None:
        target_path = _check_local_path(target_run, label="target-run")
        if not target_path.exists() or not target_path.is_dir():
            raise ValueError(f"target-run path must be an existing directory: {target_path}")
        phase51v_validation_command = (
            "python3 tools/phase51v_forward_capture_bundle_readiness.py "
            f"--target-run {target_path} "
            f"--candidate-manifest {candidate_manifest_path} "
            "--output-root runs/phase51v_forward_capture_bundle_readiness "
            f"--run-id {run_id}-PHASE51V-COMPOSED-HOLD"
        )

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
        "baseline_commit": BASELINE_COMMIT,
        "gate_status": "HOLD",
        "gate_reason": "phase51ae_candidate_manifest_composed_nonlive_hold",
        "input_candidate_manifest_count": len(input_manifest_infos),
        "input_candidate_manifest_infos": input_manifest_infos,
        "direct_source_spec_count": len(source_specs),
        "direct_source_link_spec_count": len(source_link_specs),
        "source_count": len(sources),
        "source_counts_by_venue": _count_by_venue(sources),
        "source_link_count": len(source_links),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": _sha256_file(candidate_manifest_path),
        "phase51v_validation_command": phase51v_validation_command,
        "next_required_action": "run_phase51v_against_composed_candidate_manifest",
        "promotion_boundary": "Phase 5.1v target-ready counts only",
        "clears_phase51_blockers": False,
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
    _write_json(summary_path, summary)
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "generated_at_utc": _timestamp_ns_to_utc(timestamp_ns),
            "baseline_commit": BASELINE_COMMIT,
            "gate_status": "HOLD",
            "artifacts": [
                {
                    "path": path.relative_to(out_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
                for path in (candidate_manifest_path, labels_path, summary_path)
            ],
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
        },
    )
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", action="append", type=Path, default=[])
    parser.add_argument("--source", action="append", default=[], help="SOURCE_ID=VENUE_ID=PATH")
    parser.add_argument("--source-link", action="append", default=[], help="SOURCE_LINK_ID=PATH")
    parser.add_argument("--target-run", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"PHASE51AE-CANDIDATE-MANIFEST-COMPOSE-HOLD-{_utc_stamp()}")
    parser.add_argument("--timestamp-ns", type=int)
    args = parser.parse_args()

    timestamp_ns = args.timestamp_ns
    if timestamp_ns is None:
        timestamp_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    try:
        out_dir = build_candidate_manifest_composition(
            candidate_manifests=args.candidate_manifest,
            source_specs=args.source,
            source_link_specs=args.source_link,
            output_root=args.output_root,
            run_id=args.run_id,
            timestamp_ns=timestamp_ns,
            target_run=args.target_run,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should fail closed with a concise error
        print(f"phase51ae_candidate_manifest_compose: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"phase51ae_candidate_manifest_compose: wrote {out_dir}")
    print("phase51ae_candidate_manifest_compose: status HOLD (candidate manifest composition only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
