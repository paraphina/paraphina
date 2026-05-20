#!/usr/bin/env python3
"""Extract a canonical V2 live order-path probe row with provenance.

The runtime V2 stream may contain the single admitted order-path probe row plus
subsequent HOLD rows. The authority validator intentionally rejects that mixed
stream as an ingestion artifact. This extractor creates the canonical single-row
artifact deterministically from the full stream and records the source hash and
lineage so the extraction is auditable rather than ad hoc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # Support direct `python3 tools/...` execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import v2_authority_decision_validator as validator


class V2OrderPathProbeExtractionError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_info(path: Path, artifact_root: Path) -> dict[str, Any]:
    data = path.read_bytes()
    try:
        rel = path.resolve().relative_to(artifact_root.resolve()).as_posix()
    except ValueError as exc:
        raise V2OrderPathProbeExtractionError(
            f"artifact must be under manifest root: {path}"
        ) from exc
    return {
        "path": rel,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise V2OrderPathProbeExtractionError(
                    f"line {line_no}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise V2OrderPathProbeExtractionError(f"line {line_no}: row must be object")
            rows.append((line_no, row))
    if not rows:
        raise V2OrderPathProbeExtractionError("no V2 decision rows found")
    return rows


def _is_probe_row(row: dict[str, Any]) -> bool:
    return row.get("order_path_probe_is_admission") is True


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def _linked_artifacts(run_root: Path | None, artifact_root: Path) -> list[dict[str, Any]]:
    if run_root is None:
        return []
    linked: list[dict[str, Any]] = []
    for name in [
        "build_info.json",
        "config_resolved.json",
        "summary.json",
        "telemetry.jsonl",
    ]:
        path = run_root / name
        if path.exists() and path.is_file():
            linked.append(_file_info(path, artifact_root))
    return linked


def extract_probe(
    source_decisions: Path,
    probe_output: Path,
    manifest_output: Path,
    *,
    run_root: Path | None = None,
) -> dict[str, Any]:
    artifact_root = manifest_output.parent
    rows = _read_jsonl(source_decisions)
    probe_rows = [(line_no, row) for line_no, row in rows if _is_probe_row(row)]
    if len(probe_rows) != 1:
        raise V2OrderPathProbeExtractionError(
            f"expected exactly one order-path probe row, found {len(probe_rows)}"
        )

    source_line_no, probe_row = probe_rows[0]
    tmp_output = probe_output.with_name(f".{probe_output.name}.tmp")
    try:
        _write_jsonl(tmp_output, probe_row)
        summary = validator.validate_v2_authority_decisions(tmp_output)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise
    if summary.live_canary_order_path_probe_rows != 1:
        tmp_output.unlink(missing_ok=True)
        raise V2OrderPathProbeExtractionError(
            "selected row did not validate as live canary order-path probe"
        )
    tmp_output.replace(probe_output)

    manifest = {
        "artifact_type": "v2_order_path_probe_ingestion_manifest",
        "schema_version": 1,
        "decision_validation_status": "pass",
        "source": _file_info(source_decisions, artifact_root),
        "output": _file_info(probe_output, artifact_root),
        "linked_artifacts": _linked_artifacts(run_root, artifact_root),
        "extraction": {
            "rule": (
                "select exactly one V2_ADMISSION_DECISION row where "
                "order_path_probe_is_admission=true, then validate the isolated "
                "row with tools.v2_authority_decision_validator"
            ),
            "source_line_no": source_line_no,
            "source_row_index_zero_based": source_line_no - 1,
            "source_decision_sha256": _sha256(source_decisions),
            "selected_authority_scope": probe_row.get("authority_scope"),
            "selected_admission_reason": probe_row.get("admission_reason"),
            "selected_order_intent_output_count": probe_row.get("order_intent_output_count"),
        },
        "governance": {
            "gate_status": "LIVE_CANARY_ORDER_PATH_PROBE",
            "probe_only": True,
            "approved_for_promotion": False,
            "approved_for_live": False,
            "approved_for_capital_escalation": False,
            "capital_change_allowed": False,
            "blocker_cleared": False,
            "pressure_complete_claim": False,
        },
        "v2_authority_contract": {
            "authority_scope": "live_canary_single_venue_order_path_probe",
            "can_filter_existing_intents": True,
            "can_create_new_intents": False,
            "can_mutate_live_orders": False,
            "baseline_intent_filter_only": True,
            "order_path_probe_only": True,
            "full_live_promotion": False,
            "fast_hedge_enabled": False,
        },
        "validation": summary.to_manifest_validation(),
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-decisions", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = extract_probe(
            args.source_decisions,
            args.probe_output,
            args.manifest_output,
            run_root=args.run_root,
        )
    except (V2OrderPathProbeExtractionError, validator.V2AuthorityValidationError) as exc:
        print(f"V2_ORDER_PATH_PROBE_EXTRACTION_FAIL: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        print(f"V2_ORDER_PATH_PROBE_EXTRACTOR_ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "V2_ORDER_PATH_PROBE_EXTRACTION_PASS "
        f"source_line_no={manifest['extraction']['source_line_no']} "
        f"output={args.probe_output} manifest={args.manifest_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
