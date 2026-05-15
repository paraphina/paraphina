#!/usr/bin/env python3
"""Merge Phase 5.1 forward-refresh source-owner rows without mutating inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

FORBIDDEN_FIELD_NAMES = {
    "order_id",
    "orderId",
    "raw_order_id",
    "client_order_id",
    "clientOrderId",
    "cloid",
    "raw_client_order_id",
    "trade_id",
    "tradeId",
    "fill_id",
    "fillId",
    "tx_hash",
    "txHash",
    "hash",
    "api_key",
    "apiKey",
    "jwt",
    "token",
    "signed_payload",
    "signature",
    "secret",
}

FORBIDDEN_KEY_FRAGMENTS = (
    "secret",
    "api_key",
    "apikey",
    "jwt",
    "token",
    "signed_payload",
    "signature",
    ".env",
)

UNSAFE_TRUE_FLAGS = {
    "approved_for_live",
    "approved_for_canary",
    "approved_for_model_training",
    "approved_for_capital_escalation",
    "admissible_for_financial_claim",
    "admissible_for_ev_admission",
    "live_orders_allowed",
    "capital_change_allowed",
    "risk_limit_relaxation_allowed",
}


class MergeReject(Exception):
    pass


def reject_symlink_path(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current = current / part
        if current.exists() or current.is_symlink():
            if current.is_symlink():
                raise MergeReject(f"rejecting symlink path component: {current}")


def require_safe_path(path: Path, *, must_exist: bool) -> None:
    if must_exist and not path.exists():
        raise MergeReject(f"required input does not exist: {path}")
    reject_symlink_path(path)
    if path.exists() and not path.is_file():
        raise MergeReject(f"path is not a regular file: {path}")
    lower_parts = [part.lower() for part in path.parts]
    if any(part == ".env" or part.endswith(".env") for part in lower_parts):
        raise MergeReject(f"path must not reference .env content: {path}")


def reject_forbidden_fields(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower()
            if key in FORBIDDEN_FIELD_NAMES or any(
                fragment in normalized for fragment in FORBIDDEN_KEY_FRAGMENTS
            ):
                raise MergeReject(f"{location}: forbidden field {key}")
            if key in UNSAFE_TRUE_FLAGS and child is True:
                raise MergeReject(f"{location}: unsafe true flag {key}")
            reject_forbidden_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_fields(child, f"{location}[{index}]")


def row_key(row: dict[str, Any], location: str) -> tuple[str, str, str, str]:
    target_type = row.get("target_type")
    venue_id = row.get("venue_id")
    canonical_group_id = row.get("canonical_group_id")
    order_key = row.get("order_key")
    if row.get("no_live_flag") is not True:
        raise MergeReject(f"{location}: no_live_flag must be true")
    if target_type not in {"native_role", "lighter_native_limit"}:
        raise MergeReject(f"{location}: unsupported target_type {target_type!r}")
    for field, value in (
        ("venue_id", venue_id),
        ("canonical_group_id", canonical_group_id),
        ("order_key", order_key),
    ):
        if not isinstance(value, str) or not value.strip():
            raise MergeReject(f"{location}: missing non-empty {field}")
    return (
        str(target_type),
        str(venue_id).strip().lower(),
        str(canonical_group_id).strip(),
        str(order_key).strip(),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            location = f"{path}:{line_number}"
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise MergeReject(f"{location}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise MergeReject(f"{location}: row must be a JSON object")
            reject_forbidden_fields(row, location)
            row_key(row, location)
            rows.append(row)
    return rows


def reject_duplicates(rows: list[dict[str, Any]], source: str) -> set[tuple[str, str, str, str]]:
    seen: set[tuple[str, str, str, str]] = set()
    for index, row in enumerate(rows, start=1):
        key = row_key(row, f"{source}:{index}")
        if key in seen:
            raise MergeReject(f"{source}:{index}: duplicate row key {key}")
        seen.add(key)
    return seen


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() and path.is_symlink():
        raise MergeReject(f"output path is a symlink: {path}")
    output_parent = path.parent if str(path.parent) else Path(".")
    output_parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_path(output_parent)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    os.replace(tmp_path, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--remaining", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument(
        "--replace-base",
        action="store_true",
        help="after a successful merge, replace --base with --output atomically",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    try:
        for input_path in (args.base, args.remaining):
            require_safe_path(input_path, must_exist=True)
        require_safe_path(args.output, must_exist=False)
        require_safe_path(summary_path, must_exist=False)

        base_rows = load_jsonl(args.base)
        remaining_rows = load_jsonl(args.remaining)
        base_keys = reject_duplicates(base_rows, "base")
        remaining_keys = reject_duplicates(remaining_rows, "remaining")
        overlap = base_keys & remaining_keys
        if overlap:
            raise MergeReject(f"duplicate row key across base and remaining: {sorted(overlap)[0]}")

        merged_rows = base_rows + remaining_rows
        write_jsonl_atomic(args.output, merged_rows)
        summary = {
            "base_rows": len(base_rows),
            "remaining_rows": len(remaining_rows),
            "merged_rows": len(merged_rows),
            "output_path": str(args.output),
            "output_sha256": sha256_file(args.output),
            "base_replaced": bool(args.replace_base),
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_path(summary_path.parent)
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        if args.replace_base:
            reject_symlink_path(args.base)
            with tempfile.NamedTemporaryFile(
                "wb", dir=args.base.parent, delete=False
            ) as handle:
                tmp_base = Path(handle.name)
                with args.output.open("rb") as source:
                    shutil.copyfileobj(source, handle)
            os.replace(tmp_base, args.base)
        print(json.dumps(summary, sort_keys=True))
        return 0
    except MergeReject as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
