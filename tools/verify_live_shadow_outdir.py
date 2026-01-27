#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def read_outdir(arg_outdir: str | None) -> Path:
    if arg_outdir:
        return Path(arg_outdir)
    marker = Path("/tmp/paraphina_last_outdir.txt")
    if not marker.exists():
        raise RuntimeError("OUTDIR not provided and /tmp/paraphina_last_outdir.txt missing")
    return Path(marker.read_text().strip())


def count_log_lines(path: Path, needle: str) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if needle in line:
                count += 1
    return count


def run_contract_check(telemetry_path: Path) -> None:
    tool = Path(__file__).resolve().parent / "check_telemetry_contract.py"
    result = subprocess.run(
        [sys.executable, str(tool), str(telemetry_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "telemetry contract failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify all-5 live shadow OUTDIR")
    parser.add_argument("outdir", nargs="?", help="OUTDIR path (optional)")
    args = parser.parse_args()

    errors: list[str] = []
    try:
        outdir = read_outdir(args.outdir)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    run_log = outdir / "run.log"
    telemetry_path = outdir / "telemetry.jsonl"

    if not run_log.exists():
        errors.append(f"missing run.log: {run_log}")
    if not telemetry_path.exists():
        errors.append(f"missing telemetry.jsonl: {telemetry_path}")

    if errors:
        print("ERROR: " + "; ".join(errors), file=sys.stderr)
        return 1

    applied_count = count_log_lines(run_log, "APPLIED_BOOK venue=")
    if applied_count != 5:
        errors.append(f"APPLIED_BOOK count {applied_count} != 5")

    first_update_count = count_log_lines(run_log, "FIRST_BOOK_UPDATE venue=")
    if first_update_count != 5:
        errors.append(f"FIRST_BOOK_UPDATE count {first_update_count} != 5")

    try:
        run_contract_check(telemetry_path)
    except Exception as exc:
        errors.append(str(exc))

    last_line = None
    with telemetry_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if not last_line:
        errors.append("telemetry.jsonl contains no records")
    else:
        record = json.loads(last_line)
        venue_mid = record.get("venue_mid_usd", [])
        venue_spread = record.get("venue_spread_usd", [])
        venue_age = record.get("venue_age_ms", [])
        venue_status = record.get("venue_status", [])
        venue_toxicity = record.get("venue_toxicity", [])
        healthy_used = record.get("healthy_venues_used", [])

        for name, arr in [
            ("venue_mid_usd", venue_mid),
            ("venue_spread_usd", venue_spread),
            ("venue_age_ms", venue_age),
            ("venue_status", venue_status),
            ("venue_toxicity", venue_toxicity),
        ]:
            if not isinstance(arr, list) or len(arr) != 5:
                errors.append(f"{name} expected length 5, got {len(arr) if isinstance(arr, list) else 'non-list'}")

        if isinstance(venue_status, list) and isinstance(venue_age, list):
            expected_healthy = [
                idx
                for idx, status in enumerate(venue_status)
                if status == "Healthy"
                and idx < len(venue_age)
                and isinstance(venue_age[idx], (int, float))
                and venue_age[idx] >= 0
            ]
            if healthy_used != expected_healthy:
                errors.append(
                    f"healthy_venues_used {healthy_used} != expected {expected_healthy}"
                )

        print("Last telemetry per venue index:")
        for idx in range(5):
            mid = venue_mid[idx] if idx < len(venue_mid) else None
            spread = venue_spread[idx] if idx < len(venue_spread) else None
            age = venue_age[idx] if idx < len(venue_age) else None
            status = venue_status[idx] if idx < len(venue_status) else None
            tox = venue_toxicity[idx] if idx < len(venue_toxicity) else None
            if isinstance(age, (int, float)) and age < -1:
                errors.append(f"venue_age_ms[{idx}] negative: {age}")
            print(
                f"{idx}: mid={mid} spread={spread} age_ms={age} status={status} tox={tox}"
            )
        print(f"healthy_venues_used: {healthy_used}")

    if errors:
        print("FAIL", file=sys.stderr)
        print("ERROR: " + "; ".join(errors), file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
