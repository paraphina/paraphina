#!/usr/bin/env python3
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class StepBCommand:
    command: str
    env: Optional[Dict[str, str]] = None


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value[:80] or "command"


def last_non_empty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return "No output."


def run_command(
    repo_root: Path, evidence_dir: Path, idx: int, spec: StepBCommand
) -> Dict[str, object]:
    started_at = iso_now()
    slug = slugify(spec.command)
    evidence_path = evidence_dir / f"{idx:02d}_{slug}.log"
    env = os.environ.copy()
    if spec.env:
        env.update(spec.env)
    result = subprocess.run(
        spec.command,
        shell=True,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    evidence_path.write_text(result.stdout or "", encoding="utf-8")
    finished_at = iso_now()
    summary = last_non_empty_line(result.stdout or "")
    return {
        "command": spec.command,
        "exit_code": result.returncode,
        "summary": summary,
        "evidence": str(evidence_path.relative_to(repo_root)),
        "started_at": started_at,
        "finished_at": finished_at,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    evidence_dir = repo_root / "agent-tools"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    commands: List[StepBCommand] = [
        StepBCommand("cargo test --all"),
        StepBCommand("cargo test --features event_log"),
        StepBCommand("cargo test -p paraphina --features live"),
        StepBCommand("cargo test -p paraphina --features live,live_hyperliquid"),
        StepBCommand("cargo test -p paraphina --features live,live_lighter"),
        StepBCommand("python3 -m pytest -q"),
        StepBCommand("python3 tools/check_docs_truth_drift.py"),
        StepBCommand("python3 tools/check_docs_integrity.py"),
        StepBCommand(
            "cargo run --bin paraphina -- --ticks 300 --seed 42",
            env={
                "PARAPHINA_TELEMETRY_MODE": "jsonl",
                "PARAPHINA_TELEMETRY_PATH": "/tmp/telemetry.jsonl",
            },
        ),
        StepBCommand("python3 tools/check_telemetry_contract.py /tmp/telemetry.jsonl"),
    ]

    results = []
    for idx, spec in enumerate(commands, start=1):
        results.append(run_command(repo_root, evidence_dir, idx, spec))

    output_path = evidence_dir / "step_b_results.json"
    output_path.write_text(
        json.dumps(
            {"generated_at": iso_now(), "results": results}, indent=2, sort_keys=False
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
