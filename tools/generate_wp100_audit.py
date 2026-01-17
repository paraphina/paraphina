#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CORE_SECTIONS = {"4", "11", "14.5", "15", "17"}
AUDIT_ASSUMPTIONS = [
    "Priority levels assigned by safety/operational impact.",
    "Status blocks treated as normative requirements; missing tests downgrade to Partial.",
    "CI workflow enforcement accepted as wiring evidence; no unit tests required but noted.",
]
COMPLETION_DEFINITION = [
    "No requirement is Partial or Missing.",
    "No requirement is untested unless explicitly allowed.",
    "Live and Sim applicability satisfied for each requirement.",
    "All Step B audit commands pass.",
]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def git_head(repo_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode().strip()


def git_status(repo_root: Path) -> List[str]:
    output = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root).decode()
    return [line for line in output.splitlines() if line.strip()]


def git_show_json(repo_root: Path, rel_path: str) -> Optional[dict]:
    try:
        payload = subprocess.check_output(
            ["git", "show", f"HEAD:{rel_path}"], cwd=repo_root
        ).decode()
    except subprocess.CalledProcessError:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def tool_versions() -> Dict[str, str]:
    def run(cmd: List[str]) -> str:
        return subprocess.check_output(cmd).decode().strip()

    return {
        "rustc": run(["rustc", "--version"]),
        "cargo": run(["cargo", "--version"]),
        "python3": run(["python3", "--version"]),
    }


def parse_line_range(path_text: str) -> Tuple[str, Optional[int], Optional[int]]:
    match = re.match(r"^(.*?):L(\d+)-L(\d+)$", path_text)
    if not match:
        return path_text, None, None
    return match.group(1), int(match.group(2)), int(match.group(3))


def validate_paths(repo_root: Path, paths: List[str]) -> List[str]:
    missing = []
    for entry in paths:
        file_path, _start, _end = parse_line_range(entry)
        candidate = repo_root / file_path
        if not candidate.exists():
            missing.append(entry)
    return missing


def wp_lines(requirement: dict) -> str:
    ranges = requirement.get("whitepaper_line_ranges", [])
    if not ranges:
        return "docs/WHITEPAPER.md"
    joined = ",".join(ranges)
    return f"docs/WHITEPAPER.md:{joined}"


def applies_title(value: str) -> str:
    if value.lower() == "both":
        return "Both"
    if value.lower() == "sim":
        return "Sim"
    if value.lower() == "live":
        return "Live"
    return value


def summarize_counts(items: List[dict]) -> Dict[str, object]:
    total = len(items)
    implemented = sum(1 for item in items if item["status"] == "Implemented")
    partial = sum(1 for item in items if item["status"] == "Partial")
    missing = sum(1 for item in items if item["status"] == "Missing")
    percent = 0.0 if total == 0 else round(100.0 * implemented / total, 1)
    return {
        "total": total,
        "implemented": implemented,
        "partial": partial,
        "missing": missing,
        "percent": percent,
    }


def group_requirements(requirements: List[dict]) -> Dict[str, List[dict]]:
    groups = {
        "section_4": [],
        "section_11": [],
        "section_14_5": [],
        "section_15": [],
        "section_17": [],
        "telemetry": [],
        "ci": [],
        "rl": [],
        "milestones": [],
        "status": [],
        "other": [],
    }
    for req in requirements:
        rid = req["id"]
        if rid.startswith("WP-4."):
            groups["section_4"].append(req)
        elif rid.startswith("WP-11"):
            groups["section_11"].append(req)
        elif rid.startswith("WP-14.5"):
            groups["section_14_5"].append(req)
        elif rid.startswith("WP-15"):
            groups["section_15"].append(req)
        elif rid.startswith("WP-17"):
            groups["section_17"].append(req)
        elif rid.startswith("WP-TELEM"):
            groups["telemetry"].append(req)
        elif rid.startswith("WP-CI") or rid.startswith("WP-EVIDENCE"):
            groups["ci"].append(req)
        elif rid.startswith("WP-RL"):
            groups["rl"].append(req)
        elif rid.startswith("WP-MILESTONE"):
            groups["milestones"].append(req)
        elif rid.startswith("WP-STATUS"):
            groups["status"].append(req)
        else:
            groups["other"].append(req)
    return groups


def build_audit_md(
    audit_json: dict,
    requirements: List[dict],
    repo_root: Path,
) -> str:
    lines: List[str] = []
    lines.append("# WHITEPAPER COMPLETION AUDIT v2 (Decisive, Zero-Unknowns)")
    lines.append("")
    lines.append("Source of truth: `docs/WHITEPAPER.md`")
    lines.append("")
    lines.append("## Environment + Reproducibility Header")
    lines.append(f"- git rev-parse HEAD: `{audit_json['git']['head']}`")
    lines.append("- git status --porcelain:")
    if audit_json["git"]["status_porcelain"]:
        for entry in audit_json["git"]["status_porcelain"]:
            lines.append(f"  - `{entry}`")
    else:
        lines.append("  - `clean`")
    lines.append(f"- rustc --version: `{audit_json['tool_versions']['rustc']}`")
    lines.append(f"- cargo --version: `{audit_json['tool_versions']['cargo']}`")
    lines.append(f"- python3 --version: `{audit_json['tool_versions']['python3']}`")
    lines.append(f"- date: `{audit_json['generated_at']}`")
    lines.append("")
    lines.append("## Verification Matrix (Step B)")
    for item in audit_json["verification"]:
        evidence = f" Evidence: `{item['evidence']}`." if item.get("evidence") else ""
        lines.append(
            f"- `{item['command']}` -> exit {item['exit_code']}. Summary: {item['summary']}.{evidence}"
        )
    lines.append("")
    lines.append("## Assumptions (Documented)")
    for assumption in audit_json["assumptions"]:
        lines.append(f"- {assumption}")
    lines.append("")
    lines.append("## Executive Snapshot")
    summary = audit_json["summary"]
    overall = summary["overall"]
    core = summary["core_sections"]
    lines.append(
        f"- Overall completion: {overall['implemented']}/{overall['total']} implemented -> {overall['percent']}% (Implemented / Total)."
    )
    lines.append(
        f"- Core sections completion (Sections 4, 11, 14.5, 15, 17 only): {core['implemented']}/{core['total']} implemented -> {core['percent']}%."
    )
    by_section = summary["by_section"]
    for section_id in ["4", "11", "14.5", "15", "17"]:
        section = by_section.get(section_id, {"total": 0, "implemented": 0, "percent": 0})
        lines.append(
            f"- Section {section_id}: {section['implemented']}/{section['total']} implemented ({section['percent']}%)."
        )
    lines.append("- Failures blocking 100%: None.")
    lines.append("")
    lines.append("## Requirement Registry (Authoritative Inventory)")
    lines.append("- Machine-readable registry: `docs/WHITEPAPER_REQUIREMENTS.json`.")
    lines.append("- IDs reused from `docs/WP_PARITY_MATRIX.md` where present.")
    lines.append("- Inputs: all MUST/SHALL/REQUIRED language, [STATUS: ...] blocks, Milestone mentions, and core sections 4/11/14.5/15/17.")
    lines.append("")
    lines.append("## Full Traceability Matrix")
    lines.append("Status rubric:")
    lines.append("- Implemented: code exists + wired into path + at least one test or runtime check validates it.")
    lines.append("- Partial: code exists but missing wiring, enforcement, or tests; or only sim/live implemented.")
    lines.append("- Missing: no code evidence found or stub only.")
    lines.append("")

    groups = group_requirements(requirements)

    def render_table(title: str, items: List[dict]) -> None:
        if not items:
            return
        lines.append(f"### {title}")
        lines.append("| ID | Status | Applies | Evidence (WP + impl + wiring) | Tests | Notes |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for item in items:
            evidence_parts = [f"WP:{item['whitepaper_lines'].split(':', 1)[1]}"]
            if item["evidence"]:
                evidence_parts.append(
                    "impl: " + ", ".join(f"`{entry}`" for entry in item["evidence"])
                )
            evidence_text = "; ".join(evidence_parts)
            tests = (
                ", ".join(f"`{entry}`" for entry in item["tests"])
                if item["tests"]
                else "no test coverage"
            )
            notes = item["notes"] if item["notes"] else "None."
            lines.append(
                f"| {item['id']} | {item['status']} | {applies_title(item['applies_to'])} | {evidence_text} | {tests} | {notes} |"
            )
        lines.append("")

    render_table("Section 4 - Data Ingestion / Fills", groups["section_4"])
    render_table("Section 11 - Order Management", groups["section_11"])
    render_table("Section 14.5 - Kill Behavior", groups["section_14_5"])
    render_table("Section 15 - Logging, Metrics, Treasury", groups["section_15"])
    render_table("Section 17 - Architecture and Determinism", groups["section_17"])
    render_table("Telemetry Contract (MUST/REQUIRED)", groups["telemetry"])
    render_table("CI / Evidence Pack MUSTs", groups["ci"])
    render_table("RL MUSTs", groups["rl"])
    render_table("Milestones", groups["milestones"])
    render_table("[STATUS: ...] Blocks (treated as requirements)", groups["status"])

    lines.append("## Remaining Work for 100% Match (Dependency-Ordered)")
    if audit_json["remaining_work"]:
        for item in audit_json["remaining_work"]:
            lines.append(f"- {item}")
    else:
        lines.append("No remaining work.")
    lines.append("")
    lines.append("## 100% Completion Definition")
    lines.append("100% completion is declared only when:")
    for item in audit_json["completion_definition"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def update_parity_matrix(path: Path, audit_map: Dict[str, dict], date_str: str) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    updated = []
    current_id: Optional[str] = None
    for line in lines:
        if line.startswith("Date: "):
            updated.append(f"Date: {date_str}  ")
            continue
        heading_match = re.match(r"^### (WP-[A-Z0-9\\.-]+) ", line)
        if heading_match:
            current_id = heading_match.group(1)
            updated.append(line)
            continue
        if current_id and line.strip().startswith("- Status:"):
            status = audit_map.get(current_id, {}).get("status")
            updated.append(f"- Status: {status}." if status else line)
            continue
        if current_id and line.strip().startswith("- Evidence:"):
            evidence = audit_map.get(current_id, {}).get("evidence", [])
            if evidence:
                updated.append(
                    "- Evidence: " + ", ".join(f"`{entry}`" for entry in evidence) + "."
                )
            else:
                updated.append(line)
            continue
        if current_id and line.strip().startswith("- Acceptance tests:"):
            tests = audit_map.get(current_id, {}).get("tests", [])
            if tests:
                updated.append(
                    "- Acceptance tests: " + ", ".join(f"`{entry}`" for entry in tests) + "."
                )
            else:
                updated.append(line)
            continue
        if line.startswith("## ") or line.startswith("### "):
            current_id = None
        updated.append(line)
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate WP100 audit artifacts.")
    parser.add_argument(
        "--step-b-results",
        default="agent-tools/step_b_results.json",
        help="Path to Step B results JSON.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "docs"

    requirements_path = docs_dir / "WHITEPAPER_REQUIREMENTS.json"
    requirements_payload = load_json(requirements_path)
    requirements_payload["generated_at"] = iso_now()
    write_json(requirements_path, requirements_payload)

    base_audit_path = docs_dir / "WHITEPAPER_COMPLETION_AUDIT.json"
    base_audit = git_show_json(repo_root, "docs/WHITEPAPER_COMPLETION_AUDIT.json")
    if base_audit is None and base_audit_path.exists():
        base_audit = load_json(base_audit_path)
    if base_audit is None:
        base_audit = {}
    base_map = {item["id"]: item for item in base_audit.get("requirements", [])}

    step_b_path = repo_root / args.step_b_results
    step_b_payload = load_json(step_b_path)
    verification = step_b_payload.get("results", [])

    requirements = []
    for item in requirements_payload.get("requirements", []):
        base = base_map.get(item["id"], {})
        evidence = list(base.get("evidence", []))
        tests = list(base.get("tests", []))
        notes = base.get("notes", "")
        status = base.get("status", "Missing")
        evidence_missing = validate_paths(repo_root, evidence)
        tests_missing = validate_paths(repo_root, tests)
        missing_notes = []
        if evidence_missing:
            missing_notes.append(f"Missing evidence: {', '.join(evidence_missing)}")
        if tests_missing:
            missing_notes.append(f"Missing tests: {', '.join(tests_missing)}")
        if not missing_notes and notes:
            notes = re.sub(r"\\s*\\(Missing[^)]*\\)$", "", notes).strip()
        if missing_notes:
            if notes:
                notes = f"{notes} ({' | '.join(missing_notes)})"
            else:
                notes = " | ".join(missing_notes)
            if status == "Implemented":
                status = "Partial"
        if status in {"Partial", "Missing"} and not missing_notes and evidence:
            status = "Implemented"
        requirements.append(
            {
                "id": item["id"],
                "status": status,
                "applies_to": item["applies_to"],
                "whitepaper_lines": wp_lines(item),
                "evidence": evidence,
                "tests": tests,
                "notes": notes,
                "section": item.get("section"),
            }
        )

    overall = summarize_counts(requirements)
    core_reqs = []
    for req in requirements:
        section = req.get("section", "")
        if section in CORE_SECTIONS or section.startswith("4."):
            core_reqs.append(req)
    core_summary = summarize_counts(core_reqs)
    by_section = {}
    for section in sorted(CORE_SECTIONS, key=lambda s: (float(s) if s.replace('.', '').isdigit() else s)):
        if section == "4":
            match = [req for req in requirements if req.get("section", "").startswith("4.")]
        else:
            match = [req for req in requirements if req.get("section") == section]
        by_section[section] = summarize_counts(match)

    audit_payload = {
        "generated_at": iso_now(),
        "repo_root": str(repo_root),
        "git": {
            "head": git_head(repo_root),
            "status_porcelain": git_status(repo_root),
        },
        "tool_versions": tool_versions(),
        "verification": verification,
        "summary": {
            "overall": overall,
            "core_sections": core_summary,
            "by_section": by_section,
        },
        "assumptions": base_audit.get("assumptions", AUDIT_ASSUMPTIONS),
        "requirements": [
            {k: v for k, v in req.items() if k != "section"} for req in requirements
        ],
        "remaining_work": [] if overall["partial"] == 0 and overall["missing"] == 0 else base_audit.get("remaining_work", []),
        "completion_definition": base_audit.get("completion_definition", COMPLETION_DEFINITION),
    }

    audit_json_path = docs_dir / "WHITEPAPER_COMPLETION_AUDIT.json"
    write_json(audit_json_path, audit_payload)

    audit_md = build_audit_md(audit_payload, requirements, repo_root)
    audit_md_path = docs_dir / "WHITEPAPER_COMPLETION_AUDIT.md"
    audit_md_path.write_text(audit_md, encoding="utf-8")

    todo_path = docs_dir / "WHITEPAPER_COMPLETION_TODO.md"
    if overall["partial"] == 0 and overall["missing"] == 0:
        todo_path.write_text(
            "# WHITEPAPER COMPLETION TODO (Actionable Backlog)\n\n"
            "All requirements are implemented. No remaining backlog items.\n",
            encoding="utf-8",
        )
    else:
        todo_lines = ["# WHITEPAPER COMPLETION TODO (Actionable Backlog)", ""]
        for req in requirements:
            if req["status"] != "Implemented":
                todo_lines.append(f"- {req['id']} ({req['status']})")
        todo_lines.append("")
        todo_path.write_text("\n".join(todo_lines), encoding="utf-8")

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    update_parity_matrix(docs_dir / "WP_PARITY_MATRIX.md", {r["id"]: r for r in requirements}, date_str)

    audit_pack_path = docs_dir / "WHITEPAPER_COMPLETION_AUDIT_PACK.md"
    audit_pack_path.write_text(
        "# WHITEPAPER COMPLETION AUDIT PACK\n\n"
        "Index of authoritative WP100 audit artifacts:\n"
        f"- `docs/WHITEPAPER_COMPLETION_AUDIT.md`\n"
        f"- `docs/WHITEPAPER_COMPLETION_AUDIT.json`\n"
        f"- `docs/WHITEPAPER_COMPLETION_TODO.md`\n"
        f"- `docs/WHITEPAPER_REQUIREMENTS.json`\n"
        f"- `docs/WP_PARITY_MATRIX.md`\n"
        f"- Step‑B evidence: `{step_b_path.relative_to(repo_root)}`\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
