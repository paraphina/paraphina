#!/usr/bin/env python3
"""
Integrity Gate: Unicode Control Character Scanner

Scans all git-tracked files for forbidden Unicode control characters that
can be used in Trojan-Source style attacks (invisible text manipulation,
bidirectional overrides, etc.).

Exit codes:
    0 - Clean: no forbidden characters found
    1 - Internal error
    2 - Forbidden characters detected

Usage:
    python3 tools/check_unicode_controls.py
"""

import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import NamedTuple


class Finding(NamedTuple):
    """A single forbidden character finding."""
    path: str
    line: int
    column: int
    codepoint: int
    name: str
    category: str
    line_content: str


# Forbidden Unicode control characters
# These are banned because they can be used for Trojan-Source attacks
# or to hide malicious content in code reviews.

FORBIDDEN_CODEPOINTS: set[int] = {
    # Bidi controls - can reverse text direction for code review attacks
    0x202A,  # LEFT-TO-RIGHT EMBEDDING
    0x202B,  # RIGHT-TO-LEFT EMBEDDING
    0x202C,  # POP DIRECTIONAL FORMATTING
    0x202D,  # LEFT-TO-RIGHT OVERRIDE
    0x202E,  # RIGHT-TO-LEFT OVERRIDE
    0x2066,  # LEFT-TO-RIGHT ISOLATE
    0x2067,  # RIGHT-TO-LEFT ISOLATE
    0x2068,  # FIRST STRONG ISOLATE
    0x2069,  # POP DIRECTIONAL ISOLATE
    # Direction marks - can affect text ordering
    0x200E,  # LEFT-TO-RIGHT MARK
    0x200F,  # RIGHT-TO-LEFT MARK
    0x061C,  # ARABIC LETTER MARK
    # Zero-width/invisible characters - can hide content
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0xFEFF,  # BYTE ORDER MARK (when not at file start, it's ZERO WIDTH NO-BREAK SPACE)
    # Soft hyphen - invisible in most contexts
    0x00AD,  # SOFT HYPHEN
}


def get_unicode_info(codepoint: int) -> tuple[str, str]:
    """Get Unicode name and category for a codepoint."""
    try:
        char = chr(codepoint)
        name = unicodedata.name(char, f"UNKNOWN-{codepoint:04X}")
        category = unicodedata.category(char)
    except ValueError:
        name = f"UNKNOWN-{codepoint:04X}"
        category = "??"
    return name, category


def format_line_with_placeholder(line: str, column: int, codepoint: int) -> str:
    """Replace the forbidden character with a visible placeholder."""
    placeholder = f"⟦U+{codepoint:04X}⟧"
    # column is 1-based, convert to 0-based index
    idx = column - 1
    if 0 <= idx < len(line):
        return line[:idx] + placeholder + line[idx + 1:]
    return line


def scan_content(content: str, path: str) -> list[Finding]:
    """Scan text content for forbidden Unicode characters."""
    findings: list[Finding] = []
    lines = content.split('\n')
    
    for line_num, line in enumerate(lines, start=1):
        for col_num, char in enumerate(line, start=1):
            codepoint = ord(char)
            if codepoint in FORBIDDEN_CODEPOINTS:
                name, category = get_unicode_info(codepoint)
                findings.append(Finding(
                    path=path,
                    line=line_num,
                    column=col_num,
                    codepoint=codepoint,
                    name=name,
                    category=category,
                    line_content=line,
                ))
    
    return findings


def scan_file(path: Path) -> list[Finding]:
    """Scan a single file for forbidden Unicode characters."""
    try:
        raw = path.read_bytes()
    except (OSError, IOError) as e:
        print(f"WARNING: Could not read {path}: {e}", file=sys.stderr)
        return []
    
    # Skip binary files (contain null bytes)
    if b'\x00' in raw:
        return []
    
    try:
        content = raw.decode('utf-8')
    except UnicodeDecodeError:
        # Try with replacement for partially valid UTF-8
        try:
            content = raw.decode('utf-8', errors='replace')
        except Exception:
            return []
    
    return scan_content(content, str(path))


def get_tracked_files() -> list[Path]:
    """Get list of git-tracked files using git ls-files."""
    try:
        result = subprocess.run(
            ['git', 'ls-files', '-z'],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR: git ls-files failed: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: git not found in PATH", file=sys.stderr)
        sys.exit(1)
    
    # Split on null bytes, filter empty strings
    files = [Path(f) for f in result.stdout.split('\0') if f]
    
    # Filter out runs/ directory (as specified)
    files = [f for f in files if not str(f).startswith('runs/')]
    
    return files


def print_finding(finding: Finding) -> None:
    """Print a single finding in the required format."""
    # Format: PATH:LINE:COLUMN U+XXXX <UNICODE_NAME> (<CATEGORY>)
    print(f"{finding.path}:{finding.line}:{finding.column} "
          f"U+{finding.codepoint:04X} {finding.name} ({finding.category})")
    
    # Show line with placeholder
    marked_line = format_line_with_placeholder(
        finding.line_content, 
        finding.column, 
        finding.codepoint
    )
    print(f"  {marked_line}")


def main() -> int:
    """Main entry point."""
    try:
        files = get_tracked_files()
    except Exception as e:
        print(f"ERROR: Failed to get tracked files: {e}", file=sys.stderr)
        return 1
    
    all_findings: list[Finding] = []
    
    for path in files:
        findings = scan_file(path)
        all_findings.extend(findings)
    
    if all_findings:
        # Sort findings for deterministic output
        all_findings.sort(key=lambda f: (f.path, f.line, f.column))
        
        print(f"FAILED: Found {len(all_findings)} forbidden Unicode control character(s):\n")
        for finding in all_findings:
            print_finding(finding)
            print()
        
        return 2
    
    print(f"OK: no forbidden Unicode controls found in {len(files)} tracked files.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

